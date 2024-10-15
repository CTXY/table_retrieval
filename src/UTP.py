from abc import abstractmethod
from typing import List, Dict, Union, Optional, Any

import logging
from pathlib import Path
from copy import deepcopy

import os
import numpy as np
from tqdm import tqdm

import pandas as pd

from haystack.schema import Document, FilterType
from haystack.document_stores import BaseDocumentStore
from haystack.nodes.retriever import DenseRetriever
from haystack.utils.early_stopping import EarlyStopping
from haystack.telemetry import send_event
from haystack.lazy_imports import LazyImport
from .modeling.model.language_model import get_language_model, UniEncoder
from .modeling.data_handler.data_silo import DataSiloForUni, StreamDataSilo
from .modeling.data_handler.processor import TripletSimilarityProcessor
from .modeling.model.adaptive_model import TripletModel
from .modeling.training.base import UniTrainer
from .modeling.model.prediction_head import TripletSimilarityHead, MLMHead
from transformers.models.tapas import TapasTokenizer
logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from torch.nn import DataParallel
    from torch.utils.data.sampler import SequentialSampler
    from transformers import AutoConfig, AutoTokenizer
    from haystack.modeling.data_handler.dataloader import NamedDataLoader
    from haystack.modeling.model.optimization import initialize_optimizer
    from haystack.modeling.utils import initialize_device_settings  # pylint: disable=ungrouped-imports



class UniversalTableRetriever(DenseRetriever):
    """
    Retriever that uses a single encoder (the same transformer for query and passage).
    See the original paper for more details:
    "Bridge the Gap between Language models and Tabular Understanding" (https://arxiv.org/pdf/2302.09302).
    """

    def __init__(
        self,
        document_store: Optional[BaseDocumentStore] = None,
        embedding_model: Union[Path, str] = "bert-base-uncased",
        model_version: Optional[str] = None,
        max_seq_len: int = 512,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        use_fast_tokenizers: bool = True,
        similarity_function: str = "dot_product",
        global_loss_buffer_size: int = 150000,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format


        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or remote name of encoder checkpoint. The format equals the
                                      one used by hugging-face transformers' modelhub models
                                      Currently available remote names: ``"bert-base-uncased"``
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param max_seq_len: Longest length of each sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
        :param top_k: How many documents to return per query.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_title: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
                            This is the approach used in the original paper and is likely to improve performance if your
                            titles contain meaningful information for retrieval (topic, entities etc.) .
                            The title is expected to be present in doc.meta["name"] and can be supplied in the documents
                            before writing them to the DocumentStore like this:
                            {"text": "my text", "meta": {"name": "my title"}}.
        :param use_fast_tokenizers: Whether to use fast Rust tokenizers
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
                        Note: as multi-GPU training is currently not implemented for DPR, training
                        will only use the first device provided in this list.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        torch_and_transformers_import.check()
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.scale_score = scale_score
        self.use_auth_token = use_auth_token

        if document_store and document_store.similarity != "dot_product":
            logger.warning(
                "You are using a Dense Passage Retriever model with the %s function. "
                "We recommend you use dot_product instead. "
                "This can be set when initializing the DocumentStore",
                document_store.similarity,
            )

        # Init & Load Encoders
        self.tokenizer = TapasTokenizer.from_pretrained(
            pretrained_model_name_or_path=embedding_model,
            revision=model_version,
            do_lower_case=True,
            use_fast=use_fast_tokenizers,
            use_auth_token=use_auth_token,
        )
        
        self.encoder = UniEncoder(
            pretrained_model_name_or_path=embedding_model,
            model_type="TapasForMaskedEncoder",
            use_auth_token=use_auth_token,
        )
        
        self.processor = TripletSimilarityProcessor(
            tokenizer=self.tokenizer,
            max_seq_len=max_seq_len,
            embed_title=embed_title,
            num_hard_negatives=1,
            num_positives=1,

        )

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path='/home/yangchenyu/pre-trained-models/google:tapas-base')

        prediction_heads = [
            TripletSimilarityHead(
                similarity_function=similarity_function, global_loss_buffer_size=global_loss_buffer_size
            ),
            MLMHead(
                hidden_size=config.hidden_size, vocab_size=config.vocab_size
            )
        ]

        self.model = TripletModel(
            language_model=self.encoder,
            prediction_heads=prediction_heads,
            embeds_dropout_prob=0.1,
            lm_output_types=["per_sequence", "per_tokens"],
            device=self.devices[0],
        )

        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=False)

        if len(self.devices) > 1:
            self.model = DataParallel(self.model, device_ids=self.devices)  # type: ignore [assignment]

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through the documents in a DocumentStore and return a small number of documents
        that are most relevant to the query.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic API_KEY'} for basic authentication)
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(queries=[query])
        documents = document_store.query_by_embedding(
            query_emb=query_emb[0], top_k=top_k, filters=filters, index=index, headers=headers, scale_score=scale_score
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through the documents in a DocumentStore and return a small number of documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).

                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic API_KEY'} for basic authentication)
        :param batch_size: Number of queries to embed at a time.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )

        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score

        query_embs: List[np.ndarray] = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(queries=batch))

        documents = document_store.query_by_embedding_batch(
            query_embs=query_embs, top_k=top_k, filters=filters, index=index, headers=headers, scale_score=scale_score
        )

        return documents

    def _get_predictions(self, dicts: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + formatting).

        :param dicts: list of dictionaries
        examples:[{'query': "where is florida?"}, {'query': "who wrote lord of the rings?"}, ...]
                [{'passages': [{
                    "title": 'Big Little Lies (TV series)',
                    "text": 'series garnered several accolades. It received..',
                    "label": 'positive',
                    "external_id": '18768923'},
                    {"title": 'Framlingham Castle',
                    "text": 'Castle on the Hill "Castle on the Hill" is a song by English..',
                    "label": 'positive',
                    "external_id": '19930582'}, ...]
        :return: dictionary of embeddings for "passages" and "query"
        """
        dataset, tensor_names, _, _ = self.processor.dataset_from_dicts(
            dicts, indices=list(range(len(dicts))), return_baskets=True
        )

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        query_embeddings_batched = []
        passage_embeddings_batched = []
        self.model.eval()

        # When running evaluations etc., we don't want a progress bar for every single query
        if len(dataset) == 1:
            disable_tqdm = True
        else:
            disable_tqdm = not self.progress_bar
       
        with tqdm(
            total=len(data_loader) * self.batch_size,
            unit=" Docs",
            desc="Create embeddings",
            position=1,
            leave=False,
            disable=disable_tqdm,
        ) as progress_bar:
            for raw_batch in data_loader:
                batch = {key: raw_batch[key].to(self.devices[0]) for key in raw_batch}

                # get logits
                with torch.inference_mode():
                    query_embeddings, passage_embeddings, _ = self.model.forward(
                        query_input_ids=batch.get("query_input_ids", None),
                        query_segment_ids=batch.get("query_segment_ids", None),
                        query_attention_mask=batch.get("query_attention_mask", None),
                        table_input_ids=batch.get("passage_input_ids", None),
                        table_segment_ids=batch.get("passage_segment_ids", None),
                        table_attention_mask=batch.get("passage_attention_mask", None),
                        text_table_input_ids=batch.get("text_table_input_ids", None),
                        text_table_segment_ids=batch.get("text_table_segment_ids", None),
                        text_table_attention_mask=batch.get("text_table_attention_mask", None),
                    )[0]
                    if query_embeddings is not None:
                        query_embeddings_batched.append(query_embeddings.cpu().numpy())
                    if passage_embeddings is not None:
                        passage_embeddings_batched.append(passage_embeddings.cpu().numpy())
                progress_bar.update(self.batch_size)

        all_embeddings: Dict[str, np.ndarray] = {}
        if passage_embeddings_batched:
            all_embeddings["passages"] = np.concatenate(passage_embeddings_batched)
        if query_embeddings_batched:
            all_embeddings["query"] = np.concatenate(query_embeddings_batched)
        return all_embeddings

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries using the query encoder.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        query_dicts = [{"query": q} for q in queries]
        result = self._get_predictions(query_dicts)["query"]
        return result

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents using the passage encoder.

        :param documents: List of documents to embed.
        :return: Embeddings of documents, one per input document, shape: (documents, embedding_dim)
        """
        if self.processor.num_hard_negatives != 0:
            logger.warning(
                "'num_hard_negatives' is set to %s, but inference does "
                "not require any hard negatives. Setting num_hard_negatives to 0.",
                self.processor.num_hard_negatives,
            )
            self.processor.num_hard_negatives = 0
        
        # 需要处理内容
        passages = [
            {
                "passages": [
                    {
                        "title": "",
                        "text": {
                            "title": d.meta["title"] if "title" in d.meta else "",
                            "columns": d.meta["columns"],
                            "rows": d.meta["rows"],
                        },
                        "label": d.meta["label"] if d.meta and "label" in d.meta else "positive",
                        "external_id": d.id,
                    }
                ]
            }
            for d in documents
        ]
        embeddings = self._get_predictions(passages)["passages"]
        return embeddings

    def train(
        self,
        data_dir: str,
        train_filename: str,
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        max_samples: Optional[int] = None,
        max_processes: int = 128,
        multiprocessing_strategy: Optional[str] = None,
        dev_split: float = 0,
        batch_size: int = 2,
        eval_batch_size: int = 32,
        embed_title: bool = True,
        num_hard_negatives: int = 1,
        num_positives: int = 1,
        n_epochs: int = 3,
        evaluate_every: int = 1000,
        n_gpu: int = 1,
        learning_rate: float = 1e-5,
        epsilon: float = 1e-08,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 100,
        grad_acc_steps: int = 1,
        use_amp: bool = False,
        optimizer_name: str = "AdamW",
        optimizer_correct_bias: bool = True,
        save_dir: str = "../saved_models/dpr",
        encoder_save_dir: str = "encoder",
        checkpoint_root_dir: Path = Path("model_checkpoints"),
        checkpoint_every: Optional[int] = None,
        checkpoints_to_keep: int = 3,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        """
        train a UniversalTableRetriever model
        :param data_dir: Directory where training file, dev file and test file are present
        :param train_filename: training filename
        :param dev_filename: development set filename, file to be used by model in eval step of training
        :param test_filename: test set filename, file to be used by model in test step after training
        :param max_samples: maximum number of input samples to convert. Can be used for debugging a smaller dataset.
        :param max_processes: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing or make debugging easier.
        :param multiprocessing_strategy: Set the multiprocessing sharing strategy, this can be one of file_descriptor/file_system depending on your OS.
                                         If your system has low limits for the number of open file descriptors, and you can’t raise them,
                                         you should use the file_system strategy.
        :param dev_split: The proportion of the train set that will be sliced. Only works if dev_filename is set to None
        :param batch_size: total number of samples in 1 batch of data
        :param embed_title: whether to concatenate passage title with each passage. The default setting in official DPR embeds passage title with the corresponding passage
        :param num_hard_negatives: number of hard negative passages(passages which are very similar(high score by BM25) to query but do not contain the answer
        :param num_positives: number of positive passages
        :param n_epochs: number of epochs to train the model on
        :param evaluate_every: number of training steps after evaluation is run
        :param n_gpu: number of gpus to train on
        :param learning_rate: learning rate of optimizer
        :param epsilon: epsilon parameter of optimizer
        :param weight_decay: weight decay parameter of optimizer
        :param grad_acc_steps: number of steps to accumulate gradient over before back-propagation is done
        :param use_amp: Whether to use automatic mixed precision (AMP) natively implemented in PyTorch to improve
                        training speed and reduce GPU memory usage.
                        For more information, see (Haystack Optimization)[https://haystack.deepset.ai/guides/optimization]
                        and (Automatic Mixed Precision Package - Torch.amp)[https://pytorch.org/docs/stable/amp.html].
        :param optimizer_name: what optimizer to use (default: AdamW)
        :param num_warmup_steps: number of warmup steps
        :param optimizer_correct_bias: Whether to correct bias in optimizer
        :param save_dir: directory where models are saved
        :param encoder_save_dir: directory inside save_dir where encoder model files are saved
        :param checkpoint_root_dir: The Path of a directory where all train checkpoints are saved. For each individual
                checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
        :param checkpoint_every: Save a train checkpoint after this many steps of training.
        :param checkpoints_to_keep: The maximum number of train checkpoints to save.
        :param early_stopping: An initialized EarlyStopping object to control early stopping and saving of the best models.

        Checkpoints can be stored via setting `checkpoint_every` to a custom number of steps.
        If any checkpoints are stored, a subsequent run of train() will resume training from the latest available checkpoint.
        """
        send_event(event_name="Training", event_properties={"class": self.__class__.__name__, "function_name": "train"})
        self.processor.embed_title = embed_title
        self.processor.data_dir = Path(data_dir)
        self.processor.train_filename = train_filename
        self.processor.dev_filename = dev_filename
        self.processor.test_filename = test_filename
        self.processor.max_samples = max_samples
        self.processor.dev_split = dev_split
        self.processor.num_hard_negatives = num_hard_negatives
        self.processor.num_positives = num_positives

        if isinstance(self.model, DataParallel):
            self.model.module.connect_heads_with_processor(self.processor.tasks, require_labels=True)  # type: ignore [operator]
        else:
            self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)

        data_silo = DataSiloForUni(
            processor=self.processor,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            distributed=False,
            max_processes=max_processes,
            multiprocessing_strategy=multiprocessing_strategy,
        )


        # 5. Create an optimizer
        self.model, optimizer, lr_schedule = initialize_optimizer(
            model=self.model,  # type: ignore [arg-type]
            learning_rate=learning_rate,
            optimizer_opts={
                "name": optimizer_name,
                "correct_bias": optimizer_correct_bias,
                "weight_decay": weight_decay,
                "eps": epsilon,
            },
            schedule_opts={"name": "LinearWarmup", "num_warmup_steps": num_warmup_steps},
            n_batches=len(data_silo.loaders["train"]), # data_silo.n_train_batches,
            n_epochs=n_epochs,
            grad_acc_steps=grad_acc_steps,
            device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
        )
        

        # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        trainer = UniTrainer.create_or_load_checkpoint(
            model=self.model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
            use_amp=use_amp,
            checkpoint_root_dir=Path(checkpoint_root_dir),
            checkpoint_every=checkpoint_every,
            checkpoints_to_keep=checkpoints_to_keep,
            early_stopping=early_stopping,
            evaluator_test=False,
        )

        # 7. Let it grow! Watch the tracked metrics live on experiment tracker (e.g. Mlflow)
        trainer.train()

        self.model.save(Path(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        
        if len(self.devices) > 1 and not isinstance(self.model, DataParallel):
            self.model = DataParallel(self.model, device_ids=self.devices)  # type: ignore [assignment]

    def save(
        self,
        save_dir: Union[Path, str],
        encoder_dir: str = "encoder",
    ):
        """
        Save DensePassageRetriever to the specified directory.

        :param save_dir: Directory to save to.
        :param query_encoder_dir: Directory in save_dir that contains query encoder model.
        :param passage_encoder_dir: Directory in save_dir that contains passage encoder model.
        :return: None
        """
        save_dir = Path(save_dir)
        self.model.save(Path(save_dir), lm_name=encoder_dir)
        save_dir = str(save_dir)
        self.tokenizer.save_pretrained(f"{save_dir}/{encoder_dir}")

    @classmethod
    def load(
        cls,
        load_dir: Union[Path, str],
        document_store: BaseDocumentStore,
        max_seq_len: int = 512,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        use_fast_tokenizers: bool = True,
        similarity_function: str = "dot_product",
        encoder_dir: str = "encoder",
    ):
        """
        Load DensePassageRetriever from the specified directory.
        """
        load_dir = Path(load_dir)
        dpr = cls(
            document_store=document_store,
            embedding_model=Path(load_dir) / encoder_dir,
            max_seq_len=max_seq_len,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_title=embed_title,
            use_fast_tokenizers=use_fast_tokenizers,
            similarity_function=similarity_function,
        )
        logger.info("DPR model loaded from %s", load_dir)

        return dpr

