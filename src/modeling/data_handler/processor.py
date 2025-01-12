from typing import Optional, Dict, List, Union, Any, Iterable, Type, IO, Tuple

import os
import json
import uuid
import inspect
import logging
import random
import tarfile
import tempfile
from pathlib import Path
from inspect import signature
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import requests
from tqdm import tqdm
from torch.utils.data import TensorDataset
import transformers
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers import BertTokenizer, TapasTokenizer
import numbers
from haystack.modeling.utils import flatten_list

from haystack.errors import HaystackError
from haystack.modeling.model.feature_extraction import (
    tokenize_batch_question_answering,
    tokenize_with_metadata,
    truncate_sequences,
)

from haystack.modeling.data_handler.samples import (
    Sample,
    SampleBasket,
    get_passage_offsets,
    offset_to_token_idx_vecorized,
)
from haystack.modeling.data_handler.input_features import sample_to_features_text
from haystack.utils.experiment_tracking import Tracker as tracker

DOWNSTREAM_TASK_MAP = {
    "squad20": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz",
    "covidqa": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/covidqa.tar.gz",
}
logger = logging.getLogger(__name__)


class Processor(ABC):
    """
    Base class for low level data processors to convert input text to PyTorch Datasets.
    """

    subclasses: dict = {}

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        train_filename: Optional[Union[Path, str]],
        dev_filename: Optional[Union[Path, str]],
        test_filename: Optional[Union[Path, str]],
        dev_split: float,
        data_dir: Optional[Union[Path, str]],
        tasks: Optional[Dict] = None,
        proxies: Optional[Dict] = None,
        multithreading_rust: Optional[bool] = True,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing test data.
        :param dev_split: The proportion of the train set that will be sliced. Only works if `dev_filename` is set to `None`.
        :param data_dir: The directory in which the train, test and perhaps dev files can be found.
        :param tasks: Tasks for which the processor shall extract labels from the input data.
                      Usually this includes a single, default task, e.g. text classification.
                      In a multitask setting this includes multiple tasks, e.g. 2x text classification.
                      The task name will be used to connect with the related PredictionHead.
        :param proxies: proxy configuration to allow downloads of remote datasets.
                    Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param multithreading_rust: Whether to allow multithreading in Rust, e.g. for FastTokenizers.
                                    Note: Enabling multithreading in Rust AND multiprocessing in python might cause
                                    deadlocks.
        """
        if tasks is None:
            tasks = {}
        if not multithreading_rust:
            os.environ["RAYON_RS_NUM_CPUS"] = "1"

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.proxies = proxies

        # data sets
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = None  

        self._log_params()
        self.problematic_sample_ids: set = set()

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific Processor implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(
        cls,
        processor_name: str,
        data_dir: str,  # TODO revert ignore
        tokenizer,  
        max_seq_len: int,
        train_filename: str,
        dev_filename: Optional[str],
        test_filename: str,
        dev_split: float,
        **kwargs,
    ):
        """
        Loads the class of processor specified by processor name.

        :param processor_name: The class of processor to be loaded.
        :param data_dir: Directory where data files are located.
        :param tokenizer: A tokenizer object
        :param max_seq_len: Sequences longer than this will be truncated.
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data.
                             If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing test data.
        :param dev_split: The proportion of the train set that will be sliced.
                          Only works if dev_filename is set to None
        :param kwargs: placeholder for passing generic parameters
        :return: An instance of the specified processor.
        """

        sig = signature(cls.subclasses[processor_name])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(
            "Got more parameters than needed for loading %s: %s. Those won't be used!", processor_name, unused_args
        )
        processor = cls.subclasses[processor_name](
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            **kwargs,
        )

        return processor

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Infers the specific type of Processor from a config file (e.g. SquadProcessor) and loads an instance of it.

        :param load_dir: directory that contains a 'processor_config.json'
        :return: An instance of a Processor Subclass (e.g. SquadProcessor)
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        with open(processor_config_file) as f:
            config = json.load(f)
        config["inference"] = True
        # init tokenizer
        if "lower_case" in config:
            logger.warning(
                "Loading tokenizer from deprecated config. "
                "If you used `custom_vocab` or `never_split_chars`, this won't work anymore."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                load_dir, tokenizer_class=config["tokenizer"], do_lower_case=config["lower_case"]
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(load_dir, tokenizer_class=config["tokenizer"])

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name=config["processor"], **config)

        for task_name, task in config["tasks"].items():
            processor.add_task(
                name=task_name,
                metric=task["metric"],
                label_list=task["label_list"],
                label_column_name=task["label_column_name"],
                text_column_name=task.get("text_column_name", None),
                task_type=task["task_type"],
            )

        if processor is None:
            raise Exception

        return processor

    @classmethod
    def convert_from_transformers(
        cls,
        tokenizer_name_or_path,
        task_type,
        max_seq_len,
        doc_stride,
        revision=None,
        tokenizer_class=None,
        tokenizer_args=None,
        use_fast=True,
        max_query_length=64,
        **kwargs,
    ):
        tokenizer_args = tokenizer_args or {}
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            tokenizer_class=tokenizer_class,
            use_fast=use_fast,
            revision=revision,
            **tokenizer_args,
            **kwargs,
        )

        # TODO infer task_type automatically from config (if possible)
        if task_type == "question_answering":
            processor = SquadProcessor(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                label_list=["start_token", "end_token"],
                metric="squad",
                data_dir="data",
                doc_stride=doc_stride,
                max_query_length=max_query_length,
            )
        elif task_type == "embeddings":
            processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)

        else:
            raise ValueError(
                f"`task_type` {task_type} is not supported yet. "
                f"Valid options for arg `task_type`: 'question_answering', "
                f"'embeddings', "
            )

        return processor

    def save(self, save_dir: str):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def generate_config(self):
        """
        Generates config file from Class and instance attributes (only for sensible config parameters).
        """
        config = {}
        # self.__dict__ doesn't give parent class attributes
        for key, value in inspect.getmembers(self):
            if _is_json(value) and key[0] != "_":
                if issubclass(type(value), Path):
                    value = str(value)
                config[key] = value
        return config

    # TODO potentially remove tasks from code - multitask learning is not supported anyways
    def add_task(
        self, name, metric, label_list, label_column_name=None, label_name=None, task_type=None, text_column_name=None
    ):
        if type(label_list) is not list:
            raise ValueError(f"Argument `label_list` must be of type list. Got: f{type(label_list)}")

        if label_name is None:
            label_name = f"{name}_label"
        label_tensor_name = label_name + "_ids"
        self.tasks[name] = {
            "label_list": label_list,
            "metric": metric,
            "label_tensor_name": label_tensor_name,
            "label_name": label_name,
            "label_column_name": label_column_name,
            "text_column_name": text_column_name,
            "task_type": task_type,
        }

    @abstractmethod
    def file_to_dicts(self, file: str) -> List[dict]:
        raise NotImplementedError()

    @abstractmethod
    def dataset_from_dicts(
        self, dicts: List[Dict], indices: Optional[List[int]] = None, return_baskets: bool = False, debug: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    def _create_dataset(self, baskets: List[SampleBasket]):
        raise NotImplementedError

    @staticmethod
    def log_problematic(problematic_sample_ids):
        if problematic_sample_ids:
            n_problematic = len(problematic_sample_ids)
            problematic_id_str = ", ".join([str(i) for i in problematic_sample_ids])
            logger.error(
                "Unable to convert %s samples to features. Their ids are : %s", n_problematic, problematic_id_str
            )

    @staticmethod
    def _check_sample_features(basket: SampleBasket):
        """
        Check if all samples in the basket has computed its features.

        :param basket: the basket containing the samples

        :return: True if all the samples in the basket has computed its features, False otherwise
        """

        return basket.samples and not any(sample.features is None for sample in basket.samples)

    def _log_samples(self, n_samples: int, baskets: List[SampleBasket]):
        logger.debug("*** Show %s random examples ***", n_samples)
        if len(baskets) == 0:
            logger.debug("*** No samples to show because there are no baskets ***")
            return
        for _ in range(n_samples):
            random_basket = random.choice(baskets)
            random_sample = random.choice(random_basket.samples)  
            logger.debug(random_sample)

    def _log_params(self):
        params = {"processor": self.__class__.__name__, "tokenizer": self.tokenizer.__class__.__name__}
        names = ["max_seq_len", "dev_split"]
        for name in names:
            value = getattr(self, name)
            params.update({name: str(value)})
        tracker.track_params(params)


class SquadProcessor(Processor):
    """
    Convert QA data (in SQuAD Format)
    """

    def __init__(
        self,
        tokenizer,  
        max_seq_len: int,
        data_dir: Optional[Union[Path, str]],
        label_list: Optional[List[str]] = None,
        metric="squad",  
        train_filename: Optional[Union[Path, str]] = Path("train-v2.0.json"),
        dev_filename: Optional[Union[Path, str]] = Path("dev-v2.0.json"),
        test_filename: Optional[Union[Path, str]] = None,
        dev_split: float = 0,
        doc_stride: int = 128,
        max_query_length: int = 64,
        proxies: Optional[dict] = None,
        max_answers: Optional[int] = None,
        **kwargs,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automatically
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `haystack.basics.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/haystack/blob/main/haystack/basics/data_handler/utils.py>`_.
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :param metric: name of metric that shall be used for evaluation, can be "squad" or "top_n_accuracy"
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: None
        :param dev_split: The proportion of the train set that will be sliced. Only works if `dev_filename` is set to `None`.
        :param doc_stride: When the document containing the answer is too long it gets split into part, strided by doc_stride
        :param max_query_length: Maximum length of the question (in number of subword tokens)
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_answers: Number of answers to be converted. QA sets can contain multi-way annotations, which are converted to arrays of max_answer length.
                            Adjusts to maximum number of answers in the first processed datasets if not set.
                            Truncates or pads to max_answer length if set.
        :param kwargs: placeholder for passing generic parameters
        """
        self.ph_output_type = "per_token_squad"

        # validate max_seq_len
        assert max_seq_len <= tokenizer.model_max_length, (
            "max_seq_len cannot be greater than the maximum sequence length handled by the model: "
            f"got max_seq_len={max_seq_len}, while the model maximum length is {tokenizer.model_max_length}. "
            "Please adjust max_seq_len accordingly or use another model "
        )

        assert doc_stride < (max_seq_len - max_query_length), (
            "doc_stride ({}) is longer than max_seq_len ({}) minus space reserved for query tokens ({}). \nThis means that there will be gaps "
            "as the passage windows slide, causing the model to skip over parts of the document.\n"
            "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384)\n "
            "Or decrease max_query_length".format(doc_stride, max_seq_len, max_query_length)
        )

        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_answers = max_answers
        super(SquadProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        self._initialize_special_tokens_count()
        if metric and label_list:
            self.add_task("question_answering", metric, label_list)
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: Optional[List[int]] = None, return_baskets: bool = False, debug: bool = False
    ):
        """
        Convert input dictionaries into a pytorch dataset for Question Answering.
        For this we have an internal representation called "baskets".
        Each basket is a question-document pair.
        Each stage adds or transforms specific information to our baskets.

        :param dicts: dict, input dictionary with SQuAD style information present
        :param indices: list, indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: boolean, whether to return the baskets or not (baskets are needed during inference)
        """
        if indices is None:
            indices = []
        # Convert to standard format
        pre_baskets = [self.convert_qa_input_dict(x) for x in dicts]  # TODO move to input object conversion

        # Tokenize documents and questions
        baskets = tokenize_batch_question_answering(pre_baskets, self.tokenizer, indices)

        # Split documents into smaller passages to fit max_seq_len
        baskets = self._split_docs_into_passages(baskets)

        # Determine max_answers if not set
        max_answers = (
            self.max_answers
            if self.max_answers is not None
            else max(*(len(basket.raw["answers"]) for basket in baskets), 1)
        )

        # Convert answers from string to token space, skip this step for inference
        if not return_baskets:
            baskets = self._convert_answers(baskets, max_answers)

        # Convert internal representation (nested baskets + samples with mixed types) to pytorch features (arrays of numbers)
        baskets = self._passages_to_pytorch_features(baskets, return_baskets, max_answers)

        # Convert features into pytorch dataset, this step also removes potential errors during preprocessing
        dataset, tensor_names, baskets = self._create_dataset(baskets)

        # Logging
        if indices and 0 in indices:
            self._log_samples(n_samples=1, baskets=baskets)

        # During inference we need to keep the information contained in baskets.
        if return_baskets:
            return dataset, tensor_names, self.problematic_sample_ids, baskets
        else:
            return dataset, tensor_names, self.problematic_sample_ids

    def file_to_dicts(self, file: str) -> List[dict]:
        nested_dicts = _read_squad_file(filename=file)
        dicts = [y for x in nested_dicts for y in x["paragraphs"]]
        return dicts

    # TODO use Input Objects instead of this function, remove Natural Questions (NQ) related code
    def convert_qa_input_dict(self, infer_dict: dict) -> Dict[str, Any]:
        """Input dictionaries in QA can either have ["context", "qas"] (internal format) as keys or
        ["text", "questions"] (api format). This function converts the latter into the former. It also converts the
        is_impossible field to answer_type so that NQ and SQuAD dicts have the same format.
        """
        # validate again max_seq_len
        assert self.max_seq_len <= self.tokenizer.model_max_length, (
            "max_seq_len cannot be greater than the maximum sequence length handled by the model: "
            f"got max_seq_len={self.max_seq_len}, while the model maximum length is {self.tokenizer.model_max_length}. "
            "Please adjust max_seq_len accordingly or use another model "
        )

        # check again for doc stride vs max_seq_len when. Parameters can be changed for already initialized models (e.g. in haystack)
        assert self.doc_stride < (self.max_seq_len - self.max_query_length), (
            "doc_stride ({}) is longer than max_seq_len ({}) minus space reserved for query tokens ({}). \nThis means that there will be gaps "
            "as the passage windows slide, causing the model to skip over parts of the document.\n"
            "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384)\n "
            "Or decrease max_query_length".format(self.doc_stride, self.max_seq_len, self.max_query_length)
        )

        try:
            # Check if infer_dict is already in internal json format
            if "context" in infer_dict and "qas" in infer_dict:
                return infer_dict
            # converts dicts from inference mode to data structure used in Haystack
            questions = infer_dict["questions"]
            text = infer_dict["text"]
            uid = infer_dict.get("id", None)
            qas = [{"question": q, "id": uid, "answers": [], "answer_type": None} for i, q in enumerate(questions)]
            converted = {"qas": qas, "context": text}
            return converted
        except KeyError:
            raise Exception("Input does not have the expected format")

    def _initialize_special_tokens_count(self):
        vec = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=["a"], token_ids_1=["b"])
        self.sp_toks_start = vec.index("a")
        self.sp_toks_mid = vec.index("b") - self.sp_toks_start - 1
        self.sp_toks_end = len(vec) - vec.index("b") - 1

    def _split_docs_into_passages(self, baskets: List[SampleBasket]):
        """
        Because of the sequence length limitation of Language Models, the documents need to be divided into smaller
        parts that we call passages.
        """
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        for basket in baskets:
            samples = []
            ########## perform some basic checking
            # TODO, eventually move checking into input validation functions
            # ignore samples with empty context
            if basket.raw["document_text"] == "":
                logger.warning("Ignoring sample with empty context")
                continue
            ########## end checking

            # Calculate the number of tokens that can be reserved for the passage. This is calculated by considering
            # the max_seq_len, the number of tokens in the question and the number of special tokens that will be added
            # when the question and passage are joined (e.g. [CLS] and [SEP])
            passage_len_t = (
                self.max_seq_len - len(basket.raw["question_tokens"][: self.max_query_length]) - n_special_tokens
            )

            # passage_spans is a list of dictionaries where each defines the start and end of each passage
            # on both token and character level
            try:
                passage_spans = get_passage_offsets(
                    basket.raw["document_offsets"], self.doc_stride, passage_len_t, basket.raw["document_text"]
                )
            except Exception as e:
                logger.warning(
                    "Could not divide document into passages. Document: %s\nWith error: %s",
                    basket.raw["document_text"][:200],
                    e,
                )
                passage_spans = []

            for passage_span in passage_spans:
                # Unpack each variable in the dictionary. The "_t" and "_c" indicate
                # whether the index is on the token or character level
                passage_start_t = passage_span["passage_start_t"]
                passage_end_t = passage_span["passage_end_t"]
                passage_start_c = passage_span["passage_start_c"]
                passage_end_c = passage_span["passage_end_c"]

                passage_start_of_word = basket.raw["document_start_of_word"][passage_start_t:passage_end_t]
                passage_tokens = basket.raw["document_tokens"][passage_start_t:passage_end_t]
                passage_text = basket.raw["document_text"][passage_start_c:passage_end_c]

                clear_text = {
                    "passage_text": passage_text,
                    "question_text": basket.raw["question_text"],
                    "passage_id": passage_span["passage_id"],
                }
                tokenized = {
                    "passage_start_t": passage_start_t,
                    "passage_start_c": passage_start_c,
                    "passage_tokens": passage_tokens,
                    "passage_start_of_word": passage_start_of_word,
                    "question_tokens": basket.raw["question_tokens"][: self.max_query_length],
                    "question_offsets": basket.raw["question_offsets"][: self.max_query_length],
                    "question_start_of_word": basket.raw["question_start_of_word"][: self.max_query_length],
                }
                # The sample ID consists of internal_id and a passage numbering
                sample_id = f"{basket.id_internal}-{passage_span['passage_id']}"
                samples.append(Sample(id=sample_id, clear_text=clear_text, tokenized=tokenized))

            basket.samples = samples

        return baskets

    def _convert_answers(self, baskets: List[SampleBasket], max_answers: int):
        """
        Converts answers that are pure strings into the token based representation with start and end token offset.
        Can handle multiple answers per question document pair as is common for development/text sets
        """
        for basket in baskets:
            error_in_answer = False
            for sample in basket.samples:  # type: ignore
                # Dealing with potentially multiple answers (e.g. Squad dev set)
                # Initializing a numpy array of shape (max_answers, 2), filled with -1 for missing values
                label_idxs = np.full((max_answers, 2), fill_value=-1)

                if error_in_answer or (len(basket.raw["answers"]) == 0):
                    # If there are no answers we set
                    label_idxs[0, :] = 0
                else:
                    # For all other cases we use start and end token indices, that are relative to the passage
                    for i, answer in enumerate(basket.raw["answers"]):
                        if i >= max_answers:
                            logger.warning(
                                "Found a sample with more answers (%d) than "
                                "max_answers (%d). These will be ignored.",
                                len(basket.raw["answers"]),
                                max_answers,
                            )
                            break
                        # Calculate start and end relative to document
                        answer_len_c = len(answer["text"])
                        answer_start_c = answer["answer_start"]
                        answer_end_c = answer_start_c + answer_len_c - 1

                        # Convert character offsets to token offsets on document level
                        answer_start_t = offset_to_token_idx_vecorized(basket.raw["document_offsets"], answer_start_c)
                        answer_end_t = offset_to_token_idx_vecorized(basket.raw["document_offsets"], answer_end_c)

                        # Adjust token offsets to be relative to the passage
                        answer_start_t -= sample.tokenized["passage_start_t"]  # type: ignore
                        answer_end_t -= sample.tokenized["passage_start_t"]  # type: ignore

                        # Initialize some basic variables
                        question_len_t = len(sample.tokenized["question_tokens"])  # type: ignore
                        passage_len_t = len(sample.tokenized["passage_tokens"])  # type: ignore

                        # Check that start and end are contained within this passage
                        # answer_end_t is 0 if the first token is the answer
                        # answer_end_t is passage_len_t if the last token is the answer
                        if passage_len_t > answer_start_t >= 0 and passage_len_t >= answer_end_t >= 0:
                            # Then adjust the start and end offsets by adding question and special token
                            label_idxs[i][0] = self.sp_toks_start + question_len_t + self.sp_toks_mid + answer_start_t
                            label_idxs[i][1] = self.sp_toks_start + question_len_t + self.sp_toks_mid + answer_end_t
                        # If the start or end of the span answer is outside the passage, treat passage as no_answer
                        else:
                            label_idxs[i][0] = 0
                            label_idxs[i][1] = 0

                        ########## answer checking ##############################
                        # TODO, move this checking into input validation functions and delete wrong examples there
                        # Cases where the answer is not within the current passage will be turned into no answers by the featurization fn
                        if answer_start_t < 0 or answer_end_t >= passage_len_t:
                            pass
                        else:
                            doc_text = basket.raw["document_text"]
                            answer_indices = doc_text[answer_start_c : answer_end_c + 1]
                            answer_text = answer["text"]
                            # check if answer string can be found in context
                            if answer_text not in doc_text:
                                logger.warning(
                                    "Answer '%s' not contained in context.\n"
                                    "Example will not be converted for training/evaluation.",
                                    answer["text"],
                                )
                                error_in_answer = True
                                label_idxs[i][0] = -100  # TODO remove this hack also from featurization
                                label_idxs[i][1] = -100
                                break  # Break loop around answers, so the error message is not shown multiple times
                            if answer_indices.strip() != answer_text.strip():
                                logger.warning(
                                    "Answer using start/end indices is '%s' while gold label text is '%s'.\n"
                                    "Example will not be converted for training/evaluation.",
                                    answer_indices,
                                    answer_text,
                                )
                                error_in_answer = True
                                label_idxs[i][0] = -100  # TODO remove this hack also from featurization
                                label_idxs[i][1] = -100
                                break  # Break loop around answers, so the error message is not shown multiple times
                        ########## end of checking ####################

                sample.tokenized["labels"] = label_idxs  # type: ignore

        return baskets

    def _passages_to_pytorch_features(self, baskets: List[SampleBasket], return_baskets: bool, max_answers: int):
        """
        Convert internal representation (nested baskets + samples with mixed types) to python features (arrays of numbers).
        We first join question and passages into one large vector.
        Then we add vectors for: - input_ids (token ids)
                                 - segment_ids (does a token belong to question or document)
                                 - padding_mask
                                 - span_mask (valid answer tokens)
                                 - start_of_word
        """
        for basket in baskets:
            # Add features to samples
            for sample in basket.samples:  # type: ignore
                # Initialize some basic variables
                if sample.tokenized is not None:
                    question_tokens = sample.tokenized["question_tokens"]
                    question_start_of_word = sample.tokenized["question_start_of_word"]
                    question_len_t = len(question_tokens)
                    passage_start_t = sample.tokenized["passage_start_t"]
                    passage_tokens = sample.tokenized["passage_tokens"]
                    passage_start_of_word = sample.tokenized["passage_start_of_word"]
                    passage_len_t = len(passage_tokens)
                    sample_id = [int(x) for x in sample.id.split("-")]

                    # - Combines question_tokens and passage_tokens into a single vector called input_ids
                    # - input_ids also contains special tokens (e.g. CLS or SEP tokens).
                    # - It will have length = question_len_t + passage_len_t + n_special_tokens. This may be less than
                    #   max_seq_len but never greater since truncation was already performed when the document was chunked into passages
                    question_input_ids = sample.tokenized["question_tokens"]
                    passage_input_ids = sample.tokenized["passage_tokens"]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(
                    token_ids_0=question_input_ids, token_ids_1=passage_input_ids
                )

                segment_ids = self.tokenizer.create_token_type_ids_from_sequences(
                    token_ids_0=question_input_ids, token_ids_1=passage_input_ids
                )
                # To make the start index of passage tokens the start manually
                seq_2_start_t = self.sp_toks_start + question_len_t + self.sp_toks_mid

                start_of_word = (
                    [0] * self.sp_toks_start
                    + question_start_of_word
                    + [0] * self.sp_toks_mid
                    + passage_start_of_word
                    + [0] * self.sp_toks_end
                )

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                padding_mask = [1] * len(input_ids)

                # The span_mask has 1 for tokens that are valid start or end tokens for QA spans.
                # 0s are assigned to question tokens, mid special tokens, end special tokens, and padding
                # Note that start special tokens are assigned 1 since they can be chosen for a no_answer prediction
                span_mask = [1] * self.sp_toks_start
                span_mask += [0] * question_len_t
                span_mask += [0] * self.sp_toks_mid
                span_mask += [1] * passage_len_t
                span_mask += [0] * self.sp_toks_end

                # Pad up to the sequence length. For certain models, the pad token id is not 0 (e.g. Roberta where it is 1)
                pad_idx = self.tokenizer.pad_token_id
                padding = [pad_idx] * (self.max_seq_len - len(input_ids))
                zero_padding = [0] * (self.max_seq_len - len(input_ids))

                input_ids += padding
                padding_mask += zero_padding
                segment_ids += zero_padding
                start_of_word += zero_padding
                span_mask += zero_padding

                # TODO possibly remove these checks after input validation is in place
                len_check = (
                    len(input_ids) == len(padding_mask) == len(segment_ids) == len(start_of_word) == len(span_mask)
                )
                id_check = len(sample_id) == 3
                label_check = return_baskets or len(sample.tokenized.get("labels", [])) == max_answers  # type: ignore
                # labels are set to -100 when answer cannot be found
                label_check2 = return_baskets or np.all(sample.tokenized["labels"] > -99)  # type: ignore
                if len_check and id_check and label_check and label_check2:
                    # - The first of the labels will be used in train, and the full array will be used in eval.
                    # - start_of_word and spec_tok_mask are not actually needed by model.forward() but are needed for
                    #   model.formatted_preds() during inference for creating answer strings
                    # - passage_start_t is index of passage's first token relative to document
                    feature_dict = {
                        "input_ids": input_ids,
                        "padding_mask": padding_mask,
                        "segment_ids": segment_ids,
                        "passage_start_t": passage_start_t,
                        "start_of_word": start_of_word,
                        "labels": sample.tokenized.get("labels", []),  # type: ignore
                        "id": sample_id,
                        "seq_2_start_t": seq_2_start_t,
                        "span_mask": span_mask,
                    }
                    # other processor's features can be lists
                    sample.features = [feature_dict]  # type: ignore
                else:
                    self.problematic_sample_ids.add(sample.id)
                    sample.features = None
        return baskets

    def _create_dataset(self, baskets: List[SampleBasket]):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat: List[dict] = []
        basket_to_remove = []
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:  # type: ignore
                    features_flat.extend(sample.features)  # type: ignore
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, baskets


class TextSimilarityProcessor(Processor):
    """
    Used to handle the Dense Passage Retrieval (DPR) datasets that come in json format, example: biencoder-nq-train.json, biencoder-nq-dev.json, trivia-train.json, trivia-dev.json

    Datasets can be downloaded from the official DPR github repository (https://github.com/facebookresearch/DPR)
    dataset format: list of dictionaries with keys: 'dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'
    Each sample is a dictionary of format:
    {"dataset": str,
    "question": str,
    "answers": list of str
    "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }

    """

    def __init__(
        self,
        query_tokenizer,  # type: ignore
        passage_tokenizer,  # type: ignore
        max_seq_len_query: int,
        max_seq_len_passage: int,
        data_dir: str = "",
        metric=None,  # type: ignore
        train_filename: str = "train.json",
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = "test.json",
        dev_split: float = 0.1,
        proxies: Optional[dict] = None,
        max_samples: Optional[int] = None,
        embed_title: bool = True,
        num_positives: int = 1,
        num_hard_negatives: int = 1,
        shuffle_negatives: bool = True,
        shuffle_positives: bool = False,
        label_list: Optional[List[str]] = None,
        query_structure: Optional[str] = 'global',
        table_structure: Optional[str] = 'global',
        linearization: str = "default",
        linearization_direction: str = "row",
        **kwargs,
    ):
        """
        :param query_tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a passage (str) into tokens.
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automatically
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `haystack.basics.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/haystack/blob/main/haystack/basics/data_handler/utils.py>`_.
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: None
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_samples: maximum number of samples to use
        :param embed_title: Whether to embed title in passages during tensorization (bool),
        :param num_hard_negatives: maximum number to hard negative context passages in a sample
        :param num_positives: maximum number to positive context passages in a sample
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the num_hard_negative number of passages
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the num_positive number of passages
        :param label_list: list of labels to predict. Usually ["hard_negative", "positive"]
        :param kwargs: placeholder for passing generic parameters
        """
        # TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.max_samples = max_samples
        self.query_tokenizer = query_tokenizer
        self.passage_tokenizer = passage_tokenizer
        self.embed_title = embed_title
        self.num_hard_negatives = num_hard_negatives
        self.num_positives = num_positives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passage = max_seq_len_passage
        self.query_structure = query_structure
        self.table_structure = table_structure
        self.linearization = linearization
        self.linearization_direction = linearization_direction

        super(TextSimilarityProcessor, self).__init__(
            tokenizer=None,  # type: ignore
            max_seq_len=0,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric:
            self.add_task(
                name="text_similarity",
                metric=metric,
                label_list=label_list,
                label_name="label",
                task_type="text_similarity",
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Overwriting method from parent class to **always** load the TextSimilarityProcessor instead of the specific class stored in the config.

        :param load_dir: directory that contains a 'processor_config.json'
        :return: An instance of an TextSimilarityProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        with open(processor_config_file) as f:
            config = json.load(f)
        # init tokenizers
        query_tokenizer_class: Type[PreTrainedTokenizer] = getattr(transformers, config["query_tokenizer"])
        query_tokenizer = query_tokenizer_class.from_pretrained(
            pretrained_model_name_or_path=load_dir, subfolder="query"
        )
        passage_tokenizer_class: Type[PreTrainedTokenizer] = getattr(transformers, config["passage_tokenizer"])
        passage_tokenizer = passage_tokenizer_class.from_pretrained(
            pretrained_model_name_or_path=load_dir, subfolder="passage"
        )

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["query_tokenizer"]
        del config["passage_tokenizer"]

        processor = cls.load(
            query_tokenizer=query_tokenizer,
            passage_tokenizer=passage_tokenizer,
            processor_name="TextSimilarityProcessor",
            **config,
        )
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def save(self, save_dir: Union[str, Path]):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["query_tokenizer"] = self.query_tokenizer.__class__.__name__
        config["passage_tokenizer"] = self.passage_tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.query_tokenizer.save_pretrained(str(save_dir / "query"))
        self.passage_tokenizer.save_pretrained(str(save_dir / "passage"))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: Optional[List[int]] = None, return_baskets: bool = False, debug: bool = False
    ):
        """
        Convert input dictionaries into a pytorch dataset for TextSimilarity (e.g. DPR).
        For conversion we have an internal representation called "baskets".
        Each basket is one query and related text passages (positive passages fitting to the query and negative
        passages that do not fit the query)
        Each stage adds or transforms specific information to our baskets.

        :param dicts: input dictionary with DPR-style content
                        {"query": str,
                         "passages": List[
                                        {'title': str,
                                        'text': str,
                                        'label': 'hard_negative',
                                        'external_id': str},
                                        ....
                                        ]
                         }
        :param indices: indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: whether to return the baskets or not (baskets are needed during inference)
        :return: dataset, tensor_names, problematic_ids, [baskets]
        """
        if indices is None:
            indices = []
        # Take the dict and insert into our basket structure, this stages also adds an internal IDs
        baskets = self._fill_baskets(dicts, indices)

        # Separate conversion of query
        baskets = self._convert_queries(baskets=baskets)

        # and context passages. When converting the context the label is also assigned.
        baskets = self._convert_contexts(baskets=baskets)

        # Convert features into pytorch dataset, this step also removes and logs potential errors during preprocessing
        dataset, tensor_names, problematic_ids, baskets = self._create_dataset(baskets)

        if problematic_ids:
            logger.error(
                "There were %s errors during preprocessing at positions: %s", len(problematic_ids), problematic_ids
            )

        if return_baskets:
            return dataset, tensor_names, problematic_ids, baskets
        else:
            return dataset, tensor_names, problematic_ids

    def file_to_dicts(self, file: str) -> List[dict]:
        """
        Converts a Dense Passage Retrieval (DPR) data file in json format to a list of dictionaries.

        :param file: filename of DPR data in json format
                Each sample is a dictionary of format:
                {"dataset": str,
                "question": str,
                "answers": list of str
                "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                }


        Returns:
        list of dictionaries: List[dict]
            each dictionary:
            {"query": str,
            "passages": [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
            {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
            ...]}
        """
        dicts = _read_dpr_json(
            file,
            max_samples=self.max_samples,
            num_hard_negatives=self.num_hard_negatives,
            num_positives=self.num_positives,
            shuffle_negatives=self.shuffle_negatives,
            shuffle_positives=self.shuffle_positives,
        )

        # shuffle dicts to make sure that similar positive passages do not end up in one batch
        #############  We delete the shuffling here  #############
        dicts = random.sample(dicts, len(dicts))
        return dicts

    def _fill_baskets(self, dicts: List[dict], indices: Optional[List[int]]):
        baskets = []
        if not indices:
            indices = list(range(len(dicts)))
        for d, id_internal in zip(dicts, indices):
            basket = SampleBasket(id_external=None, id_internal=id_internal, raw=d)
            baskets.append(basket)
        return baskets

    def _create_biased_id(self, row_ids: torch.Tensor, column_ids: torch.Tensor): 
        """Compute relation-based bias id. 
        args: 
            row_ids: <seq-len> 
            column_ids: <seq-len> 
            valid_len: <int> 
        ret: 
            bias_id: <seq-len, seq-len> 
            
        notes: 
        - title:  row-id = 0, col-id = 0 
        - header: row-id = 0, col-id = 1-indexed 
        - cell:   row-id = 1-indexed, col-id = 1-indexed 
        """
    
        n = row_ids.size(0)
        row_ids = row_ids.unsqueeze(1).expand(n, n)
        column_ids = column_ids.unsqueeze(1).expand(n, n)
        
        bias_id = torch.zeros((n, n), dtype=torch.long)
        
        # Header
        bias_id[(row_ids == 0) & (column_ids != 0)] = 1
        bias_id[(row_ids.T == 0) & (column_ids.T == 0)] = 3
        bias_id[(row_ids == 0) & (row_ids.T == 0) & (column_ids == column_ids.T)] = 4
        bias_id[(row_ids == 0) & (row_ids.T == 0) & (column_ids != column_ids.T)] = 5
        
        # Cell
        bias_id[(row_ids != 0) & (column_ids.T == 0)] = 7
        bias_id[(row_ids != 0) & (row_ids.T == 0)] = 8
        bias_id[(row_ids == row_ids.T) & (column_ids == column_ids.T)] = 9
        bias_id[(row_ids == row_ids.T) & (column_ids != column_ids.T)] = 10
        bias_id[(row_ids != row_ids.T) & (column_ids == column_ids.T)] = 11
        bias_id[(row_ids != row_ids.T) & (column_ids != column_ids.T)] = 12
        
        # Sentence
        bias_id[(row_ids == 0) & (column_ids == 0)] = 0

        return bias_id
    
    def _create_rowcol_attn_mask(
        self,
        row_ids: List[int], 
        column_ids: List[int], 
        title_len: int, 
    ) -> torch.LongTensor:
        # tokens within the same row OR column are mutually visible 
        mask = (row_ids == row_ids.transpose(0, 1)) & (column_ids == column_ids.transpose(0, 1)) 
        mask = mask.long() 
        # title tokens are globally visible 
        mask[: title_len, :] = 1 
        mask[:, : title_len] = 1 
        return mask 
    
    def _create_global_attn_mask(self, max_seq_length: int): 
        mask = torch.ones(max_seq_length, max_seq_length).long() 
        return mask 

    def _convert_queries(self, baskets: List[SampleBasket]):
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features = [{}]  # type: ignore
            
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                # try:
                query = self._normalize_question(basket.raw["query"])

                if isinstance(self.query_tokenizer, TapasTokenizer):

                    tokenized_query = self.query_tokenizer.tokenize(query)
                    # print('--------------------------------------')
                    # print(tokenized_query)
                    # 如果token数量超过256，截断tokenized_query到256个token
                    if len(tokenized_query) > self.max_seq_len_query-2:
                        # 使用截断后的token重新构建query字符串
                        query = self.query_tokenizer.convert_tokens_to_string(tokenized_query[:self.max_seq_len_query-2])
                    
                    # print(query)
                    
                    encoding = self.query_tokenizer(
                        table=pd.DataFrame(),
                        queries=query,
                        add_special_tokens=True,
                        truncation='drop_rows_to_fit',
                        max_length=self.max_seq_len_query,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    # print(encoding["input_ids"][0])

                   
                    #     # encoding = self.query_tokenizer(
                    #     #     table=pd.DataFrame(),
                    #     #     queries=query,
                    #     #     add_special_tokens=True,
                    #     #     truncation=True,
                    #     #     truncation_strategy="longest_first",
                    #     #     max_length=self.max_seq_len_query,
                    #     #     padding="max_length",
                    #     #     return_tensors="pt"
                    #     # )
                    
                    query_inputs = {
                        "input_ids": encoding["input_ids"][0],
                        "token_type_ids": encoding["token_type_ids"][0],
                        "attention_mask": encoding["attention_mask"][0],
                    }

                    # print(encoding["input_ids"][0].shape)
                    # print(encoding["token_type_ids"][0].shape)
                    # print(encoding["attention_mask"][0].shape)         
                
                else:
                    
                    query_inputs = self.query_tokenizer(
                        query,
                        max_length=self.max_seq_len_query,
                        add_special_tokens=True,
                        truncation=True,
                        truncation_strategy="longest_first",
                        padding="max_length",
                        return_token_type_ids=True,
                    )

                    # print(query_inputs["input_ids"])
                    
                
                row_ids = torch.LongTensor(self._to_max_len([], 0, self.max_seq_len_query))
                col_ids = torch.LongTensor(self._to_max_len([], 0, self.max_seq_len_query))

                if self.query_structure == "bias": 
                    row_ids = self._create_biased_id(row_ids, col_ids) 
                

                
                
                # tokenize query
                tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_inputs["input_ids"])
                
                    
                if len(tokenized_query) == 0:
                    logger.warning(
                        "The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize"
                    )
                    return None
                
                # print('--------------Query Input-----------------')
                # print(len(query_inputs["input_ids"]))
                # print(len(query_inputs["token_type_ids"]))
                # print(len(query_inputs["attention_mask"]))
                # print(len(row_ids))

                # clear_text["query_text"] = query
                tokenized["query_tokens"] = tokenized_query
                features[0]["query_input_ids"] = query_inputs["input_ids"]
                features[0]["query_segment_ids"] = query_inputs["token_type_ids"]
                features[0]["query_attention_mask"] = query_inputs["attention_mask"]
                features[0]["query_row_ids"] = row_ids
                features[0]["query_col_ids"] = col_ids
                # except Exception:
                #     features = None  # type: ignore
            
            
            sample = Sample(id="", clear_text=clear_text, tokenized=tokenized, features=features)  # type: ignore
            basket.samples = [sample]
        return baskets

    def _to_max_len(self, seq_list: List[int], pad_id: int, max_len: int): 

        if len(seq_list) < max_len: 
            seq_list.extend([pad_id for _ in range(max_len-len(seq_list))])
        seq_list = seq_list[: max_len]
        
        return seq_list 
    
    def _serialized_table(self, line: dict):
        if self.linearization == 'default':
            if self.linearization_direction == 'row':
                serialized_tables = ''

                title = line['title'].strip()
                serialized_tables += 'title: ' + title + '\n'
                
                # Retrieve column headers
                columns = [col['text'] for col in line['columns']]
                serialized_tables += '| ' + ' | '.join(columns) + ' |\n'
                
                # Initialize the markdown table header separator
                serialized_tables += '| ' + ' | '.join(['---'] * len(columns)) + ' |\n'

                # Process rows and cells
                filtered_rows = []  # Store filtered rows
                for row_key, row_value in line['rows'].items():
                    current_row = []  # Store current row cells
                    for cell in row_value['cells']:
                        cell_text = cell['text'].strip()
                        # Limit the cell text to the first 10 words
                        if len(cell_text.split()) > 10:
                            cell_text = ' '.join(cell_text.split()[:10])
                        current_row.append(cell_text)
                    
                    # Append the formatted row to the result if it's not a noise row (all cells same)
                    if len(set(current_row)) != 1:
                        filtered_rows.append('| ' + ' | '.join(current_row) + ' |')

                serialized_tables += '\n'.join(filtered_rows)

            elif self.linearization_direction == 'column':
                serialized_tables = ''

                title = line['title'].strip()
                serialized_tables += 'title: ' + title + '\n'
                
                # Retrieve column headers and initialize containers for column data
                columns = [col['text'] for col in line['columns']]
                column_data = {col: [] for col in columns}  # Dictionary to store column data
                
                # Process each row and distribute cell data to the appropriate column
                for row_key, row_value in line['rows'].items():
                    for idx, cell in enumerate(row_value['cells']):
                        cell_text = cell['text'].strip()
                        # Limit the cell text to the first 10 words
                        if len(cell_text.split()) > 10:
                            cell_text = ' '.join(cell_text.split()[:10])
                        column_data[columns[idx]].append(cell_text)
                
                
                # Serialize each column with its data
                for col in columns:
                    serialized_tables += f"## {col}: "  # Markdown header for each column with entries inline
                    serialized_tables += ', '.join(column_data[col]) + ".\n"


        elif self.linearization == 'separator':
            if self.linearization_direction == 'row':
                serialized_tables = ''

                title = line['title'].strip()
                serialized_tables += '[Title]' + title 

                columns = [col['text'] for col in line['columns']]
                serialized_tables += '[Header]' + '[sep]'.join(columns)

                serialized_tables += '[Rows]'
                # Process rows and cells
                filtered_rows = []  # Store filtered rows
                for row_key, row_value in line['rows'].items():
                    current_row = []  # Store current row cells
                    for cell in row_value['cells']:
                        cell_text = cell['text'].strip()
                        # Limit the cell text to the first 10 words
                        if len(cell_text.split()) > 10:
                            cell_text = ' '.join(cell_text.split()[:10])
                        current_row.append(cell_text)
                    
                    # Append the formatted row to the result if it's not a noise row (all cells same)
                    if len(set(current_row)) != 1:
                        filtered_rows.append('[Row]' + '[sep]'.join(current_row))

                serialized_tables += ''.join(filtered_rows)

                # print('-----------------------Seperator-------------------')
                # print(serialized_tables)
            
            elif self.linearization_direction == 'column':
                serialized_tables = ''

                title = line['title'].strip()
                serialized_tables += '[Title]' + title 

                columns = [col['text'] for col in line['columns']]

                column_data = {col: [] for col in columns}  # Dictionary to store column data
                
                serialized_tables += '[Columns]'
                # Process each row and distribute cell data to the appropriate column
                for row_key, row_value in line['rows'].items():
                    for idx, cell in enumerate(row_value['cells']):
                        cell_text = cell['text'].strip()
                        # Limit the cell text to the first 10 words
                        if len(cell_text.split()) > 10:
                            cell_text = ' '.join(cell_text.split()[:10])
                        column_data[columns[idx]].append(cell_text)

                # Serialize each column with its data, separated by defined separators
                for col in columns:
                    serialized_tables += f"[Column]{col}[sep]"  # Column header with separator
                    for entry in column_data[col]:
                        serialized_tables += f"{entry}[sep]"  # Each entry followed by a separator

                    # Optionally, remove the last separator or handle it differently if required
                    serialized_tables = serialized_tables.rstrip('[sep]')

                # print('-----------------------separator-------------------')
                # print(serialized_tables)

        elif self.linearization == 'template':
            if self.linearization_direction == 'row':
                title = line['title'].strip()
                
                serialized_tables = f"Given the table of \"{title}\". "

                columns = [col['text'] for col in line['columns']]

                filtered_rows = []  # Store filtered rows
                for row_key, row_value in line['rows'].items():
                    current_row = []  # Store current row cells
                    for cell in row_value['cells']:
                        cell_text = cell['text'].strip()
                        # Limit the cell text to the first 10 words
                        if len(cell_text.split()) > 10:
                            cell_text = ' '.join(cell_text.split()[:10])
                        current_row.append(cell_text)
                    
                    # Append the formatted row to the result if it's not a noise row (all cells same)
                    if len(set(current_row)) != 1:
                        serialized_tables += f"In the row {int(row_key)+1}, "
                        for j in range(len(columns)):
                            serialized_tables += f'the {columns[j]} is {current_row[j]}' + ', '
                        
                        serialized_tables = serialized_tables[:-2] + '. '
            
            elif self.linearization_direction == 'column':
                title = line['title'].strip()
                serialized_tables = f"Given the table titled \"{title}\", "

                columns = [col['text'] for col in line['columns']]
                column_data = {col: [] for col in columns}  # Dictionary to store column data

                # Process each row and distribute cell data to the appropriate column
                for row_key, row_value in line['rows'].items():
                    for idx, cell in enumerate(row_value['cells']):
                        cell_text = cell['text'].strip()
                        # Limit the cell text to the first 10 words
                        if len(cell_text.split()) > 10:
                            cell_text = ' '.join(cell_text.split()[:10])
                        column_data[columns[idx]].append(cell_text)

                # Serialize each column with its data in narrative form
                for col in columns:
                    serialized_tables += f"In the column \"{col}\", the rows contain: "
                    entries = ', '.join(f"row {idx + 1}: {entry}" for idx, entry in enumerate(column_data[col]))
                    serialized_tables += entries + '. '

                # Optionally, remove any trailing spaces and finalize the serialization
                serialized_tables = serialized_tables.strip()
            
            # print('-----------------------Template-------------------')
            # print(serialized_tables)
        
        elif self.linearization == 'direct':
            if self.linearization_direction == 'row':
                serialized_tables = ''

                serialized_tables += line['title'].strip() + ' '
                
                # Retrieve column headers
                columns = [col['text'] for col in line['columns']]
                serialized_tables += ' '.join(columns) + ' '
                
                # Process rows and cells
                filtered_rows = []  # Store filtered rows
                for row_key, row_value in line['rows'].items():
                    current_row = []  # Store current row cells
                    for cell in row_value['cells']:
                        cell_text = cell['text'].strip()
                        # Limit the cell text to the first 10 words
                        if len(cell_text.split()) > 10:
                            cell_text = ' '.join(cell_text.split()[:10])
                        current_row.append(cell_text)
                    
                    # Append the formatted row to the result if it's not a noise row (all cells same)
                    if len(set(current_row)) != 1:
                        filtered_rows.append(' '.join(current_row) + ' ')

                serialized_tables += ' '.join(filtered_rows)


            elif self.linearization_direction == 'column':
                serialized_tables = ''

                serialized_tables += line['title'].strip() + ' '
                
                # Retrieve column headers and initialize containers for column data
                columns = [col['text'] for col in line['columns']]
                column_data = {col: [] for col in columns}  # Dictionary to store column data
                
                # Process each row and distribute cell data to the appropriate column
                for row_key, row_value in line['rows'].items():
                    for idx, cell in enumerate(row_value['cells']):
                        cell_text = cell['text'].strip()
                        # Limit the cell text to the first 10 words
                        if len(cell_text.split()) > 10:
                            cell_text = ' '.join(cell_text.split()[:10])
                        column_data[columns[idx]].append(cell_text)
                
                
                # Serialize each column with its data
                for col in columns:
                    serialized_tables += ' '.join(column_data[col]) + " "
            
                    
        return serialized_tables
  
    def _convert_contexts(self, baskets: List[SampleBasket]):
        for basket in baskets:
            if "passages" in basket.raw:
                # try:
                positive_context = list(filter(lambda x: x["label"] == "positive", basket.raw["passages"]))
                if self.shuffle_positives:
                    random.shuffle(positive_context)
                positive_context = positive_context[: self.num_positives]
                hard_negative_context = list(
                    filter(lambda x: x["label"] == "hard_negative", basket.raw["passages"])
                )
                if self.shuffle_negatives:
                    random.shuffle(hard_negative_context)
                hard_negative_context = hard_negative_context[: self.num_hard_negatives]

                ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives

                all_contexts = positive_context + hard_negative_context

                all_ctx = []
                ctx_inputs = {"input_ids": [], "token_type_ids": [], "attention_mask": [], 'row_ids': [], 'col_ids': []}

                # if len(all_contexts) != 8:
                #     print(len(hard_negative_context), len(positive_context))
                #     print(positive_context)

                for table in all_contexts:
                    table = table['text']
                    title = table['title'].strip() if 'title' in table else ''
                    columns = [col['text'].strip() for col in table['columns']] if 'columns' in table else []
                    columns = columns[:50]

                    if isinstance(self.passage_tokenizer, TapasTokenizer):
                    
                        table_df = pd.DataFrame(columns=columns)

                        for row_key, row_value in table['rows'].items():
                            row_idx = int(row_key)  # converting the row key to an integer index
                            cell_data = row_value['cells']
                            if row_idx > 200:
                                continue  # Skip rows with index greater than 255

                            # Ensure the row exists in the DataFrame
                            if row_idx not in table_df.index:
                                table_df.loc[row_idx] = [None] * len(columns)
                            
                            for col_idx, cell in enumerate(cell_data):
                                if col_idx >= len(columns):
                                    continue  # Skip columns that are out of the defined range
                                cell_text = ' '.join(str(cell['text']).strip().split()[:8])  # Limit cell text length
                                
                                # Place the cell text in the correct location in the DataFrame
                                table_df.iat[row_idx, col_idx] = cell_text
                        
                        try:
                            encoding = self.passage_tokenizer(
                                table=table_df,
                                queries=title,
                                truncation="drop_rows_to_fit",
                                padding="max_length",
                                max_length=self.max_seq_len_passage,
                                return_tensors="pt"
                            )
                        except Exception as e:
                            encoding = self.passage_tokenizer(
                                table=pd.DataFrame(),
                                queries=title[:self.max_seq_len_passage],
                                truncation="drop_rows_to_fit",
                                padding="max_length",
                                max_length=self.max_seq_len_passage,
                                return_tensors="pt"
                            )

                        # print('--------------Table Input-----------------')
                        # print(encoding["input_ids"])

                        ctx_inputs["input_ids"].append(encoding["input_ids"][0])
                        ctx_inputs["token_type_ids"].append(encoding["token_type_ids"][0])
                        ctx_inputs["attention_mask"].append(encoding["attention_mask"][0])
                        row_ids = torch.LongTensor(self._to_max_len([], 0, self.max_seq_len_passage))
                        col_ids = torch.LongTensor(self._to_max_len([], 0, self.max_seq_len_passage))
                        ctx_inputs["row_ids"].append(row_ids)
                        ctx_inputs["col_ids"].append(col_ids)

                    else:

                        if self.table_structure == 'global':
                            serialized_tables = self._serialized_table(table)
                            all_ctx.append(serialized_tables)

                        else:
                            token_ids = []
                            row_ids, col_ids = [] ,[]

                            title_token_ids = self.passage_tokenizer.encode(
                                title,
                                add_special_tokens=True,
                                max_length=self.max_seq_len_passage,
                                truncation=True,
                                pad_to_max_length=False
                            )
                            
                            title_len = len(title_token_ids)
                            token_ids.extend(title_token_ids)
                            row_ids.extend([0 for _ in title_token_ids])
                            col_ids.extend([0 for _ in title_token_ids])

                            for col_index, col in enumerate(columns):
                                col_token_ids = self.passage_tokenizer.encode(
                                    col.strip(),
                                    add_special_tokens=False,
                                    max_length=self.max_seq_len_passage,
                                    truncation=True,
                                    pad_to_max_length=False
                                )
                                token_ids.extend(col_token_ids)
                                row_ids.extend([0] * len(col_token_ids))  # Header is considered at row 0
                                col_ids.extend([col_index + 1] * len(col_token_ids))  # Column indices start from 1

                            # Process cells in rows
                            rows = table['rows']
                            for row_key, row_value in rows.items():
                                row_idx = int(row_key) + 1  # Adding 1 to differentiate from the header row (0)
                                for col_idx, cell in enumerate(row_value['cells']):
                                    cell_text = cell['text'].strip()
                                    # Limit the cell text to the first 10 words
                                    if len(cell_text.split()) > 10:
                                        cell_text = ' '.join(cell_text.split()[:10])

                                    cell_token_ids = self.passage_tokenizer.encode(
                                        cell_text,
                                        add_special_tokens=False,
                                        max_length=self.max_seq_len_passage,
                                        truncation=True,
                                        pad_to_max_length=False
                                    )
                                    token_ids.extend(cell_token_ids)
                                    row_ids.extend([row_idx] * len(cell_token_ids))
                                    col_ids.extend([col_idx + 1] * len(cell_token_ids))  # Column indices start from 1
                            
                            assert len(token_ids) == len(row_ids) == len(col_ids) 
                            
                            if len(token_ids) > self.max_seq_len_passage:
                                token_ids = token_ids[:self.max_seq_len_passage-1]

                            token_ids.append(self.passage_tokenizer.sep_token_id)

                            valid_len = min(len(token_ids), self.max_seq_len_passage)

                            token_ids = self._to_max_len(token_ids, self.passage_tokenizer.pad_token_id, self.max_seq_len_passage)

                            token_ids = torch.LongTensor(token_ids)
                            row_ids = torch.LongTensor(self._to_max_len(row_ids, 0, self.max_seq_len_passage))
                            col_ids = torch.LongTensor(self._to_max_len(col_ids, 0, self.max_seq_len_passage))
                            
                            if self.table_structure == "rowcol": 
                                attn_mask = self._create_rowcol_attn_mask(row_ids.unsqueeze(0), col_ids.unsqueeze(0), title_len)
                            else: 
                                attn_mask = self._create_global_attn_mask(self.max_seq_len_passage)

                            if self.table_structure == "bias": 
                                row_ids = self._create_biased_id(row_ids, col_ids) 
            
                            # set pad positions to invisible 
                            attn_mask[valid_len: , :] = 0 
                            attn_mask[:, valid_len: ] = 0 
                            token_type_ids = torch.zeros_like(token_ids)


                            ctx_inputs["input_ids"].append(token_ids)
                            ctx_inputs["token_type_ids"].append(token_type_ids)
                            ctx_inputs["attention_mask"].append(attn_mask)
                            ctx_inputs["row_ids"].append(row_ids)
                            ctx_inputs["col_ids"].append(col_ids)
                                            
                if len(all_ctx) != 0:
                    ctx_inputs = self.passage_tokenizer(
                        all_ctx,
                        add_special_tokens=True,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_seq_len_passage,
                        return_token_type_ids=True,
                    )

                    ctx_segment_ids = [[0] * len(ctx_inputs["token_type_ids"][0])] * len(ctx_inputs["token_type_ids"])

                    ctx_inputs["token_type_ids"] = ctx_segment_ids
                    ctx_inputs["row_ids"] = [[0] * len(ctx_inputs["input_ids"][0])] * len(ctx_inputs["input_ids"])
                    ctx_inputs["col_ids"] = [[0] * len(ctx_inputs["input_ids"][0])] * len(ctx_inputs["input_ids"])

                else:
                    for key, value in ctx_inputs.items():
                        if value is not None:
                            ctx_inputs[key] = torch.stack(value)
                
                # print('------A sample-------')
                # print(self.table_structure)
                # print(ctx_inputs["row_ids"])
                # print(ctx_inputs["col_ids"])
                # print(ctx_inputs["input_ids"])
                # print(ctx_inputs["token_type_ids"])
                # print(ctx_inputs["attention_mask"])

                sample = basket.samples[0]  # type: ignore

                # sample.clear_text["passages"] = positive_context + hard_negative_context
                # sample.tokenized["passages_tokens"] = tokenized_passage  # type: ignore

                sample.features[0]["passage_input_ids"] = ctx_inputs["input_ids"]  # type: ignore
                sample.features[0]["passage_segment_ids"] = ctx_inputs["token_type_ids"]  # type: ignore
                sample.features[0]["passage_attention_mask"] = ctx_inputs["attention_mask"]  # type: ignore
                sample.features[0]["passage_row_ids"] = ctx_inputs["row_ids"]  
                sample.features[0]["passage_col_ids"] = ctx_inputs["col_ids"]
                sample.features[0]["label_ids"] = ctx_label  # type: ignore

                # print(sample.features[0]["passage_segment_ids"].shape)
                # except Exception:
                #     basket.samples[0].features = None  # type: ignore

        return baskets
    

    def _create_dataset(self, baskets: List[SampleBasket]):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat: List[dict] = []
        basket_to_remove = []
        problematic_ids: set = set()
        
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:  # type: ignore
                    features_flat.extend(sample.features)  # type: ignore
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                problematic_ids.add(basket.id_internal)
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, problematic_ids, baskets

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Removes '?' from queries/questions"""
        if question[-1] == "?":
            question = question[:-1]
        return question

    @staticmethod
    def _combine_title_context(titles: List[str], texts: List[str]):
        res = []
        for title, ctx in zip(titles, texts):
            if title is None:
                title = ""
                logger.warning(
                    "Couldn't find title although `embed_title` is set to True for DPR. Using title='' now. Related passage text: '%s' ",
                    ctx,
                )
            res.append((title, ctx))
        return res




class TripletSimilarityProcessor(Processor):
    """
    Used to handle the Universal Table Retrieval (UTP) datasets that come in json format, example: 

    Datasets can be downloaded from the official DPR github repository (https://github.com/facebookresearch/DPR)
    dataset format: list of dictionaries with keys: 'dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'
    Each sample is a dictionary of format:
    {
        "dataset": str,
        "question": str,
        "answers": list of str
        "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }

    """

    def __init__(
        self,
        tokenizer,  # type: ignore
        max_seq_len: int,
        data_dir: str = "",
        train_filename: str = "train.json",
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = "test.json",
        dev_split: float = 0.1,
        proxies: Optional[dict] = None,
        max_samples: Optional[int] = None,
        embed_title: bool = True,
        num_positives: int = 1,
        num_hard_negatives: int = 1,
        shuffle_negatives: bool = True,
        shuffle_positives: bool = False,
        label_list: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        :param query_tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a passage (str) into tokens.
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automatically
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `haystack.basics.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/haystack/blob/main/haystack/basics/data_handler/utils.py>`_.
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: None
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_samples: maximum number of samples to use
        :param embed_title: Whether to embed title in passages during tensorization (bool),
        :param num_hard_negatives: maximum number to hard negative context passages in a sample
        :param num_positives: maximum number to positive context passages in a sample
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the num_hard_negative number of passages
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the num_positive number of passages
        :param label_list: list of labels to predict. Usually ["hard_negative", "positive"]
        :param kwargs: placeholder for passing generic parameters
        """
        # TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        self.embed_title = embed_title
        self.num_hard_negatives = num_hard_negatives
        self.num_positives = num_positives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.max_seq_len = max_seq_len
        
        super(TripletSimilarityProcessor, self).__init__(
            tokenizer=tokenizer,  # type: ignore
            max_seq_len=self.max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        
        ## TODO: 修改add_task这里，加入两种不同的task: similarity and MLM
        self.add_task(
            name="text_similarity",
            metric="text_similarity_metric",
            label_list=["hard_negative", "positive"],
            label_name="similarity_label",
            task_type="text_similarity",
        )
        self.add_task(
            name="masked_lm",
            metric="mlm_metrics",
            label_list=['input_ids'], ## TODO: make it clear the usage of lable_list
            label_name="mlm_label",
            task_type="mlm",
        )
        

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Overwriting method from parent class to **always** load the TextSimilarityProcessor instead of the specific class stored in the config.

        :param load_dir: directory that contains a 'processor_config.json'
        :return: An instance of an TextSimilarityProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        with open(processor_config_file) as f:
            config = json.load(f)
        # init tokenizers
        tokenizer_class: Type[PreTrainedTokenizer] = getattr(transformers, config["tokenizer"])
        tokenizer = tokenizer_class.from_pretrained(
            pretrained_model_name_or_path=load_dir
        )

        processor = cls.load(
            tokenizer=tokenizer,
            processor_name="TripletSimilarityProcessor",
            **config,
        )
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def save(self, save_dir: Union[str, Path]):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))
        
        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: Optional[List[int]] = None, return_baskets: bool = False, debug: bool = False
    ):
        """
        Convert input dictionaries into a pytorch dataset for TextSimilarity (e.g. DPR).
        For conversion we have an internal representation called "baskets".
        Each basket is one query and related text passages (positive passages fitting to the query and negative
        passages that do not fit the query)
        Each stage adds or transforms specific information to our baskets.

        :param dicts: input dictionary with DPR-style content
                        {"query": str,
                         "passages": List[
                                        {'title': str,
                                        'text': str,
                                        'label': 'hard_negative',
                                        'external_id': str},
                                        ....
                                        ]
                         }
        :param indices: indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: whether to return the baskets or not (baskets are needed during inference)
        :return: dataset, tensor_names, problematic_ids, [baskets]
        """
        if indices is None:
            indices = []
        # Take the dict and insert into our basket structure, this stages also adds an internal IDs
        baskets = self._fill_baskets(dicts, indices)

        # Separate conversion of query
        baskets = self._convert_queries(baskets=baskets)

        # and context passages. When converting the context the label is also assigned.
        baskets = self._convert_contexts(baskets=baskets)
                    
        baskets = self._convert_question_table(baskets=baskets)


        # for basket in baskets:
        #     sample = basket.samples[0]
        #     for key, values in sample.features[0].items():
        #         print(key, type(values))
        #         if isinstance(values, torch.Tensor):
        #             print(values.shape)

        # Convert features into pytorch dataset, this step also removes and logs potential errors during preprocessing
        dataset, tensor_names, problematic_ids, baskets = self._create_dataset(baskets)

        if problematic_ids:
            logger.error(
                "There were %s errors during preprocessing at positions: %s", len(problematic_ids), problematic_ids
            )

        if return_baskets:
            return dataset, tensor_names, problematic_ids, baskets
        else:
            return dataset, tensor_names, problematic_ids

    def file_to_dicts(self, file: str) -> List[dict]:
        """
        Converts a Dense Passage Retrieval (DPR) data file in json format to a list of dictionaries.

        :param file: filename of DPR data in json format
                Each sample is a dictionary of format:
                {"dataset": str,
                "question": str,
                "answers": list of str
                "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                }


        Returns:
        list of dictionaries: List[dict]
            each dictionary:
            {"query": str,
            "passages": [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
            {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
            ...]}
        """
        dicts = _read_dpr_json(
            file,
            max_samples=self.max_samples,
            num_hard_negatives=self.num_hard_negatives,
            num_positives=self.num_positives,
            shuffle_negatives=self.shuffle_negatives,
            shuffle_positives=self.shuffle_positives,
        )

        # shuffle dicts to make sure that similar positive passages do not end up in one batch
        #############  We delete the shuffling here  #############
        # dicts = random.sample(dicts, len(dicts))
        return dicts

    def _fill_baskets(self, dicts: List[dict], indices: Optional[List[int]]):
        baskets = []
        if not indices:
            indices = list(range(len(dicts)))
        for d, id_internal in zip(dicts, indices):
            basket = SampleBasket(id_external=None, id_internal=id_internal, raw=d)
            baskets.append(basket)
        return baskets


    def _word_mask(self, input_ids):
        labels = input_ids.clone()

        # 获取 special tokens 的位置
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
            dtype=torch.bool
        )

        # 将input_ids转换为tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 跟踪每个单词的起始位置和长度, 这个地方需要进行修改
        word_boundaries = []
        current_word = []
        for idx, token in enumerate(tokens):
            if token in ['[CLS]', '[PAD]', '[SEP]']:
                continue
            if token.startswith("##"):
                current_word.append(idx)
            else:
                if current_word:
                    word_boundaries.append(current_word)
                current_word = [idx]
        if current_word:
            word_boundaries.append(current_word)
        
        num_words_to_mask = max(1, int(0.15 * len(word_boundaries)))  # 确保至少有一个单词被 mask

        words_to_mask = random.sample(word_boundaries, num_words_to_mask)

        # 设置 mask indices
        mask_indices = [idx for word in words_to_mask for idx in word]

        # 将选中的单词替换为 [MASK]
        input_ids[mask_indices] = self.tokenizer.mask_token_id

        # 生成 labels，将非 special token 和非 mask token 的 label 设为 -100
        labels[special_tokens_mask] = -100
        labels[~special_tokens_mask & ~(input_ids == self.tokenizer.mask_token_id)] = -100

        return input_ids, labels


    def _convert_queries(self, baskets: List[SampleBasket]):
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features = [{}]  # type: ignore
            
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                # try:
                query = self._normalize_question(basket.raw["query"])
                # 预先tokenize查询问题来估算token数量
                tokenized_query = self.tokenizer.tokenize(query)

                # 如果token数量超过256，截断tokenized_query到256个token
                if len(tokenized_query) > 256:
                    # 使用截断后的token重新构建query字符串
                    query = self.tokenizer.convert_tokens_to_string(tokenized_query[:256])

                table = pd.DataFrame()  # Provide an empty table if no table is available
                encoding = self.tokenizer(
                    table=table,
                    queries=query,
                    add_special_tokens=True,
                    truncation='drop_rows_to_fit',
                    max_length=self.max_seq_len,
                    padding="max_length",
                    return_tensors="pt"
                )

                query_inputs = {
                    "input_ids": encoding["input_ids"][0],
                    "token_type_ids": encoding["token_type_ids"][0],
                    "attention_mask": encoding["attention_mask"][0],
                }

                # Mask 15% of the words
                query_inputs["input_ids"], query_labels = self._word_mask(query_inputs["input_ids"])

                # print('-------------------------')
                # print(query_inputs['input_ids'])
                # print(query_labels)

                # tokenize query
                tokenized_query = self.tokenizer.convert_ids_to_tokens(query_inputs["input_ids"])

                if len(tokenized_query) == 0:
                    logger.warning(
                        "The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize"
                    )
                    return None
                
                # clear_text["query_text"] = query
                tokenized["query_tokens"] = query_inputs["input_ids"]
                features[0]["query_input_ids"] = query_inputs["input_ids"]
                features[0]["query_segment_ids"] = query_inputs["token_type_ids"]
                features[0]["query_attention_mask"] = query_inputs["attention_mask"]
                features[0]["query_labels"] = query_labels

                # print('*************************')
                # print(query_labels)
                # print(query_inputs["input_ids"])
            
            sample = Sample(id="", clear_text=clear_text, tokenized=tokenized, features=features)  # type: ignore
            basket.samples = [sample]
        return baskets

    def _to_max_len(self, seq_list: List[int], pad_id: int, max_len: int): 

        if len(seq_list) < max_len: 
            seq_list.extend([pad_id for _ in range(max_len-len(seq_list))])
        seq_list = seq_list[: max_len]
        
        return seq_list 
    
    def _convert_contexts(self, baskets: List[SampleBasket]):
        for basket in baskets:
            if "passages" in basket.raw:
                # try:
                positive_context = [p for p in basket.raw["passages"] if p["label"] == "positive"][:self.num_positives]

                hard_negative_context = [p for p in basket.raw["passages"] if p["label"] == "hard_negative"][:self.num_hard_negatives]

                ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives

                all_contexts = positive_context + hard_negative_context

                ctx_inputs = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "labels": []}

                for table in all_contexts:
                    table_text = table['text']

                    title = table_text.get('title', '').strip()
                    columns = [col['text'].strip() for col in table_text.get('columns', [])][:50]

                    table_df = pd.DataFrame(columns=columns)
                    # print('-----------------------------')
                    # print(table_df)
                    # print(table_text.get('rows', {}))

                    try:
                        for row_key, row_value in table_text.get('rows', {}).items():
                            row_idx = int(row_key)
                            if row_idx > 200:
                                continue

                            if row_idx not in table_df.index:
                                table_df.loc[row_idx] = [None] * len(columns)

                            for col_idx, cell in enumerate(row_value['cells']):
                                if col_idx >= len(columns):
                                    continue
                                cell_text = ' '.join(cell['text'].strip().split()[:8])
                                table_df.iat[row_idx, col_idx] = cell_text
                    except IndexError as e:
                        print(f"Error accessing row {row_idx}, column {col_idx}: {str(e)}")
                    
                    # print(f'-----------------Table----------------------')
                    # print(table_df)
                    # print(title)
                    encoding = self.tokenizer(
                        table=table_df,
                        queries=title,
                        truncation="drop_rows_to_fit",
                        padding="max_length",
                        max_length=self.max_seq_len,
                        return_tensors="pt"
                    )

                    # print("###############Table###################")
                    # print(encoding["input_ids"][0])

                    # Mask 15% of the word-piece tokens
                    
                    try:
                        input_ids, labels = self._word_mask(encoding["input_ids"][0])
                    except:
                        print(title, table_df)
                        print(table_text)

                    input_ids, labels = self._word_mask(encoding["input_ids"][0])

                    # print(input_ids)
                    # print(labels)
                
                    ctx_inputs["input_ids"].append(input_ids)
                    ctx_inputs["token_type_ids"].append(encoding["token_type_ids"][0])
                    ctx_inputs["attention_mask"].append(encoding["attention_mask"][0])
                    ctx_inputs["labels"].append(labels)

                    
                for key, value in ctx_inputs.items():
                    if value is not None:
                        ctx_inputs[key] = torch.stack(value)
                
                sample = basket.samples[0]  

                sample.features[0]["passage_input_ids"] = ctx_inputs["input_ids"] 
                sample.features[0]["passage_segment_ids"] = ctx_inputs["token_type_ids"] 
                sample.features[0]["passage_attention_mask"] = ctx_inputs["attention_mask"]  
                sample.features[0]["passage_labels"] = ctx_inputs["labels"]  
                sample.features[0]["label_ids"] = ctx_label 

                # except Exception:
                #     basket.samples[0].features = None  

        return baskets
    
    
    def _convert_question_table(self, baskets: List[SampleBasket]):
        query_cache = {}  # Cache tokenization results for queries
        for basket in baskets:
            if "query" in basket.raw and "passages" in basket.raw:
                raw_query = basket.raw["query"]
                if raw_query not in query_cache:
                    query = self._normalize_question(raw_query)
                    tokenized_query = self.tokenizer.tokenize(query)
                    if len(tokenized_query) > 256:
                        query = self.tokenizer.convert_tokens_to_string(tokenized_query[:256])
                    query_cache[raw_query] = query
                else:
                    query = query_cache[raw_query]
                
                # Simplified context handling using list comprehensions
                contexts = basket.raw["passages"]
                positive_context = [x for x in contexts if x["label"] == "positive"][:self.num_positives]
                hard_negative_context = [x for x in contexts if x["label"] == "hard_negative"][:self.num_hard_negatives]

                all_contexts = positive_context + hard_negative_context
                qt_inputs = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "labels": []}

                for table in all_contexts:
                    table = table['text']
                    title = table.get('title', '').strip()
                    columns = [col['text'].strip() for col in table.get('columns', [])][:50]

                    
                    table_df = pd.DataFrame(columns=columns)

                    for row_key, row_value in table['rows'].items():
                        row_idx = int(row_key)  # converting the row key to an integer index
                        cell_data = row_value['cells']
                        if row_idx > 200:
                            continue  # Skip rows with index greater than 255

                        # Ensure the row exists in the DataFrame
                        if row_idx not in table_df.index:
                            table_df.loc[row_idx] = [None] * len(columns)
                        
                        for col_idx, cell in enumerate(cell_data):
                            if col_idx >= len(columns):
                                continue  # Skip columns that are out of the defined range
                            cell_text = ' '.join(str(cell['text']).strip().split()[:8])  # Limit cell text length
                            
                            # Place the cell text in the correct location in the DataFrame
                            table_df.iat[row_idx, col_idx] = cell_text
                        
                    question = query + ' [SEP] ' + title

                    # print(f'-----------------Question + Table ----------------------')
                    # print(table_df)
                    # print(question)

                    encoding = self.tokenizer(
                        table=table_df,
                        queries=question,
                        truncation="drop_rows_to_fit",
                        padding="max_length",
                        max_length=self.max_seq_len,
                        return_tensors="pt"
                    )

                    input_ids, labels = self._word_mask(encoding["input_ids"][0])
                    qt_inputs["input_ids"].append(input_ids)
                    qt_inputs["token_type_ids"].append(encoding["token_type_ids"][0])
                    qt_inputs["attention_mask"].append(encoding["attention_mask"][0])
                    qt_inputs["labels"].append(labels)

                for key, value in qt_inputs.items():
                    if value:
                        qt_inputs[key] = torch.stack(value)

                sample = basket.samples[0]
                sample.features[0].update({
                    "qt_input_ids": qt_inputs["input_ids"],
                    "qt_segment_ids": qt_inputs["token_type_ids"],
                    "qt_attention_mask": qt_inputs["attention_mask"],
                    "qt_labels": qt_inputs["labels"]
                })

        return baskets

    def _create_dataset(self, baskets: List[SampleBasket]):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat: List[dict] = []
        basket_to_remove = []
        problematic_ids: set = set()
        
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:  
                    features_flat.extend(sample.features)  
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                problematic_ids.add(basket.id_internal)
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, problematic_ids, baskets

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Removes '?' from queries/questions"""
        if question[-1] == "?":
            question = question[:-1]
        return question

    @staticmethod
    def _combine_title_context(titles: List[str], texts: List[str]):
        res = []
        for title, ctx in zip(titles, texts):
            if title is None:
                title = ""
                logger.warning(
                    "Couldn't find title although `embed_title` is set to True for DPR. Using title='' now. Related passage text: '%s' ",
                    ctx,
                )
            res.append((title, ctx))
        return res




def _read_dpr_json(
    file: str,
    max_samples: Optional[int] = None,
    proxies: Optional[Any] = None,
    num_hard_negatives: int = 1,
    num_positives: int = 1,
    shuffle_negatives: bool = True,
    shuffle_positives: bool = False,
):
    """
    Reads a Dense Passage Retrieval (DPR) data file in json format and returns a list of dictionaries.

    :param file: filename of DPR data in json format

    Returns:
        list of dictionaries: List[dict]
        each dictionary: {
                    "query": str -> query_text
                    "passages": List[dictionaries] -> [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
                                {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
                                ...]
                    }
        example:
                ["query": 'who sings does he love me with reba'
                "passages" : [{'title': 'Does He Love You',
                    'text': 'Does He Love You "Does He Love You" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba\'s album "Greatest Hits Volume Two". It is one of country music\'s several songs about a love triangle. "Does He Love You" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members',
                    'label': 'positive',
                    'external_id': '11828866'},
                    {'title': 'When the Nightingale Sings',
                    'text': "When the Nightingale Sings When The Nightingale Sings is a Middle English poem, author unknown, recorded in the British Library's Harley 2253 manuscript, verse 25. It is a love poem, extolling the beauty and lost love of an unknown maiden. When þe nyhtegale singes þe wodes waxen grene.<br> Lef ant gras ant blosme springes in aueryl y wene,<br> Ant love is to myn herte gon wiþ one spere so kene<br> Nyht ant day my blod hit drynkes myn herte deþ me tene. Ich have loved al þis er þat y may love namore,<br> Ich have siked moni syk lemmon for",
                    'label': 'hard_negative',
                    'external_id': '10891637'}]
                ]

    """
    # get remote dataset if needed
    if not os.path.exists(file):
        logger.info("Couldn't find %s locally. Trying to download ...", file)
        _download_extract_downstream_data(file, proxies=proxies)

    if Path(file).suffix.lower() == ".jsonl":
        dicts = []
        with open(file, encoding="utf-8") as f:
            for line in f:
                dicts.append(json.loads(line))
    else:
        with open(file, encoding="utf-8") as f:
            dicts = json.load(f)

    if max_samples:
        dicts = random.sample(dicts, min(max_samples, len(dicts)))

    # convert DPR dictionary to standard dictionary
    query_json_keys = ["question", "questions", "query"]
    positive_context_json_keys = ["positive_contexts", "positive_ctxs", "positive_context", "positive_ctx"]
    hard_negative_json_keys = [
        "hard_negative_contexts",
        "hard_negative_ctxs",
        "hard_negative_context",
        "hard_negative_ctx",
    ]
    standard_dicts = []
    for dict in dicts:
        sample = {}
        passages = []
        for key, val in dict.items():
            if key in query_json_keys:
                sample["query"] = val
            elif key in positive_context_json_keys:
                if shuffle_positives:
                    random.shuffle(val)
                for passage in val[:num_positives]:
                    passages.append(
                        {
                            "title": passage["title"],
                            "text": passage["text"],
                            "label": "positive",
                            "passage_id": passage["passage_id"],
                            "external_id": passage.get("passage_id", uuid.uuid4().hex.upper()[0:8]),
                        }
                    )
            elif key in hard_negative_json_keys:
                if shuffle_negatives:
                    random.shuffle(val)
                for passage in val[:num_hard_negatives]:
                    passages.append(
                        {
                            "title": passage["title"],
                            "text": passage["text"],
                            "label": "hard_negative",
                            "passage_id": passage["passage_id"],
                            "external_id": passage.get("passage_id", uuid.uuid4().hex.upper()[0:8]),
                        }
                    )
        sample["passages"] = passages
        standard_dicts.append(sample)
    return standard_dicts


def _read_squad_file(filename: str, proxies=None):
    """Read a SQuAD json file"""
    if not os.path.exists(filename):
        logger.info("Couldn't find %s locally. Trying to download ...", filename)
        _download_extract_downstream_data(filename, proxies)
    with open(filename, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    return input_data


def http_get(
    url: str,
    temp_file: IO[bytes],
    proxies: Optional[Dict[str, str]] = None,
    timeout: Union[float, Tuple[float, float]] = 10.0,
):
    """
    Runs a HTTP GET requests and saves response content to file.
    :param url: URL address
    :param temp_file: file-like object open in binary mode
    :param proxies: (optional) Dictionary mapping protocol to the URL of the proxy.
    :param timeout: How many seconds to wait for the server to send data before giving up,
        as a float, or a :ref:`(connect timeout, read timeout) <timeouts>` tuple.
        Defaults to 10 seconds.
    """
    req = requests.get(url, stream=True, proxies=proxies, timeout=timeout)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def _download_extract_downstream_data(input_file: str, proxies=None):
    # download archive to temp dir and extract to correct position
    full_path = Path(os.path.realpath(input_file))
    directory = full_path.parent
    taskname = directory.stem
    datadir = directory.parent
    logger.info("downloading and extracting file %s to dir %s", taskname, datadir)
    if taskname not in DOWNSTREAM_TASK_MAP:
        logger.error("Cannot download %s. Unknown data source.", taskname)
    else:
        if os.name == "nt":  # make use of NamedTemporaryFile compatible with Windows
            delete_tmp_file = False
        else:
            delete_tmp_file = True
        with tempfile.NamedTemporaryFile(delete=delete_tmp_file) as temp_file:
            http_get(DOWNSTREAM_TASK_MAP[taskname], temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)
        # temp_file gets deleted here


def _is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False



def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        
        sample_data = [sample[t_name] for sample in features]
        if isinstance(sample_data[0], torch.Tensor) and sample_data[0].dtype == torch.long:
            # All elements must be tensors of the same size
            cur_tensor = torch.stack(sample_data)
        else:
            try:
                # Checking whether a non-integer will be silently converted to torch.long
                check = features[0][t_name]
                if isinstance(check, numbers.Number):
                    base = check
                # extract a base variable from a nested lists or tuples
                elif isinstance(check, list):
                    base = list(flatten_list(check))[0]
                # extract a base variable from numpy arrays
                else:
                    base = check.ravel()[0]
                if not np.issubdtype(type(base), np.integer):
                    logger.warning(
                        "Problem during conversion to torch tensors:\n"
                        "A non-integer value for feature '%s' with a value of: "
                        "'%s' will be converted to a torch tensor of dtype long.",
                        t_name,
                        base,
                    )
            except:
                logger.debug(
                    "Could not determine type for feature '%s'. Converting now to a tensor of default type long.", t_name
                )
            
            # if t_name == 'passage_input_ids':
            #     print('------------------------')
            #     print(len(sample_data))

            #     for i in range(len(sample_data)):
            #         print(len(sample_data[i]))
            #         # assert  len(sample_data[i]) == 8
                
            
            # Convert all remaining python objects to torch long tensors
            cur_tensor = torch.as_tensor(np.array(sample_data), dtype=torch.long)

        all_tensors.append(cur_tensor)
    
    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names

