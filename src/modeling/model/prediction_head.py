import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import AutoModelForQuestionAnswering

from haystack.modeling.data_handler.samples import SampleBasket
from haystack.modeling.model.predictions import QACandidate, QAPred
from haystack.modeling.utils import try_get, all_gather_list
from haystack.utils.scipy_utils import expit
from haystack.modeling.model.prediction_head import PredictionHead


logger = logging.getLogger(__name__)


class TripletSimilarityHead(PredictionHead):
    """
    Trains a head on predicting the similarity of two texts like in Dense Passage Retrieval.
    """

    def __init__(self, similarity_function: str = "dot_product", global_loss_buffer_size: int = 150000, **kwargs):
        """
        Init the TripletSimilarityHead.

        :param similarity_function: Function to calculate similarity between queries and passage embeddings.
                                    Choose either "dot_product" (Default) or "cosine".
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param kwargs:
        """
        super(TripletSimilarityHead, self).__init__()

        self.similarity_function = similarity_function
        self.loss_fct = NLLLoss(reduction="mean")
        self.task_name = "text_similarity"
        self.model_type = "text_similarity"
        self.ph_output_type = "per_sequence"
        self.global_loss_buffer_size = global_loss_buffer_size
        self.generate_config()

    @classmethod
    def dot_product_scores(cls, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates dot product similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                        of dimension n1 x D,
                        where n1 is the number of queries/batch size and D is embedding size
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                        of dimension n2 x D,
                        where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                        and D is embedding size

        :return: dot_product: similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        dot_product = torch.matmul(query_vectors, torch.transpose(passage_vectors, 0, 1))
        return dot_product

    @classmethod
    def cosine_scores(cls, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates cosine similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                          and D is embedding size

        :return: cosine similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        cosine_similarities = []
        passages_per_batch = passage_vectors.shape[0]
        for query_vector in query_vectors:
            query_vector_repeated = query_vector.repeat(passages_per_batch, 1)
            current_cosine_similarities = nn.functional.cosine_similarity(query_vector_repeated, passage_vectors, dim=1)
            cosine_similarities.append(current_cosine_similarities)
        return torch.stack(cosine_similarities)

    def get_similarity_function(self):
        """
        Returns the type of similarity function used to compare queries and passages/contexts
        """
        if "dot_product" in self.similarity_function:
            return TripletSimilarityHead.dot_product_scores
        elif "cosine" in self.similarity_function:
            return TripletSimilarityHead.cosine_scores
        else:
            raise AttributeError(
                f"The similarity function can only be 'dot_product' or 'cosine', not '{self.similarity_function}'"
            )

    def forward(self, query_vectors: torch.Tensor, table_vectors: torch.Tensor, text_table_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Only packs the embeddings from both language models into a tuple. No further modification.
        The similarity calculation is handled later to enable distributed training (DDP)
        while keeping the support for in-batch negatives.
        (Gather all embeddings from nodes => then do similarity scores + loss)

        :param query_vectors: Tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :param passage_vectors: Tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is the number of queries/batch size and D is embedding size
        """
        return query_vectors, table_vectors, text_table_vectors

    def _embeddings_to_scores(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates similarity scores between all given query_vectors and passage_vectors

        :param query_vectors: Tensor of queries encoded by the query encoder model
        :param passage_vectors: Tensor of passages encoded by the passage encoder model
        :return: Tensor of log softmax similarity scores of each query with each passage (dimension: n1xn2)
        """
        sim_func = self.get_similarity_function()
        scores = sim_func(query_vectors, passage_vectors)

        if len(query_vectors.size()) > 1:
            q_num = query_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = nn.functional.log_softmax(scores, dim=1)
        return softmax_scores
    
    def logits_to_loss(self, logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], label_ids, **kwargs): 
        """
        Computes the loss (Default: NLLLoss) by applying a similarity function (Default: dot product) to the input
        tuple of (query_vectors, table_vectors, query_table_vectors) and afterwards applying the loss function on similarity scores.

        :param logits: Tuple of Tensors (query_embedding, table_embedding, query_table_embedding) as returned from forward()

        :return: negative log likelihood loss from similarity scores
        """
        # Check if DDP is initialized
        try:
            if torch.distributed.is_available():
                rank = torch.distributed.get_rank()
            else:
                rank = -1
        except (AssertionError, RuntimeError):
            rank = -1

        # Prepare predicted scores
        query_vectors, table_vectors, query_table_vectors = logits

        # Prepare Labels
        # positive_idx_per_question = torch.nonzero((label_ids.view(-1) == 1), as_tuple=False)
        positive_idx_per_question = torch.nonzero((label_ids.view(-1) == 1), as_tuple=False).squeeze(-1)

        # Gather global embeddings from all distributed nodes (DDP)
        if rank != -1:
            q_vector_to_send = torch.empty_like(query_vectors).cpu().copy_(query_vectors).detach_()
            t_vector_to_send = torch.empty_like(table_vectors).cpu().copy_(table_vectors).detach_()
            qt_vector_to_send = torch.empty_like(query_table_vectors).cpu().copy_(query_table_vectors).detach_()

            global_vectors = all_gather_list(
                [q_vector_to_send, t_vector_to_send, qt_vector_to_send, positive_idx_per_question], max_size=self.global_loss_buffer_size
            )

            global_query_vectors = []
            global_table_vectors = []
            global_query_table_vectors = []
            global_positive_idx_per_question = []
            total_tables = 0
            for i, item in enumerate(global_vectors):
                q_vector, t_vector, qt_vector, positive_idx = item

                if i != rank:
                    global_query_vectors.append(q_vector.to(query_vectors.device))
                    global_table_vectors.append(t_vector.to(table_vectors.device))
                    global_query_table_vectors.append(qt_vector.to(query_table_vectors.device))
                    global_positive_idx_per_question.extend([v + total_tables for v in positive_idx])
                else:
                    global_query_vectors.append(query_vectors)
                    global_table_vectors.append(table_vectors)
                    global_query_table_vectors.append(query_table_vectors)
                    global_positive_idx_per_question.extend([v + total_tables for v in positive_idx_per_question])
                total_tables += t_vector.size(0)

            global_query_vectors = torch.cat(global_query_vectors, dim=0) 
            global_table_vectors = torch.cat(global_table_vectors, dim=0) 
            global_query_table_vectors = torch.cat(global_query_table_vectors, dim=0) 
            global_positive_idx_per_question = torch.LongTensor(global_positive_idx_per_question) 
        else:
            global_query_vectors = query_vectors 
            global_table_vectors = table_vectors 
            global_query_table_vectors = query_table_vectors 
            global_positive_idx_per_question = positive_idx_per_question 

        # Get similarity scores
        # text <-> tables [query-size, table-size]
        query_table_scores = self._embeddings_to_scores(global_query_vectors, global_table_vectors) 
        
        # table <-> table + text (需要找到对应的table才为1) [table-size, table-size]
        table_query_table_scores = self._embeddings_to_scores(global_table_vectors, global_query_table_vectors) 
        # text <-> table + text (需要找到对应的text) [table-size, query-size]
        query_table_query_scores = self._embeddings_to_scores(global_query_table_vectors, global_query_vectors)

        # Calculate losses
        targets = global_positive_idx_per_question.squeeze(-1).to(query_table_scores.device)
        
        if targets.dim() == 0:
            targets = targets.unsqueeze(0).to(query_table_scores.device)
            
        # print('--------------------Triplet Similarity---------------------')
        # print(query_table_loss)
        
        query_table_loss = self.loss_fct(query_table_scores, targets)
        
        
        max_positive_id = table_query_table_scores.shape[0]
        target_2 = torch.arange(max_positive_id, device=table_query_table_scores.device)

        # print(target_2)
        # print(table_query_table_scores.shape)
        
        table_query_table_loss = self.loss_fct(table_query_table_scores, target_2)

        # 修改部分开始
        target_3 = []
        max_positive_id = query_table_query_scores.shape[1]
        span = query_table_query_scores.shape[0] // query_table_query_scores.shape[1]
        for i in range(max_positive_id):
            target_3.extend([i] * span)
        target_3 = torch.tensor(target_3, device=query_table_query_scores.device)

        # print(query_table_query_scores.shape)
        # print(target_3)

        query_table_query_loss = self.loss_fct(query_table_query_scores, target_3)
        # print('--------------TripletLoss---------------')
        # print(query_table_loss, table_query_table_loss, query_table_query_loss)
        # Sum the losses
        total_loss = query_table_loss + table_query_table_loss + query_table_query_loss
        return total_loss

    def logits_to_preds(self, logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        """
        Returns predicted ranks(similarity) of tables for each query, queries for each table, and queries for each query-table pair

        :param logits: tuple of tensors (query_vectors, table_vectors, query_table_vectors)

        :return: tuple of predicted ranks (query_table_ranks, table_query_ranks, query_table_query_ranks)
        """
    
        query_vectors, table_vectors, query_table_vectors = logits

        query_table_scores = self._embeddings_to_scores(query_vectors, table_vectors)

        _, query_table_ranks = torch.sort(query_table_scores, dim=1, descending=True)

        table_query_scores = self._embeddings_to_scores(table_vectors, query_table_vectors)
        _, table_query_ranks = torch.sort(table_query_scores, dim=1, descending=True)

        query_table_query_scores = self._embeddings_to_scores(query_table_vectors, query_vectors)
        _, query_table_query_ranks = torch.sort(query_table_query_scores, dim=1, descending=True)

        # print('--------------------------------------------------------------')
        # print(query_table_ranks.shape)
        # print(table_query_ranks.shape)
        # print(query_table_query_ranks.shape)

        return [query_table_ranks, table_query_ranks, query_table_query_ranks]

    def prepare_labels(self, query_input_ids, passage_input_ids, label_ids, **kwargs) -> torch.Tensor: 
        """
        Returns a tensor with passage labels(0:hard_negative/1:positive) for each query

        :return: passage labels(0:hard_negative/1:positive) for each query
        """
        
        labels_1 = torch.zeros(label_ids.size(0), label_ids.numel())

        positive_indices = torch.nonzero(label_ids.view(-1) == 1, as_tuple=False)

        for i, indx in enumerate(positive_indices):
            labels_1[i, indx.item()] = 1
        
        num_passages = passage_input_ids.shape[0] * passage_input_ids.shape[1]
        num_queries = query_input_ids.shape[0]
        
        labels_2 = torch.eye(num_passages, dtype=torch.float)


        labels_3 = torch.zeros(num_passages, num_queries)
        for i in range(num_queries):
            labels_3[i * num_passages: (i + 1) * num_passages, i] = 1


        return [labels_1, labels_2, labels_3]

    def formatted_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        raise NotImplementedError("formatted_preds is not supported in TextSimilarityHead yet!")


# class MLMHead(PredictionHead):
#     """
#     Masked Language Modeling (MLM) head for predicting masked tokens in a sequence.
#     """

#     def __init__(self, hidden_size: int, vocab_size: int, hidden_act: str = "gelu", layer_norm_eps: float = 1e-12, **kwargs):
#         """
#         Init the MLMHead.

#         :param hidden_size: Size of the hidden states.
#         :param vocab_size: Size of the vocabulary.
#         :param hidden_act: Activation function to use in the feed forward layer.
#         :param layer_norm_eps: Epsilon to use in the layer normalization layers.
#         :param kwargs:
#         """
#         super(MLMHead, self).__init__()

#         self.mlm_loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.task_name = "masked_lm"
#         self.model_type = "language_modeling"
#         self.ph_output_type = "per_token"

#         self.dense = nn.Linear(hidden_size, hidden_size)
#         self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
#         self.bias = nn.Parameter(torch.zeros(vocab_size))
#         self.activation = nn.GELU()

#         self.decoder.bias = self.bias
#         self.generate_config()
        
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         print("Dense layer weights:", self.dense.weight)
#         print("Dense layer bias:", self.dense.bias)
#         print("Layer norm weights:", self.layer_norm.weight)

#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.layer_norm(hidden_states)
#         logits = self.decoder(hidden_states)
#         return logits
    
#     def logits_to_loss(self, logits: torch.Tensor, query_labels, passage_labels, qt_labels, **kwargs) -> torch.Tensor:
#         # Check if DDP is initialized
#         try:
#             if torch.distributed.is_available():
#                 rank = torch.distributed.get_rank()
#             else:
#                 rank = -1
#         except (AssertionError, RuntimeError):
#             rank = -1

#         # Prepare predicted scores
#         mlm_logits_q, mlm_logits_t, mlm_logits_qt = logits

#         # Gather global embeddings from all distributed nodes (DDP)
#         if rank != -1:
#             q_logits_to_send = torch.empty_like(mlm_logits_q).cpu().copy_(mlm_logits_q).detach_()
#             t_logits_to_send = torch.empty_like(mlm_logits_t).cpu().copy_(mlm_logits_t).detach_()
#             qt_logits_to_send = torch.empty_like(mlm_logits_qt).cpu().copy_(mlm_logits_qt).detach_()

#             global_logits = all_gather_list(
#                 [q_logits_to_send, t_logits_to_send, qt_logits_to_send, query_labels, passage_labels, qt_labels],
#                 max_size=self.global_loss_buffer_size
#             )

#             global_mlm_logits_q = []
#             global_mlm_logits_t = []
#             global_mlm_logits_qt = []
#             global_query_labels = []
#             global_passage_labels = []
#             global_qt_labels = []

#             for i, item in enumerate(global_logits):
#                 q_logits, t_logits, qt_logits, q_labels, p_labels, qt_labels = item

#                 if i != rank:
#                     global_mlm_logits_q.append(q_logits.to(mlm_logits_q.device))
#                     global_mlm_logits_t.append(t_logits.to(mlm_logits_t.device))
#                     global_mlm_logits_qt.append(qt_logits.to(mlm_logits_qt.device))
#                     global_query_labels.append(q_labels.to(query_labels.device))
#                     global_passage_labels.append(p_labels.to(passage_labels.device))
#                     global_qt_labels.append(qt_labels.to(qt_labels.device))
#                 else:
#                     global_mlm_logits_q.append(mlm_logits_q)
#                     global_mlm_logits_t.append(mlm_logits_t)
#                     global_mlm_logits_qt.append(mlm_logits_qt)
#                     global_query_labels.append(query_labels)
#                     global_passage_labels.append(passage_labels)
#                     global_qt_labels.append(qt_labels)

#             global_mlm_logits_q = torch.cat(global_mlm_logits_q, dim=0)
#             global_mlm_logits_t = torch.cat(global_mlm_logits_t, dim=0)
#             global_mlm_logits_qt = torch.cat(global_mlm_logits_qt, dim=0)
#             global_query_labels = torch.cat(global_query_labels, dim=0)
#             global_passage_labels = torch.cat(global_passage_labels, dim=0)
#             global_qt_labels = torch.cat(global_qt_labels, dim=0)
#         else:
#             global_mlm_logits_q = mlm_logits_q
#             global_mlm_logits_t = mlm_logits_t
#             global_mlm_logits_qt = mlm_logits_qt
#             global_query_labels = query_labels
#             global_passage_labels = passage_labels
#             global_qt_labels = qt_labels

#         max_len = global_query_labels.shape[1]

#         # 修改 global_passage_labels 的形状
#         # global_passage_labels = global_passage_labels.view(-1, max_len)
#         # 修改 global_qt_labels 的形状
#         # global_qt_labels = global_qt_labels.view(-1, max_len)
        
#         # print('----------------------')
#         # print(global_mlm_logits_q)
#         # print(global_mlm_logits_q.view(-1, self.vocab_size).shape)
        
#         # print(global_query_labels.view(-1))
#         # print(global_query_labels)
#         non_negative_hundred_count = torch.sum(global_query_labels.view(-1) != -100).item()

#         # 打印结果
#         # print("非-100的元素数量：", non_negative_hundred_count)
#         # print(global_query_labels.view(-1).shape)

#         # print(global_passage_labels.shape)
#         # print(global_qt_labels.shape)
#         # print(global_mlm_logits_t.shape, global_passage_labels.shape)
#         # print(global_mlm_logits_qt.shape, global_qt_labels.shape)

#         masked_lm_loss_q = self.mlm_loss_fct(global_mlm_logits_q.view(-1, self.vocab_size), global_query_labels.view(-1))
#         masked_lm_loss_t = self.mlm_loss_fct(global_mlm_logits_t.view(-1, self.vocab_size), global_passage_labels.view(-1))
#         masked_lm_loss_qt = self.mlm_loss_fct(global_mlm_logits_qt.view(-1, self.vocab_size), global_qt_labels.view(-1))
        
#         print('----------------------MLM Loss----------------------')
#         print(masked_lm_loss_q, masked_lm_loss_t, masked_lm_loss_qt)
        
#         return masked_lm_loss_q + masked_lm_loss_t + masked_lm_loss_qt


#     def logits_to_preds(self, logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:
#         # Prepare predicted scores
#         mlm_logits_q, mlm_logits_t, mlm_logits_qt = logits
#         preds_q = torch.argmax(mlm_logits_q, dim=-1)
#         preds_t = torch.argmax(mlm_logits_t, dim=-1)
#         preds_qt = torch.argmax(mlm_logits_qt, dim=-1)
#         return [preds_q, preds_t, preds_qt]
    
#     # TODO: label_ids 不一定只为token_ids的长度，可能已经进行了处理
#     def prepare_labels(self, query_labels, passage_labels, qt_labels, **kwargs):
#         return [query_labels, passage_labels, qt_labels]

#     def formatted_preds(self, logits: torch.Tensor, **kwargs):
#         preds = torch.argmax(logits, dim=-1)
#         return preds
  

class MLMHead(PredictionHead):
    """
    Masked Language Modeling (MLM) head for predicting masked tokens in a sequence.
    """

    def __init__(self, hidden_size: int, vocab_size: int, hidden_act: str = "gelu", layer_norm_eps: float = 1e-12, **kwargs):
        """
        Init the MLMHead.

        :param hidden_size: Size of the hidden states.
        :param vocab_size: Size of the vocabulary.
        :param hidden_act: Activation function to use in the feed forward layer.
        :param layer_norm_eps: Epsilon to use in the layer normalization layers.
        :param kwargs:
        """
        super(MLMHead, self).__init__()

        self.mlm_loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.task_name = "masked_lm"
        self.model_type = "language_modeling"
        self.ph_output_type = "per_token"

        self.generate_config()
        
    def forward(self, logits_q: torch.Tensor, logits_t: torch.Tensor, logits_qt: torch.Tensor) -> torch.Tensor:
        return logits_q, logits_t, logits_qt
    
    def logits_to_loss(self, logits: torch.Tensor, query_labels, passage_labels, qt_labels, **kwargs) -> torch.Tensor:
        # Check if DDP is initialized
        try:
            if torch.distributed.is_available():
                rank = torch.distributed.get_rank()
            else:
                rank = -1
        except (AssertionError, RuntimeError):
            rank = -1

        # Prepare predicted scores
        mlm_logits_q, mlm_logits_t, mlm_logits_qt = logits

        # Gather global embeddings from all distributed nodes (DDP)
        if rank != -1:
            q_logits_to_send = torch.empty_like(mlm_logits_q).cpu().copy_(mlm_logits_q).detach_()
            t_logits_to_send = torch.empty_like(mlm_logits_t).cpu().copy_(mlm_logits_t).detach_()
            qt_logits_to_send = torch.empty_like(mlm_logits_qt).cpu().copy_(mlm_logits_qt).detach_()

            global_logits = all_gather_list(
                [q_logits_to_send, t_logits_to_send, qt_logits_to_send, query_labels, passage_labels, qt_labels],
                max_size=self.global_loss_buffer_size
            )

            global_mlm_logits_q = []
            global_mlm_logits_t = []
            global_mlm_logits_qt = []
            global_query_labels = []
            global_passage_labels = []
            global_qt_labels = []

            for i, item in enumerate(global_logits):
                q_logits, t_logits, qt_logits, q_labels, p_labels, qt_labels = item

                if i != rank:
                    global_mlm_logits_q.append(q_logits.to(mlm_logits_q.device))
                    global_mlm_logits_t.append(t_logits.to(mlm_logits_t.device))
                    global_mlm_logits_qt.append(qt_logits.to(mlm_logits_qt.device))
                    global_query_labels.append(q_labels.to(query_labels.device))
                    global_passage_labels.append(p_labels.to(passage_labels.device))
                    global_qt_labels.append(qt_labels.to(qt_labels.device))
                else:
                    global_mlm_logits_q.append(mlm_logits_q)
                    global_mlm_logits_t.append(mlm_logits_t)
                    global_mlm_logits_qt.append(mlm_logits_qt)
                    global_query_labels.append(query_labels)
                    global_passage_labels.append(passage_labels)
                    global_qt_labels.append(qt_labels)

            global_mlm_logits_q = torch.cat(global_mlm_logits_q, dim=0)
            global_mlm_logits_t = torch.cat(global_mlm_logits_t, dim=0)
            global_mlm_logits_qt = torch.cat(global_mlm_logits_qt, dim=0)
            global_query_labels = torch.cat(global_query_labels, dim=0)
            global_passage_labels = torch.cat(global_passage_labels, dim=0)
            global_qt_labels = torch.cat(global_qt_labels, dim=0)
        else:
            global_mlm_logits_q = mlm_logits_q
            global_mlm_logits_t = mlm_logits_t
            global_mlm_logits_qt = mlm_logits_qt
            global_query_labels = query_labels
            global_passage_labels = passage_labels
            global_qt_labels = qt_labels

        masked_lm_loss_q = self.mlm_loss_fct(global_mlm_logits_q.view(-1, self.vocab_size), global_query_labels.view(-1))
        masked_lm_loss_t = self.mlm_loss_fct(global_mlm_logits_t.view(-1, self.vocab_size), global_passage_labels.view(-1))
        masked_lm_loss_qt = self.mlm_loss_fct(global_mlm_logits_qt.view(-1, self.vocab_size), global_qt_labels.view(-1))
        
        # print('----------------------MLM Loss----------------------')
        # print(masked_lm_loss_q, masked_lm_loss_t, masked_lm_loss_qt)
        
        return masked_lm_loss_q + masked_lm_loss_t + masked_lm_loss_qt


    def logits_to_preds(self, logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:
        # Prepare predicted scores
        mlm_logits_q, mlm_logits_t, mlm_logits_qt = logits
        preds_q = torch.argmax(mlm_logits_q, dim=-1)
        preds_t = torch.argmax(mlm_logits_t, dim=-1)
        preds_qt = torch.argmax(mlm_logits_qt, dim=-1)
        return [preds_q, preds_t, preds_qt]
    
    # TODO: label_ids 不一定只为token_ids的长度，可能已经进行了处理
    def prepare_labels(self, query_labels, passage_labels, qt_labels, **kwargs):
        return [query_labels, passage_labels, qt_labels]

    def formatted_preds(self, logits: torch.Tensor, **kwargs):
        preds = torch.argmax(logits, dim=-1)
        return preds
  