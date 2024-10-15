# coding=utf-8
# Copyright 2018 DPR Authors, The Hugging Face Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DPR model for Open Domain Question Answering."""


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging, ModelOutput
from transformers.models.tapas.modeling_tapas import TapasConfig, TapasModel, TapasOnlyMLMHead, TapasPreTrainedModel


logger = logging.get_logger(__name__)


class TapasEncoder(TapasModel):
    def __init__(self, config):
        TapasModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        row_ids: Optional[Tensor] = None,
        col_ids: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        pooled: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        pooled_output = outputs.pooler_output
        
        
        if not pooled:
            pooled_output = sequence_output[:, 0, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=hidden_states,
        )


class TapasForMaskedEncoderOutput(ModelOutput):
    mlm_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class TapasForMaskedEncoder(TapasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")  # 使用异常处理替代断言
        self.tapas = TapasModel(config, add_pooling_layer=False)
        self.cls = TapasOnlyMLMHead(config)
        self.post_init()
        
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        row_ids: Optional[Tensor] = None,
        col_ids: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        pooled: bool = False,
    ) -> Union[TapasForMaskedEncoderOutput, Tuple[Tensor, ...]]:
        
        outputs = self.tapas(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        pooled_output = outputs.pooler_output

        prediction_scores = self.cls(sequence_output)
        
        # print(self.cls.predictions.decoder.weight)
        
        if not pooled:
            pooled_output = sequence_output[:, 0, :]

        return TapasForMaskedEncoderOutput(
            last_hidden_state=sequence_output, 
            pooler_output=pooled_output, 
            hidden_states=hidden_states, 
            mlm_logits=prediction_scores
        )
    

       