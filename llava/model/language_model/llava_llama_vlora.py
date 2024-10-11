#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from ..VLoRA.modeling_llama_vlora import LlamaVLoRAModel, LlamaVLoRAForCausalLM
                         
from llava.constants import IMAGE_TOKEN_INDEX
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..llava_arch_vlora import LlavaVLoRAMetaModel, LlavaVLoRAMetaForCausalLM

from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput

class LlavaVLoRAConfig(LlamaConfig):
    model_type = "llava_vlora"


class LlavaLlamaVLoRAModel(LlavaVLoRAMetaModel, LlamaVLoRAModel):
    config_class = LlavaVLoRAConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaVLoRAModel, self).__init__(config)


class LlavaLlamaVLoRAForCausalLM(LlamaVLoRAForCausalLM, LlavaVLoRAMetaForCausalLM):
    config_class = LlavaVLoRAConfig

    def __init__(self, config):
        super(LlamaVLoRAForCausalLM, self).__init__(config)
        self.model = LlavaLlamaVLoRAModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_vlora_weights(self, images, input_ids):
        if images is None:
            return None
        vlora_weights = self.encode_images(images)

        # fill zero for language-only data
        mask = input_ids == IMAGE_TOKEN_INDEX
        mask = (mask.sum(1) > 0).long().reshape(-1, 1, 1)
        for vlora_weights_sub in vlora_weights:
            for key in vlora_weights_sub:
                if vlora_weights_sub[key] != (None, None):
                    vlora_weights_sub[key] = (vlora_weights_sub[key][0] * mask, 
                                              vlora_weights_sub[key][1])

        return vlora_weights

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        vlora_weights = None,
        img_token_idx = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if vlora_weights is None:
            vlora_weights = self.get_vlora_weights(images, input_ids)
        
        assert vlora_weights is not None

        # TODO: an ugly way to handle IMAGE_TOKEN_INDEX
        mask = input_ids == IMAGE_TOKEN_INDEX
        input_ids[mask] = 0
        if mask.shape != attention_mask.shape:  # not equal when .generate()
            assert img_token_idx is not None
            for i in range(img_token_idx.shape[0]):
                if img_token_idx[i] >= 0:
                    attention_mask[i, img_token_idx[i]] = 0
        else:
            attention_mask[mask] = 0
        
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vlora_weights=vlora_weights
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        images = kwargs.pop("images", None)
        vlora_weights = self.get_vlora_weights(images, inputs)

        img_token_idx_list = []
        for i in range(inputs.shape[0]):
            mask = (inputs[i] == IMAGE_TOKEN_INDEX).int()
            if mask.sum() > 0:
                img_token_idx_list.append(mask.argmax().item())
            else:
                img_token_idx_list.append(-1)
        img_token_idx = torch.Tensor(img_token_idx_list).long().to(images.device)

        return super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            # assistant_model, # fixme: remove for transformers 4.28.0
            streamer,
            vlora_weights=vlora_weights,
            img_token_idx=img_token_idx,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        vlora_weights = kwargs.pop("vlora_weights", None)
        img_token_idx = kwargs.pop("img_token_idx", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if vlora_weights is not None:
            _inputs['vlora_weights'] = vlora_weights
        if img_token_idx is not None:
            _inputs['img_token_idx'] = img_token_idx
        return _inputs

AutoConfig.register("llava_vlora", LlavaVLoRAConfig)
AutoModelForCausalLM.register(LlavaVLoRAConfig, LlavaLlamaVLoRAForCausalLM)
