# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""
EAGLE-3 draft model for Llama model

Adapted from
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama_eagle3.py
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mindspore as ms
import torch
from mindspore import Tensor, dtype, jit, mint, mutable, nn, ops
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

from sgl_mindspore.layers import (
    ColParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RMSNorm,
    VocabParallelEmbedding,
)
from sgl_mindspore.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP
from sgl_mindspore.utils import get_ms_dtype, tensor_torch2ms

logger = logging.getLogger(__name__)

LlamaConfig = None


class LlamaDecoderLayerEagle3(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            param_dtype=config.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.mlp = LlamaMLP(config, quant_config, prefix)

        self.hidden_norm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

    def construct(
        self,
        embeds: Tensor,
        hidden_states: Tensor,
        residual: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        layer_idx: int,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = mint.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            layer_idx=layer_idx,
            attn_mask=attn_mask,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
            block_tables=block_tables,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModelEagle3(nn.Cell):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(config=config)

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = ReplicatedLinear(
            input_size=self.hidden_size_in * 3,
            output_size=config.hidden_size,
            bias=getattr(config, "bias", False),
            param_dtype=config.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )

        self.midlayer = LlamaDecoderLayerEagle3(config, 0, quant_config, prefix)

        self.norm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

    def construct(
        self,
        input_ids,
        hidden_states=None,
        position_ids=None,
        attention_mask=None,
        batch_valid_length=None,
        is_prefill=True,
        q_seq_lens=None,
        key_cache=None,
        value_cache=None,
        out_cache_loc=None,
        block_tables=None,
    ):
        embeds = self.embed_tokens(input_ids)
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            embeds=embeds,
            hidden_states=hidden_states,
            residual=residual,
            positions=position_ids,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            layer_idx=0,
            attn_mask=attention_mask,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache[0],
            value_cache=value_cache[0],
            out_cache_loc=out_cache_loc,
            block_tables=block_tables,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.config = config
        self.quant_config = quant_config

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        if self.config.dtype:
            param_dtype = get_ms_dtype(self.config.dtype)
        else:
            param_dtype = ms.dtype.bfloat16
        setattr(self.config, "param_dtype", param_dtype)

        self.model = LlamaModelEagle3(config, quant_config, add_prefix("model", prefix))

        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ColParallelLinear(
                input_size=self.config.hidden_size,
                output_size=self.config.draft_vocab_size,
                bias=False,
                param_dtype=self.config.param_dtype,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = self.parameters_dict()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                loaded_weight = tensor_torch2ms(loaded_weight).move_to("Ascend")
                self.hot_token_id = loaded_weight + mint.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    assert hasattr(param, "weight_load")
                    weight_load = getattr(param, "weight_load")
                    weight_load(param, loaded_weight, shard_id)
                    param.set_data(param.move_to("Ascend"))
                    break
            else:
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    if hasattr(param, "weight_load"):
                        weight_load = getattr(
                            param, "weight_load", default_weight_loader
                        )
                        weight_load(param, loaded_weight)
                        param.set_data(param.move_to("Ascend"))
                    else:
                        param.set_data(tensor_torch2ms(loaded_weight).move_to("Ascend"))
                    # Make sure the weight is loaded on device, so the kv cache calculation is correct.

    def get_hot_token_id(self):
        return self.hot_token_id

    def prepare_inputs(self, forward_batch: ForwardBatch, model_inputs: Dict[str, Any]):
        if forward_batch.spec_info:
            model_inputs["hidden_states"] = tensor_torch2ms(
                forward_batch.spec_info.hidden_states
            )
        if forward_batch.capture_hidden_mode:
            model_inputs["capture_hidden_mode"] = forward_batch.capture_hidden_mode
        return model_inputs


EntryClass = [LlamaForCausalLMEagle3]
