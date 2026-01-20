# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import logging
import math
import os
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
import numpy as np
import torch
from mindspore import Parameter, Tensor, dtype, jit, mint, mutable, nn, ops
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.utils import divide
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sgl_mindspore.layers import (
    BaseRotaryEmbedding,
    ColParallelLinear,
    MLPColParallelLinear,
    MsNativeAttnBackend,
    QKVParallelLinear,
    RMSNorm,
    RowParallelLinear,
    SwiGLU,
    VocabParallelEmbedding,
    YaRNScalingRotaryEmbedding,
)
from sgl_mindspore.layers.quantization.base_config import get_ms_quant_config
from sgl_mindspore.models.mindspore_model_base import MindSporeModelBase
from sgl_mindspore.utils import (
    _get_tp_group_name,
    add_prefix,
    get_ms_dtype,
    tensor_torch2ms,
)

logger = logging.getLogger(__name__)

LlamaConfig = None


class LlamaMLP(nn.Cell):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.param_dtype = config.param_dtype

        self.gate_up_proj = MLPColParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size * 2,
            param_dtype=self.param_dtype,
            bias=False,
            output_sizes=[self.intermediate_size] * 2,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            param_dtype=config.param_dtype,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SwiGLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class LlamaAttention(nn.Cell):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_heads
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.scaling = float(self.head_dim**-0.5)
        self.rope_theta = int(config.rope_theta)
        self.param_dtype = config.param_dtype
        self.max_position = config.max_position_embeddings
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling["rope_type"]
            self.rope_factor = config.rope_scaling["factor"]
            self.rope_max_position_embeddings = config.rope_scaling[
                "original_max_position_embeddings"
            ]
        else:
            self.rope_type = "default_rope"

        self.attn = MsNativeAttnBackend(
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            param_dtype=self.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = None
        if self.rope_type == "yarn":
            self.rotary_emb = YaRNScalingRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.rope_max_position_embeddings,
                base=self.rope_theta,
                is_neox_style=True,
                scaling_factor=self.rope_factor,
                dtype=self.param_dtype,
            )
        else:
            self.rotary_emb = BaseRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.max_position,
                base=self.rope_theta,
                dtype=self.param_dtype,
            )

    def construct(
        self,
        hidden_states: Tensor,
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
    ) -> Tensor:
        token_lens, hidden_dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.q_size,
                self.kv_size,
                self.kv_size,
            ],
            dim=-1,
        )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        q, k = self.rotary_emb(
            positions,
            q,
            k,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
        )

        key_out = self.attn(
            k,
            v,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
        )
        q = ops.depend(q, key_out)

        if is_prefill:
            attn_output = self.attn.extend(
                q,
                k,
                v,
                attn_mask,
                None,
                None,
                None,
                batch_valid_length,
                batch_valid_length,
            )
        else:
            attn_output = self.attn.decode(
                q,
                batch_valid_length,
                attn_mask,
                q_seq_lens,
                key_cache,
                value_cache,
                block_tables,
            )

        output = self.o_proj(attn_output).view(token_lens, -1)
        return output


class LlamaDecoderLayer(nn.Cell):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = LlamaMLP(
            config=config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
        )
        self.input_layernorm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("input_layernorm", prefix),
        )
        self.post_attention_layernorm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("post_attention_layernorm", prefix),
        )

    def construct(
        self,
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
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
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
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Cell):
    r"""
    llama model
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config=config, prefix=add_prefix("embed_tokens", prefix)
        )

        self.layers = nn.CellList()
        self.layers_to_capture = []

        for i in range(self.num_hidden_layers):
            layer = LlamaDecoderLayer(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix(f"layers.{i}", prefix),
            )
            self.layers.append(layer)

        self.norm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("norm", prefix),
        )

    # pylint: disable=W0613
    @jit
    def construct(
        self,
        input_ids,
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
        """
        Forward of llama model.
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        aux_hidden_states = []
        for i in range(self.num_hidden_layers):
            if i in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            layer = self.layers[i]
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=position_ids,
                batch_valid_length=batch_valid_length,
                is_prefill=is_prefill,
                layer_idx=i,
                attn_mask=attention_mask,
                q_seq_lens=q_seq_lens,
                key_cache=key_cache[i],
                value_cache=value_cache[i],
                out_cache_loc=out_cache_loc,
                block_tables=block_tables,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class GatherLastDim(nn.Cell):
    def __init__(self):
        super().__init__()
        tp_group_name = _get_tp_group_name()
        self.all_gather = ops.AllGather(group=tp_group_name)
        self.world_size = get_tensor_model_parallel_world_size()
        self.split = ops.Split(axis=0, output_num=self.world_size)

    def construct(self, input: Tensor) -> Tensor:
        output = self.all_gather(input)
        tensor_list = self.split(output)
        output = mint.cat(tensor_list, dim=-1)
        return output


class LlamaForCausalLM(MindSporeModelBase):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prev_prefill = False
        self.config = config
        quant_config = get_ms_quant_config(quant_config)
        if self.config.dtype:
            param_dtype = get_ms_dtype(self.config.dtype)
        else:
            param_dtype = ms.dtype.bfloat16
        setattr(self.config, "param_dtype", param_dtype)
        self.model = LlamaModel(
            self.config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        self.lm_head = ColParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.vocab_size,
            param_dtype=self.config.param_dtype,
            bias=False,
            prefix=add_prefix("lm_head", prefix),
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.all_gather = GatherLastDim()

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

        # for best performance of MindSpore for Llama
        os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = (
            "FlashAttentionScore,PagedAttention"
        )
        os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "RmsNorm"

    def prepare_inputs(self, forward_batch, model_inputs):
        if forward_batch.capture_hidden_mode:
            model_inputs["capture_hidden_mode"] = forward_batch.capture_hidden_mode
        return model_inputs

    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None], dtype=dtype.int32)
        dyn_position_ids = Tensor(shape=[None], dtype=dtype.int64)

        head_size = self.config.head_dim
        # use pa, if use ifa, the shape should (None, None, head_size)
        kv_cache_shape = (None, None, None, head_size)

        kv_cache_dtype = self.config.param_dtype

        num_layers = self.config.num_hidden_layers

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable([dyn_value_cache for _ in range(num_layers)])

        dyn_out_cache_loc = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dynamic_attention_mask = Tensor(
            shape=[None, None], dtype=self.config.param_dtype
        )
        dyn_batch_valid_length = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dyn_q_seq_lens = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dyn_block_tables = Tensor(shape=[None, None], dtype=dtype.int32)
        # dyn_intermediate_tensors = None
        # dyn_inputs_embeds = None
        self.model.set_inputs(
            input_ids=dyn_input_ids,
            position_ids=dyn_position_ids,
            attention_mask=dynamic_attention_mask,
            batch_valid_length=dyn_batch_valid_length,
            is_prefill=is_prefill,
            q_seq_lens=dyn_q_seq_lens,
            key_cache=dyn_key_caches,
            value_cache=dyn_value_caches,
            out_cache_loc=dyn_out_cache_loc,
            block_tables=dyn_block_tables,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        param_dict = self.parameters_dict()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]

        for name, weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in param_dict:
                    param = param_dict[name]
                    assert hasattr(param, "weight_load")
                    weight_load = getattr(param, "weight_load")
                    weight_load(param, weight, shard_id)
                    param.set_data(param.move_to("Ascend"))
                    break
            else:
                if name in param_dict:
                    param = param_dict[name]
                    if hasattr(param, "weight_load"):
                        weight_load = getattr(param, "weight_load")
                        weight_load(param, weight)
                        param.set_data(param.move_to("Ascend"))
                    else:
                        param.set_data(tensor_torch2ms(weight).move_to("Ascend"))
                    # Make sure the weight is loaded on device, so the kv cache calculation is correct.

    def construct(self, **model_inputs) -> Tensor:
        q_seq_lens = model_inputs["q_seq_lens"]
        is_prefill = model_inputs["is_prefill"]
        if "capture_hidden_mode" in model_inputs:
            capture_hidden_mode = model_inputs.pop("capture_hidden_mode")
        if "forward_mode" in model_inputs:
            forward_mode = model_inputs.pop("forward_mode")

        if self.prev_prefill != is_prefill:
            self.set_model_inputs(is_prefill)
        self.prev_prefill = is_prefill

        if is_prefill:
            self.model.phase = "prefill"
        else:
            self.model.phase = "increment"

        hidden_states = self.model(**model_inputs)

        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
            if capture_hidden_mode.need_capture():
                if capture_hidden_mode.is_full():
                    aux_hidden_states = mint.cat(aux_hidden_states, dim=-1)
                elif capture_hidden_mode.is_last():
                    aux_hidden_states = mint.cat(aux_hidden_states, dim=-1)
                    aux_hidden_states = mint.index_select(
                        aux_hidden_states, 0, mint.cumsum(q_seq_lens, 0) - 1
                    )
                else:
                    assert False, "Unsupported capture hidden mode"

        # TODO: In pure decode scenarios, cumsum and gather operations will be redundant .
        q_seq_lens = mint.cumsum(q_seq_lens, 0)
        if not forward_mode.is_target_verify():
            # In target verify mode, all tokens' logits are needed.
            hidden_states = mint.index_select(hidden_states, 0, q_seq_lens - 1)

        logits = self.lm_head(hidden_states)
        if self.tp_size:
            logits = self.all_gather(logits)
        logits = mint.reshape(logits, (-1, logits.shape[-1]))

        if self.capture_aux_hidden_states:
            return logits, aux_hidden_states
        else:
            return logits

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        ms.runtime.empty_cache()
        ms.runtime.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        ms.runtime.empty_cache()
        ms.runtime.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = LlamaForCausalLM
