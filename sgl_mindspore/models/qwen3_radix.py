# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import logging
import math
import os
from functools import lru_cache
from typing import Iterable, Optional, Tuple, Type, Union

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
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sgl_mindspore.layers import (
    BaseRotaryEmbedding,
    ColParallelLinear,
    MLPColParallelLinear,
    MsNativeAttnBackendRadix,
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
    format_cast,
    is_310p
)

logger = logging.getLogger(__name__)

Qwen3Config = None


class Qwen3MLP(nn.Cell):
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

class LMHead(ColParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        param_dtype: Optional[ms.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = (
            tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
        )
        self.param_dtype = param_dtype
        self.input_size = input_size
        self.output_size = output_size // self.tp_size
        self.enable_bias = bias

        self.matmul = ops.MatMul(transpose_b=True)
        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.param_dtype,
            weight_load=self.weight_load,
        )

        if self.enable_bias:
            self.bias = Parameter(mint.zeros(self.output_size, dtype=self.param_dtype))
            setattr(self.bias, "weight_load", self.weight_load)

    @jit
    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        bias = self.bias if self.enable_bias else None
        x = self.quant_method.apply(self, input, bias)
        return x

class Qwen3Attention(nn.Cell):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.attn_tp_size = attn_tp_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % attn_tp_size == 0
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.total_num_heads
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

        self.local_num_heads = self.total_num_heads // attn_tp_size
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
            self.local_num_kv_heads = self.total_num_kv_heads // attn_tp_size
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
            self.local_num_kv_heads = 1

        self.local_q_size = self.local_num_heads * self.head_dim
        self.local_kv_size = self.local_num_kv_heads * self.head_dim
        self.q_size = self.total_num_heads * self.head_dim
        self.kv_size = self.total_num_kv_heads * self.head_dim

        self.attn = MsNativeAttnBackendRadix(
            self.local_num_heads,
            self.head_dim,
            self.local_num_kv_heads,
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
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.q_norm = RMSNorm(
            norm_dim=config.head_dim,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("q_norm", prefix),
        )
        self.k_norm = RMSNorm(
            norm_dim=config.head_dim,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("k_norm", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=self.q_size,
            output_size=self.hidden_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
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
        hidden_state: Tensor,
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
        req_to_token: Tensor,
        req_pool_indices: Tensor,
        seq_lens: Tensor,
        extend_prefix_lens: Tensor,
        extend_seq_lens: Tensor,
    ) -> Tensor:
        token_lens, hidden_dim = hidden_state.shape

        qkv = self.qkv_proj(hidden_state)
        q, k, v = qkv.split(
            [
                self.local_q_size,
                self.local_kv_size,
                self.local_kv_size,
            ],
            dim=-1,
        )

        q = q.view(-1, self.head_dim).contiguous()
        k = k.view(-1, self.head_dim).contiguous()
        v = v.view(-1, self.local_kv_size).contiguous()

        q = self.q_norm(q).view(-1, self.local_q_size)
        k = self.k_norm(k).view(-1, self.local_kv_size)

        q, k = self.rotary_emb(
            positions,
            q,
            k,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
        )

        key_cache, value_cache = self.attn(
            k,
            v,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
        )

        key_cache = key_cache.reshape(key_cache.shape[0], 8, 128)
        value_cache = value_cache.reshape(value_cache.shape[0], 8, 128)
        q = q.reshape(q.shape[0], 32, 128)
        L, S = q.shape[-2], key_cache.shape[-2]
        attn_bias = np.zeros((L,S), dtype=np.float16)
        if is_prefill:
            attn_output = self.attn.extend(
                key_cache, value_cache, q, req_to_token, req_pool_indices, seq_lens, extend_prefix_lens, extend_seq_lens, None, attn_bias
            )
        else:
            attn_bias = ms.Tensor(attn_bias, dtype=ms.float16)
            attn_output = self.attn.decode(
                key_cache, value_cache, q, req_to_token, req_pool_indices, seq_lens, extend_prefix_lens, extend_seq_lens, None, attn_bias
            )
        attn_output = attn_output.reshape(attn_output.shape[0], 4096)
        output = self.o_proj(attn_output).view(token_lens, -1)
        return output


class Qwen3DecoderLayer(nn.Cell):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3MLP(
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
        hidden_state: Tensor,
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
        req_to_token: Tensor,
        req_pool_indices: Tensor,
        seq_lens: Tensor,
        extend_prefix_lens: Tensor,
        extend_seq_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)
        else:
            hidden_state, residual = self.input_layernorm(hidden_state, residual)
        hidden_state = self.self_attn(
            hidden_state=hidden_state,
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
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
        )
        hidden_state, residual = self.post_attention_layernorm(hidden_state, residual)
        hidden_state = self.mlp(hidden_state)

        return hidden_state, residual


class Qwen3Model(nn.Cell):
    r"""
    qwen3 model
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

        for i in range(self.num_hidden_layers):
            layer = Qwen3DecoderLayer(
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
        req_to_token=None,
        req_pool_indices=None,
        seq_lens=None,
        extend_prefix_lens=None,
        extend_seq_lens=None,
    ):
        """
        Forward of qwen model.
        """
        hidden_state = self.embed_tokens(input_ids)
        residual = None
        for i in range(self.num_hidden_layers):
            layer = self.layers[i]
            hidden_state, residual = layer(
                hidden_state=hidden_state,
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
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
            )

        hidden_state, _ = self.norm(hidden_state, residual)

        return hidden_state


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


class Qwen3ForCausalLMRadix(MindSporeModelBase):
    def __init__(
        self,
        config: Qwen3Config,
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
        if param_dtype == ms.bfloat16 and is_310p:
            param_dtype = ms.float16
            logger.warning(
                "Ascend 310P does not support bfloat16, will convert to float16"
            )
        setattr(self.config, "param_dtype", param_dtype)
        self.model = Qwen3Model(
            self.config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        self.lm_head = LMHead(
            input_size=self.config.hidden_size,
            output_size=self.config.vocab_size,
            param_dtype=self.config.param_dtype,
            bias=False,
            prefix=add_prefix("lm_head", prefix),
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.all_gather = GatherLastDim()

        # for best performance of MindSpore for Qwen3
        os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = (
            "FlashAttentionScore,PagedAttention"
        )
        os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "RmsNorm"
        if is_310p():
            os.environ["MS_ENABLE_INTERNAL_BOOST"] = "off"

    def set_model_inputs(self, is_prefill, max_context_len):
        dyn_input_ids = Tensor(shape=[None], dtype=dtype.int32)
        dyn_position_ids = Tensor(shape=[None], dtype=dtype.int64)

        head_size = self.config.head_dim
        num_kv_heads = self.config.num_key_value_heads
        # use pa, if use ifa, the shape should (None, None, head_size)
        kv_cache_shape = (None, num_kv_heads * head_size)

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
            req_to_token=Tensor(shape=[None, max_context_len], dtype=dtype.int32),
            req_pool_indices=Tensor(shape=[1], dtype=dtype.int32),
            seq_lens=Tensor(shape=[1], dtype=dtype.int32),
            extend_prefix_lens=Tensor(shape=[1], dtype=dtype.int32),
            extend_seq_lens=Tensor(shape=[1],dtype=dtype.int32),
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

        def adjust_weight(params_dict):
            target_keywords = [
                "qkv_proj.weight",
                "o_proj.weight",
                "gate_up_proj.weight",
                "down_proj.weight",
                "lm_head.weight",
            ]

            for name, param in params_dict.items():
                if any(name.endswith(keyword) for keyword in target_keywords):
                    cast_weight = format_cast(param, "nz")
                    ms.runtime.synchronize()
                    param.set_data(cast_weight)

        if is_310p():
            ms.runtime.synchronize()
            print(param_dict.keys())
            adjust_weight(param_dict)
            ms.runtime.synchronize()

    def prepare_inputs(self, forward_batch, model_inputs):
        model_inputs["req_to_token"] = tensor_torch2ms(forward_batch.req_to_token_pool.req_to_token)
        model_inputs["req_pool_indices"] = tensor_torch2ms(forward_batch.req_pool_indices).to(ms.int32)
        model_inputs["seq_lens"] = tensor_torch2ms(forward_batch.seq_lens).to(ms.int32)
        is_prefill = model_inputs["is_prefill"]
        if is_prefill:
            model_inputs["extend_prefix_lens"] = tensor_torch2ms(forward_batch.extend_prefix_lens)
            model_inputs["extend_seq_lens"] = tensor_torch2ms(forward_batch.extend_seq_lens)
        else:
            model_inputs["extend_prefix_lens"] = mint.ones(forward_batch.batch_size, dtype = ms.int32)
            model_inputs["extend_seq_lens"] = model_inputs["batch_valid_length"] - model_inputs["extend_seq_lens"]
        return model_inputs

    def construct(self, **model_inputs) -> Tensor:
        q_seq_lens = model_inputs["q_seq_lens"]
        is_prefill = model_inputs["is_prefill"]
        max_context_len = model_inputs['req_to_token'].shape[1]

        if self.prev_prefill != is_prefill:
            self.set_model_inputs(is_prefill, max_context_len)
        self.prev_prefill = is_prefill

        if is_prefill:
            self.model.phase = "prefill"
        else:
            self.model.phase = "increment"

        hidden_state = self.model(**model_inputs)

        # TODO: In pure decode scenarios, cumsum and gather operations will be redundant .
        q_seq_lens = mint.cumsum(q_seq_lens, 0)
        hidden_state = mint.index_select(hidden_state, 0, q_seq_lens - 1)

        logits = self.lm_head(hidden_state)
        if self.tp_size:
            logits = self.all_gather(logits)
        logits = mint.reshape(logits, (-1, logits.shape[-1]))
        return logits


EntryClass = Qwen3ForCausalLMRadix
