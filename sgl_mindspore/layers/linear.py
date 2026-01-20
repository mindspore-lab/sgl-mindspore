# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import logging
from os import pread
from typing import Iterable, Optional, Tuple, Type, Union

import mindspore as ms
import numpy as np
import torch
from mindspore import Parameter, Tensor, from_numpy, mint, nn, ops
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.utils import divide
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sgl_mindspore.layers.quantization.base_config import QuantizeMethodBase
from sgl_mindspore.layers.quantization.unquant import UnquantizedLinearMethod
from sgl_mindspore.utils import _get_tp_group_name, tensor_torch2ms

logger = logging.getLogger(__name__)


class LinearBase(nn.Cell):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        param_dtype: Optional[ms.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = quant_config
        if param_dtype is None:
            param_dtype = ms.float32
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        raise NotImplementedError()


class ColParallelLinear(LinearBase):
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

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        bias = self.bias if self.enable_bias else None
        x = self.quant_method.apply(self, input, bias)
        return x

    def weight_load(self, param: Tensor, weight: torch.Tensor) -> None:
        output_dim = getattr(param, "output_dim", 0)
        shard_size = param.shape[output_dim]
        start_idx = self.tp_rank * shard_size
        weight = weight.narrow(output_dim, start_idx, shard_size).contiguous()

        param.set_data(tensor_torch2ms(weight))
        return None


class QKVParallelLinear(ColParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        param_dtype: Optional[Type] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_head = total_num_kv_heads

        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)

        if tp_size > self.total_num_kv_head:
            assert tp_size % self.total_num_kv_head == 0
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_head)
        else:
            self.num_kv_heads = divide(self.total_num_kv_head, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = ( self.num_heads + 2 * self.num_kv_heads ) * tp_size * self.head_dim

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def get_shard_offset_and_size(self, shard_id: str):
        assert shard_id in ["q", "k", "v"]
        if shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_dim
        elif shard_id == "k":
            shard_offset = self.num_heads * self.head_dim
            shard_size = self.num_kv_heads * self.head_dim
        elif shard_id == "v":
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_dim
            shard_size = self.num_kv_heads * self.head_dim
        return shard_offset, shard_size

    def weight_load(
        self, param: Parameter, weight: torch.Tensor, shard_id: Optional[str] = None
    ) -> None:
        if param.size == 1:
            param.set_data(tensor_torch2ms(weight))
            return None

        output_dim = getattr(param, "output_dim", None)
        tp_rank = self.tp_rank

        shard_offset, shard_size = self.get_shard_offset_and_size(shard_id=shard_id)

        if shard_id == "q":
            shard_idx = tp_rank
        else:
            shard_idx = tp_rank // self.num_kv_head_replicas
        start_idx = shard_idx * shard_size

        weight = weight.narrow(output_dim, start_idx, shard_size).contiguous()
        param[shard_offset : shard_offset + shard_size, ...] = tensor_torch2ms(weight)


class MLPColParallelLinear(ColParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        output_sizes: list,
        param_dtype: Optional[Type] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            param_dtype=param_dtype,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.output_sizes = output_sizes

    def _get_shard_idx(self, shard_id: str) -> int:
        if shard_id == "gate":
            return 0

        if shard_id == "up":
            return 1

        return -1

    def weight_load(self, param: Tensor, weight: torch.Tensor, shard_id: str) -> None:
        if param.shape[0] == 1:
            param.set_data(tensor_torch2ms(weight))
            return None

        shard_idx = self._get_shard_idx(shard_id=shard_id)
        assert shard_idx != -1

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None and shard_idx is not None:
            assert shard_idx < len(self.output_sizes)
            shard_offset = sum(self.output_sizes[:shard_idx]) // tp_size
            shard_size = self.output_sizes[shard_idx] // tp_size
            param_data = param.data
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = tp_rank * shard_size
            weight = weight.narrow(output_dim, start_idx, shard_size).contiguous()
            assert param_data.shape == weight.shape
            param[shard_offset : shard_offset + shard_size, ...] = tensor_torch2ms(
                weight
            )
        else:
            param.set_data(tensor_torch2ms(weight))


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        param_dtype: Optional[Type] = None,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
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
        self.input_size = input_size // self.tp_size
        self.output_size = output_size
        self.enable_bias = bias
        self.reduce_results = reduce_results

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
        tp_group_name = _get_tp_group_name()
        self.all_reduce = ops.AllReduce(group=tp_group_name)

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        bias = self.bias if self.enable_bias else None
        x = self.quant_method.apply(self, input, bias)
        if self.reduce_results and self.tp_size > 1:
            x = self.all_reduce(x)
        return x

    def weight_load(self, param: Tensor, weight: torch.Tensor) -> None:
        if weight.dim() > 1 and weight.shape[1] > 1:
            input_dim = getattr(param, "input_dim", 1)
            shard_size = param.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            weight = weight.narrow(input_dim, start_idx, shard_size).contiguous()

        param.set_data(tensor_torch2ms(weight))
        return None


class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        param_dtype: Optional[ms.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.param_dtype = param_dtype
        self.enable_bias = bias

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

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        bias = self.bias if self.enable_bias else None
        x = self.quant_method.apply(self, input, bias)
        return x

    def weight_load(self, param: Tensor, weight: torch.Tensor) -> None:
        if weight.dim() == 0:
            weight = weight.reshape(1)
        assert param.shape == weight.shape, (
            f"Tried to load weights of size {weight.size()}"
            f"to a parameter of size {param.size()}"
        )
        param.set_data(tensor_torch2ms(weight))
        return None


class MoeReplicatedLinear(nn.Cell):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        param_dtype: Optional[Type] = None,
        optim_tp_ep_gating_perf: bool = False,
        expert_start_index: Optional[Type] = None,
        expert_end_index: Optional[Type] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.enable_bias = bias
        self.param_dtype = param_dtype
        self.optim_tp_ep_gating_perf = optim_tp_ep_gating_perf
        self.expert_start_index = expert_start_index
        self.expert_end_index = expert_end_index

        self.matmul = ops.MatMul(transpose_b=True)

        self.weight = Parameter(
            mint.zeros((self.output_size, self.input_size), dtype=self.param_dtype),
            requires_grad=False,
        )
        setattr(self.weight, "weigth_load", self.weight_load)

        if self.enable_bias:
            self.bias = Parameter(mint.zeros(self.output_size, dtype=self.param_dtype))
            setattr(self.bias, "weight_load", self.weight_load)

    def construct(self, input: Tensor) -> Tensor:
        origin_shape = input.shape
        x = self.matmul(input.view(-1, origin_shape[-1]), self.weight)
        if self.enable_bias:
            x = mint.add(x, self.bias)
        return x.view(*origin_shape[:-1], -1)

    def weight_load(self, param: Parameter, weight: torch.Tensor):
        weight = weight.contiguous().to(torch.float32).numpy()
        weight = weight[:]
        if len(weight) == 0:
            weight = weight.reshape(1)

        assert param.shape == weight.shape, (
            f"Tried to load weights of size {weight.size()}"
            f"to a parameter of size {param.size()}"
        )

        if self.optim_tp_ep_gating_perf:
            if self.expert_start_index is None or self.expert_end_index is None:
                raise ValueError(
                    "If setting optim_tp_ep_gating_perf, expert_start_index "
                    "and expert_end_index must be set too."
                )
                rearange_weight = [
                    weight[self.expert_start_index : self.expert_end_index],
                    weight[: self.expert_start_index],
                    weight[self.expert_end_index :],
                ]
                weight = np.concatenate(rearange_weight, axis=0)

        param.set_data(from_numpy(weight).to(self.param_dtype))
