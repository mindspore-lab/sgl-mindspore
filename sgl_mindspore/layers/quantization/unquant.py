from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, ops
from mindspore.ops.auto_generate import (
    GroupedMatmulV4,
)
from mindspore.ops.function.array_func import split_ext

from sgl_mindspore.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizeMethodBase,
)
from sgl_mindspore.utils import set_weight_attrs


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ):
        weight = ms.Parameter(
            mint.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.weight = weight
        self.matmul = ops.MatMul(transpose_b=True)

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        return

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        origin_shape = x.shape
        x = self.matmul(x.view(-1, origin_shape[-1]), layer.weight)
        if bias is not None:
            x = mint.add(x, bias)
        return x.view(*origin_shape[:-1], -1)


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = ms.Parameter(
            mint.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.weight = weight

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        return mint.linear(x, layer.weight, bias)

    def embedding(self, layer: ms.nn.Cell, input_: ms.Tensor) -> ms.Tensor:
        return mint.index_select(layer.weight, 0, input_)


class UnquantizedFusedMoEFFNMethod(QuantizeMethodBase):
    def __init__(self):
        super().__init__()
        self.with_bias = False

    def create_weights(
        self,
        layer: ms.nn.Cell,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: ms.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)

        w13_weight = ms.Parameter(
            mint.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = ms.Parameter(
                mint.empty(num_experts, w13_up_dim, dtype=ms.float32),
                requires_grad=False,
            )
            layer.insert_param_to_cell("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )

        w2_weight = ms.Parameter(
            mint.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = ms.Parameter(
                mint.empty(num_experts, hidden_size, dtype=ms.float32),
                requires_grad=False,
            )
            layer.insert_param_to_cell("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)
        
        self.group_matmul_op = GroupedMatmulV4()
    
    def _group_matmul(self, hidden_states, weight, group_list):
        return self.group_matmul_op(
            [hidden_states],
            [weight],
            None,
            None,
            None,
            None,
            None,
            None,
            group_list,
            split_item=3,
            group_type=0,
            group_list_type=1,
        )[0]
    
    def _gate_activation(self, gate, activation):
        if activation == "silu":
            return mint.nn.functional.silu(gate)
        elif activation == "gelu":
            return mint.nn.functional.gelu(gate)
        else:
            raise ValueError(f"unsupported activation function: {activation}")

    def apply(self, layer, hidden_states, group_list, activation="silu"):
        w1 = layer.w13_weight
        w2 = layer.w2_weight
        # gmm1: gate_up_proj
        gate_hidden_out = self._group_matmul(hidden_states, w1, group_list)
        gate_hidden_out = self._group_matmul(
            hidden_states=hidden_states, weight=w1, group_list=group_list
        )
        gate, hidden = split_ext(
            gate_hidden_out, (w1.shape[2] // 2, w1.shape[2] // 2), -1
        )
        gate = self._gate_activation(gate=gate, activation=activation)
        hidden = mint.mul(hidden, gate)
        expert_output = self._group_matmul(
            hidden_states=hidden, weight=w2, group_list=group_list
        )
        expert_output = mint.nan_to_num(expert_output, 0, 0, 0)
        return expert_output

