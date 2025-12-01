from typing import List, Optional

import mindspore as ms
from mindspore import mint
from mindspore import dtype as msdtype
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore.ops.auto_generate import (
    WeightQuantBatchMatmul,
    QuantBatchMatmul,
    DynamicQuantExt,
    GroupedMatmulV4,
    DequantSwigluQuant,
)
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
)
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

from sgl_mindspore.layers.linear import RowParallelLinear
from sgl_mindspore.layers.moe.fused_moe import FusedMoe
from sgl_mindspore.layers.quantization.base_config import QuantizeMethodBase
from sgl_mindspore.layers.quantization.unquant import UnquantizedLinearMethod
from sgl_mindspore.utils import set_weight_attrs


class MsW8A8Int8Config(W8A8Int8Config):
    def __init__(self, quant_config: W8A8Int8Config):
        super().__init__(quant_config.quant_description)

    def get_quant_method(
        self,
        layer: ms.nn.Cell,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sgl_mindspore.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            key = "model"
            if "vision_model" in prefix:
                key = "vision_model"
            elif "visual" in prefix:
                key = "visual"
            packed_modules_mapping_subset = self.packed_modules_mapping.get(key, {})
            prefix_in_quant_config = prefix
            proj_name = prefix.split(".")[-1]
            if proj_name in packed_modules_mapping_subset:
                prefix_in_quant_config = prefix.replace(
                    proj_name, packed_modules_mapping_subset[proj_name][0]
                )
            self.is_dynamic = (
                self.quant_description[prefix_in_quant_config + ".weight"]
                == "W8A8_DYNAMIC"
            )
            if self.is_layer_skipped(prefix, packed_modules_mapping_subset):
                return UnquantizedLinearMethod()
            return MSW8A8DynamicLinearMethod(self) if self.is_dynamic else MSW8A8LinearMethod(self)
        elif isinstance(layer, FusedMoe):
            return MSW8A8FusedMoEFFNMethod(self)
        
        return None


class MSW8A8LinearMethod(LinearMethodBase):
    """Linear method for NPU quantization.

    This class search for specific quantization
    implementation supported on NPU hardware for linear methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        q_weight_dict = {
            "weight": ms.mint.zeros(
                (sum(output_partition_sizes), input_size_per_partition), dtype=ms.int8
            ),
        }
        per_tensor_weight_dict = {
            "input_scale": ms.mint.zeros(1, dtype=ms.float32),
            "input_offset": ms.mint.zeros(1, dtype=ms.float32),
        }
        per_channel_weight_dict = {
            "quant_bias": ms.mint.zeros(output_size_per_partition, dtype=ms.int32),
            "deq_scale": ms.mint.zeros(
                output_size_per_partition,
                dtype=ms.float32 if params_dtype == ms.bfloat16 else ms.int64,
            ),
            "weight_scale": ms.mint.zeros(
                [output_size_per_partition, 1], dtype=params_dtype
            ),
            "weight_offset": ms.mint.zeros(
                [output_size_per_partition, 1], dtype=params_dtype
            ),
        }

        for name, data in q_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        for name, data in per_tensor_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        for name, data in per_channel_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        self.matmul = ms.ops.auto_generate.QuantBatchMatmul(
            transpose_x1=False, transpose_x2=True, dtype=params_dtype
        )
        self.quant = QuantV2()

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        input_scale_reciprocal = ms.Parameter(
            1.0 / layer.input_scale, requires_grad=False
        )
        layer.insert_param_to_cell("input_scale_reciprocal", input_scale_reciprocal)

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        original_dtype = x.dtype
        if original_dtype != ms.int8:
            x = x.to(layer.input_scale.dtype)
            qx = self.quant(
                x,
                layer.input_scale_reciprocal,
                layer.input_offset,
                False,
                "ROUND",
                ms.dtype.int8,
            )
        else:
            qx = x
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = ms.mint.zeros_like(layer.quant_bias)
        else:
            quant_bias = layer.quant_bias
        output = self.matmul(
            qx,
            layer.weight,
            layer.deq_scale,
            None,
            quant_bias,
            None,
        )
        if bias is not None:
            output = output + bias
        return output


class MSW8A8DynamicLinearMethod(LinearMethodBase):
    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        q_weight_dict = {
            "weight": ms.mint.zeros(
                (sum(output_partition_sizes), input_size_per_partition), dtype=ms.int8
            ),
        }
        per_tensor_weight_dict = {}
        per_channel_weight_dict = {
            "weight_scale": ms.mint.zeros(
                [output_size_per_partition, 1], dtype=params_dtype
            ),
            "weight_offset": ms.mint.zeros(
                [output_size_per_partition, 1], dtype=params_dtype
            ),
        }

        for name, data in q_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        for name, data in per_tensor_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        for name, data in per_channel_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        self.dynamic_quant = DynamicQuantExt()
        self.matmul = QuantBatchMatmul(
            transpose_x1=False, transpose_x2=True, dtype=params_dtype
        )

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        layer.weight_scale = layer.weight_scale.flatten()

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        qx, x_scale = self.dynamic_quant(x)
        return self.matmul(
            qx,
            layer.weight,
            layer.weight_scale,
            None,
            None,
            x_scale,
        )


class MSW8A8FusedMoEFFNMethod(QuantizeMethodBase):
    """MoE method for NPU quantization.

    This class search for specific quantization
    implementations supported on NPU hardware for moe methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: ms.nn.Cell,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.num_experts = num_experts

        # weight
        w13_weight = ms.Parameter(
            mint.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=msdtype.int8,
            ),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = ms.Parameter(
            mint.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=msdtype.int8,
            ),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        # scale
        w13_weight_scale = ms.Parameter(
            mint.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=msdtype.float32
            ),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = ms.Parameter(
            mint.empty(num_experts, hidden_size, 1, dtype=msdtype.float32),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        # offset
        w13_weight_offset = ms.Parameter(
            mint.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=msdtype.float32
            ),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)
        w2_weight_offset = ms.Parameter(
            mint.empty(num_experts, hidden_size, 1, dtype=msdtype.float32),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

        self.dynamic_quant = DynamicQuantExt()
        self.group_matmul = GroupedMatmulV4()
        self.dequant_swiglu_quant = DequantSwigluQuant()

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        pass

    def apply(
        self,
        layer,
        hidden_states,
        group_list,
        activation: str = "silu",
    ) -> ms.Tensor:
        if activation != "silu":
            raise ValueError("MSW8A8FusedMoEFFNMethod only activation function: silu")
        output_dtype = ms.bfloat16
        group_list_type = 1

        hidden_states, hidden_states_scale = self.dynamic_quant(hidden_states)

        # gmm1: gate_up_proj
        hidden_states = self.group_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=msdtype.int32,
        )[0]

        # act_fn: swiglu
        hidden_states, swiglu_out_scale = self.dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=layer.w13_weight_scale.to(msdtype.float32),
            activation_scale=hidden_states_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
        )

        # gmm2: down_proj
        hidden_states = self.group_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale.to(output_dtype)],
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
        )[0]
        return hidden_states
