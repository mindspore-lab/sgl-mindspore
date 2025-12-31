# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
from abc import abstractmethod
from typing import Any, Dict, Optional

import mindspore as ms
from mindspore import Parameter, Tensor, dtype, jit, mint, mutable, nn, ops
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MindSporeModelBase(ms.nn.Cell):
    @abstractmethod
    def construct(self, **model_inputs) -> ms.Tensor:
        raise NotImplementedError

    def prepare_inputs(
        self, forward_batch: ForwardBatch, model_inputs: Dict[str, Any]
    ) -> ms.Tensor:
        return model_inputs

    def set_model_inputs(self, is_prefill: bool):
        """
        Set shared inputs for all models. If the derived model has additional inputs,
        it should override this method and call super().set_model_inputs(is_prefill)
        """
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
        model_inputs = {
            "input_ids": dyn_input_ids,
            "position_ids": dyn_position_ids,
            "attention_mask": dynamic_attention_mask,
            "batch_valid_length": dyn_batch_valid_length,
            "is_prefill": is_prefill,
            "q_seq_lens": dyn_q_seq_lens,
            "key_cache": dyn_key_caches,
            "value_cache": dyn_value_caches,
            "out_cache_loc": dyn_out_cache_loc,
            "block_tables": dyn_block_tables,
        }
        self.model.set_inputs(kwargs=model_inputs)
