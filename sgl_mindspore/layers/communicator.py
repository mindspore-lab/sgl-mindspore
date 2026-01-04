# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

import logging

from mindspore import Tensor, mint, nn, ops
from sglang.srt.layers.communicator import (
    CommunicateContext,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import get_attention_dp_rank

from sgl_mindspore.utils import _get_attn_tp_group_name, _get_tp_group_name

logger = logging.getLogger(__name__)


class MindsporeLayerCommunicator:
    """
    Mindspore implementation of LayerCommunicator.
    For collective communication, we use ops as it supports graph mode.
    """

    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: nn.Cell,
        post_attention_layernorm: nn.Cell,
    ):
        self.layer_scatter_modes = layer_scatter_modes
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

        self._context = CommunicateContext.init_new()
        self._context.attn_dp_rank = get_attention_dp_rank()
        self.attn_tp_all_gather = ops.AllGather(group=_get_attn_tp_group_name())
        self.attn_tp_reduce_scatter = ops.ReduceScatter(group=_get_attn_tp_group_name())
        self.tp_all_gather = ops.AllGather(group=_get_tp_group_name())
        self.tp_all_reduce = ops.AllReduce(group=_get_tp_group_name())

    def get_group_size(self, scatter_mode: ScatterMode):
        if scatter_mode == ScatterMode.TP_ATTN_FULL:
            return self._context.attn_tp_size
        elif scatter_mode == ScatterMode.FULL:
            return self._context.tp_size
        else:
            return 1

    def prepare_attn(
        self,
        hidden_states: Tensor,
        residual: Tensor,
    ):
        input_mode = self.layer_scatter_modes.layer_input_mode
        output_mode = self.layer_scatter_modes.layer_output_mode

        # If the last layer is moe, input mode is scattered.
        assert input_mode in [
            ScatterMode.SCATTERED,
            ScatterMode.TP_ATTN_FULL,
        ], input_mode
        assert output_mode == ScatterMode.TP_ATTN_FULL, output_mode

        if hidden_states.shape[0] == 0:  # idle batch
            return hidden_states, hidden_states.clone()

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if input_mode == ScatterMode.SCATTERED:
            hidden_states = self.attn_tp_all_gather(hidden_states)

        return hidden_states, residual

    def prepare_mlp(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        dp_attn_info: dict,
    ):
        hidden_states_input_mode = self.layer_scatter_modes.attn_mode
        residual_input_mode = self.layer_scatter_modes.layer_input_mode
        hidden_states_output_mode = self.layer_scatter_modes.mlp_mode
        residual_output_mode = self.layer_scatter_modes.middle_residual_mode
        context = self._context

        # case 1: no DP Attention, only do layernorm
        if (
            self.get_group_size(hidden_states_input_mode)
            == self.get_group_size(hidden_states_output_mode)
            and self.get_group_size(residual_input_mode)
            == self.get_group_size(residual_output_mode)
            and context.attn_tp_size == 1
        ):
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
            return hidden_states, residual

        # case 2: FFN is dense or MoE TP, gather hidden states and residual
        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.FULL)
            and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            # residual: SCATTERED -> TP_ATTN_FULL
            if (
                residual_input_mode == ScatterMode.SCATTERED
                and context.attn_tp_size > 1
            ):
                residual = self.attn_tp_all_gather(residual)

            if context.attn_dp_size > 1:
                if context.attn_tp_rank == 0:
                    hidden_states += residual
                # hidden states: TP_ATTN_FULL -> FULL
                if dp_attn_info["is_max_len"]:
                    hidden_states = self.dp_gather_via_all_gather(hidden_states, True)
                else:
                    hidden_states = self.dp_gather_via_all_reduce(
                        hidden_states, dp_attn_info, True
                    )
                # residual: FULL -> TP_ATTN_FULL
                self.dp_scatter(residual, hidden_states, dp_attn_info)
                if hidden_states.shape[0] != 0:
                    hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                # attn tp group = tp group. replaces o_proj's all_reduce
                hidden_states = self.tp_all_reduce(hidden_states)
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
            return hidden_states, residual

        # case 3: FFN is MoE EP, scatter hidden states and residual
        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.SCATTERED)
            and (residual_output_mode == ScatterMode.SCATTERED)
        ):
            hidden_states = self.attn_tp_reduce_scatter(hidden_states)
            if residual_input_mode == ScatterMode.TP_ATTN_FULL:
                start = len(residual) // context.attn_tp_size * context.attn_tp_rank
                end = start + len(residual) // context.attn_tp_size
                residual = residual[start:end]
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )

            return hidden_states, residual

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    def postprocess_layer(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        dp_attn_info: dict,
    ):
        hidden_states_input_mode = self.layer_scatter_modes.mlp_mode
        residual_input_mode = self.layer_scatter_modes.middle_residual_mode
        output_mode = self.layer_scatter_modes.layer_output_mode
        context = self._context

        # case 1: no DP attention, do nothing
        if (
            self.get_group_size(hidden_states_input_mode)
            == self.get_group_size(residual_input_mode)
            == self.get_group_size(output_mode)
        ):
            return hidden_states, residual

        # case 2: FFN is dense or MoE TP
        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            hidden_states, global_hidden_states = (
                dp_attn_info["dp_buffer"][
                    : dp_attn_info["input_len"]
                ].contiguous(),
                hidden_states,
            )

            self.dp_scatter(hidden_states, global_hidden_states, dp_attn_info)
            return hidden_states, residual

        # case 3: FFN is MoE EP
        if (
            (hidden_states_input_mode == ScatterMode.SCATTERED)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            hidden_states += residual
            residual = None
            hidden_states = self.attn_tp_all_gather(hidden_states)

        # case 4: TODO figure out use case
        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            assert residual is None, "not yet handled residual!=None"
            tensor_list = list(hidden_states.tensor_split(context.attn_tp_size))
            hidden_states = tensor_list[context.attn_tp_rank]
            return hidden_states, residual

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    def dp_scatter(
        self,
        local_tokens: Tensor,
        global_tokens: Tensor,
        dp_attn_info: dict,
    ):
        """
        Scatter global tokens to each DP rank. Copy from global_tokens to local_tokens without device communication.
        """
        local_tokens.fill_(0)
        assert local_tokens.is_contiguous()
        assert global_tokens.is_contiguous()
        if local_tokens.shape[0] > 0:
            assert (
                local_tokens.untyped_storage() is not global_tokens.untyped_storage()
            ), "aliasing between global_tokens and local_tokens not allowed"
            local_start_pos, local_num_tokens = self.get_dp_local_info(dp_attn_info)
            memcpy_dim0(
                local_tokens, global_tokens, local_start_pos, local_num_tokens, True
            )

    def dp_gather_via_all_gather(
        self,
        local_tokens: Tensor,
        is_partial: bool,
    ):
        """
        Gather on DP group when DP padding mode is max len. First scatter on attn tp group, then gather on tp group.

        is partial: True if each attn tp rank has partial data and needs to be reduced first. If false, erase data for all ranks > 0.
        """

        if not is_partial and self._context.attn_tp_rank > 0:
            local_tokens.fill_(0)

        scattered_local_tokens = self.attn_tp_reduce_scatter(local_tokens)
        return self.tp_all_gather(scattered_local_tokens)

    def dp_gather_via_all_reduce(
        self,
        local_tokens: Tensor,
        dp_attn_info: dict,
        is_partial: bool,
    ):
        """
        Gather on DP group when DP padding mode is sum len.
        Length of local_tokens may be different on different dp ranks. Therefore, we need to copy values from local tokens to global tokens and do allreduce.

        is partial: True if each attn tp rank has partial data and needs to be reduced first. If false, erase data for all ranks > 0.
        """
        global_tokens = dp_attn_info["dp_buffer"]
        global_tokens.fill_(0)
        assert local_tokens.is_contiguous()
        assert global_tokens.is_contiguous()

        if local_tokens.shape[0] > 0 and (
            is_partial or self._context.attn_tp_rank == 0
        ):
            assert (
                local_tokens.untyped_storage() is not global_tokens.untyped_storage()
            ), "aliasing between global_tokens and local_tokens not allowed"
            local_start_pos, local_num_tokens = self.get_dp_local_info(dp_attn_info)
            memcpy_dim0(
                global_tokens, local_tokens, local_start_pos, local_num_tokens, False
            )

        return self.tp_all_reduce(global_tokens)

    def get_dp_local_info(self, dp_attn_info):
        dp_rank = self._context.attn_dp_rank

        if "dp_local_start_pos" not in dp_attn_info:
            cumtokens = mint.cumsum(dp_attn_info["global_num_tokens_gpu"], dim=0)
            if dp_rank == 0:
                local_start_pos = mint.zeros_like(cumtokens[0])
            else:
                local_start_pos = cumtokens[dp_rank - 1]
            local_num_tokens = dp_attn_info["global_num_tokens_gpu"][dp_rank]
            dp_attn_info["dp_local_start_pos"] = local_start_pos
            dp_attn_info["dp_local_num_tokens"] = local_num_tokens

        return dp_attn_info["dp_local_start_pos"], dp_attn_info["dp_local_num_tokens"]


def memcpy_dim0(dst, src, offset, sz, offset_src):
    assert (
        src.shape[1:] == dst.shape[1:]
    ), "src and dst must have same shape for dims > 0"

    if offset_src:
        dst[:] = src[offset : offset + sz]
    else:
        dst[offset : offset + sz] = src[:]
