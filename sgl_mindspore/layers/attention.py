# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import math
from typing import Optional

from mindspore import Tensor, nn, ops, mint
from mindspore.ops.operations.nn_ops import (
    FlashAttentionScore,
    PagedAttention,
    ReshapeAndCache,
)
import swft
from swft.core import *
from swft.api import *
CORE_NUM = 8
NUM_HEADS = 8
NUM_HEADS_Q = 32
HEAD_SIZE = 128
MAX_CONTEXT_LEN = 40964
tor = 0.13523377

@sub_kernel(core_num=8)
def gather_kv_fused(gm_k_cache, gm_v_cache, gm_query, req_to_token, req_pool_indices, seq_lens, extend_prefix_lens,
                    extend_seq_lens, gm_output, gm_attn_bias,m,n,t,r):
    m1 = m + 0
    n1 = n + 0
    t1 = t + 0
    r1 = r + 0
    global tor
    current_start_q = Scalar("INT32", 0)
    block_idx = get_block_idx()
    loop_count = (m + 63) // 64

    for block in dynamic_loop(loop_count):
        begin_index = block * 64
        current_block_size = Scalar("INT32", 64)
        if block == loop_count - 1:
            current_block_size = m - begin_index
        ub_req_pool_indices = slice_to_ub(req_pool_indices, [begin_index], [64])
        ub_extend_seq_lens = slice_to_ub(extend_seq_lens, [begin_index], [64])
        ub_extend_prefix_lens = slice_to_ub(extend_prefix_lens, [begin_index], [64])
        ub_seq_lens = slice_to_ub(seq_lens, [begin_index], [64])
        now_core = Scalar("INT32", 0)
        for i in dynamic_loop(current_block_size):
            req_pool_idx = move_to_scalar(ub_req_pool_indices[i])
            extend_seq_len_q = move_to_scalar(ub_extend_seq_lens[i])
            prefill_seq_len_q = move_to_scalar(ub_extend_prefix_lens[i])
            seq_len_kv = move_to_scalar(ub_seq_lens[i])
            current_end_q = current_start_q + extend_seq_len_q
            block_num = 8
            for m in dynamic_loop(extend_seq_len_q):
                now_core.load(now_core + 1)
                if (now_core % 8 != block_idx):
                    continue

                query_idx = current_start_q + m
                for b in dynamic_loop(block_num):
                    num_heads = NUM_HEADS_Q // block_num
                    gm = vector_dup(Scalar("FP16", -65500.0), [num_heads, 16, 1], False)
                    gm = change_view(gm, new_format="NZ")
                    gl = vector_dup(Scalar("FP32", 0), [num_heads, 16, 1], False)
                    gl = change_view(gl, new_format="NZ")
                    go = vector_dup(Scalar("FP32", 0.0), [num_heads, 16, HEAD_SIZE], False)
                    go = change_view(go, new_format="NZ")
                    current_q = slice_to_ub(gm_query, [current_start_q + m, num_heads * b, 0], [1, num_heads, HEAD_SIZE])
                    current_q = change_view(current_q, [num_heads, 1, HEAD_SIZE])
                    current_q = pad_to_ub(current_q, [num_heads, 16, HEAD_SIZE])
                    current_q_nz = nd_to_nz(current_q)
                    l1_q = move_to_l1(current_q_nz)
                    block_size = Scalar("INT32", 16)

                    for n in dynamic_loop(seq_len_kv):
                        key_idx = n
                        block_start = (n // block_size) * block_size
                        ub_req_to_token_block = slice_to_ub(req_to_token, [req_pool_idx, block_start], [1, block_size])
                        offset_in_block = n - block_start
                        token_idx = move_to_scalar(ub_req_to_token_block[0, offset_in_block])
                        current_k = slice_to_ub(gm_k_cache, [token_idx, b, 0], [1, 1, HEAD_SIZE])
                        current_k = concat([current_k, current_k, current_k, current_k], 0)
                        current_k_nz = nd_to_nz(current_k)
                        current_v = slice_to_ub(gm_v_cache, [token_idx, b, 0], [1, 1, HEAD_SIZE])
                        current_v = concat([current_v, current_v, current_v, current_v], 0)
                        current_v_nz = nd_to_nz(current_v)
                        l1_k = move_to_l1(current_k_nz)
                        l1_v = move_to_l1(current_v_nz)
                        l0a = move_to_l0A(l1_q)
                        l0b = move_to_l0B(l1_k, Transpose=True)
                        l0c = mmad(l0a, l0b)
                        ub_qk = move_to_ub(l0c, "FP16")
                        ls = vmuls(ub_qk, tor)

                        ub_attn_bias = slice_to_ub(gm_attn_bias, [query_idx, key_idx], [1, 1])
                        attn_bias_val = move_to_scalar(ub_attn_bias[0, 0])
                        ls = vadds(ls, attn_bias_val)

                        lm = vcmax(ls, -1)
                        hm = vmax(lm, gm)
                        dm = vsub(gm, hm)
                        dm = vconv(dm, "FP32")
                        dm = vexp(dm)
                        gm = move_to_ub(hm)
                        hm = vbrcb(hm, -1, ub_qk.shape[-1])
                        ls = vsub(ls, hm)
                        ls_f32 = vconv(ls, "FP32")
                        ls_f32 = vexp(ls_f32)
                        lp = vconv(ls_f32, "FP16")
                        ll = vcadd(ls_f32, -1)
                        gl = vmul(gl, dm)
                        gl = vadd(gl, ll)
                        l1_qk = move_to_l1(lp)
                        l0a = move_to_l0A(l1_qk)
                        l0b = move_to_l0B(l1_v)
                        l0c = mmad(l0a, l0b)
                        lo = move_to_ub(l0c, "FP32")
                        dm = vbrcb(dm, -1, go.shape[-1])
                        go = vmul(go, dm)
                        go = vadd(go, lo)
                    gl = vbrcb(gl, -1, go.shape[-1])
                    ub_out = vdiv(go, gl)
                    ub_out = vconv(ub_out, "FP16")
                    ub_out = nz_to_nd(ub_out)
                    ub_out = slice_to_ub(ub_out, [0,0,0], [num_heads, 1, HEAD_SIZE])
                    ub_out = change_view(ub_out, [1, num_heads, HEAD_SIZE])
                    insert_to_gm(gm_output, ub_out, [current_start_q + m, num_heads * b, 0], [1, num_heads, HEAD_SIZE])
            current_start_q.load(current_end_q)

class MsNativeAttnBackendRadix(nn.Cell):
    """MindSpore Attention Manager."""

    def __init__(self, n_heads: int, head_dim: int, n_kv_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.scale_value = 1 / math.sqrt(self.head_dim)
        self.attention_layout = "TH"
        
        self.flash_attention = FlashAttentionScore(
            head_num=self.n_heads,
            scale_value=self.scale_value,
            next_tokens=0,
            input_layout=self.attention_layout,
        )
        self.paged_attention = PagedAttention(
            head_num=self.n_heads,
            scale_value=self.scale_value,
            kv_head_num=self.n_kv_heads,
        )
        self.scatter_update = ops.ScatterUpdate()
        m = Scalar("INT32")
        n = Scalar("INT32")
        t = Scalar("INT32")
        r = Scalar("INT32")
        self.gather_kv_fused = ssft.aot(kernel=gather_kv_fused, core_num=CORE_NUM, arg_attrs={
            "gm_k_cache": {"type": "tensor", "shape": [t, NUM_HEADS, HEAD_SIZE], "dtype": "float16", "format": "ND"},
            "gm_v_cache": {"type": "tensor", "shape": [t, NUM_HEADS, HEAD_SIZE], "dtype": "float16", "format": "ND"},
            "gm_query": {"type": "tensor", "shape": [n, NUM_HEADS_Q, HEAD_SIZE], "dtype": "float16", "format": "ND"},
            "req_to_token": {"type": "tensor", "shape": [r, MAX_CONTEXT_LEN], "dtype": "int32", "format": "ND"},
            "req_pool_indices": {"type": "tensor", "shape": [m], "dtype": "int32", "format": "ND"},
            "seq_lens": {"type": "tensor", "shape": [m], "dtype": "int32", "format": "ND"},
            "extend_prefix_lens": {"type": "tensor", "shape": [m], "dtype": "int32", "format": "ND"},
            "extend_seq_lens": {"type": "tensor", "shape": [m], "dtype": "int32", "format": "ND"},
            "gm_output": {"type": "tensor", "shape": [n, NUM_HEADS_Q, HEAD_SIZE], "dtype": "float16", "format": "ND"},
            "gm_attn_bias": {"type": "tensor", "shape": [NUM_HEADS_Q, NUM_HEADS], "dtype": "float16", "format": "ND"},
            "m": {"type": "int", "value": m},
            "n": {"type": "int", "value": n},
            "t": {"type": "int", "value": t},
            "r": {"type": "int", "value": r}
        }, locals = globals())

    # pylint: disable=W0613
    def construct(
        self,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor=None,
        value_cache: Tensor=None,
        out_cache_loc: Tensor=None,
        k_scale: float=None,
        v_scale: float=None,
    ) -> Tensor:
        if k_scale is not None:
            key = key / k_scale
        if v_scale is not None:
            value = value / v_scale
        key_cache = self.scatter_update(key_cache, out_cache_loc, key)
        value_cache = self.scatter_update(value_cache, out_cache_loc, value)
        return key_cache, value_cache


    def extend(
        self,
        gm_k_cache: Tensor=None,
        gm_v_cache: Tensor=None,
        gm_query: Tensor=None,
        req_to_token: Tensor=None,
        req_pool_indices: Tensor=None,
        seq_lens: Tensor=None,
        extend_prefix_lens: Tensor=None,
        extend_seq_lens: Tensor=None,
        gm_output: Tensor=None,
        gm_attn_bias: Tensor=None,
    ) -> Tensor:
        gm_output = mint.zeros_like(gm_query)
        gm_k_cache = gm_k_cache.contiguous()
        gm_v_cache = gm_v_cache.contiguous()
        gm_query = gm_query.contiguous()
        req_to_token = req_to_token.contiguous()
        req_pool_indices = req_pool_indices.contiguous()
        seq_lens = seq_lens.contiguous()
        extend_prefix_lens = extend_prefix_lens.contiguous()
        extend_seq_lens = extend_seq_lens.contiguous()
        gm_output = gm_output.contiguous()
        gm_attn_bias = gm_attn_bias.contiguous()
        self.gather_kv_fused(
            gm_k_cache, gm_v_cache, gm_query, req_to_token,req_pool_indices,
            seq_lens, extend_prefix_lens, extend_seq_lens, gm_output, gm_attn_bias,
            seq_lens.shape[0], gm_query.shape[0], gm_v_cache.shape[0], req_to_token.shape[0]
        )
        return gm_output

    def decode(
        self,
        gm_k_cache: Tensor=None,
        gm_v_cache: Tensor=None,
        gm_query: Tensor=None,
        req_to_token: Tensor=None,
        req_pool_indices: Tensor=None,
        seq_lens: Tensor=None,
        extend_prefix_lens: Tensor=None,
        extend_seq_lens: Tensor=None,
        gm_output: Tensor=None,
        gm_attn_bias: Tensor=None,
    ) -> Tensor:
        gm_output = mint.zeros_like(gm_query)
        gm_k_cache = gm_k_cache.contiguous()
        gm_v_cache = gm_v_cache.contiguous()
        gm_query = gm_query.contiguous()
        req_to_token = req_to_token.contiguous()
        req_pool_indices = req_pool_indices.contiguous()
        seq_lens = seq_lens.contiguous()
        extend_prefix_lens = extend_prefix_lens.contiguous()
        extend_seq_lens = extend_seq_lens.contiguous()
        gm_output = gm_output.contiguous()
        gm_attn_bias = gm_attn_bias.contiguous()
        self.gather_kv_fused(
            gm_k_cache, gm_v_cache, gm_query, req_to_token,req_pool_indices,
            seq_lens, extend_prefix_lens, extend_seq_lens, gm_output, gm_attn_bias,
            seq_lens.shape[0], gm_query.shape[0], gm_v_cache.shape[0], req_to_token.shape[0]
        )
        return gm_output
    

class MsNativeAttnBackend(nn.Cell):
    """MindSpore Attention Manager."""

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        scale_value: Optional[float] = None,
        mla_v_dim: int = 0,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.scale_value = (
            1 / math.sqrt(self.head_dim) if scale_value is None else scale_value
        )
        self.attention_layout = "TH"

        self.flash_attention = FlashAttentionScore(
            head_num=self.n_heads,
            scale_value=self.scale_value,
            next_tokens=0,
            input_layout=self.attention_layout,
        )
        self.paged_attention = PagedAttention(
            head_num=self.n_heads,
            scale_value=self.scale_value,
            kv_head_num=self.n_kv_heads,
            mla_v_dim=mla_v_dim,
        )
        self.reshape_and_cache = ReshapeAndCache()

    # pylint: disable=W0613
    def construct(
        self,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor = None,
        value_cache: Tensor = None,
        out_cache_loc: Tensor = None,
        k_scale: float = None,
        v_scale: float = None,
    ) -> Tensor:
        if key.dtype != key_cache.dtype:
            if k_scale is not None:
                key = key / k_scale
            key = key.to(key_cache.dtype)
        if value.dtype != value_cache.dtype:
            if v_scale is not None:
                value = value / v_scale
            value = value.to(value_cache.dtype)
        cache_out = self.reshape_and_cache(
            key, value, key_cache, value_cache, out_cache_loc
        )
        key = ops.depend(key, cache_out)

        return key

    def extend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor = None,
        alibi_mask: Tensor = None,
        prefix=None,
        padding_mask: Tensor = None,
        q_seq_lens: Tensor = None,
        batch_valid_length: Tensor = None,
    ) -> Tensor:
        _, _, _, output = self.flash_attention(
            query,
            key,
            value,
            alibi_mask,
            None,
            padding_mask,
            attn_mask,
            prefix,
            q_seq_lens,
            batch_valid_length,
        )
        return output

    def decode(
        self,
        query: Tensor,
        batch_valid_length: Tensor,
        attn_mask: Tensor = None,
        q_seq_lens: Tensor = None,
        key_cache: Tensor = None,
        value_cache: Tensor = None,
        block_tables: Tensor = None,
    ) -> Tensor:
        output = self.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            batch_valid_length,
            None,
            None,
            attn_mask,
            q_seq_lens,
        )
        return output
