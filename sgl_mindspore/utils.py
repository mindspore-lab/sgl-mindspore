# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import mindspore as ms
import torch
import torch_npu
from mindspore._c_expression import MSContext
from mindspore.utils.dlpack import from_dlpack as ms_from_dlpack
from mindspore.utils.dlpack import to_dlpack as ms_to_dlpack
from sglang.srt.distributed import get_tp_group, get_world_group

FORMAT_TYPE = {
    "nz": 29,  # TODO: need a variable or enum from mindspore to keep consistency
}

def is_910b():
    device = MSContext.get_instance().get_ascend_soc_version()
    return device in ['910b', 'ascend910b']


def tensor_torch2ms(x: torch.Tensor):
    if x is None or not isinstance(x, torch.Tensor):
        return x

    # torch tensor -> dlpack -> mindspore tensor
    pt_dlpack = torch.utils.dlpack.to_dlpack(x)
    ms_tensor = ms_from_dlpack(pt_dlpack)
    return ms_tensor


def tensor_ms2torch(x: ms.Tensor):
    if x is None or not isinstance(x, ms.Tensor):
        return x

    # ms tensor -> dlpack -> torch tensor
    ms_dlpack = ms_to_dlpack(x)
    torch_tensor = torch.utils.dlpack.from_dlpack(ms_dlpack)
    torch_npu.npu.synchronize()
    return torch_tensor


def split_loaded_weight(loaded_weight, shard_dim, start_idx, shard_size):
    if shard_dim is None:
        loaded_weight = loaded_weight[:]
        return loaded_weight

    end_idx = start_idx + shard_size
    if shard_dim == 0:
        loaded_weight = loaded_weight[start_idx:end_idx]
    elif shard_dim == 1:
        loaded_weight = loaded_weight[:, start_idx:end_idx]
    elif shard_dim == 2:
        loaded_weight = loaded_weight[:, :, start_idx:end_idx]
    else:
        raise ValueError("shard_dim:{} is not supported.".format(shard_dim))
    return loaded_weight


def _get_tp_group_name():
    return get_tp_group().unique_name


def _get_world_group_name():
    return get_world_group().unique_name


def set_weight_attrs(weight, weight_attrs):
    if not weight_attrs:
        return
    for key, value in weight_attrs.items():
        setattr(weight, key, value)


def get_ms_dtype(dtype: torch.dtype):
    type_name = str(dtype).split(".")[-1]
    if hasattr(ms.dtype, type_name):
        return getattr(ms.dtype, type_name)
    raise ValueError(f"MindSpore dtype {type_name} is not supported.")


def add_prefix(name: str, prefix: str) -> str:
    """Add a weight path prefix to a module name.

    Args:
        name: base module name.
        prefix: weight prefix str to added to the front of `name` concatenated with `.`.

    Returns:
        The string `prefix.name` if prefix is non-empty, otherwise just `name`.
    """
    return name if not prefix else f"{prefix}.{name}"

def format_cast(x: ms.Tensor, format: str):
    if format in FORMAT_TYPE:
        return ms.ops.auto_generate.format_cast(x, FORMAT_TYPE[format])
    else:
        raise ValueError(f"Unknown format {format}")

def get_ascend_soc_version():
    from mindspore._c_expression import MSContext

    return MSContext.get_instance().get_ascend_soc_version()

def is_310p():
    device = get_ascend_soc_version()
    return device in ["310p", "ascend310p"]

def patch_triton_310p():
    """
    Triton-Ascend is not supported on Ascend 310P.
    """
    import torch
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
        alloc_extend_naive,
    )
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    AttentionBackend.support_triton = lambda x:False

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        num_new_pages = (
            (seq_lens + self.roundup) // self.page_size
            - (prefix_lens + self.roundup) // self.page_size
        ).sum()
        num_new_pages_item = num_new_pages.item()
        if self.need_sort and num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages_item > len(self.free_pages):
            return None

        out_indices = torch.empty(
            (extend_num_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        alloc_extend_naive(
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.page_size,
            self.device,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices.int()

    NPUPagedTokenToKVPoolAllocator.alloc_extend = alloc_extend

def patch_memory_pool_310p():
    """
    Memory-pool-npu optimization on Ascend 310P.
    """
    import torch
    import mindspore as ms
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from sgl_mindspore.utils import is_310p

    def _create_buffers_nz(self):
        def create_kv_cache(kv_shape, dtype):
            if len(kv_shape) != 4:
                raise ValueError(f"Format_Cast op need kv_cache shape be"
                                f"(batch_size, num_heads, seq_len, head_dim),"
                                f"but got {len(kv_shape)} dimensions: {kv_shape}")
            batch_size, num_heads, seq_len, head_dim = kv_shape
            reshaped_for_nz = (batch_size, num_heads, seq_len * head_dim)
            zeros_tensor = ms.mint.zeros(reshaped_for_nz, dtype=ms.float16)
            return ms.ops.auto_generate.format_cast(zeros_tensor, 29)
        
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            kv_shape =  (
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            self.k_buffer = [
                create_kv_cache(kv_shape, self.store_dtype)
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                create_kv_cache(kv_shape, self.store_dtype)
                for _ in range(self.layer_num)
            ]
            self.kv_buffer = (self.k_buffer, self.v_buffer)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            if is_310p:
                self._create_buffers_nz()
                return

            self.kv_buffer = torch.zeros(
                (
                    2,
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer = self.kv_buffer[0]
            self.v_buffer = self.kv_buffer[1]

            if self.use_fia:
                self.k_buffer = []
                self.v_buffer = []
                for i in range(self.layer_num):
                    k_buffer_layer = self.kv_buffer[0][i].view(
                        -1, 1, self.head_num, self.head_dim
                    )
                    v_buffer_layer = self.kv_buffer[1][i].view(
                        -1, 1, self.head_num, self.head_dim
                    )
                    self.k_buffer.append(k_buffer_layer)
                    self.v_buffer.append(v_buffer_layer)

    NPUMHATokenToKVPool._create_buffers = _create_buffers
    NPUMHATokenToKVPool._create_buffers_nz = _create_buffers_nz
    
