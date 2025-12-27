# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import mindspore as ms
import torch
import torch_npu
from mindspore.utils.dlpack import from_dlpack as ms_from_dlpack
from mindspore.utils.dlpack import to_dlpack as ms_to_dlpack
from mindspore._c_expression import MSContext
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