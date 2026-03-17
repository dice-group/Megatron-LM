# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Parameter layout utilities for DDP buffers.

This module defines how parameters are laid out in contiguous buffers,
independently of buffer allocation and optimizer logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ..fp8_utils import is_float8tensor


@dataclass
class ParamLayout:
    """Layout for parameters within a single contiguous buffer.

    Describes how parameters should be laid out in the contiguous buffer.

    Attributes:
        param_index_map: Mapping from parameter to (start_index, end_index, bucket_id) in buffer.
        bucket_indices: List of (start_index, end_index) for each bucket.
        per_bucket_numel_unpadded: Number of unpadded elements per bucket.
    """

    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(default_factory=dict)
    bucket_indices: List[Tuple[int, int]] = field(default_factory=list)
    per_bucket_numel_unpadded: List[int] = field(default_factory=list)


@dataclass
class ParamLayoutMap:
    """Layout for all parameters across all dtype groups in a model chunk.

    Maps (param_dtype, grad_dtype) keys to per-buffer ParamLayout objects.
    Each ParamLayout has its own independent index space since different
    dtype groups are physically separate buffers.

    Attributes:
        layouts: Mapping from (param_dtype, grad_dtype) to ParamLayout.
    """

    layouts: Dict[Tuple[torch.dtype, torch.dtype], ParamLayout] = field(default_factory=dict)


def default_param_layout(
    params: List[torch.nn.Parameter],
    bucket_size: Optional[int],
) -> ParamLayout:
    """Compute parameter layout for the non-distributed-optimizer case.

    No padding is applied. Parameters are iterated in reverse order (backprop order)
    and grouped into buckets of approximately `bucket_size` elements.

    Args:
        params: List of parameters to lay out.
        bucket_size: Approximate number of elements per bucket, or None for a single bucket.

    Returns:
        ParamLayout with the computed mapping.
    """
    param_index_map = {}
    bucket_indices = []
    per_bucket_numel_unpadded = []

    param_start_index = 0
    bucket_start_index = 0
    bucket_params = set()
    bucket_id = 0

    for param in params[::-1]:
        this_numel = param.data.nelement()
        param_end_index = param_start_index + this_numel
        param_index_map[param] = (param_start_index, param_end_index, bucket_id)
        bucket_params.add(param)

        if bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size:
            per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
            bucket_indices.append((bucket_start_index, param_end_index))
            bucket_start_index = param_end_index
            bucket_params = set()
            bucket_id += 1
            param_start_index = param_end_index
        else:
            param_start_index = param_end_index

    # Add remaining params to a new bucket.
    if len(bucket_params) > 0:
        per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
        bucket_indices.append((bucket_start_index, param_end_index))

    return ParamLayout(
        param_index_map=param_index_map,
        bucket_indices=bucket_indices,
        per_bucket_numel_unpadded=per_bucket_numel_unpadded,
    )


def group_params_by_dtype(
    params: List[torch.nn.Parameter],
    grad_reduce_in_fp32: bool,
) -> Dict[Tuple[torch.dtype, torch.dtype], Tuple[List[torch.nn.Parameter], List[int]]]:
    """Group parameters by their (param_dtype, grad_dtype) for buffer allocation.

    For FP8 parameters, the param_dtype is torch.uint8 (the actual storage dtype).
    Returns a dict mapping (param_dtype, grad_dtype) to (params_list, param_indices).

    The param_indices track each parameter's position among same-dtype params (using
    the "fake" high-precision dtype for FP8 params), needed for loading non-native-fp8
    checkpoints in native-fp8 mode.

    Args:
        params: List of parameters to group.
        grad_reduce_in_fp32: Whether gradients are reduced in FP32.

    Returns:
        Dict mapping (param_dtype, grad_dtype) to (params_list, param_indices).
    """
    dtype_to_params = {}
    dtype_to_offsets = {}
    dtype_to_indices = {}

    for param in params:
        assert param.requires_grad

        param_dtype = param.dtype
        if is_float8tensor(param):
            param_dtype = torch.uint8
        grad_dtype = torch.float if grad_reduce_in_fp32 else param.dtype

        param_list = dtype_to_params.get((param_dtype, grad_dtype), [])
        param_list.append(param)
        dtype_to_params[(param_dtype, grad_dtype)] = param_list

        offset = dtype_to_offsets.get((param.dtype, grad_dtype), 0)
        dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
        indices = dtype_to_indices.get((param_dtype, grad_dtype), [])
        indices.append(offset)
        dtype_to_indices[(param_dtype, grad_dtype)] = indices

    result = {}
    for key, param_list in dtype_to_params.items():
        result[key] = (param_list, dtype_to_indices[key])
    return result
