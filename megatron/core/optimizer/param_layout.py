# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameter layout dataclasses for optimizer-driven buffer layout.

These dataclasses describe how parameters are laid out in contiguous buffers.
Each distributed optimizer implementation (e.g., DistributedOptimizer) is
responsible for computing these layouts via a compute_param_layout method,
applying its own padding, alignment, and bucket splitting rules. DDP and
buffers consume the resulting layouts without any optimizer-specific knowledge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


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
