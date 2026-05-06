# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split pointwise ops whose device memory exceeds the hardware span limit.

Runs after ``propagate_spyre_tensor_layouts`` and ``insert_restickify`` so
that every ``ComputedBuffer`` already carries a ``FixedTiledLayout``.
``core_division_planning`` sees the resulting chunks as normal ops.
"""

import math

import torch
import torch._inductor.lowering as ind_lowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Scatter,
)
from torch._inductor.virtualized import V

from . import config
from .core_division import MAX_SPAN_BYTES
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("chunk_large_tensors")


def _needs_chunking(op: ComputedBuffer, max_cores: int) -> int | None:
    """Return total device bytes when they exceed *MAX_SPAN_BYTES * max_cores*.

    ``op.layout`` is already a ``FixedTiledLayout`` at this point so we can
    read ``device_size`` directly.  Returns ``None`` when chunking is not
    needed.
    """
    device_layout = op.layout.device_layout
    total_bytes = (
        math.prod(int(s) for s in device_layout.device_size) * op.layout.dtype.itemsize
    )
    return total_bytes if total_bytes > MAX_SPAN_BYTES * max_cores else None


def _find_split_dim(op: ComputedBuffer) -> int:
    """Return the host dimension index to split on.

    Walks device dimensions outermost-to-innermost (skipping the last
    within-stick dimension) and returns the first host dimension whose
    stride matches and whose size is greater than one.
    """
    layout = op.layout
    device_layout = layout.device_layout
    host_size = [int(s) for s in layout.size]
    host_stride = [int(s) for s in layout.stride]

    for device_dim in range(len(device_layout.device_size) - 1):
        stride_val = int(device_layout.stride_map[device_dim])
        if stride_val <= 0:
            continue
        host_dim = next(
            (d for d, s in enumerate(host_stride) if s == stride_val),
            None,
        )
        if host_dim is not None and host_size[host_dim] > 1:
            return host_dim

    return max(range(len(host_size)), key=lambda d: host_size[d])


def _make_chunk_layout(
    original_ftl: FixedTiledLayout,
    split_dim: int,
    chunk_size: int,
) -> FixedTiledLayout:
    """Build a ``FixedTiledLayout`` for a single chunk.

    The host size along *split_dim* is shrunk to *chunk_size*; strides are
    recomputed for a contiguous layout.
    """
    from torch_spyre._C import SpyreTensorLayout

    chunk_host_size = list(original_ftl.size)
    chunk_host_size[split_dim] = chunk_size

    chunk_stride = [1] * len(chunk_host_size)
    for d in range(len(chunk_host_size) - 2, -1, -1):
        chunk_stride[d] = chunk_stride[d + 1] * chunk_host_size[d + 1]

    chunk_stl = SpyreTensorLayout(
        [int(s) for s in chunk_host_size],
        original_ftl.dtype,
    )
    return FixedTiledLayout(
        original_ftl.device,
        original_ftl.dtype,
        chunk_host_size,
        chunk_stride,
        chunk_stl,
    )


def _make_chunk_inner_fn(orig_fn, dim: int, offset: int):
    """Return an ``inner_fn`` that shifts *dim* by *offset*."""

    def inner_fn(index):
        idx = list(index)
        idx[dim] = idx[dim] + offset
        return orig_fn(idx)

    return inner_fn


def _make_overwrite_inner_fn(loader, overwrite_fn):
    """Return an ``inner_fn`` wrapping *loader* through the overwrite op."""

    def inner_fn(index):
        return overwrite_fn(loader(index))

    return inner_fn


def _make_output_indexer(dim: int, offset: int):
    """Return an ``output_indexer`` that shifts *dim* by *offset*."""

    def output_indexer(index):
        out_index = [*index]
        out_index[dim] += offset
        return out_index

    return output_indexer


def _insert_op(
    buf: ComputedBuffer,
    operations: list[Operation],
    insert_pos: int,
) -> int:
    """Register *buf* in the graph and insert it into *operations*.

    Returns the next insertion position.
    """
    buf.name = V.graph.register_buffer(buf)
    V.graph.register_operation(buf)
    if buf in operations:
        operations.remove(buf)
    operations.insert(insert_pos, buf)
    return insert_pos + 1


def _chunk_op(
    op: ComputedBuffer,
    max_cores: int,
    operations: list[Operation],
    op_index: int,
    total_bytes: int,
) -> None:
    """Split *op* into chunks along the best device dimension."""
    original_ftl = op.layout
    original_ranges = list(op.data.ranges)
    original_inner_fn = op.data.inner_fn

    split_dim = _find_split_dim(op)
    full_size = int(original_ranges[split_dim])
    chunk_size = math.ceil(
        full_size / math.ceil(total_bytes / (MAX_SPAN_BYTES * max_cores))
    )
    num_chunks = math.ceil(full_size / chunk_size)

    logger.info(
        "Chunking %s: dim=%d, full_size=%d, num_chunks=%d, "
        "chunk_size=%d, total_bytes=%.2fGB",
        op.get_name(),
        split_dim,
        full_size,
        num_chunks,
        chunk_size,
        total_bytes / (1024**3),
    )

    overwrite_fn = ind_lowering.ops_wrapper(torch.ops.spyre.overwrite.__name__)

    # --- chunk 0: shrink the original op's ranges in-place ---
    # The layout stays full-size so the buffer storage is allocated at the
    # original size; chunk 0 writes the first ``chunk_size`` rows directly.
    chunk0_size = min(chunk_size, full_size)
    chunk0_ranges = list(original_ranges)
    chunk0_ranges[split_dim] = chunk0_size
    object.__setattr__(op.data, "ranges", chunk0_ranges)

    # --- chunks 1..N-1: new compute buffer + overwrite into buf0 ---
    insert_pos = op_index + 1

    for c in range(1, num_chunks):
        offset = c * chunk_size
        this_chunk_size = min(chunk_size, full_size - offset)

        chunk_ranges = list(original_ranges)
        chunk_ranges[split_dim] = this_chunk_size

        # -- pointwise compute for this chunk --
        chunk_pw = Pointwise(
            device=op.data.device,
            dtype=op.data.dtype,
            inner_fn=_make_chunk_inner_fn(original_inner_fn, split_dim, offset),
            ranges=chunk_ranges,
        )
        object.__setattr__(chunk_pw, "origins", op.data.origins)
        object.__setattr__(chunk_pw, "traceback", op.data.traceback)

        chunk_layout = _make_chunk_layout(original_ftl, split_dim, this_chunk_size)
        chunk_buf = ComputedBuffer(name=None, layout=chunk_layout, data=chunk_pw)
        chunk_buf.origins = op.origins
        insert_pos = _insert_op(chunk_buf, operations, insert_pos)

        # -- scatter-overwrite into the original full-size buffer --
        chunk_loader = chunk_buf.make_loader()
        overwrite_data = Scatter(
            device=op.data.device,
            dtype=op.data.dtype,
            inner_fn=_make_overwrite_inner_fn(chunk_loader, overwrite_fn),
            ranges=chunk_ranges,
            output_indexer=_make_output_indexer(split_dim, offset),
        )

        overwrite_buf = ComputedBuffer(
            name=None,
            layout=MutationLayoutSHOULDREMOVE(op),
            data=overwrite_data,
        )
        overwrite_buf.origins = op.origins
        insert_pos = _insert_op(overwrite_buf, operations, insert_pos)


def chunk_large_tensors(operations: list[Operation]) -> None:
    """Split pointwise ops whose device memory exceeds the hardware span limit.

    Only three-dimensional pointwise ``ComputedBuffer`` nodes with a
    ``FixedTiledLayout`` are considered.  The pass runs **after**
    ``propagate_spyre_tensor_layouts`` and ``insert_restickify`` so that
    layouts are already assigned; ``core_division_planning`` runs afterwards
    and treats the resulting chunks as ordinary operations.
    """
    max_cores = config.sencores
    i = 0
    while i < len(operations):
        op = operations[i]
        if (
            isinstance(op, ComputedBuffer)
            and isinstance(op.data, Pointwise)
            and isinstance(op.layout, FixedTiledLayout)
            and len(op.data.ranges) == 3
        ):
            total_bytes = _needs_chunking(op, max_cores)
            if total_bytes is not None:
                _chunk_op(op, max_cores, operations, i, total_bytes)
        i += 1
