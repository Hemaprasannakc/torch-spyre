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

"""Split oversized pointwise and reduction ops into memory-safe chunks.

Runs after ``propagate_spyre_tensor_layouts`` / ``insert_restickify`` and
before ``span_reduction``.

Pointwise:
    Controlling-dim split selection.  Chunk 0 is the original op shrunk
    in-place; chunks 1..N-1 are new Scatter buffers written back into the
    original buffer via MutationLayoutSHOULDREMOVE.  All inserted Scatter
    nodes are skipped by the main loop (``i += n_inserted``) to avoid
    redundant work.

Reduction:
    Same controlling-dim + _choose_split_dim pipeline as the pointwise path.
    Chunk 0 keeps the original op's full-size allocation but receives shrunken
    ranges.  Chunks 1..N-1 receive a physically correct FixedTiledLayout built
    from the original layout's stride_map so each chunk buffer allocates only
    chunk-sized device memory rather than the full tensor.  A Scatter with
    MutationLayoutSHOULDREMOVE then writes each chunk back into the original
    full buffer at the correct offset.

    ``_compute_num_chunks`` calculates the required number of chunks in a
    single pass (accounting for both per-core span and total-bytes limits).
    All inserted chunk_buf and Scatter nodes are skipped by the main loop
    (``i += n_inserted``) — identical to the pointwise path — so inserted
    nodes are never re-examined.  Re-examining them would trigger recursive
    splitting and produce an exponential number of buffers.

Known limitations (warnings fire at runtime when hit):
    [REDUCTION_DIM_AS_CONTROLLING] controlling dim maps to a reduction dim.
    [GLOBAL_SCALAR_REDUCTION] no splittable output dim found.
    [MULTI_OUTPUT_REDUCTION] argmax/argmin have two write deps.
    [FP32_REDUCTION_UNSUPPORTED] fp32 max/min unsupported by Spyre backend.
    [UNCOVERED_TENSOR] chosen split dim does not index all over-limit tensors.
    [MULTI_DIM_REDUCTION_WRONG_SPLIT] 4D+ ops with small outer batch axis.
"""

import math
from dataclasses import dataclass

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    Scatter,
)
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout

from . import config
from .ir import FixedTiledLayout, SpyreReduction
from .logging_utils import get_inductor_logger
from .pass_utils import device_coordinates, host_coordinates
from .views import matching_dim
from .work_division import MAX_SPAN_BYTES

logger = get_inductor_logger("chunk_large_tensors")

# Reduction types unsupported on IEEE_FP32 by the Spyre backend.
_UNSUPPORTED_FP32_REDUCTION_TYPES = frozenset({"max", "min"})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingInfo:
    total_bytes: int
    per_core_span: int
    best_split: int
    dev_dim_size: int
    dev_dim_stride: int
    host_dim: int
    stick_elems: int


def _find_best_split(dim_size: int, max_cores: int) -> int:
    """Return largest divisor of dim_size that is <= max_cores."""
    for i in range(min(max_cores, dim_size), 0, -1):
        if dim_size % i == 0:
            return i
    return 1


def _find_controlling_dim(
    layout: FixedTiledLayout,
    max_cores: int,
) -> tuple[int, int, int, int] | None:
    """Find the outermost splittable device dim and its best core split.

    Walks device dims outer-to-inner (skipping stick dim). For each device dim,
    uses stride_map to find the corresponding host dim. Returns the first host
    dim with size > 1 that is at least stick_elems in the device space.

    Returns
    -------
    (host_dim_idx, dev_dim_size, dev_dim_stride, best_split) or None.
    """
    stl = layout.device_layout
    device_size = [int(s) for s in stl.device_size]
    host_size = [int(s) for s in layout.size]
    host_stride = [int(s) for s in layout.stride]
    stick_elems = stl.elems_per_stick()

    for device_dim in range(len(device_size) - 1):  # skip last = stick
        sm = int(stl.stride_map[device_dim])
        if sm <= 0:
            continue
        matching_dims = [
            d for d, s in enumerate(host_stride) if s == sm and host_size[d] > 1
        ]
        if not matching_dims:
            continue
        host_dim = matching_dims[0]
        dev_dim_size = device_size[device_dim]
        dev_dim_stride = math.prod(device_size[device_dim + 1:])
        if dev_dim_size < stick_elems:
            logger.debug(
                "Skipping device_dim %d: host_dim %d size=%d "
                "< stick_elems=%d, cannot chunk",
                device_dim, host_dim, host_size[host_dim], stick_elems,
            )
            continue
        best_split = _find_best_split(dev_dim_size, max_cores)
        return host_dim, dev_dim_size, dev_dim_stride, best_split

    return None


def _compute_num_chunks(
    chunking_info: ChunkingInfo,
    max_cores: int,
) -> int:
    """Compute number of chunks needed.

    Uses two estimates and picks the best:
    - num_from_span:  per_core_span fits in 256 MB
    - num_from_total: total fits in max_cores * 256 MB

    If chunking by num_from_total improves dim divisibility (chunk dim gets a
    better core split), the total formula is used.
    """
    if chunking_info.dev_dim_size == 0:
        return max(
            1, math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))
        )

    num_from_span = math.ceil(chunking_info.per_core_span / MAX_SPAN_BYTES)
    num_from_total = math.ceil(
        chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores)
    )

    # Check if chunking by num_from_total improves divisibility.
    total_sticks_preview = math.ceil(
        chunking_info.dev_dim_size / chunking_info.stick_elems
    )
    chunk_sticks_preview = math.ceil(total_sticks_preview / max(num_from_total, 1))
    chunk_dim_preview = chunk_sticks_preview * chunking_info.stick_elems
    chunk_best_split = _find_best_split(chunk_dim_preview, max_cores)
    if chunk_best_split > chunking_info.best_split and num_from_total > 1:
        return num_from_total
    return max(num_from_span, num_from_total)


def _needs_chunking(
    layout: FixedTiledLayout,
    max_cores: int,
    controlling: tuple[int, int, int, int] | None,
) -> ChunkingInfo | None:
    """Return ChunkingInfo if this layout exceeds memory limits, else None.

    Two cases trigger chunking:
      1. per_core_span > 256 MB after best split on controlling dim.
      2. total_bytes > 256 MB * max_cores.
    """
    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    total_bytes = math.prod(device_size) * itemsize
    stick_elems = layout.device_layout.elems_per_stick()

    if controlling is None:
        if total_bytes > MAX_SPAN_BYTES * max_cores:
            host_size = [int(s) for s in layout.size]
            fallback_host_dim = max(
                range(len(host_size)), key=lambda d: host_size[d]
            )
            return ChunkingInfo(
                total_bytes=total_bytes,
                per_core_span=total_bytes,
                best_split=1,
                dev_dim_size=0,
                dev_dim_stride=0,
                host_dim=fallback_host_dim,
                stick_elems=stick_elems,
            )
        return None

    host_dim, dev_dim_size, dev_dim_stride, best_split = controlling
    per_core_span = math.ceil(dev_dim_size / best_split) * dev_dim_stride * itemsize
    needs_chunk_for_span = per_core_span > MAX_SPAN_BYTES
    needs_chunk_for_total = total_bytes > MAX_SPAN_BYTES * max_cores

    if needs_chunk_for_span or needs_chunk_for_total:
        logger.info(
            "Op needs chunking: dev_dim_size=%d best_split=%d "
            "per_core_span=%.2fMB total=%.2fGB "
            "(span_limit=256MB total_limit=%.2fGB)",
            dev_dim_size,
            best_split,
            per_core_span / (1024**2),
            total_bytes / (1024**3),
            (MAX_SPAN_BYTES * max_cores) / (1024**3),
        )
        return ChunkingInfo(
            total_bytes=total_bytes,
            per_core_span=per_core_span,
            best_split=best_split,
            dev_dim_size=dev_dim_size,
            dev_dim_stride=dev_dim_stride,
            host_dim=host_dim,
            stick_elems=stick_elems,
        )
    return None


def _contiguous_stride(size: list[int]) -> list[int]:
    """Return a contiguous row-major stride for the given size."""
    stride = [1] * len(size)
    for dim in range(len(size) - 2, -1, -1):
        stride[dim] = stride[dim + 1] * size[dim + 1]
    return stride


def _make_output_indexer(offset: int, split_dim: int):
    """Return a Scatter output indexer shifted by offset on split_dim."""
    def output_indexer(index):
        out = list(index)
        out[split_dim] += offset
        return out
    return output_indexer


def _register_and_insert(
    buf: ComputedBuffer,
    op: ComputedBuffer,
    operations: list[Operation],
    insert_pos: int,
) -> int:
    """Register buf in the graph and insert it at insert_pos."""
    buf.name = V.graph.register_buffer(buf)
    V.graph.register_operation(buf)
    buf.origins = op.origins
    if hasattr(op, "origin_node"):
        buf.origin_node = op.origin_node
    if buf in operations:
        operations.remove(buf)
    operations.insert(insert_pos, buf)
    return insert_pos + 1


def _sticks_first_chunk_size(
    full_size: int,
    num_chunks: int,
    stick_elems: int,
) -> tuple[int, int]:
    """Compute stick-aligned chunk_size and recomputed num_chunks."""
    total_sticks = math.ceil(full_size / stick_elems)
    sticks_per_chunk = math.ceil(total_sticks / num_chunks)
    chunk_size = sticks_per_chunk * stick_elems
    num_chunks = math.ceil(total_sticks / sticks_per_chunk)
    return chunk_size, num_chunks


def _copy_loop_metadata(src, dst) -> None:
    """Copy origins and traceback from src IR node to dst IR node."""
    if hasattr(src, "origins"):
        object.__setattr__(dst, "origins", src.origins)
    if hasattr(src, "traceback"):
        object.__setattr__(dst, "traceback", src.traceback)


# ---------------------------------------------------------------------------
# Pointwise 
# ---------------------------------------------------------------------------


def _make_chunk_fn(orig_fn, dim: int, offset: int):
    """Return an inner_fn that shifts the split dim by offset."""
    def inner_fn(index):
        idx = list(index)
        idx[dim] = idx[dim] + offset
        return orig_fn(idx)
    return inner_fn


def _chunk_op(
    op: ComputedBuffer,
    max_cores: int,
    operations: list[Operation],
    op_index: int,
    chunking_info: ChunkingInfo,
    original_ftl: FixedTiledLayout,
) -> int:
    """Split a Pointwise op into memory-safe chunks.

    Chunk 0 is the original op shrunk in-place (ranges only; layout stays
    full-size so the scheduler finds it by name). Chunks 1..N-1 are direct
    Scatter mutation buffers that compute each chunk inline and write it into
    the corresponding region of the original output.
    """
    original_ranges = list(op.data.ranges)
    original_inner_fn = op.data.inner_fn

    split_dim_idx = chunking_info.host_dim
    if split_dim_idx >= len(original_ranges):
        logger.warning(
            "%s: controlling host_dim=%d out of range for %d-D ranges",
            op.get_name(), split_dim_idx, len(original_ranges),
        )
        return 0

    if len(original_ranges) >= 4:  # noqa: PLR2004
        host_size = [int(s) for s in original_ftl.size]
        if host_size[split_dim_idx] < max(host_size):
            logger.warning(
                "[TODO MULTI_DIM_REDUCTION_WRONG_SPLIT] %s: ndim=%d, "
                "split_dim=%d is not the largest — per-chunk span or "
                "inner_fn offset may be wrong.",
                op.get_name(), len(original_ranges), split_dim_idx,
            )

    split_dim_full_size = int(original_ranges[split_dim_idx])
    stick_elems = chunking_info.stick_elems

    num_chunks = _compute_num_chunks(chunking_info, max_cores)
    chunk_size, num_chunks = _sticks_first_chunk_size(
        split_dim_full_size, num_chunks, stick_elems
    )

    logger.info(
        "Chunking %s: split_dim=%d full_size=%d "
        "chunk_size=%d num_chunks=%d total=%.2fGB",
        op.get_name(), split_dim_idx, split_dim_full_size,
        chunk_size, num_chunks, chunking_info.total_bytes / (1024**3),
    )

    # Chunk 0: shrink original op in-place.
    chunk0_size = min(chunk_size, split_dim_full_size)
    chunk0_ranges = list(original_ranges)
    chunk0_ranges[split_dim_idx] = chunk0_size
    object.__setattr__(op.data, "ranges", chunk0_ranges)

    insert_pos = op_index + 1
    n_inserted = 0

    # Chunks 1..N-1: Scatter mutation into the original output.
    for chunk_idx in range(1, num_chunks):
        chunk_offset = chunk_idx * chunk_size
        remaining_elems = max(0, split_dim_full_size - chunk_offset)
        remaining_sticks = math.ceil(remaining_elems / stick_elems)
        this_chunk_size = min(remaining_sticks * stick_elems, chunk_size)

        chunk_ranges = list(original_ranges)
        chunk_ranges[split_dim_idx] = this_chunk_size

        mutation_data = Scatter(
            device=op.data.device,
            dtype=op.data.dtype,
            inner_fn=_make_chunk_fn(original_inner_fn, split_dim_idx, chunk_offset),
            ranges=chunk_ranges,
            output_indexer=_make_output_indexer(chunk_offset, split_dim_idx),
        )
        mutation_buf = ComputedBuffer(
            name=None,
            layout=MutationLayoutSHOULDREMOVE(op),
            data=mutation_data,
        )
        insert_pos = _register_and_insert(mutation_buf, op, operations, insert_pos)
        n_inserted += 1

    return n_inserted


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reduction: tensor-info helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorInfo:
    dep: MemoryDep
    layout: FixedTiledLayout
    total_bytes: int


def _device_bytes(layout: FixedTiledLayout) -> int:
    return (
        math.prod(int(s) for s in layout.device_layout.device_size)
        * layout.dtype.itemsize
    )


def _collect_tensor_infos(op: ComputedBuffer) -> list[TensorInfo]:
    """Collect FixedTiledLayout infos for all read deps and the write dep."""
    rw = op.get_read_writes()
    if len(rw.writes) > 1:
        logger.warning(
            "[TODO MULTI_OUTPUT_REDUCTION] %s: op has %d write deps; "
            "only the first is handled.",
            op.get_name(), len(rw.writes),
        )

    infos: list[TensorInfo] = []
    for dep in rw.reads:
        if not isinstance(dep, MemoryDep):
            continue
        buf = V.graph.get_buffer(dep.name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            continue
        infos.append(TensorInfo(dep, layout, _device_bytes(layout)))

    write_dep = next(iter(rw.writes))
    infos.append(TensorInfo(write_dep, op.layout, _device_bytes(op.layout)))
    return infos


# ---------------------------------------------------------------------------
# Reduction: inner_fn and data construction
# ---------------------------------------------------------------------------


def _make_shifted_reduction_fn(orig_fn, split_dim: int, offset: int):
    """Return an inner_fn that shifts output index's split_dim by offset."""
    def inner_fn(index, reduction_index):
        shifted = list(index)
        shifted[split_dim] += offset
        return orig_fn(shifted, reduction_index)
    return inner_fn


def _make_chunk_reduction_data(
    original_data,
    split_dim: int,
    offset: int,
    chunk_ranges: list[int],
):
    """Build a chunk Reduction/SpyreReduction data node.

    Preserves SpyreReduction.op_info when the original is a SpyreReduction.
    Warns (but continues) when fp32 max/min is encountered.
    """
    if (
        getattr(original_data, "src_dtype", None) == torch.float32
        and getattr(original_data, "reduction_type", None)
        in _UNSUPPORTED_FP32_REDUCTION_TYPES
    ):
        logger.warning(
            "[TODO FP32_REDUCTION_UNSUPPORTED] reduction_type=%s on fp32 "
            "is not supported by the Spyre backend; chunking proceeds but "
            "codegen will reject this op.",
            original_data.reduction_type,
        )

    fn = (
        original_data.inner_fn
        if offset == 0
        else _make_shifted_reduction_fn(original_data.inner_fn, split_dim, offset)
    )
    kwargs = dict(
        device=original_data.device,
        dtype=original_data.dtype,
        inner_fn=fn,
        ranges=chunk_ranges,
        reduction_ranges=list(original_data.reduction_ranges),
        reduction_type=original_data.reduction_type,
        src_dtype=original_data.src_dtype,
        reduction_hint=original_data.reduction_hint,
    )
    chunk_data = (
        SpyreReduction(op_info=original_data.op_info, **kwargs)
        if isinstance(original_data, SpyreReduction)
        else Reduction(**kwargs)
    )
    _copy_loop_metadata(original_data, chunk_data)
    return chunk_data


# ---------------------------------------------------------------------------
# Reduction: split-dim selection
# ---------------------------------------------------------------------------


def _choose_split_dim(
    op: ComputedBuffer,
    tensor_infos: list[TensorInfo],
    total_limit_bytes: int,
) -> tuple[int, sympy.Symbol] | None:
    """Pick the best output dim to split along for a Reduction op.

    Skips the within-stick host dim (it is an atomic hardware unit).
    Ranks candidates by (covers_all_offending, total_affected_bytes, size, -dim).
    """
    write_dep = next(iter(op.get_read_writes().writes))
    output_symbols = list(write_dep.ranges.keys())
    output_ranges = [int(size) for size in op.data.ranges]

    # Identify the within-stick host dim so we can exclude it.
    stick_host_dim: int | None = None
    if isinstance(op.layout, FixedTiledLayout):
        try:
            out_coords = host_coordinates(op.layout, write_dep)
            dev_coords = device_coordinates(op.layout.device_layout, write_dep)
            stick_host_dim = matching_dim(out_coords, dev_coords[-1])
        except Exception:
            pass

    offending = [
        info for info in tensor_infos if info.total_bytes > total_limit_bytes
    ]
    sym_idx = 0
    best, best_dim, best_sym = None, None, None

    for dim, size in enumerate(output_ranges):
        if size <= 1:
            continue
        if sym_idx >= len(output_symbols):
            break
        sym = output_symbols[sym_idx]
        sym_idx += 1
        if dim == stick_host_dim:
            continue

        affected = [
            info for info in tensor_infos if sym in info.dep.index.free_symbols
        ]
        if not affected:
            continue
        covers_all = all(
            sym in info.dep.index.free_symbols for info in offending
        )
        score = sum(info.total_bytes for info in affected)
        # Prefer: covers_all first, then larger score, then larger size, then outer dim.
        rank = (covers_all, score, size, -dim)
        if best is None or rank > best:
            best, best_dim, best_sym = rank, dim, sym

    if best_dim is None or best_sym is None:
        return None
    return best_dim, best_sym


# ---------------------------------------------------------------------------
# Reduction: core chunking with physical layouts + recursive guard
# ---------------------------------------------------------------------------


def _input_span_is_already_safe(
    output_ranges: list[int],
    chunking_info: ChunkingInfo,
    input_itemsize: int,
    max_cores: int,
) -> bool:
    """Return True when the current output range already brings the input span safe.

    Used to avoid re-chunking an op whose per-core span is already within limits.
    dev_dim_size == 0 is the fallback path (no controlling dim found).
    """
    host_dim = chunking_info.host_dim

    if chunking_info.dev_dim_size > 0:
        if host_dim >= len(output_ranges):
            return False
        actual_out_size = output_ranges[host_dim]
        min_per_core = (
            math.ceil(actual_out_size / chunking_info.best_split)
            * chunking_info.dev_dim_stride
            * input_itemsize
        )
        return min_per_core <= MAX_SPAN_BYTES

    total_out = math.prod(output_ranges) * input_itemsize
    return total_out <= MAX_SPAN_BYTES * max_cores


def _make_reduction_chunk_layout(
    op_layout: FixedTiledLayout,
    split_dim: int,
    chunk_size: int,
    full_size: int,
) -> FixedTiledLayout:
    """Build a correctly-scaled FixedTiledLayout for a reduction chunk buffer.

    Using op.layout unchanged for chunk_buf causes wrong device-address
    computation: device_size still references full_size and stride_map retains
    the original batch stride (full_size), so make_loader() reads from wrong
    positions (e.g. batch stride 8193 instead of 4096 for a half-split).

    This function builds a fresh layout via the 3-arg
    SpyreTensorLayout(device_size, stride_map, device_dtype) constructor,
    with two adjustments for the device dim that maps to split_dim:

      device_size[k] := chunk_size
          — the device extent of that dim shrinks from full_size to chunk_size.

      stride_map[k] for outer dims := original * chunk_size // full_size
          — outer host strides contain full_size as a factor; after chunking
            that factor becomes chunk_size (e.g. batch stride 8193 → 4096).

    Chunk 0 keeps op.layout unchanged because it IS op and writes directly
    into op's full allocation at the correct strided positions.  Only
    chunks 1..N-1 need their own correctly-sized layout.
    """
    stl = op_layout.device_layout
    orig_device_size = [int(s) for s in stl.device_size]
    orig_stride_map  = [int(s) for s in stl.stride_map]
    host_stride      = [int(s) for s in op_layout.stride]

    chunk_host_size   = [int(s) for s in op_layout.size]
    chunk_host_size[split_dim] = chunk_size
    chunk_host_stride = _contiguous_stride(chunk_host_size)

    split_host_stride = host_stride[split_dim]
    # In a contiguous host layout, strides of dims outer to split_dim are all
    # multiples of (split_host_stride * full_size).
    outer_threshold = split_host_stride * full_size

    new_device_size = list(orig_device_size)
    new_stride_map  = list(orig_stride_map)

    for k in range(len(orig_stride_map) - 1):  # skip last = stick dim
        sm = orig_stride_map[k]
        if sm <= 0:
            continue  # sentinel for size-1 or virtual dims

        if sm == split_host_stride:
            # Device dim k maps directly to split_dim.
            # Its element count shrinks from full_size to chunk_size.
            new_device_size[k] = chunk_size

        elif outer_threshold > 0 and sm % outer_threshold == 0:
            # Device dim k maps to a host dim outer to split_dim.
            # Its host stride embeds full_size as a factor; replace with
            # chunk_size so the batch (or outer) stride is correct.
            new_stride_map[k] = sm * chunk_size // full_size

    chunk_stl = SpyreTensorLayout(new_device_size, new_stride_map, stl.device_dtype)
    return FixedTiledLayout(
        op_layout.device,
        op_layout.dtype,
        chunk_host_size,
        chunk_host_stride,
        chunk_stl,
    )


def _chunk_reduction_op(
    op: ComputedBuffer,
    operations: list[Operation],
    op_index: int,
    tensor_infos: list[TensorInfo],
    total_limit_bytes: int,
    chunking_info: ChunkingInfo,
) -> int:
    """Split a Reduction op into memory-safe chunks along an output dim.

    Chunk 0: replaces op.data with a smaller-range Reduction; op.layout is
    kept unchanged (full-size allocation so downstream reads find the buffer).

    Chunks 1..N-1: each gets a correctly-scaled FixedTiledLayout via
    _make_reduction_chunk_layout (device_size and stride_map proportionally
    adjusted for chunk_size), followed by a Scatter with
    MutationLayoutSHOULDREMOVE(op) that writes the result back into the
    original full buffer at the correct device offset.

    Returns the number of new operations inserted (0 if chunking is skipped).
    """
    max_cores = config.sencores
    output_ranges = [int(r) for r in op.data.ranges]
    host_dim = chunking_info.host_dim

    # Guard: if the controlling dim maps to a reduction dim (not an output dim),
    # we cannot safely split — chunking along a reduction dim would produce
    # partial sums rather than independent sub-reductions.
    host_dim_is_output_dim = (
        chunking_info.dev_dim_size > 0
        and host_dim < len(output_ranges)
        and output_ranges[host_dim] > 1
    )
    if chunking_info.dev_dim_size > 0 and not host_dim_is_output_dim:
        logger.warning(
            "[TODO REDUCTION_DIM_AS_CONTROLLING] %s: "
            "controlling host_dim=%d maps to a reduction dim; "
            "chunking skipped, EAR overflow likely.",
            op.get_name(), host_dim,
        )
        return 0

    input_itemsize = getattr(op.data, "src_dtype", op.data.dtype).itemsize
    if _input_span_is_already_safe(output_ranges, chunking_info, input_itemsize, max_cores):
        logger.debug(
            "%s: input span already safe for current output range, skipping re-chunk.",
            op.get_name(),
        )
        return 0

    result = _choose_split_dim(op, tensor_infos, total_limit_bytes)
    if result is None:
        logger.warning(
            "[TODO GLOBAL_SCALAR_REDUCTION] %s: no splittable output dim found; "
            "chunking skipped.",
            op.get_name(),
        )
        return 0
    split_dim, split_sym = result

    # Warn when the chosen dim does not index all over-limit tensors.
    uncovered = [
        info.dep.name
        for info in tensor_infos
        if info.total_bytes > total_limit_bytes
        and split_sym not in info.dep.index.free_symbols
    ]
    if uncovered:
        logger.warning(
            "[TODO UNCOVERED_TENSOR] %s: split_sym=%s does not index tensors %s; "
            "those tensors remain over-limit.",
            op.get_name(), split_sym, uncovered,
        )

    if len(output_ranges) >= 4:  # noqa: PLR2004
        host_size = [int(s) for s in op.layout.size]
        if split_dim < len(host_size) and host_size[split_dim] < max(host_size):
            logger.warning(
                "[TODO MULTI_DIM_REDUCTION_WRONG_SPLIT] %s: ndim=%d, "
                "split_dim=%d is not the largest — per-chunk span or "
                "inner_fn offset may be wrong.",
                op.get_name(), len(output_ranges), split_dim,
            )

    original_data = op.data
    full_size = output_ranges[split_dim]
    if full_size <= 1:
        return 0

    # Use the OUTPUT layout's stick granularity for chunk alignment so that
    # each chunk boundary is device-addressable.
    out_stick_elems = (
        op.layout.device_layout.elems_per_stick()
        if isinstance(op.layout, FixedTiledLayout)
        else chunking_info.stick_elems
    )

    num_chunks = _compute_num_chunks(chunking_info, max_cores)
    chunk_size, num_chunks = _sticks_first_chunk_size(
        full_size, num_chunks, out_stick_elems
    )
    if num_chunks <= 1:
        return 0

    logger.info(
        "Chunking reduction %s: split_dim=%d full_size=%d "
        "chunk_size=%d num_chunks=%d per_core_span=%.2fMB total=%.2fGB",
        op.get_name(), split_dim, full_size, chunk_size, num_chunks,
        chunking_info.per_core_span / (1024**2),
        chunking_info.total_bytes / (1024**3),
    )

    # --- Chunk 0: replace op.data in-place; keep op.layout for allocation ---
    chunk0_size = min(chunk_size, full_size)
    chunk0_ranges = list(output_ranges)
    chunk0_ranges[split_dim] = chunk0_size
    object.__setattr__(
        op,
        "data",
        _make_chunk_reduction_data(original_data, split_dim, 0, chunk0_ranges),
    )
    ComputedBuffer.get_default_sizes_body.clear_cache(op)

    n_inserted = 0
    insert_pos = op_index + 1

    # --- Chunks 1..N-1: physical chunk layout + Scatter write-back ---
    for chunk_idx in range(1, num_chunks):
        offset = chunk_idx * chunk_size
        remaining_elems = max(0, full_size - offset)
        remaining_sticks = math.ceil(remaining_elems / out_stick_elems)
        this_chunk_size = min(remaining_sticks * out_stick_elems, chunk_size)

        chunk_ranges = list(output_ranges)
        chunk_ranges[split_dim] = this_chunk_size

        chunk_buf = ComputedBuffer(
            name=None,
            layout=_make_reduction_chunk_layout(
                op.layout, split_dim, this_chunk_size, full_size
            ),
            data=_make_chunk_reduction_data(
                original_data, split_dim, offset, chunk_ranges
            ),
        )
        chunk_buf.origins = op.origins
        insert_pos = _register_and_insert(chunk_buf, op, operations, insert_pos)
        n_inserted += 1

        # Scatter: reads chunk_buf, writes into op's memory at the correct offset.
        loader = chunk_buf.make_loader()

        def _scatter_inner(index, _loader=loader):
            return _loader(index)

        scatter_buf = ComputedBuffer(
            name=None,
            layout=MutationLayoutSHOULDREMOVE(op),
            data=Scatter(
                device=original_data.device,
                dtype=original_data.dtype,
                inner_fn=_scatter_inner,
                ranges=chunk_ranges,
                output_indexer=_make_output_indexer(offset, split_dim),
            ),
        )
        _copy_loop_metadata(original_data, scatter_buf.data)
        scatter_buf.origins = op.origins
        insert_pos = _register_and_insert(scatter_buf, op, operations, insert_pos)
        n_inserted += 1

    return n_inserted


# ---------------------------------------------------------------------------
# Public pass
# ---------------------------------------------------------------------------


def chunk_large_tensors(operations: list[Operation]) -> None:
    """Split pointwise and reduction ops whose device footprint exceeds limits.

    Must run after ``propagate_spyre_tensor_layouts`` / ``insert_restickify``
    and before ``span_reduction``.

    Both pointwise and reduction paths advance ``i`` by the number of nodes
    inserted (``i += n_inserted``) so the main loop never re-examines inserted
    Scatter or chunk_buf nodes.  ``_compute_num_chunks`` determines the correct
    number of chunks in a single pass for both per-core span and total-bytes
    limits, so no recursive re-chunking is required.
    """
    max_cores = config.sencores
    total_limit_bytes = MAX_SPAN_BYTES * max_cores
    i = 0
    while i < len(operations):
        op = operations[i]

        if isinstance(op, ComputedBuffer) and isinstance(op.layout, FixedTiledLayout):

            if isinstance(op.data, Pointwise) and len(op.data.ranges) >= 2:
                controlling = _find_controlling_dim(op.layout, max_cores)
                chunking_info = _needs_chunking(op.layout, max_cores, controlling)
                if chunking_info is not None:
                    n_inserted = _chunk_op(
                        op, max_cores, operations, i, chunking_info, op.layout
                    )
                    # Skip inserted Scatter nodes — they are not Pointwise/Reduction
                    # and do not need chunking examination.
                    i += n_inserted

            elif isinstance(op.data, Reduction):
                try:
                    tensor_infos = _collect_tensor_infos(op)
                except RuntimeError as e:
                    logger.debug(
                        "%s: _collect_tensor_infos failed (%s); skipping",
                        op.get_name(), e,
                    )
                    i += 1
                    continue

                if not tensor_infos:
                    i += 1
                    continue

                read_infos = [
                    info for info in tensor_infos if info.dep.name != op.get_name()
                ]
                dominant_layout = (
                    max(read_infos, key=lambda t: t.total_bytes).layout
                    if read_infos
                    else op.layout
                )

                controlling = _find_controlling_dim(dominant_layout, max_cores)
                chunking_info = _needs_chunking(dominant_layout, max_cores, controlling)

                if chunking_info is not None:
                    n_inserted = _chunk_reduction_op(
                        op, operations, i, tensor_infos, total_limit_bytes, chunking_info
                    )
                    # Skip inserted chunk_buf and scatter nodes — identical to
                    # the pointwise path.  _compute_num_chunks already produces
                    # the correct number of chunks in one pass; re-examining
                    # chunk_bufs caused exponential buffer growth via recursion.
                    i += n_inserted

        i += 1

