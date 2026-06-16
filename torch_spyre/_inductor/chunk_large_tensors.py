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

"""Split oversized pointwise/scalar-reduction ops into memory-safe chunks.

Runs after ``propagate_spyre_tensor_layouts`` / ``insert_restickify`` and
before ``span_reduction``.  Each chunk becomes a normal
``ComputedBuffer`` that work-division handles without special-casing.
"""

import itertools
import math

from dataclasses import dataclass
from typing import cast
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    Scatter,
)
from torch._inductor.virtualized import V

from . import config
from .constants import BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP, TOPK_OPS
from .errors import Unsupported
from .work_division import MAX_SPAN_BYTES
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .loop_info import copy_op_metadata
from .pass_utils import host_coordinates


logger = get_inductor_logger("chunk_large_tensors")


# Scalar reductions have one tensor operand and produce independent output
# elements for each entry in Reduction.ranges.  We can chunk those output ranges
# without touching Reduction.reduction_ranges.
SUPPORTED_SCALAR_REDUCTIONS = {
    "sum",
    "mean",
    "max",
    "amax",
    "min",
    "amin",
}

UNSUPPORTED_CHUNKED_REDUCTIONS = {
    BATCH_MATMUL_OP,
    BATCH_MATMUL_FP8_OP,
    "any",
    "prod",
    "argmax",
    "argmin",
    "welford_reduce",
    "welford_combine",
    "xor_sum",
    "exx2",
    *TOPK_OPS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingInfo:
    total_bytes: int
    per_core_span: int
    unsplit_per_core_span: int
    core_split_estimate: int
    selected_device_dim_size: int
    selected_device_span_stride_elems: int
    selected_host_dim: int
    stick_elems: int
    outer_device_span_prefix_elems: int = 1
    reason: str | None = None


@dataclass(frozen=True)
class SpanDimInfo:
    selected_host_dim: int
    selected_device_dim_size: int
    selected_device_span_stride_elems: int
    core_split_estimate: int
    outer_device_span_prefix_elems: int = 1
    skipped_outer_device_dims: tuple[int, ...] = ()


@dataclass(frozen=True)
class TensorInfo:
    dep: MemoryDep
    layout: FixedTiledLayout
    device_bytes: int


def _device_bytes(layout: FixedTiledLayout) -> int:
    return (
        math.prod(int(s) for s in layout.device_layout.device_size)
        * layout.dtype.itemsize
    )


def _collect_tensor_infos(op: ComputedBuffer) -> list[TensorInfo]:
    """Collect FixedTiledLayout read/write tensors for an op.

    Reduction input span checks must use physical Spyre layout sizes, not logical
    ``ranges * reduction_ranges``. Device layouts include stick padding and the
    actual dtype size used by the backend.
    """
    rw = op.get_read_writes()
    infos: list[TensorInfo] = []

    for dep in rw.reads:
        if not isinstance(dep, MemoryDep):
            continue
        try:
            buf = V.graph.get_buffer(dep.name)
        except Exception:
            logger.debug(
                "Could not resolve read buffer %s for %s",
                dep.name,
                op.get_name(),
            )
            continue

        layout = buf.get_layout()
        if isinstance(layout, FixedTiledLayout):
            infos.append(TensorInfo(dep, layout, _device_bytes(layout)))

    if len(rw.writes) > 1:
        logger.warning(
            "%s has %d write deps; chunk_large_tensors only inspects the first",
            op.get_name(),
            len(rw.writes),
        )

    write_dep = next(iter(rw.writes), None)
    if isinstance(write_dep, MemoryDep) and isinstance(op.layout, FixedTiledLayout):
        infos.append(TensorInfo(write_dep, op.layout, _device_bytes(op.layout)))

    return infos


def _find_max_divisible_core_split(dim_size: int, max_cores: int) -> int:
    """Return largest divisor of dim_size that is <= max_cores."""
    for i in range(min(max_cores, dim_size), 0, -1):
        if dim_size % i == 0:
            return i
    return 1


def _find_outermost_span_reduction_dim(
    layout: FixedTiledLayout,
    max_cores: int,
) -> SpanDimInfo | None:
    """Find the outermost splittable device dim and its best core split.

    Walks device dims outer-to-inner (skipping stick dim). For each
    device dim, uses stride_map to find the corresponding host dim.
    Returns the first host dim with size > 1.

    The span-reduction dim determines per-core memory span.
    Splitting inner dims increases parallelism but does NOT reduce span.
    Splitting the stick dim is not supported (atomic memory unit).

    Returns a ``SpanDimInfo`` with the selected host dim, selected physical dim,
    inner physical stride, and the best core split.  If small unchunkable outer
    device dims are skipped, their product is recorded as
    ``outer_device_span_prefix_elems`` so later span math still accounts for
    that outer span.  Returns None if no valid splittable dim is found.

    Example::

        host_size   = [32,   8193,  1740]
        device_size = [8193, 28,    32,  64]
        stride_map  = [1740, 64, 14255820,  1]

        device_dim 0: stride_map=1740 -> selected_host_dim 1 (M=8193), size>1
        8193 = 3 * 2731 -> core_split_estimate = 3 (largest divisor <= 32)
        selected_device_span_stride_elems = 28 * 32 * 64 = 57344

        -> returns SpanDimInfo(selected_host_dim=1,
                                selected_device_dim_size=8193,
                                selected_device_span_stride_elems=57344,
                                core_split_estimate=3)
    """
    stl = layout.device_layout
    device_size = [int(s) for s in stl.device_size]
    host_size = [int(s) for s in layout.size]
    host_stride = [int(s) for s in layout.stride]
    stick_elems = stl.elems_per_stick()

    outer_device_span_prefix_elems = 1
    skipped_outer_device_dims: list[int] = []

    for device_dim in range(len(device_size) - 1):  # skip last=stick
        sm = int(stl.stride_map[device_dim])
        if sm <= 0:
            outer_device_span_prefix_elems *= device_size[device_dim]
            skipped_outer_device_dims.append(device_dim)
            continue

        # Collect host dims matching this device stride.  For a non-stick
        # host dim the stride_map entry is the host stride.  For the
        # outer-stick device dim, stride_map is host_stride * stick_elems
        # while the within-stick dim carries host_stride.  Example:
        # shape=[8192, 17408] may have device_size=[272, 8192, 64]
        # and stride_map=[64, 17408, 1]; device dim 0 still maps to
        # host dim 1 because 64 == host_stride[1] * stick_elems.
        matching_dims = [
            d
            for d, s in enumerate(host_stride)
            if host_size[d] > 1 and (s == sm or s * stick_elems == sm)
        ]

        if not matching_dims:
            outer_device_span_prefix_elems *= device_size[device_dim]
            skipped_outer_device_dims.append(device_dim)
            continue

        selected_host_dim = matching_dims[0]
        selected_device_dim_size = device_size[device_dim]
        # Match work_division.get_per_core_span(): each device coord's span
        # uses the inner physical stride for that coord. Outer device dims are
        # considered as separate coords/splits by work_division, not multiplied
        # into this selected coord's stride.
        selected_device_span_stride_elems = math.prod(device_size[device_dim + 1 :])

        # Device dims smaller than one stick cannot produce stick-aligned
        # chunks on that dim.  Instead of failing immediately, carry their
        # physical size as an outer prefix and keep walking inward.  A later
        # chunkable dim can still make the whole prefixed span safe, e.g.
        # attention layouts with heads=32 followed by a large sequence dim.
        if selected_device_dim_size < stick_elems:
            outer_device_span_prefix_elems *= selected_device_dim_size
            skipped_outer_device_dims.append(device_dim)
            logger.debug(
                "Skipping device_dim %d: selected_host_dim %d size=%d "
                "< stick_elems=%d, carrying outer prefix=%d",
                device_dim,
                selected_host_dim,
                host_size[selected_host_dim],
                stick_elems,
                outer_device_span_prefix_elems,
            )
            continue

        # Find the largest divisor of selected_device_dim_size that fits within
        # max_cores.  This simulates the best split work_division can
        # do on this dim.
        core_split_estimate = _find_max_divisible_core_split(
            selected_device_dim_size, max_cores
        )
        return SpanDimInfo(
            selected_host_dim=selected_host_dim,
            selected_device_dim_size=selected_device_dim_size,
            selected_device_span_stride_elems=selected_device_span_stride_elems,
            core_split_estimate=core_split_estimate,
            outer_device_span_prefix_elems=outer_device_span_prefix_elems,
            skipped_outer_device_dims=tuple(skipped_outer_device_dims),
        )

    return None


def _choose_sticks_per_chunk(total_sticks: int, min_sticks: int) -> int:
    """Choose a stick chunk size with regular geometry when possible.

    ``min_sticks`` is the smallest chunk size implied by the required chunk
    count, usually ``ceil(total_sticks / num_chunks)``.  This helper keeps that
    as the lower safety target, then looks for a nearby divisor of
    ``total_sticks`` so chunk sizes are regular.

    Example: ``total_sticks=100`` and ``num_chunks=6`` gives
    ``min_sticks=17``.  Using 17 would produce chunk stick counts
    ``17, 17, 17, 17, 17, 15``.  This helper prefers 20 because it divides 100,
    producing ``20, 20, 20, 20, 20``.  Later safety caps can still reduce the
    chosen size if this regular chunk would violate span limits.

    If no divisor exists at or above ``min_sticks``, try a nearby smaller
    divisor, but not so small that it explodes the number of chunks for
    prime-like sizes.  Fall back to ``min_sticks`` when no regular option is
    available.
    """
    if min_sticks >= total_sticks:
        return total_sticks

    for sticks in range(min_sticks, total_sticks):
        if total_sticks % sticks == 0:
            return sticks

    lower_bound = max(1, math.ceil(min_sticks / 2))
    for sticks in range(min_sticks - 1, lower_bound - 1, -1):
        if total_sticks % sticks == 0:
            return sticks

    return min_sticks


def _compute_num_chunks(
    chunking_info: ChunkingInfo,
    max_cores: int,
) -> int:
    """Compute number of chunks needed.

    Uses three estimates and picks the safe larger value:
    - num_from_span: per_core_span fits in 256MB
    - num_from_unsplit_span: span fits if work_division cannot split this dim
    - num_from_total: total fits in max_cores x 256MB
    """

    # Fallback path: no span-reduction dim found
    if chunking_info.selected_device_dim_size == 0:
        return max(
            1, math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))
        )
    num_from_span = math.ceil(chunking_info.per_core_span / MAX_SPAN_BYTES)
    num_from_unsplit_span = math.ceil(
        chunking_info.unsplit_per_core_span / MAX_SPAN_BYTES
    )
    num_from_total = math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))

    return max(num_from_span, num_from_unsplit_span, num_from_total)


def _cap_chunk_size_for_unsplit_span(
    chunking_info: ChunkingInfo,
    original_ftl: FixedTiledLayout,
    split_dim_full_size: int,
    chunk_size: int,
) -> tuple[int, int]:
    """Cap host chunk size for the no-extra-core-split case.

    The flow is:

    1. ``_chunk_op`` chooses a split dimension.
    2. ``_chunk_op`` chooses an initial, regular, stick-aligned ``chunk_size``
       for that dimension.
    3. This helper asks whether that chunk would still fit under the 256MB
       hardware span limit if work division does not split this same dimension
       any further.
    4. If the chunk is already safe, keep ``chunk_size`` unchanged.
    5. If the chunk is too large, reduce ``chunk_size`` to the largest host
       element count that is physically safe.
    6. Recompute and return ``num_chunks`` from the final ``chunk_size``.

    The selected split dim comes from host/output ranges, so ``chunk_size`` is
    measured in logical host elements.  The hardware span limit is measured in
    the physical device layout.  This helper converts the physical 256MB span
    limit back to host elements for the selected split dimension.

    This is called the unsplit-span cap because it assumes work division cannot
    spend additional cores on this same selected dimension after chunking.  If a
    chunk is safe under that pessimistic assumption, it is also safe when work
    division later does add a useful split.

    Two common mappings exist:
    - host dim and selected device dim have the same size, so one physical unit
      corresponds to one host element;
    - selected device dim is stick-packed, so one physical unit corresponds to
      one stick, i.e. ``stick_elems`` host elements.
    """
    if (
        chunking_info.selected_device_dim_size == 0
        or chunking_info.selected_device_span_stride_elems == 0
    ):
        return chunk_size, math.ceil(split_dim_full_size / chunk_size)

    stick_elems = chunking_info.stick_elems
    host_elems_per_selected_device_unit = (
        1
        if chunking_info.selected_device_dim_size == split_dim_full_size
        else stick_elems
    )
    # If we skipped small outer device dims before selecting this chunkable
    # dim, every unit of the selected dim still carries that outer physical
    # prefix.  Include it when converting the 256MB span limit into host elems.
    bytes_per_selected_device_unit = (
        chunking_info.outer_device_span_prefix_elems
        * chunking_info.selected_device_span_stride_elems
        * original_ftl.dtype.itemsize
    )
    max_safe_device_units_per_chunk = max(
        1,
        MAX_SPAN_BYTES // bytes_per_selected_device_unit,
    )
    max_safe_host_elems = (
        max_safe_device_units_per_chunk * host_elems_per_selected_device_unit
    )
    if chunk_size > max_safe_host_elems:
        chunk_size = max_safe_host_elems
    num_chunks = math.ceil(split_dim_full_size / chunk_size)
    return chunk_size, num_chunks


def _cap_reduction_chunk_size_for_input_span(
    reduction: Reduction,
    original_ranges: list[int],
    split_dim_idx: int,
    split_dim_full_size: int,
    stick_elems: int,
    chunk_size: int,
) -> tuple[int, int]:
    """Cap reduction chunk size so input reads fit within 256MB.

    The unsplit-span cap protects the chunk's selected output/write layout.  A
    reduction also has input-read pressure: each output element reads all
    ``reduction_ranges`` elements.  For one unit of the selected output split
    dim, the input read covers:

    ``product(other output ranges) * product(reduction_ranges)`` elements.

    This helper converts that read volume into a maximum safe host chunk size,
    keeps nonzero offsets stick-aligned, and returns the updated chunk count.
    """
    itemsize = reduction.dtype.itemsize
    reduction_elems = math.prod(int(r) for r in reduction.reduction_ranges)
    non_split_output_elems = math.prod(
        size for dim, size in enumerate(original_ranges) if dim != split_dim_idx
    )
    input_elems_per_split_dim_unit = max(1, non_split_output_elems * reduction_elems)
    max_chunk_size_for_input_span = max(
        1, MAX_SPAN_BYTES // (input_elems_per_split_dim_unit * itemsize)
    )
    if chunk_size > max_chunk_size_for_input_span:
        stick_aligned_max_chunk_size = max(
            1, (max_chunk_size_for_input_span // stick_elems) * stick_elems
        )
        chunk_size = min(chunk_size, stick_aligned_max_chunk_size)
    num_chunks = math.ceil(split_dim_full_size / chunk_size)
    return chunk_size, num_chunks


def _add_reduction_secondary_chunk_plans(
    *,
    op_name: str,
    reduction: Reduction,
    original_ranges: list[int],
    original_ftl: FixedTiledLayout,
    max_cores: int,
    primary_split_dim_idx: int,
    per_dim_chunk_plans: list[list[tuple[int, int]]],
    stick_elems: int,
) -> None:
    """Add secondary output-dim chunks needed for reduction input reads.

    ``primary_split_dim_idx`` is the main dimension chosen before entering
    ``_chunk_op``.  It may have been chosen from either output-span pressure or
    input-read pressure, but it is always an output dimension because reductions
    only chunk ``Reduction.ranges``.

    After the primary dimension is chunked, earlier output dims are still full
    size by default.  For scalar reductions that can still make the input read
    span exceed 256MB because the input keeps the original full-output stride
    and also carries ``reduction_ranges``.

    Before adding manual secondary chunks, estimate whether work_division can
    split that earlier dim across SENCORES.  If that split would already bring
    the input-read span under 256MB, leave the dim whole.  This avoids exploding
    the chunk grid from ``primary_chunks`` to
    ``primary_chunks * secondary_chunks`` when the scheduler can handle the
    secondary dimension naturally.

    Secondary chunking is limited to dims before the primary dim because the
    chunk grid is built around the primary span-reduction dim.  That is why the
    caller prefers the later dim as primary when input and output span analysis
    choose different dims.
    """
    itemsize = reduction.dtype.itemsize
    reduction_elems = math.prod(int(r) for r in reduction.reduction_ranges)

    for secondary_dim_idx in range(primary_split_dim_idx):
        secondary_dim_size = original_ranges[secondary_dim_idx]
        if secondary_dim_size <= 1:
            continue

        # ``stride[secondary_dim_idx]`` says how many output elements are
        # covered/skipped by one step in this dim.  For reductions, each of
        # those output elements also reads every element in ``reduction_ranges``.
        # Together, this is the input read span caused by one unit of this
        # secondary dim.
        input_read_elems_per_secondary_unit = (
            int(original_ftl.stride[secondary_dim_idx]) * reduction_elems
        )
        full_secondary_input_span = (
            secondary_dim_size * input_read_elems_per_secondary_unit * itemsize
        )
        if full_secondary_input_span <= MAX_SPAN_BYTES:
            continue

        best_secondary_core_split = _find_max_divisible_core_split(
            secondary_dim_size,
            max_cores,
        )
        per_core_secondary_input_span = (
            math.ceil(secondary_dim_size / best_secondary_core_split)
            * input_read_elems_per_secondary_unit
            * itemsize
        )
        if per_core_secondary_input_span <= MAX_SPAN_BYTES:
            logger.info(
                "Skipping reduction input-span secondary chunking %s: "
                "dim=%d full_size=%d full_input_span=%.2fMB "
                "estimated_work_division_split=%d estimated_per_core_span=%.2fMB",
                op_name,
                secondary_dim_idx,
                secondary_dim_size,
                full_secondary_input_span / (1024**2),
                best_secondary_core_split,
                per_core_secondary_input_span / (1024**2),
            )
            continue

        max_safe_secondary_chunk_size = max(
            1,
            MAX_SPAN_BYTES // (input_read_elems_per_secondary_unit * itemsize),
        )
        secondary_chunk_size = min(
            secondary_dim_size,
            max_safe_secondary_chunk_size,
        )

        # If this output dimension's row stride is not stick-aligned, arbitrary
        # chunk offsets on the dim become non-stick-aligned storage offsets. That
        # lowers to unsupported stick coordinates like ``Mod(d1, 64) + k``. Keep
        # every nonzero offset a multiple of a full stick so the within-stick
        # coordinate stays ``Mod(var,64)``.
        if int(original_ftl.stride[secondary_dim_idx]) % stick_elems != 0:
            aligned_chunk_size = (
                max_safe_secondary_chunk_size // stick_elems
            ) * stick_elems
            if aligned_chunk_size < 1:
                raise Unsupported(
                    f"Cannot chunk {op_name} (reduction:"
                    f"{reduction.reduction_type}) along dim "
                    f"{secondary_dim_idx}: max safe chunk size "
                    f"{max_safe_secondary_chunk_size} is smaller than stick "
                    f"size {stick_elems}, and stride "
                    f"{int(original_ftl.stride[secondary_dim_idx])} is not "
                    "stick-aligned."
                )
            secondary_chunk_size = min(secondary_dim_size, aligned_chunk_size)

        secondary_dim_chunk_plan: list[tuple[int, int]] = []
        for offset in range(0, secondary_dim_size, secondary_chunk_size):
            secondary_dim_chunk_plan.append(
                (
                    offset,
                    min(secondary_chunk_size, secondary_dim_size - offset),
                )
            )
        per_dim_chunk_plans[secondary_dim_idx] = secondary_dim_chunk_plan

        logger.info(
            "Reduction input-span secondary chunking %s: dim=%d full_size=%d "
            "chunk_size=%d num_chunks=%d full_input_span=%.2fMB",
            op_name,
            secondary_dim_idx,
            secondary_dim_size,
            secondary_chunk_size,
            len(secondary_dim_chunk_plan),
            full_secondary_input_span / (1024**2),
        )


def _full_destination_write_span_failure(
    layout: FixedTiledLayout,
    max_cores: int,
) -> str | None:
    """Return a reason if a full-layout mutation write cannot fit.

    Chunking shrinks loop ranges, but chunk 0 and Scatter mutation chunks still
    write through the original destination layout.  work_division currently
    checks that full physical layout, not the smaller Scatter slice.  This guard
    models the outermost write-coordinate span that work_division must satisfy
    before inner dimensions can help.
    """
    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize

    # This is intentionally conservative.  The failing mutation-layout path is
    # the full-destination write span, and observed work_division behavior for
    # these scatter writes may reserve/distribute cores so the outer write coord
    # only gets a 4-way split even when SENCORES is larger.  Using all
    # ``max_cores`` here would miss the exact failure we are trying to report
    # early.  If work_division becomes Scatter-slice-aware, this guard can go
    # away.
    max_full_write_coord_split = min(max_cores, 4)

    for device_dim, dim_size in enumerate(device_size[:-1]):
        if dim_size <= 1:
            continue

        best_split = _find_max_divisible_core_split(
            dim_size,
            max_full_write_coord_split,
        )
        per_core_span = (
            math.ceil(dim_size / best_split)
            * math.prod(device_size[device_dim + 1 :])
            * itemsize
        )
        if per_core_span <= MAX_SPAN_BYTES:
            return None

        return (
            f"full destination write layout still has per-core span "
            f"{per_core_span} bytes after best split d{device_dim}:{best_split}; "
            f"limit is {MAX_SPAN_BYTES} bytes. device_size={device_size}"
        )

    total_bytes = math.prod(device_size) * itemsize
    if total_bytes > MAX_SPAN_BYTES * max_cores:
        return (
            f"full destination write layout has total size {total_bytes} bytes; "
            f"limit is {MAX_SPAN_BYTES * max_cores} bytes for SENCORES={max_cores}. "
            f"device_size={device_size}"
        )
    return None


def _chunk_bytes_fit(
    layout: FixedTiledLayout,
    max_cores: int,
) -> bool:
    """Return True if a planned chunk is already small enough.

    Chunk planning is intentionally done once: pick chunk sizes, build the full
    chunk grid, then validate every generated chunk.  If this returns False, the
    planned chunk would still trigger ``_needs_chunking`` and we fail with a
    clear ``Unsupported`` instead of recursively chunking a chunk.
    """
    layout_span_reduction_dim_info = _find_outermost_span_reduction_dim(
        layout, max_cores
    )
    return (
        _needs_chunking(
            layout,
            max_cores,
            layout_span_reduction_dim_info,
        )
        is None
    )


def _needs_chunking(
    layout: FixedTiledLayout,
    max_cores: int,
    span_reduction_dim_info: SpanDimInfo | None,
    *,
    op_name: str | None = None,
    trigger: str = "output_span",
) -> ChunkingInfo | None:
    """Return chunking info if this layout exceeds span/total limits.

    Span checks use ``core_split_estimate`` because work division can often
    split the same span-reduction dimension.  Reduction input span checks are
    translated back to output ranges only when that selected input dimension is
    controlled by an output index.

    Three cases trigger chunking:

    1. ``per_core_span > 256 MB`` on the span-reduction dim.

    2. ``unsplit_per_core_span > 256 MB`` on the same dim, because
       work division may have already spent the core budget on another tensor.

    3. ``total_bytes > 256 MB * max_cores``.

    Falls back to total_bytes threshold if no span-reduction dim found.
    """
    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    total_bytes = math.prod(device_size) * itemsize
    stick_elems = layout.device_layout.elems_per_stick()

    if span_reduction_dim_info is None:
        if total_bytes > MAX_SPAN_BYTES * max_cores:
            host_size = [int(s) for s in layout.size]
            fallback_selected_host_dim = max(
                range(len(host_size)), key=lambda d: host_size[d]
            )
            return ChunkingInfo(
                total_bytes=total_bytes,
                per_core_span=total_bytes,
                unsplit_per_core_span=total_bytes,
                core_split_estimate=1,
                selected_device_dim_size=0,
                selected_device_span_stride_elems=0,
                selected_host_dim=fallback_selected_host_dim,
                stick_elems=stick_elems,
                outer_device_span_prefix_elems=1,
                reason=(
                    "no device dimension could be mapped to a splittable "
                    "host dimension via stride_map; using largest host dim "
                    "as fallback"
                ),
            )
        return None

    selected_host_dim = span_reduction_dim_info.selected_host_dim
    selected_device_dim_size = span_reduction_dim_info.selected_device_dim_size
    selected_device_span_stride_elems = (
        span_reduction_dim_info.selected_device_span_stride_elems
    )
    core_split_estimate = span_reduction_dim_info.core_split_estimate
    outer_device_span_prefix_elems = (
        span_reduction_dim_info.outer_device_span_prefix_elems
    )

    per_core_span = (
        outer_device_span_prefix_elems
        * math.ceil(selected_device_dim_size / core_split_estimate)
        * selected_device_span_stride_elems
        * itemsize
    )
    unsplit_per_core_span = (
        outer_device_span_prefix_elems
        * selected_device_dim_size
        * selected_device_span_stride_elems
        * itemsize
    )

    needs_chunk_for_span = per_core_span > MAX_SPAN_BYTES
    needs_chunk_for_unsplit_span = unsplit_per_core_span > MAX_SPAN_BYTES
    needs_chunk_for_total = total_bytes > MAX_SPAN_BYTES * max_cores

    if needs_chunk_for_span or needs_chunk_for_unsplit_span or needs_chunk_for_total:
        logger.info(
            "[chunk_large_tensors] trigger=%s op=%s "
            "selected_host_dim=%d selected_device_dim_size=%d selected_device_span_stride_elems=%d "
            "outer_device_span_prefix_elems=%d skipped_outer_device_dims=%s "
            "core_split_estimate=%d per_core_span=%.2f MB unsplit_per_core_span=%.2f MB "
            "total=%.2f GB (shape=%s, dtype=%s, device_size=%s, "
            "span_limit=%.2f MB, total_limit=%.2f GB)",
            trigger,
            op_name or "<unknown>",
            selected_host_dim,
            selected_device_dim_size,
            selected_device_span_stride_elems,
            outer_device_span_prefix_elems,
            span_reduction_dim_info.skipped_outer_device_dims,
            core_split_estimate,
            per_core_span / (1024**2),
            unsplit_per_core_span / (1024**2),
            total_bytes / (1024**3),
            list(layout.size),
            layout.dtype,
            device_size,
            MAX_SPAN_BYTES / (1024**2),
            (MAX_SPAN_BYTES * max_cores) / (1024**3),
        )
        return ChunkingInfo(
            total_bytes=total_bytes,
            per_core_span=per_core_span,
            unsplit_per_core_span=unsplit_per_core_span,
            core_split_estimate=core_split_estimate,
            selected_device_dim_size=selected_device_dim_size,
            selected_device_span_stride_elems=selected_device_span_stride_elems,
            selected_host_dim=selected_host_dim,
            stick_elems=stick_elems,
            outer_device_span_prefix_elems=outer_device_span_prefix_elems,
            reason=(
                "skipped unchunkable outer device dims "
                f"{span_reduction_dim_info.skipped_outer_device_dims} with "
                f"outer prefix {outer_device_span_prefix_elems}"
                if span_reduction_dim_info.skipped_outer_device_dims
                else None
            ),
        )
    return None


def _output_dim_for_input_selected_host_dim(
    op: ComputedBuffer,
    input_info: TensorInfo,
    input_selected_host_dim: int,
) -> int | None:
    """Map an input host dim to an output dim when it uses output symbols.

    Reduction input dependencies contain both output-index symbols and reduction-index
    symbols.  We only trust input-span work-division hints when the input host
    coordinate for the span-reduction dim uses the same symbol set as an output
    coordinate.  If it maps to a reduction symbol, chunking output ranges cannot
    force that split.
    """
    rw = op.get_read_writes()
    write_dep = next(iter(rw.writes), None)
    if not isinstance(write_dep, MemoryDep):
        return None

    try:
        input_coords = host_coordinates(input_info.layout, input_info.dep)
        output_coords = host_coordinates(cast(FixedTiledLayout, op.layout), write_dep)
    except Exception as exc:
        logger.debug(
            "Could not map input host dim for %s through host coordinates: %s",
            op.get_name(),
            exc,
        )
        return None

    if input_selected_host_dim >= len(input_coords):
        return None

    input_symbols = input_coords[input_selected_host_dim].free_symbols
    if not input_symbols or not input_symbols <= write_dep.index.free_symbols:
        return None

    for output_dim, output_coord in enumerate(output_coords):
        if input_symbols == output_coord.free_symbols:
            return output_dim
    return None


def _reduction_input_span_chunking(
    op: ComputedBuffer,
    max_cores: int,
    output_span_reduction_dim_info: SpanDimInfo | None,
) -> ChunkingInfo | None:
    """Return chunking info when a reduction input span exceeds limits.

    Output-span checks can miss reductions whose result is small but whose
    input tensor is physically large after Spyre tiling/padding.  This path uses
    the same span limit check as output tensors, then translates the selected
    input dimension to an output chunk dimension when possible.
    """
    reduction = cast(Reduction, op.data)
    output_layout = cast(FixedTiledLayout, op.layout)
    output_ranges = [int(r) for r in reduction.ranges]
    if not output_ranges:
        return None

    input_infos = [
        info
        for info in _collect_tensor_infos(op)
        if info.dep.name != op.get_name() and info.device_bytes > 0
    ]
    largest_input = max(input_infos, key=lambda info: info.device_bytes, default=None)
    if largest_input is None:
        logger.debug(
            "[chunk_large_tensors] trigger=reduction_input_span skipped op=%s: "
            "no FixedTiledLayout input tensor found",
            op.get_name(),
        )
        return None

    input_span_reduction_dim_info = _find_outermost_span_reduction_dim(
        largest_input.layout, max_cores
    )
    input_span_chunking_info = _needs_chunking(
        largest_input.layout,
        max_cores,
        input_span_reduction_dim_info,
        op_name=f"{op.get_name()}:{largest_input.dep.name}",
        trigger="reduction_input_span",
    )
    if input_span_chunking_info is None:
        return None

    candidates = [(size, dim) for dim, size in enumerate(output_ranges) if size > 1]
    if not candidates:
        raise Unsupported(
            f"Cannot chunk {op.get_name()} (reduction:{reduction.reduction_type}) "
            f"for input tensor {largest_input.dep.name}: no splittable output "
            "dimension is available."
        )

    stick_elems = output_layout.device_layout.elems_per_stick()
    itemsize = largest_input.layout.dtype.itemsize
    input_span_exceeds_limit = input_span_chunking_info.per_core_span > MAX_SPAN_BYTES
    input_total_exceeds_limit = (
        input_span_chunking_info.total_bytes > MAX_SPAN_BYTES * max_cores
    )
    input_dim_maps_to_output = False
    selected_host_dim = None

    if input_span_reduction_dim_info is not None:
        input_selected_host_dim = input_span_reduction_dim_info.selected_host_dim
        output_selected_host_dim = _output_dim_for_input_selected_host_dim(
            op, largest_input, input_selected_host_dim
        )
        if (
            output_selected_host_dim is not None
            and output_ranges[output_selected_host_dim] > 1
        ):
            selected_host_dim = output_selected_host_dim
            input_dim_maps_to_output = True

    if selected_host_dim is None:
        if input_span_exceeds_limit:
            raise Unsupported(
                f"Cannot chunk {op.get_name()} (reduction:{reduction.reduction_type}) "
                f"below the hardware memory span limit for input tensor "
                f"{largest_input.dep.name}: the input tensor needs span reduction "
                "on a dimension that does not map to an output index. "
                "Chunking output ranges cannot split reduction_ranges."
            )

        if output_span_reduction_dim_info is not None:
            selected_host_dim = output_span_reduction_dim_info.selected_host_dim
            dim_size = output_span_reduction_dim_info.selected_device_dim_size
            core_split_estimate = output_span_reduction_dim_info.core_split_estimate
        else:
            dim_size, selected_host_dim = max(candidates)
            core_split_estimate = 1
        mapping_note = (
            "input total exceeds limit; using output span-reduction dim"
            if input_total_exceeds_limit
            else "input span requires output range chunking"
        )
    else:
        dim_size = output_ranges[selected_host_dim]
        core_split_estimate = input_span_chunking_info.core_split_estimate
        mapping_note = (
            "input span-reduction dim maps to output dim; using input span "
            "core split estimate"
        )

    source = largest_input.dep.name
    input_shape = list(largest_input.layout.size)
    input_dtype = largest_input.layout.dtype
    input_device_size = list(largest_input.layout.device_layout.device_size)
    logger.info(
        "[chunk_large_tensors] trigger=reduction_input_span op=%s input=%s "
        "selected_host_dim=%d dim_size=%d core_split_estimate=%d "
        "per_core_span=%.2f MB input_total=%.2f GB total_limit=%.2f GB "
        "input_dim_maps_to_output=%s reason='%s' "
        "(input_shape=%s, input_dtype=%s, input_device_size=%s, "
        "output_shape=%s, reduction_ranges=%s)",
        op.get_name(),
        source,
        selected_host_dim,
        dim_size,
        core_split_estimate,
        input_span_chunking_info.per_core_span / (1024**2),
        input_span_chunking_info.total_bytes / (1024**3),
        (MAX_SPAN_BYTES * max_cores) / (1024**3),
        input_dim_maps_to_output,
        mapping_note,
        input_shape,
        input_dtype,
        input_device_size,
        output_ranges,
        list(reduction.reduction_ranges),
    )
    return ChunkingInfo(
        total_bytes=input_span_chunking_info.total_bytes,
        per_core_span=input_span_chunking_info.per_core_span,
        unsplit_per_core_span=input_span_chunking_info.unsplit_per_core_span,
        core_split_estimate=core_split_estimate,
        selected_device_dim_size=dim_size,
        selected_device_span_stride_elems=max(
            1,
            input_span_chunking_info.total_bytes // max(dim_size * itemsize, 1),
        ),
        selected_host_dim=selected_host_dim,
        stick_elems=stick_elems,
        outer_device_span_prefix_elems=1,
        reason=mapping_note,
    )


def _make_layout_for_size(
    original_ftl: FixedTiledLayout,
    host_size: list[int],
) -> FixedTiledLayout:
    """Build a chunk layout while preserving the original device rank.

    Reconstructing ``SpyreTensorLayout`` from ``host_size`` alone can pick a
    different physical rank from the original tensor.  Reduction chunk temps are
    later scatter-copied into the original output, and DDL expects compatible
    device-dim structures for that mapping.
    """
    from torch_spyre._C import SpyreTensorLayout

    # Keep the original host strides because the preserved device stride_map
    # still describes the original physical row geometry.  Recomputing compact
    # strides for a smaller chunk can create fractional coordinate expressions
    # when later passes map between host and device coordinates.
    host_stride = [int(s) for s in original_ftl.stride]

    original_stl = original_ftl.device_layout
    device_size = [int(s) for s in original_stl.device_size]
    stride_map = [int(s) for s in original_stl.stride_map]
    original_host_size = [int(s) for s in original_ftl.size]
    original_host_stride = [int(s) for s in original_ftl.stride]
    stick_elems = original_stl.elems_per_stick()

    for device_dim, sm in enumerate(stride_map[:-1]):
        if sm <= 0:
            continue

        for selected_host_dim, stride in enumerate(original_host_stride):
            if (
                original_host_size[selected_host_dim] <= 1
                and host_size[selected_host_dim] <= 1
            ):
                continue
            if sm == stride:
                device_size[device_dim] = int(host_size[selected_host_dim])
                break
            if sm == stride * stick_elems:
                device_size[device_dim] = math.ceil(
                    int(host_size[selected_host_dim]) / stick_elems
                )
                break

    if device_size:
        device_size[-1] = int(original_stl.device_size[-1])

    stl = SpyreTensorLayout(device_size, stride_map, original_stl.device_dtype)
    return FixedTiledLayout(
        original_ftl.device,
        original_ftl.dtype,
        host_size,
        host_stride,
        stl,
    )


def _make_chunk_fn(orig_fn, dim: int, offset: int):
    """Return a pointwise ``inner_fn`` shifted on the split dim."""

    def inner_fn(index):
        idx = list(index)
        idx[dim] = idx[dim] + offset
        return orig_fn(idx)

    return inner_fn


def _make_reduction_chunk_fn(orig_fn, offsets: list[int]):
    """Return a reduction ``inner_fn`` shifted only on output indices."""

    def inner_fn(index, rindex):
        idx = list(index)
        for dim, offset in enumerate(offsets):
            if offset:
                idx[dim] = idx[dim] + offset
        return orig_fn(idx, rindex)

    return inner_fn


def _make_reduction_data(
    original: Reduction,
    inner_fn,
    ranges: list,
) -> Reduction:
    """Clone reduction metadata while replacing ``inner_fn`` and ranges."""
    kwargs = {
        "device": original.device,
        "dtype": original.dtype,
        "inner_fn": inner_fn,
        "ranges": ranges,
        "reduction_ranges": list(original.reduction_ranges),
        "reduction_type": original.reduction_type,
        "src_dtype": original.src_dtype,
        "reduction_hint": original.reduction_hint,
    }
    if hasattr(original, "op_info"):
        kwargs["op_info"] = original.op_info
    return type(original)(**kwargs)


def _make_pointwise_data(
    original: Pointwise,
    inner_fn,
    ranges: list,
) -> Pointwise:
    """Clone pointwise metadata while replacing ``inner_fn`` and ranges."""
    return type(original)(
        device=original.device,
        dtype=original.dtype,
        inner_fn=inner_fn,
        ranges=ranges,
    )


def _replace_computed_buffer_data(
    op: ComputedBuffer,
    new_data,
    operations: list[Operation],
    op_index: int,
) -> ComputedBuffer:
    """Replace ``op`` with a fresh ComputedBuffer using ``new_data``.

    ``ComputedBuffer.get_default_sizes_body`` is cached on the buffer instance.
    Chunk 0 changes the loop ranges, so reconstruct the buffer instead of
    mutating ``op.data`` in place and then update graph lookup tables.
    """
    new_op = ComputedBuffer(
        name=op.get_name(),
        layout=op.layout,
        data=new_data,
        _split_size=op._split_size,
        _original_inner_fn=op._original_inner_fn,
        _original_ranges=op._original_ranges,
        _original_reduction_ranges=op._original_reduction_ranges,
    )
    new_op.operation_name = op.operation_name
    new_op.origins = op.origins
    if hasattr(op, "origin_node"):
        new_op.origin_node = op.origin_node
    copy_op_metadata(op, new_op)

    if operations[op_index] is op:
        operations[op_index] = new_op
    else:
        operations[operations.index(op)] = new_op
    V.graph.name_to_buffer[new_op.get_name()] = new_op
    ComputedBuffer.get_default_sizes_body.clear_cache(new_op)
    return new_op


def _is_supported_scalar_reduction(data: Reduction) -> bool:
    """Return True for scalar reductions that can be chunked by output dims."""
    reduction_type = str(data.reduction_type)
    if reduction_type in UNSUPPORTED_CHUNKED_REDUCTIONS:
        return False
    if reduction_type in SUPPORTED_SCALAR_REDUCTIONS:
        return True

    logger.debug(
        "Skipping reduction chunking for unsupported reduction_type=%s",
        reduction_type,
    )
    return False


def _make_output_indexer(offset: int, split_dim: int):
    """Return a scatter output indexer shifted by *offset* on *split_dim*."""

    def output_indexer(index):
        out = list(index)
        out[split_dim] = out[split_dim] + offset
        return out

    return output_indexer


def _make_multi_output_indexer(offsets: list[int]):
    """Return a scatter output indexer shifted by offsets on each dim."""

    def output_indexer(index):
        out = list(index)
        for dim, offset in enumerate(offsets):
            if offset:
                out[dim] = out[dim] + offset
        return out

    return output_indexer


def _register_and_insert(
    buf: ComputedBuffer,
    op: ComputedBuffer,
    operations: list[Operation],
    insert_pos: int,
) -> int:
    """Register *buf* in the graph and insert it at *insert_pos*.

    ``V.graph.register_operation`` appends to the same ``operations``
    list, so the duplicate is removed before the positioned insert.

    Returns the next insert position.
    """
    buf.name = V.graph.register_buffer(buf)
    V.graph.register_operation(buf)
    buf.origins = op.origins
    if buf in operations:
        operations.remove(buf)
    operations.insert(insert_pos, buf)
    return insert_pos + 1


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------


def _chunk_op(
    op: ComputedBuffer,
    max_cores: int,
    operations: list[Operation],
    op_index: int,
    chunking_info: ChunkingInfo,
    num_chunks: int,
    original_ftl: FixedTiledLayout,
) -> int:
    """Split *op* into memory-safe chunks along the span-reduction dim.

    Chunk 0 is the original op shrunk in-place (ranges only; layout
    stays full-size so the scheduler finds it by name).  Pointwise chunks
    1..N-1 are direct ``Scatter`` mutation buffers.  Reduction chunks
    1..N-1 first compute a reduced temporary chunk, then scatter-copy the
    temporary into the corresponding region of the original output.

    Reduction chunks may form a small grid.  The primary split dimension fixes
    the output span.  Earlier output dimensions may also be chunked because
    input reads keep the original full-tensor row stride even when the output
    column range is smaller.
    """
    data = cast(Reduction | Pointwise, op.data)
    is_reduction = isinstance(data, Reduction)
    reduction = cast(Reduction, data) if is_reduction else None
    pointwise = cast(Pointwise, data) if not is_reduction else None
    original_ranges = [int(r) for r in data.ranges]
    original_inner_fn = data.inner_fn
    primary_split_dim_idx = chunking_info.selected_host_dim
    primary_split_dim_full_size = int(original_ranges[primary_split_dim_idx])
    stick_elems = chunking_info.stick_elems

    # The caller already selected num_chunks while choosing between output-span
    # and reduction-input-span chunking plans.
    # -- Step 1: stick-aligned chunk size on the primary dim --
    total_sticks = math.ceil(primary_split_dim_full_size / stick_elems)
    min_sticks_per_chunk = math.ceil(total_sticks / num_chunks)
    sticks_per_chunk = _choose_sticks_per_chunk(total_sticks, min_sticks_per_chunk)
    chunk_size = sticks_per_chunk * stick_elems

    # The regular-geometry preference above may choose a larger divisor and
    # reduce the effective chunk count.  Cap it so each generated chunk is safe
    # even if work_division cannot spend cores on this dimension.
    chunk_size, num_chunks = _cap_chunk_size_for_unsplit_span(
        chunking_info,
        original_ftl,
        primary_split_dim_full_size,
        chunk_size,
    )

    if reduction is not None:
        chunk_size, num_chunks = _cap_reduction_chunk_size_for_input_span(
            reduction,
            original_ranges,
            primary_split_dim_idx,
            primary_split_dim_full_size,
            stick_elems,
            chunk_size,
        )
    primary_split_dim_chunk_plan: list[tuple[int, int]] = []
    for chunk_idx in range(num_chunks):
        chunk_offset = chunk_idx * chunk_size
        remaining_elems = max(0, primary_split_dim_full_size - chunk_offset)
        if remaining_elems == 0:
            continue
        # Keep logical ranges within the original tensor size.  The chunk
        # layout may still allocate a full final stick via _make_layout_for_size,
        # but op.data.ranges and scatter ranges must not iterate over padded
        # logical elements.
        this_chunk_size = min(remaining_elems, chunk_size)
        primary_split_dim_chunk_plan.append((chunk_offset, this_chunk_size))

    per_dim_chunk_plans: list[list[tuple[int, int]]] = [
        [(0, size)] for size in original_ranges
    ]
    per_dim_chunk_plans[primary_split_dim_idx] = primary_split_dim_chunk_plan

    if reduction is not None:
        _add_reduction_secondary_chunk_plans(
            op_name=op.get_name(),
            reduction=reduction,
            original_ranges=original_ranges,
            original_ftl=original_ftl,
            max_cores=max_cores,
            primary_split_dim_idx=primary_split_dim_idx,
            per_dim_chunk_plans=per_dim_chunk_plans,
            stick_elems=stick_elems,
        )

    # ``per_dim_chunk_plans`` is one independent plan per output dimension.
    # Convert it into concrete output tiles by taking the cartesian product:
    # every dim-0 chunk with every dim-1 chunk, and so on.  Each entry in
    # ``chunk_grid`` is one real chunk operation with offsets/sizes for all
    # output dims.
    chunk_grid = []
    for entries in itertools.product(*per_dim_chunk_plans):
        offsets = [offset for offset, _ in entries]
        sizes = [size for _, size in entries]
        chunk_grid.append((offsets, sizes))

    failure_reason = f" Reason: {chunking_info.reason}." if chunking_info.reason else ""

    # If chunking skipped a small outer physical dimension and split an inner
    # dimension instead, the compute ranges may be safe while the write-back
    # still goes through the full destination layout.  work_division is not
    # Scatter-slice-aware today, so fail here with the real boundary instead of
    # generating many chunks that fail later during scheduling.
    if len(chunk_grid) > 1:
        write_span_failure = _full_destination_write_span_failure(
            original_ftl,
            max_cores,
        )
        if write_span_failure is not None:
            op_kind = (
                f"reduction:{reduction.reduction_type}"
                if reduction is not None
                else "pointwise"
            )
            raise Unsupported(
                f"Cannot chunk {op.get_name()} ({op_kind}) below the "
                "hardware memory span limit: generated chunks have smaller "
                "logical ranges, but chunk 0 and Scatter mutation chunks still "
                "write through the full output layout. work_division currently "
                "checks that full destination layout rather than the Scatter "
                "slice range/output_indexer. "
                f"{write_span_failure}{failure_reason}"
            )

    # Final guardrail: rebuild the physical layout for each planned tile and
    # verify it no longer needs chunking.  This catches cases where the primary
    # and secondary plans still cannot satisfy span/total limits.  We raise
    # instead of recursively chunking again so the pass remains predictable.
    for offsets, sizes in chunk_grid:
        chunk_layout = _make_layout_for_size(original_ftl, sizes)
        if not _chunk_bytes_fit(chunk_layout, max_cores):
            op_kind = (
                f"reduction:{reduction.reduction_type}"
                if reduction is not None
                else "pointwise"
            )
            raise Unsupported(
                f"Cannot chunk {op.get_name()} ({op_kind}) below the "
                f"hardware memory span limit "
                f"({MAX_SPAN_BYTES // (1024 * 1024)}MB per core): "
                f"SENCORES={max_cores}, offsets={offsets}, sizes={sizes}, "
                f"total={chunking_info.total_bytes} bytes, "
                f"per_core_span={chunking_info.per_core_span} bytes."
                f"{failure_reason}"
            )

    logger.info(
        "[chunk_large_tensors] apply op=%s kind=%s split_dim=%d "
        "full_size=%d split_dim_chunk_size=%d num_chunks=%d "
        "total=%.2f GB per_core_span=%.2f MB core_split_estimate=%d "
        "per_dim_chunk_plans=%s",
        op.get_name(),
        "reduction" if is_reduction else "pointwise",
        primary_split_dim_idx,
        primary_split_dim_full_size,
        chunk_size,
        len(chunk_grid),
        chunking_info.total_bytes / (1024**3),
        chunking_info.per_core_span / (1024**2),
        chunking_info.core_split_estimate,
        per_dim_chunk_plans,
    )

    # -- Chunk 0: replace original op with a shrunk fresh buffer --
    chunk0_offsets, chunk0_sizes = chunk_grid[0]
    assert all(offset == 0 for offset in chunk0_offsets)
    chunk0_data: Reduction | Pointwise
    if reduction is not None:
        chunk0_data = _make_reduction_data(
            reduction,
            original_inner_fn,
            list(chunk0_sizes),
        )
    else:
        assert pointwise is not None
        chunk0_data = _make_pointwise_data(
            pointwise,
            original_inner_fn,
            list(chunk0_sizes),
        )
    op = _replace_computed_buffer_data(op, chunk0_data, operations, op_index)

    insert_pos = op_index + 1
    n_inserted = 0

    # -- Chunks 1..N-1 --
    for offsets, sizes in chunk_grid[1:]:
        chunk_ranges = list(sizes)

        if reduction is not None:
            reduction_data = _make_reduction_data(
                reduction,
                _make_reduction_chunk_fn(original_inner_fn, offsets),
                chunk_ranges,
            )
            reduction_buf = ComputedBuffer(
                name=None,
                layout=_make_layout_for_size(original_ftl, chunk_ranges),
                data=reduction_data,
            )
            insert_pos = _register_and_insert(reduction_buf, op, operations, insert_pos)
            n_inserted += 1

            mutation_data = Scatter(
                device=reduction.device,
                dtype=reduction.dtype,
                inner_fn=reduction_buf.make_loader(),
                ranges=chunk_ranges,
                output_indexer=_make_multi_output_indexer(offsets),
            )
        else:
            assert pointwise is not None
            mutation_data = Scatter(
                device=pointwise.device,
                dtype=pointwise.dtype,
                inner_fn=_make_chunk_fn(
                    original_inner_fn,
                    primary_split_dim_idx,
                    offsets[primary_split_dim_idx],
                ),
                ranges=chunk_ranges,
                output_indexer=_make_output_indexer(
                    offsets[primary_split_dim_idx], primary_split_dim_idx
                ),
            )

        mutation_buf = ComputedBuffer(
            name=None,
            layout=MutationLayoutSHOULDREMOVE(op),
            data=mutation_data,
        )
        insert_pos = _register_and_insert(mutation_buf, op, operations, insert_pos)
        n_inserted += 1
    return n_inserted


def chunk_large_tensors(graph: GraphLowering) -> None:
    """Split oversized pointwise and scalar-reduction ops.

    Must run **after** ``propagate_spyre_tensor_layouts`` /
    ``insert_restickify`` and **before** ``span_reduction``.

    Note: ``ir.Pointwise`` is broader than ``torch.Tag.pointwise``.
    ``inner_fn`` can in theory access non-corresponding input indices,
    making chunking unsafe.  Reduction chunking is limited to scalar
    reductions and only splits output ranges, never ``reduction_ranges``.

    TODO: Use OpsHandler to verify output[i] only uses input[i].
    """
    operations = graph.operations
    max_cores = config.sencores
    i = 0
    while i < len(operations):
        op = operations[i]

        if not (
            isinstance(op, ComputedBuffer)
            and isinstance(op.layout, FixedTiledLayout)
            and isinstance(op.data, (Pointwise, Reduction))
            and len(op.data.ranges) >= 1
        ):
            i += 1
            continue

        if isinstance(op.data, Reduction) and not _is_supported_scalar_reduction(
            op.data
        ):
            i += 1
            continue
        output_span_reduction_dim_info = _find_outermost_span_reduction_dim(
            op.layout, max_cores
        )
        output_span_chunking_info = _needs_chunking(
            op.layout,
            max_cores,
            output_span_reduction_dim_info,
            op_name=op.get_name(),
        )
        final_span_chunking_info = output_span_chunking_info
        final_num_chunks = (
            _compute_num_chunks(output_span_chunking_info, max_cores)
            if output_span_chunking_info is not None
            else None
        )
        if isinstance(op.data, Reduction):
            input_span_chunking_info = _reduction_input_span_chunking(
                op, max_cores, output_span_reduction_dim_info
            )
            if final_span_chunking_info is None:
                final_span_chunking_info = input_span_chunking_info
                final_num_chunks = (
                    _compute_num_chunks(input_span_chunking_info, max_cores)
                    if input_span_chunking_info is not None
                    else None
                )
            elif input_span_chunking_info is not None:
                assert output_span_chunking_info is not None
                assert final_num_chunks is not None
                output_chunks = final_num_chunks
                input_chunks = _compute_num_chunks(input_span_chunking_info, max_cores)
                # _chunk_op can add secondary plans only for dimensions before
                # the primary split dim.  Prefer the later output dimension when
                # both output and input spans need chunking; otherwise input
                # chunking on an earlier dim can leave the output span full-size.
                input_dim = input_span_chunking_info.selected_host_dim
                output_dim = output_span_chunking_info.selected_host_dim
                if input_dim > output_dim or (
                    input_dim == output_dim and input_chunks > output_chunks
                ):
                    logger.info(
                        "[chunk_large_tensors] trigger=reduction_input_span "
                        "overrides output chunking for op=%s: "
                        "input_dim=%d output_dim=%d input_chunks=%d "
                        "output_chunks=%d",
                        op.get_name(),
                        input_dim,
                        output_dim,
                        input_chunks,
                        output_chunks,
                    )
                    final_span_chunking_info = input_span_chunking_info
                    final_num_chunks = input_chunks
        if final_span_chunking_info is not None:
            assert final_num_chunks is not None
            n_inserted = _chunk_op(
                op,
                max_cores,
                operations,
                i,
                final_span_chunking_info,
                final_num_chunks,
                op.layout,
            )
            i += n_inserted
        i += 1
