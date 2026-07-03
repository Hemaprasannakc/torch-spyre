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

"""Span-overflow tile planning for oversized output-range ops."""

from __future__ import annotations

import itertools
import math

from dataclasses import dataclass
from typing import Callable

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise, Reduction
from torch._inductor.virtualized import V

from .constants import BATCH_MATMUL_OP
from .errors import Unsupported
from .ir import FixedTiledLayout, _resize_device_layout
from .logging_utils import get_inductor_logger
from .pass_utils import (
    _fixed_read_layout,
    compute_coordinates,
    concretize_index,
    device_coordinates,
    indirect_info_from_op,
    op_out_coords,
)
from .work_division import MAX_SPAN_BYTES, core_split


logger = get_inductor_logger("span_overflow_hint_analysis")


@dataclass(frozen=True)
class ChunkingInfo:
    """Physical span facts for one op before coarse tiling."""

    total_bytes: int
    per_core_span: int
    core_split_estimate: int
    # 0 means no device dim mapped via stride_map; use the fallback host dim.
    selected_device_dim_size: int
    selected_device_span_stride_elems: int
    selected_host_dim: int
    stick_elems: int
    reason: str | None = None


@dataclass(frozen=True)
class InputSpanInfo:
    """Input span facts that can be reduced by output-range tiling."""

    chunking_info: ChunkingInfo
    dep_name: str
    controlling_symbol: sympy.Symbol


@dataclass(frozen=True)
class SpanOverflowTileLevel:
    """One output-range coarse-tiling level requested by span analysis."""

    selected_host_dim: int
    split_count: int
    is_reduction: bool = False


@dataclass(frozen=True)
class SpanOverflowTilePlan:
    """Coarse-tiling request produced by span-overflow analysis."""

    levels: tuple[SpanOverflowTileLevel, ...]
    chunking_infos: tuple[ChunkingInfo, ...]
    reason: str | None = None

    @property
    def selected_host_dim(self) -> int:
        """Compatibility accessor for historical single-level callers."""
        return self.levels[0].selected_host_dim

    @property
    def split_count(self) -> int:
        """Compatibility accessor for historical single-level callers."""
        return self.levels[0].split_count

    @property
    def is_reduction(self) -> bool:
        """Compatibility accessor for historical single-level callers."""
        return self.levels[0].is_reduction

    @property
    def chunking_info(self) -> ChunkingInfo:
        """Compatibility accessor for historical single-candidate callers."""
        return self.chunking_infos[0]


@dataclass(frozen=True)
class SpanOverflowCandidate:
    """A span overflow that can potentially be fixed by output-range tiling."""

    chunking_info: ChunkingInfo
    source: str


# Keep the search bounded; most real cases should need one or two tiled dims.
_MAX_TILE_DIMS = 3
_MAX_TILE_COMBOS = 512
_MAX_SPLITS_PER_DIM = 16


@dataclass(frozen=True)
class SpanDimInfo:
    """Mapping from the first span-controlling device dim to a host dim."""

    selected_host_dim: int
    selected_device_dim_size: int
    selected_device_span_stride_elems: int
    core_split_estimate: int
    skipped_outer_device_dims: tuple[int, ...] = ()


def _layout_has_static_span_metadata(layout: FixedTiledLayout) -> bool:
    """Return True when span planning can use concrete layout metadata."""
    try:
        for values in (
            layout.size,
            layout.stride,
            layout.device_layout.device_size,
            layout.device_layout.stride_map,
        ):
            for value in values:
                int(value)
        int(layout.device_layout.elems_per_stick())
    except (TypeError, ValueError):
        return False
    return True


def _iter_span_dim_infos(
    layout: FixedTiledLayout,
    max_cores: int,
) -> list[SpanDimInfo]:
    """Return mappable span-controlling dims in outer-to-inner order."""
    stl = layout.device_layout
    device_size = [int(s) for s in stl.device_size]
    host_size = [int(s) for s in layout.size]
    host_stride = [int(s) for s in layout.stride]
    stick_elems = stl.elems_per_stick()

    infos: list[SpanDimInfo] = []
    skipped_outer_device_dims: list[int] = []
    for device_dim in range(len(device_size) - 1):
        if device_size[device_dim] <= 1:
            skipped_outer_device_dims.append(device_dim)
            continue

        sm = int(stl.stride_map[device_dim])
        if sm <= 0:
            skipped_outer_device_dims.append(device_dim)
            continue

        exact_matches = [
            d for d, s in enumerate(host_stride) if host_size[d] > 1 and s == sm
        ]
        stick_scaled_matches = [
            d
            for d, s in enumerate(host_stride)
            if host_size[d] > 1 and s * stick_elems == sm and d not in exact_matches
        ]
        matching_dims = exact_matches or stick_scaled_matches

        if not matching_dims:
            skipped_outer_device_dims.append(device_dim)
            continue

        selected_host_dim = matching_dims[0]
        selected_device_dim_size = device_size[device_dim]
        infos.append(
            SpanDimInfo(
                selected_host_dim=selected_host_dim,
                selected_device_dim_size=selected_device_dim_size,
                selected_device_span_stride_elems=math.prod(
                    device_size[device_dim + 1 :]
                ),
                core_split_estimate=core_split(selected_device_dim_size, max_cores),
                skipped_outer_device_dims=tuple(skipped_outer_device_dims),
            )
        )

    return infos


def _find_outermost_span_dim(
    layout: FixedTiledLayout,
    max_cores: int,
) -> SpanDimInfo | None:
    """Find the outermost mapped device dim that can reduce memory span."""
    infos = _iter_span_dim_infos(layout, max_cores)
    return infos[0] if infos else None


def _compute_num_chunks(
    chunking_info: ChunkingInfo,
    max_cores: int,
) -> int:
    """Return the minimum coarse-tile count required by span and total limits."""
    if chunking_info.selected_device_dim_size == 0:
        return max(
            1, math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))
        )

    num_from_span = math.ceil(chunking_info.per_core_span / MAX_SPAN_BYTES)
    num_from_total = math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))
    return max(num_from_span, num_from_total)


def _divisible_split_candidates(full_size: int, required_count: int) -> list[int]:
    """Return exact split counts for ``full_size`` at least ``required_count``."""
    if required_count <= 1:
        return [1]
    if required_count > full_size:
        raise Unsupported(
            f"Cannot choose coarse-tile split count for dimension size {full_size}: "
            f"required count {required_count} exceeds the dimension size."
        )

    return sorted(
        {
            d
            for i in range(1, math.isqrt(full_size) + 1)
            if full_size % i == 0
            for d in (i, full_size // i)
            if d >= required_count
        }
    )


def _post_tile_layout_for_splits(
    original_layout: FixedTiledLayout,
    split_by_host_dim: dict[int, int],
    op_name: str,
) -> FixedTiledLayout:
    """Build the per-tile layout after applying one or more host-dim splits."""
    new_size = list(original_layout.size)
    for selected_host_dim, split_count in split_by_host_dim.items():
        if split_count <= 0:
            raise Unsupported(
                f"Cannot auto-tile {op_name}: split_count must be positive, "
                f"got {split_count}."
            )
        if selected_host_dim >= len(new_size):
            raise Unsupported(
                f"Cannot auto-tile {op_name}: selected host dim "
                f"{selected_host_dim} is out of bounds for layout size {new_size}."
            )

        try:
            full_size = int(new_size[selected_host_dim])
        except (TypeError, ValueError) as exc:
            raise Unsupported(
                f"Cannot auto-tile {op_name}: selected host dim "
                f"{selected_host_dim} has non-integral layout size "
                f"{new_size[selected_host_dim]!r}."
            ) from exc

        if full_size % split_count != 0:
            raise Unsupported(
                f"Cannot auto-tile {op_name}: selected host dim size {full_size} "
                f"is not divisible by split_count {split_count}."
            )
        new_size[selected_host_dim] = full_size // split_count

    new_stride = list(FlexibleLayout.contiguous_strides(new_size))

    try:
        device_layout = _resize_device_layout(
            original_layout.device_layout,
            [int(s) for s in original_layout.size],
            [int(s) for s in new_size],
        )
    except RuntimeError as exc:
        raise Unsupported(
            f"Cannot auto-tile {op_name}: post-tile device layout reconstruction "
            f"failed: {exc}"
        ) from exc
    return FixedTiledLayout(
        original_layout.device,
        original_layout.dtype,
        new_size,
        new_stride,
        device_layout,
    )


def _post_tile_layout(
    original_layout: FixedTiledLayout,
    selected_host_dim: int,
    split_count: int,
    op_name: str,
) -> FixedTiledLayout:
    """Build the per-tile layout used for single-dim post-tile validation."""
    return _post_tile_layout_for_splits(
        original_layout,
        {selected_host_dim: split_count},
        op_name,
    )


def _within_stick_host_dim(layout: FixedTiledLayout) -> int:
    """Return the host dim represented by the physical within-stick dim."""
    sm_last = int(list(layout.device_layout.stride_map)[-1])
    host_stride = [int(s) for s in layout.stride]
    return next(
        (i for i, s in enumerate(host_stride) if s == sm_last),
        len(host_stride) - 1,
    )


def _post_tile_stick_alignment_error(
    original_layout: FixedTiledLayout,
    selected_host_dim: int,
    split_count: int,
) -> str | None:
    """Return a diagnostic if coarse tiling cuts through physical sticks."""
    if split_count <= 1:
        return None

    within_stick_dim = _within_stick_host_dim(original_layout)
    if selected_host_dim != within_stick_dim:
        return None

    full_size = int(original_layout.size[selected_host_dim])
    tile_size = full_size // split_count
    stick_elems = original_layout.device_layout.elems_per_stick()
    if tile_size % stick_elems == 0:
        return None

    return (
        f"split_count {split_count} makes selected host dim {selected_host_dim} "
        f"tile size {tile_size}, which is not aligned to Spyre stick size "
        f"{stick_elems}; coarse-tile boundaries would cut through physical sticks"
    )


def _post_tile_span_ok(
    split_count: int,
    original_layout: FixedTiledLayout,
    selected_host_dim: int,
    max_cores: int,
    op_name: str,
) -> bool:
    """Return whether the post-tile layout fits span and total limits."""
    tiled_layout = _post_tile_layout(
        original_layout,
        selected_host_dim,
        split_count,
        op_name,
    )
    post_span_dim_info = _find_outermost_span_dim(tiled_layout, max_cores)
    return (
        _needs_chunking(
            tiled_layout,
            max_cores,
            post_span_dim_info,
            op_name=f"{op_name}:post_tile",
            trigger="post_tile_validation",
        )
        is None
    )


def _post_tile_validated_split_count(
    op_name: str,
    full_size: int,
    required_count: int,
    chunking_info: ChunkingInfo,
    original_layout: FixedTiledLayout,
    max_cores: int,
    extra_span_ok: Callable[[int], bool] | None = None,
) -> int:
    """Return the smallest bounded, stick-safe split count that validates."""
    candidates = _divisible_split_candidates(full_size, required_count)
    initial = candidates[0]
    selected_host_dim = chunking_info.selected_host_dim

    stick_alignment_errors: list[str] = []
    for candidate in candidates:
        stick_error = _post_tile_stick_alignment_error(
            original_layout, selected_host_dim, candidate
        )
        if stick_error is not None:
            stick_alignment_errors.append(stick_error)
            logger.debug(
                "plan_span_overflow_tile: %s candidate=%d rejected: %s",
                op_name,
                candidate,
                stick_error,
            )
            continue

        if _post_tile_span_ok(
            candidate, original_layout, selected_host_dim, max_cores, op_name
        ) and (extra_span_ok is None or extra_span_ok(candidate)):
            if candidate != initial:
                logger.debug(
                    "plan_span_overflow_tile: %s bumped split_count %d -> %d "
                    "after post-tile layout validation",
                    op_name,
                    initial,
                    candidate,
                )
            return candidate

        logger.debug(
            "plan_span_overflow_tile: %s candidate=%d still overflows after "
            "post-tile layout validation; trying next divisor",
            op_name,
            candidate,
        )

    if stick_alignment_errors:
        raise Unsupported(
            f"Cannot auto-tile {op_name}: selected_host_dim size {full_size} "
            f"has no legal split >= {required_count} that preserves Spyre "
            f"stick alignment. First rejected candidate: "
            f"{stick_alignment_errors[0]}."
        )

    raise Unsupported(
        f"Cannot auto-tile {op_name}: no divisor of selected_host_dim "
        f"size {full_size} >= {required_count} makes the post-tile layout fit "
        f"within span_limit={MAX_SPAN_BYTES / (1024**2):.0f} MB and "
        f"total_limit={(MAX_SPAN_BYTES * max_cores) / (1024**3):.2f} GB "
        f"(selected_device_dim_size={chunking_info.selected_device_dim_size}, "
        f"stride_elems={chunking_info.selected_device_span_stride_elems}, "
        f"dtype_itemsize={original_layout.dtype.itemsize})."
    )


def _needs_chunking(
    layout: FixedTiledLayout,
    max_cores: int,
    span_dim_info: SpanDimInfo | None,
    *,
    op_name: str | None = None,
    trigger: str = "output_span",
) -> ChunkingInfo | None:
    """Return chunking info when the layout exceeds span or total limits."""
    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    total_bytes = math.prod(device_size) * itemsize
    stick_elems = layout.device_layout.elems_per_stick()

    if span_dim_info is None:
        if total_bytes > MAX_SPAN_BYTES * max_cores:
            host_size = [int(s) for s in layout.size]
            fallback_selected_host_dim = max(
                range(len(host_size)), key=lambda d: host_size[d]
            )
            return ChunkingInfo(
                total_bytes=total_bytes,
                per_core_span=total_bytes,
                core_split_estimate=1,
                selected_device_dim_size=0,
                selected_device_span_stride_elems=0,
                selected_host_dim=fallback_selected_host_dim,
                stick_elems=stick_elems,
                reason=(
                    "no device dimension could be mapped to a splittable "
                    "host dimension via stride_map; using largest host dim "
                    "as fallback"
                ),
            )
        return None

    selected_device_dim_size = span_dim_info.selected_device_dim_size
    selected_device_span_stride_elems = span_dim_info.selected_device_span_stride_elems
    core_split_estimate = span_dim_info.core_split_estimate
    per_core_span = (
        math.ceil(selected_device_dim_size / core_split_estimate)
        * selected_device_span_stride_elems
        * itemsize
    )

    needs_chunk_for_span = per_core_span > MAX_SPAN_BYTES
    needs_chunk_for_total = total_bytes > MAX_SPAN_BYTES * max_cores
    if not (needs_chunk_for_span or needs_chunk_for_total):
        return None

    logger.info(
        "[span_overflow_hint_analysis] trigger=%s op=%s "
        "selected_host_dim=%d selected_device_dim_size=%d "
        "selected_device_span_stride_elems=%d core_split_estimate=%d "
        "per_core_span=%.2f MB total=%.2f GB "
        "(shape=%s, dtype=%s, device_size=%s, "
        "span_limit=%.2f MB, total_limit=%.2f GB)",
        trigger,
        op_name or "<unknown>",
        span_dim_info.selected_host_dim,
        selected_device_dim_size,
        selected_device_span_stride_elems,
        core_split_estimate,
        per_core_span / (1024**2),
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
        core_split_estimate=core_split_estimate,
        selected_device_dim_size=selected_device_dim_size,
        selected_device_span_stride_elems=selected_device_span_stride_elems,
        selected_host_dim=span_dim_info.selected_host_dim,
        stick_elems=stick_elems,
        reason=(
            "skipped unmapped outer device dims "
            f"{span_dim_info.skipped_outer_device_dims}"
            if span_dim_info.skipped_outer_device_dims
            else None
        ),
    )


def _is_batch_matmul_reduction(op: ComputedBuffer) -> bool:
    """Return True for BMM reductions handled by the output-dim policy."""
    return (
        isinstance(op.data, Reduction)
        and getattr(op.data, "reduction_type", None) == BATCH_MATMUL_OP
    )


def _output_symbol_to_dim(op: ComputedBuffer) -> dict[sympy.Symbol, int]:
    """Map output iteration symbols to output dimensions."""
    symbol_to_dim: dict[sympy.Symbol, int] = {}
    try:
        out_coords = op_out_coords(op)
    except (AttributeError, StopIteration, TypeError, RuntimeError):
        out_coords = []

    for dim, coord in enumerate(out_coords):
        for sym in getattr(coord, "free_symbols", ()):  # pragma: no branch - sympy API
            symbol_to_dim.setdefault(sym, dim)

    try:
        out_dep = next(iter(op.get_read_writes().writes))
    except (AttributeError, StopIteration, TypeError):
        return symbol_to_dim

    for dim, sym in enumerate(out_dep.ranges):
        symbol_to_dim.setdefault(sym, dim)
    return symbol_to_dim


def _bmm_output_symbol_to_dim(
    op: ComputedBuffer,
    input_deps: list[tuple[MemoryDep, FixedTiledLayout]],
) -> dict[sympy.Symbol, int]:
    """Map BMM output symbols while identifying K as reduction-only."""
    symbol_to_dim = _output_symbol_to_dim(op)
    if not symbol_to_dim:
        return {}

    reduction_symbols = {
        sym for dep, _ in input_deps for sym in dep.ranges if sym not in symbol_to_dim
    }
    if len(reduction_symbols) != 1:
        logger.debug(
            "span_overflow_bmm: op=%s skipped; expected one K symbol, got %s",
            op.get_name(),
            sorted(reduction_symbols, key=str),
        )
        return {}
    return symbol_to_dim


def _input_read_deps(op: ComputedBuffer) -> list[tuple[MemoryDep, FixedTiledLayout]]:
    """Return fixed-layout input deps that can participate in span planning."""
    try:
        reads = op.get_read_writes().reads
    except (AttributeError, TypeError):
        return []

    deps: list[tuple[MemoryDep, FixedTiledLayout]] = []
    for dep in reads:
        try:
            if (
                not isinstance(dep, MemoryDep)
                or not isinstance(dep.index, sympy.Basic)
                or dep.is_indirect()
            ):
                continue
            buf = V.graph.get_buffer(dep.name)
            layout = _fixed_read_layout(buf)
        except (AttributeError, TypeError, RuntimeError):
            continue
        deps.append((dep, layout))
    return deps


def _range_size_for_symbol(dep: MemoryDep, sym: sympy.Symbol) -> int | None:
    """Return a concrete iteration range size for ``sym`` if available."""
    try:
        return int(dep.ranges[sym])
    except (KeyError, TypeError, ValueError):
        return None


def _coordinate_span_elems(
    coord: sympy.Expr,
    dep: MemoryDep,
    split_symbol: sympy.Symbol,
    split_count: int,
) -> int | None:
    """Return the span in elements for one device coordinate expression."""
    per_core_max = 0
    per_core_min = 0
    for sym in coord.free_symbols:
        range_size = _range_size_for_symbol(dep, sym)
        if range_size is None:
            return None
        if sym == split_symbol:
            if range_size % split_count != 0:
                return None
            range_size //= split_count
        term = coord.subs({other: 0 for other in coord.free_symbols - {sym}})
        per_core_max += int(term.subs(sym, range_size - 1))
        per_core_min += int(term.subs(sym, 0))
    return per_core_max - per_core_min + 1


def _device_coordinates_for_span(
    layout: FixedTiledLayout,
    dep: MemoryDep,
) -> list[sympy.Expr]:
    """Return device coordinates for span planning.

    Span planning only inspects non-stick coordinates.  If the shared helper
    rejects the stick expression, recompute coordinates without the stick guard
    so output-controlled outer spans can still be analyzed.  TODO: replace this
    with a shared non-stick coordinate helper in pass_utils.
    """
    try:
        return device_coordinates(layout.device_layout, dep, None)
    except Unsupported:
        index = concretize_index(dep.index, set(dep.ranges.keys()))
        return compute_coordinates(
            layout.device_layout.device_size,
            layout.device_layout.stride_map,
            dep.ranges,
            index,
            None,
        )


def _input_span_infos_controlled_by_output_dims(
    op: ComputedBuffer,
    max_cores: int,
    *,
    selected_host_dim: int | None = None,
    split_count: int = 1,
    split_by_host_dim: dict[int, int] | None = None,
) -> list[InputSpanInfo]:
    """Return overflowing input spans controlled by output dimensions.

    Input spans controlled by reduction-only symbols are intentionally skipped
    because output-range coarse tiling cannot split reduction ranges without
    partial-result accumulation.  ``split_by_host_dim`` models hypothetical
    output coarse tiles during combined post-tile validation.
    """
    if split_by_host_dim is None:
        split_by_host_dim = (
            {selected_host_dim: split_count} if selected_host_dim is not None else {}
        )

    input_deps = _input_read_deps(op)
    symbol_to_dim = (
        _bmm_output_symbol_to_dim(op, input_deps)
        if _is_batch_matmul_reduction(op)
        else _output_symbol_to_dim(op)
    )
    if not symbol_to_dim:
        return []

    infos: list[InputSpanInfo] = []
    for dep, layout in input_deps:
        if not _layout_has_static_span_metadata(layout):
            continue

        device_size = [int(s) for s in layout.device_layout.device_size]
        itemsize = layout.dtype.itemsize
        stick_elems = layout.device_layout.elems_per_stick()
        try:
            device_coords = _device_coordinates_for_span(layout, dep)
        except (TypeError, ValueError, RuntimeError, Unsupported):
            continue

        for device_dim, coord in enumerate(device_coords[:-1]):
            if not coord.free_symbols:
                continue

            output_syms = [sym for sym in coord.free_symbols if sym in symbol_to_dim]
            reduction_syms = [
                sym for sym in coord.free_symbols if sym not in symbol_to_dim
            ]
            if reduction_syms:
                logger.debug(
                    "span_overflow_input: op=%s dep=%s skipped coord=%s "
                    "controlled by reduction symbols %s",
                    op.get_name(),
                    dep.name,
                    coord,
                    reduction_syms,
                )
                continue
            if len(output_syms) != 1:
                logger.debug(
                    "span_overflow_input: op=%s dep=%s skipped coord=%s "
                    "output_syms=%s reduction_syms=%s",
                    op.get_name(),
                    dep.name,
                    coord,
                    output_syms,
                    reduction_syms,
                )
                continue

            controlling_symbol = output_syms[0]
            output_dim = symbol_to_dim[controlling_symbol]
            split_for_symbol = split_by_host_dim.get(output_dim, 1)

            coord_span_elems = _coordinate_span_elems(
                coord, dep, controlling_symbol, split_for_symbol
            )
            if coord_span_elems is None:
                continue

            # Span-overflow planning is intentionally conservative: coarse tiles
            # must make the span safe even if downstream work division provides
            # no additional split for this coordinate.
            split_estimate = 1
            per_core_span = (
                coord_span_elems * math.prod(device_size[device_dim + 1 :]) * itemsize
            )
            if per_core_span <= MAX_SPAN_BYTES:
                continue

            infos.append(
                InputSpanInfo(
                    chunking_info=ChunkingInfo(
                        total_bytes=math.prod(device_size) * itemsize,
                        per_core_span=per_core_span,
                        core_split_estimate=split_estimate,
                        selected_device_dim_size=coord_span_elems,
                        selected_device_span_stride_elems=math.prod(
                            device_size[device_dim + 1 :]
                        ),
                        selected_host_dim=output_dim,
                        stick_elems=stick_elems,
                        reason=f"input span overflow for {dep.name}",
                    ),
                    dep_name=dep.name,
                    controlling_symbol=controlling_symbol,
                )
            )
    return infos


def _chunking_info_from_span_dim(
    layout: FixedTiledLayout,
    span_dim_info: SpanDimInfo,
    *,
    core_split_estimate: int,
    reason: str | None = None,
) -> ChunkingInfo:
    """Build ChunkingInfo for one span-controlling physical dimension."""
    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    per_core_span = (
        math.ceil(span_dim_info.selected_device_dim_size / core_split_estimate)
        * span_dim_info.selected_device_span_stride_elems
        * itemsize
    )
    return ChunkingInfo(
        total_bytes=math.prod(device_size) * itemsize,
        per_core_span=per_core_span,
        core_split_estimate=core_split_estimate,
        selected_device_dim_size=span_dim_info.selected_device_dim_size,
        selected_device_span_stride_elems=span_dim_info.selected_device_span_stride_elems,
        selected_host_dim=span_dim_info.selected_host_dim,
        stick_elems=layout.device_layout.elems_per_stick(),
        reason=reason,
    )


def _output_write_dep(op: ComputedBuffer) -> MemoryDep | None:
    """Return the concrete output MemoryDep used for address-math span checks."""
    try:
        dep = next(iter(op.get_read_writes().writes))
    except (AttributeError, StopIteration, TypeError):
        return None
    if not isinstance(dep, MemoryDep) or not isinstance(dep.index, sympy.Basic):
        return None
    return dep


def _output_span_candidates_from_op(
    op: ComputedBuffer,
    *,
    layout: FixedTiledLayout | None = None,
    split_by_host_dim: dict[int, int] | None = None,
    op_name: str | None = None,
) -> list[SpanOverflowCandidate]:
    """Collect output spans using physical device-coordinate address math."""
    layout = layout or op.layout
    split_by_host_dim = split_by_host_dim or {}
    out_dep = _output_write_dep(op)
    if out_dep is None:
        return _output_span_candidates(layout, op_name=op_name or op.get_name())

    symbol_to_dim = _output_symbol_to_dim(op)
    if not symbol_to_dim:
        return _output_span_candidates(layout, op_name=op_name or op.get_name())

    try:
        device_coords = _device_coordinates_for_span(layout, out_dep)
    except (TypeError, ValueError, RuntimeError, Unsupported):
        return _output_span_candidates(layout, op_name=op_name or op.get_name())

    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    candidates: list[SpanOverflowCandidate] = []
    for device_dim, coord in enumerate(device_coords[:-1]):
        if not coord.free_symbols:
            continue
        output_syms = [sym for sym in coord.free_symbols if sym in symbol_to_dim]
        if len(output_syms) != 1 or len(output_syms) != len(coord.free_symbols):
            logger.debug(
                "span_overflow_output: op=%s skipped coord=%s output_syms=%s",
                op.get_name(),
                coord,
                output_syms,
            )
            continue

        controlling_symbol = output_syms[0]
        output_dim = symbol_to_dim[controlling_symbol]
        coord_span_elems = _coordinate_span_elems(
            coord,
            out_dep,
            controlling_symbol,
            split_by_host_dim.get(output_dim, 1),
        )
        if coord_span_elems is None:
            continue
        per_core_span = (
            coord_span_elems * math.prod(device_size[device_dim + 1 :]) * itemsize
        )
        if per_core_span <= MAX_SPAN_BYTES:
            continue
        candidates.append(
            SpanOverflowCandidate(
                chunking_info=ChunkingInfo(
                    total_bytes=math.prod(device_size) * itemsize,
                    per_core_span=per_core_span,
                    core_split_estimate=1,
                    selected_device_dim_size=coord_span_elems,
                    selected_device_span_stride_elems=math.prod(
                        device_size[device_dim + 1 :]
                    ),
                    selected_host_dim=output_dim,
                    stick_elems=layout.device_layout.elems_per_stick(),
                    reason="output span overflow",
                ),
                source="output",
            )
        )

    return _log_span_candidates(
        candidates,
        layout,
        op_name=op_name or op.get_name(),
    )


def _log_span_candidates(
    candidates: list[SpanOverflowCandidate],
    layout: FixedTiledLayout,
    *,
    op_name: str,
) -> list[SpanOverflowCandidate]:
    """Log span-overflow candidates and return them unchanged."""
    device_size = [int(s) for s in layout.device_layout.device_size]
    for candidate in candidates:
        info = candidate.chunking_info
        logger.info(
            "[span_overflow_hint_analysis] trigger=%s op=%s "
            "selected_host_dim=%d selected_device_dim_size=%d "
            "selected_device_span_stride_elems=%d per_tile_span=%.2f MB "
            "total=%.2f GB shape=%s dtype=%s device_size=%s "
            "span_limit=%.2f MB",
            candidate.source,
            op_name,
            info.selected_host_dim,
            info.selected_device_dim_size,
            info.selected_device_span_stride_elems,
            info.per_core_span / (1024**2),
            info.total_bytes / (1024**3),
            list(layout.size),
            layout.dtype,
            device_size,
            MAX_SPAN_BYTES / (1024**2),
        )
    return candidates


def _output_span_candidates(
    layout: FixedTiledLayout,
    *,
    op_name: str,
) -> list[SpanOverflowCandidate]:
    """Collect output-layout spans that overflow without work-division help."""
    candidates: list[SpanOverflowCandidate] = []
    for span_dim_info in _iter_span_dim_infos(layout, max_cores=1):
        chunking_info = _chunking_info_from_span_dim(
            layout,
            span_dim_info,
            core_split_estimate=1,
            reason="output span overflow",
        )
        if chunking_info.per_core_span > MAX_SPAN_BYTES:
            candidates.append(
                SpanOverflowCandidate(chunking_info=chunking_info, source="output")
            )

    device_size = [int(s) for s in layout.device_layout.device_size]
    total_bytes = math.prod(device_size) * layout.dtype.itemsize
    if not candidates and total_bytes > MAX_SPAN_BYTES:
        host_size = [int(s) for s in layout.size]
        selected_host_dim = max(range(len(host_size)), key=lambda d: host_size[d])
        candidates.append(
            SpanOverflowCandidate(
                chunking_info=ChunkingInfo(
                    total_bytes=total_bytes,
                    per_core_span=total_bytes,
                    core_split_estimate=1,
                    selected_device_dim_size=0,
                    selected_device_span_stride_elems=0,
                    selected_host_dim=selected_host_dim,
                    stick_elems=layout.device_layout.elems_per_stick(),
                    reason=(
                        "output total span overflow; using largest host dim as fallback"
                    ),
                ),
                source="output_total",
            )
        )

    return _log_span_candidates(candidates, layout, op_name=op_name)


def _input_span_candidates(
    op: ComputedBuffer,
    max_cores: int,
    *,
    split_by_host_dim: dict[int, int] | None = None,
) -> list[SpanOverflowCandidate]:
    """Collect reduction/BMM input spans controlled by output dimensions."""
    candidates: list[SpanOverflowCandidate] = []
    for info in _input_span_infos_controlled_by_output_dims(
        op,
        max_cores,
        split_by_host_dim=split_by_host_dim,
    ):
        host_dim = info.chunking_info.selected_host_dim
        if not _host_dim_has_legal_nontrivial_split(op, host_dim):
            logger.debug(
                "span_overflow_input: op=%s dep=%s host_dim=%d has no "
                "legal non-stick coarse split; skipping input candidate",
                op.get_name(),
                info.dep_name,
                host_dim,
            )
            continue
        candidates.append(
            SpanOverflowCandidate(info.chunking_info, source=f"input:{info.dep_name}")
        )
    return candidates


def _candidate_host_dims(
    candidates: list[SpanOverflowCandidate],
) -> list[int]:
    """Return host dims in cost-search order by decreasing span pressure."""
    max_span_by_dim: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for idx, candidate in enumerate(candidates):
        dim = candidate.chunking_info.selected_host_dim
        first_seen.setdefault(dim, idx)
        max_span_by_dim[dim] = max(
            max_span_by_dim.get(dim, 0), candidate.chunking_info.per_core_span
        )
    return sorted(
        max_span_by_dim,
        key=lambda dim: (-max_span_by_dim[dim], first_seen[dim]),
    )


def _host_dim_has_legal_nontrivial_split(op: ComputedBuffer, host_dim: int) -> bool:
    """Return True when coarse tiling can legally split this output dim."""
    try:
        candidates = _split_candidates_for_host_dim(op, host_dim)
    except Unsupported:
        return False
    return any(split > 1 for split in candidates)


def _candidate_required_split_count(candidate: SpanOverflowCandidate) -> int:
    """Return the split count needed if this candidate's dim tiled alone."""
    info = candidate.chunking_info
    if info.selected_device_dim_size == 0:
        return max(1, math.ceil(info.total_bytes / MAX_SPAN_BYTES))
    return max(1, math.ceil(info.per_core_span / MAX_SPAN_BYTES))


def _required_split_counts_by_host_dim(
    candidates: list[SpanOverflowCandidate],
) -> dict[int, int]:
    """Return the strongest single-dim split requirement per host dim."""
    required_by_dim: dict[int, int] = {}
    for candidate in candidates:
        dim = candidate.chunking_info.selected_host_dim
        required_by_dim[dim] = max(
            required_by_dim.get(dim, 1),
            _candidate_required_split_count(candidate),
        )
    return required_by_dim


def _cap_split_candidates(
    legal_candidates: list[int],
    required_count: int,
) -> list[int]:
    """Bound split candidates while preserving small and required splits."""
    if len(legal_candidates) <= _MAX_SPLITS_PER_DIM:
        return legal_candidates

    selected: list[int] = []

    def add(split: int) -> None:
        if split in legal_candidates and split not in selected:
            selected.append(split)

    add(1)
    for split in legal_candidates:
        if split > 1:
            add(split)
        if len(selected) >= min(5, _MAX_SPLITS_PER_DIM):
            break

    required_idx = next(
        (idx for idx, split in enumerate(legal_candidates) if split >= required_count),
        len(legal_candidates) - 1,
    )
    for idx in range(
        max(0, required_idx - 2),
        min(len(legal_candidates), required_idx + 4),
    ):
        add(legal_candidates[idx])

    for split in reversed(legal_candidates):
        add(split)
        if len(selected) >= _MAX_SPLITS_PER_DIM:
            break

    return sorted(selected)[:_MAX_SPLITS_PER_DIM]


def _host_dim_target_symbols(
    op: ComputedBuffer,
    host_dim: int,
) -> set[sympy.Symbol]:
    """Return the output iteration symbols mapped to ``host_dim``."""
    symbol_to_dim = (
        _bmm_output_symbol_to_dim(op, _input_read_deps(op))
        if _is_batch_matmul_reduction(op)
        else _output_symbol_to_dim(op)
    )
    return {sym for sym, dim in symbol_to_dim.items() if dim == host_dim}


def _input_stick_alignment_error(
    op: ComputedBuffer,
    host_dim: int,
    split_count: int,
) -> str | None:
    """Return a diagnostic if splitting ``host_dim`` misaligns an input's sticks.

    ``host_dim`` is an index into the output op's iteration space.  The same
    iteration symbol also addresses each input dependency, but an input
    tensor's own physical layout (stride order, stick mapping) can differ
    from the output's, so stick alignment must be checked against each
    input's own layout independently of the output layout check.
    """
    if split_count <= 1:
        return None

    target_symbols = _host_dim_target_symbols(op, host_dim)
    if not target_symbols:
        return None

    for dep, layout in _input_read_deps(op):
        if not _layout_has_static_span_metadata(layout):
            continue
        dep_symbols = list(dep.ranges.keys())
        for sym in target_symbols:
            if sym not in dep_symbols:
                continue
            input_host_dim = dep_symbols.index(sym)
            error = _post_tile_stick_alignment_error(
                layout, input_host_dim, split_count
            )
            if error is not None:
                return f"input dependency {dep.name} host dim {input_host_dim}: {error}"
    return None


def _split_candidates_for_host_dim(
    op: ComputedBuffer,
    host_dim: int,
    required_count: int = 1,
) -> list[int]:
    """Return bounded legal split candidates for one output host dim."""
    ranges = list(op.data.ranges)
    if host_dim >= len(ranges):
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim {host_dim} "
            f"is out of bounds for data ranges {ranges}."
        )
    try:
        full_size = int(ranges[host_dim])
    except (TypeError, ValueError) as exc:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim {host_dim} "
            f"has non-integral range {ranges[host_dim]!r}."
        ) from exc
    if full_size <= 1:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim {host_dim} "
            f"has unsplittable range {full_size}."
        )
    candidates = sorted(
        {
            d
            for i in range(1, math.isqrt(full_size) + 1)
            if full_size % i == 0
            for d in (i, full_size // i)
        }
    )
    legal_candidates = [
        split
        for split in candidates
        if split == 1
        or (
            _post_tile_stick_alignment_error(op.layout, host_dim, split) is None
            and _input_stick_alignment_error(op, host_dim, split) is None
        )
    ]
    capped_candidates = _cap_split_candidates(legal_candidates, required_count)
    if len(legal_candidates) > len(capped_candidates):
        logger.debug(
            "span_overflow_tile_search: op=%s host_dim=%d limiting %d split "
            "candidates to %d before combo search (required_count=%d)",
            op.get_name(),
            host_dim,
            len(legal_candidates),
            len(capped_candidates),
            required_count,
        )
    return capped_candidates


def _combo_cost(combo: tuple[int, ...]) -> tuple[int, int, int, tuple[int, ...]]:
    """Prefer fewer total tiles, fewer tiled dims, and smaller maximum split."""
    return (
        math.prod(combo),
        sum(split > 1 for split in combo),
        max(combo),
        combo,
    )


def _iter_split_combos(
    split_candidates: list[list[int]],
) -> list[tuple[int, ...]]:
    """Return bounded split combinations in increasing cost order."""
    combos = sorted(itertools.product(*split_candidates), key=_combo_cost)
    if len(combos) > _MAX_TILE_COMBOS:
        logger.debug(
            "span_overflow_tile_search: truncating %d combos to %d",
            len(combos),
            _MAX_TILE_COMBOS,
        )
    return combos[:_MAX_TILE_COMBOS]


def _combined_tile_stick_alignment_error(
    op: ComputedBuffer,
    original_layout: FixedTiledLayout,
    split_by_host_dim: dict[int, int],
) -> str | None:
    """Return the first stick-alignment error for a combined split."""
    for host_dim, split_count in split_by_host_dim.items():
        error = _post_tile_stick_alignment_error(original_layout, host_dim, split_count)
        if error is not None:
            return error
        error = _input_stick_alignment_error(op, host_dim, split_count)
        if error is not None:
            return error
    return None


def _remaining_span_candidates_after_tile(
    op: ComputedBuffer,
    max_cores: int,
    split_by_host_dim: dict[int, int],
) -> list[SpanOverflowCandidate]:
    """Return spans that still overflow after applying a combined tile split."""
    tiled_layout = _post_tile_layout_for_splits(
        op.layout,
        split_by_host_dim,
        op.get_name(),
    )
    remaining = _output_span_candidates_from_op(
        op,
        layout=tiled_layout,
        split_by_host_dim=split_by_host_dim,
        op_name=f"{op.get_name()}:post_tile",
    )
    if isinstance(op.data, Reduction):
        remaining += _input_span_candidates(
            op,
            max_cores,
            split_by_host_dim=split_by_host_dim,
        )
    return remaining


def _search_min_cost_tile_plan(
    op: ComputedBuffer,
    max_cores: int,
    candidates: list[SpanOverflowCandidate],
) -> SpanOverflowTilePlan | None:
    """Find the cheapest combined coarse-tile plan that clears all spans."""
    host_dims = _candidate_host_dims(candidates)
    if not host_dims:
        return None
    if len(host_dims) > _MAX_TILE_DIMS:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: span-overflow planning found "
            f"{len(host_dims)} candidate host dims {host_dims}, exceeding the "
            f"bounded search limit {_MAX_TILE_DIMS}."
        )

    required_by_dim = _required_split_counts_by_host_dim(candidates)
    split_candidates = [
        _split_candidates_for_host_dim(op, dim, required_by_dim.get(dim, 1))
        for dim in host_dims
    ]
    first_stick_error: str | None = None
    for combo in _iter_split_combos(split_candidates):
        split_by_host_dim = {
            dim: split for dim, split in zip(host_dims, combo) if split > 1
        }
        if not split_by_host_dim:
            continue

        stick_error = _combined_tile_stick_alignment_error(
            op, op.layout, split_by_host_dim
        )
        if stick_error is not None:
            first_stick_error = first_stick_error or stick_error
            continue

        try:
            remaining = _remaining_span_candidates_after_tile(
                op,
                max_cores,
                split_by_host_dim,
            )
        except Unsupported as exc:
            logger.debug(
                "span_overflow_tile_search: op=%s combo=%s rejected: %s",
                op.get_name(),
                split_by_host_dim,
                exc,
            )
            continue
        if remaining:
            logger.debug(
                "span_overflow_tile_search: op=%s combo=%s leaves %d spans",
                op.get_name(),
                split_by_host_dim,
                len(remaining),
            )
            continue

        # Emit loop levels outer-to-inner by host dimension; the search order can
        # differ because it is driven by span pressure.
        levels = tuple(
            SpanOverflowTileLevel(
                selected_host_dim=dim,
                split_count=split_by_host_dim[dim],
            )
            for dim in sorted(split_by_host_dim)
        )
        return SpanOverflowTilePlan(
            levels=levels,
            chunking_infos=tuple(candidate.chunking_info for candidate in candidates),
            reason="; ".join(
                sorted(
                    {
                        candidate.chunking_info.reason or candidate.source
                        for candidate in candidates
                    }
                )
            ),
        )

    if first_stick_error is not None:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: no legal combined split preserves "
            f"Spyre stick alignment. First rejected candidate: {first_stick_error}."
        )
    raise Unsupported(
        f"Cannot auto-tile {op.get_name()}: no combined split among host dims "
        f"{host_dims} makes all spans fit within "
        f"{MAX_SPAN_BYTES / (1024**2):.0f} MB after trying at most "
        f"{_MAX_TILE_COMBOS} combinations."
    )


def _has_indirect_reads(op: ComputedBuffer) -> bool:
    """Return True if the op uses indirect/gather-style input reads."""
    try:
        _, _, indirect_sizes = indirect_info_from_op(op)
    except (AttributeError, RuntimeError, TypeError, Unsupported):
        return False
    return indirect_sizes is not None


def plan_span_overflow_tile(
    op: ComputedBuffer,
    max_cores: int,
) -> SpanOverflowTilePlan | None:
    """Return an automatic output-range coarse-tile plan if needed."""
    if not (
        isinstance(op, ComputedBuffer)
        and isinstance(op.layout, FixedTiledLayout)
        and isinstance(op.data, (Pointwise, Reduction))
    ):
        return None

    if not _layout_has_static_span_metadata(op.layout):
        return None

    if isinstance(op.data, Pointwise):
        # Gather/indirect-access ops can lower as Pointwise ComputedBuffers, but
        # they require the dedicated indirect-access SDSC path rather than automatic
        # output coarse tiling.
        if _has_indirect_reads(op):
            return None
        candidates = _output_span_candidates_from_op(op, op_name=op.get_name())
        return _search_min_cost_tile_plan(op, max_cores, candidates)

    if isinstance(op.data, Reduction):
        # Gather/scatter-indexed reductions (e.g. scatter-reduce) require the
        # dedicated indirect-access SDSC path rather than automatic coarse
        # tiling, same as the Pointwise case above.
        if _has_indirect_reads(op):
            return None
        if not list(op.data.ranges):
            return None
        candidates = _output_span_candidates_from_op(op, op_name=op.get_name())
        candidates += _input_span_candidates(op, max_cores)
        return _search_min_cost_tile_plan(op, max_cores, candidates)

    return None


def _plan_reduction_output_span_overflow_tile(
    op: ComputedBuffer,
    max_cores: int,
) -> SpanOverflowTilePlan | None:
    """Compatibility wrapper for reduction span planning."""
    if not list(op.data.ranges):
        return None
    candidates = _output_span_candidates_from_op(op, op_name=op.get_name())
    candidates += _input_span_candidates(op, max_cores)
    return _search_min_cost_tile_plan(op, max_cores, candidates)


def _output_chunking_info(
    op: ComputedBuffer,
    max_cores: int,
) -> ChunkingInfo | None:
    """Return the first output chunking info when the output layout overflows."""
    candidates = _output_span_candidates_from_op(op, op_name=op.get_name())
    return candidates[0].chunking_info if candidates else None


def _plan_output_span_overflow_tile(
    op: ComputedBuffer,
    max_cores: int,
) -> SpanOverflowTilePlan | None:
    """Compatibility wrapper for output-only span planning."""
    candidates = _output_span_candidates_from_op(op, op_name=op.get_name())
    return _search_min_cost_tile_plan(op, max_cores, candidates)


def _plan_tile_from_chunking_info(
    op: ComputedBuffer,
    max_cores: int,
    chunking_info: ChunkingInfo,
    *,
    required_count: int | None = None,
    reason: str | None = None,
    extra_span_ok: Callable[[int], bool] | None = None,
) -> SpanOverflowTilePlan | None:
    """Build a single-dim tile plan from selected chunking info.

    Kept for private tests and callers that exercise the historical one-dim
    path.  The production planner uses the multi-dim cost search above.
    """
    selected_host_dim = chunking_info.selected_host_dim
    ranges = list(op.data.ranges)
    if selected_host_dim >= len(ranges):
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim "
            f"{selected_host_dim} is out of bounds for given data ranges {ranges}."
        )

    try:
        full_size = int(ranges[selected_host_dim])
    except (TypeError, ValueError) as exc:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim "
            f"{selected_host_dim} has non-integral range "
            f"{ranges[selected_host_dim]!r}."
        ) from exc

    if full_size <= 1:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim "
            f"{selected_host_dim} has unsplittable range {full_size}."
        )

    if required_count is None:
        required_count = _compute_num_chunks(chunking_info, max_cores)
    if required_count <= 1:
        return None

    split_count = _post_tile_validated_split_count(
        op.get_name(),
        full_size,
        required_count,
        chunking_info,
        op.layout,
        max_cores,
        extra_span_ok=extra_span_ok,
    )

    return SpanOverflowTilePlan(
        levels=(
            SpanOverflowTileLevel(
                selected_host_dim=selected_host_dim,
                split_count=split_count,
            ),
        ),
        chunking_infos=(chunking_info,),
        reason=reason or chunking_info.reason or "output span overflow",
    )
