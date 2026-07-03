# Span-Overflow Hint Analysis

## Background

Spyre kernels must keep each core's memory-address span within the hardware
limit (`MAX_SPAN_BYTES`, normally 256 MiB).  Normal `work_division` splits work
across cores, but some physical layouts still expose a span that is too large
for one core.  When that happens, backend compilation can fail with Work
Division warnings, deeptools mutable-address failures, or immediate/EAR boundary
errors.

`span_overflow_hint_analysis.py` is the compiler-generated version of a user
`spyre_hint`: it detects span-risky `ComputedBuffer` ops and asks existing
coarse tiling to run those ops in smaller output-range tiles.

The planner does not mutate IR directly.  It computes a `SpanOverflowTilePlan`.
The adapter in `coarse_tile.py` converts that plan into the same `DimHint` and
coarse-tile group format used by manual `spyre_hint`.

```text
span_overflow_hint_analysis
  -> coarse_tile.span_overflow_groups
  -> coarse_tile
  -> CountedLoopSchedulerNode
  -> LoopSpec codegen
```

## High-Level Contract

The planner answers this question for one op:

> If Work Division gives this op no useful span reduction, can output-range
> coarse tiling make every relevant output/input span fit under
> `MAX_SPAN_BYTES`?

The important design choice is conservative validation: automatic tiles must be
safe even if Work Division later chooses a different split dimension or cannot
split the span-driving coordinate at all.  Work Division may still improve
parallelism, but correctness must not depend on it.

## Concrete Examples

This section gives examples for the main cases the pass handles.  Shapes are
representative; exact split counts can vary with Spyre physical layout, dtype,
`SENCORES`, and `MAX_SPAN_BYTES`.

### 1. Pointwise Output Span Overflow

Example operation:

```python
def fn(x, y):
    return x + y

x.shape == y.shape == (1, 8195, 256, 64)
```

The output is large enough that one physical output coordinate can span more
than the limit.  The Pointwise path collects an output candidate and emits an
output-range tile, for example:

```text
selected_host_dim = 1
split_count       = 5
```

Conceptually this is equivalent to:

```python
with spyre_hint(num_tiles_per_dim={"H": 5}):
    out = x + y
```

Code flow:

```text
plan_span_overflow_tile
  -> Pointwise branch
  -> _has_indirect_reads(op) == False
  -> _output_span_candidates_from_op
  -> _search_min_cost_tile_plan
  -> _post_tile_layout_for_splits
  -> _remaining_span_candidates_after_tile
  -> SpanOverflowTilePlan(levels=(host_dim=1, split_count=5))
```

### 2. Pointwise With Indirect/Gather Read

Example pattern:

```python
def fn(x, index):
    return x[index] + 1
```

Even if the resulting output layout is large, automatic span-overflow tiling is
skipped because indirect access needs the dedicated indirect-access path.  The
planner returns `None`.

Code flow:

```text
plan_span_overflow_tile
  -> Pointwise branch
  -> _has_indirect_reads(op) == True
  -> return None
```

### 3. Reduction Output Span Overflow

Example operation:

```python
def fn(x):
    return x.sum(dim=-1)

x.shape == (1, 8195, 256, 64)
out.shape == (1, 8195, 256)
```

The reduction output itself can have an unsafe physical span.  The planner uses
the same output-candidate path as Pointwise, but through the Reduction branch.
If a safe output-range tile exists, it emits a normal output-dim plan.

Code flow:

```text
plan_span_overflow_tile
  -> Reduction branch
  -> _has_indirect_reads(op) == False
  -> op.data.ranges is not empty
  -> _output_span_candidates_from_op
  -> _input_span_candidates
  -> _search_min_cost_tile_plan
```

### 4. Reduction Input Span Controlled By Output Dim

Example operation:

```python
def fn(x):
    return x.sum(dim=0)

x.shape   == (2, 20, 16, 64)
out.shape == (20, 16, 64)
```

The output may be smaller than the input, but reading `x` can still span too
much memory.  If the overflowing input physical coordinate is controlled by an
output symbol, such as the output `20` dimension, output-range tiling can reduce
the input span.

A small codegen test uses a patched span limit and checks for:

```text
LoopSpec(
count=sympify('10')
op='sum'
tiled_symbols=[[sympify('c0')]]
```

Code flow:

```text
plan_span_overflow_tile
  -> Reduction branch
  -> _output_span_candidates_from_op
  -> _input_span_candidates
       -> _input_span_infos_controlled_by_output_dims
          -> device coordinate contains output symbol
          -> create input-derived SpanOverflowCandidate
  -> _search_min_cost_tile_plan
```

### 5. BMM / LM-Head Input Span Controlled By Output Dim

Example operation:

```python
def lm_head(x, weight):
    return torch.nn.functional.linear(x, weight)

x.shape      == (2, 64)
weight.shape == (1024, 64)
out.shape    == (2, 1024)
```

Lowering can represent this as a batch-matmul-style reduction.  The BMM symbol
model is:

```text
C[b, m, n] = reduce_k A[b, m, k] * B[b, k, n]
```

Spans controlled by `b`, `m`, or `n` are output-range tileable.  Spans
controlled by `k` are not.  A codegen test checks for:

```text
LoopSpec(
count=sympify('16')
op='batchmatmul'
tiled_symbols=[[sympify('c0')]]
```

Code flow:

```text
plan_span_overflow_tile
  -> Reduction branch
  -> _is_batch_matmul_reduction(op) == True
  -> _bmm_output_symbol_to_dim
       -> verify exactly one reduction-only symbol, k
  -> _input_span_candidates
  -> _search_min_cost_tile_plan
```

### 6. Multiple Overflowing Host Dims

Example pattern:

```python
x.shape == (4096, 4096, 4096, 64)
out = x + 1
```

More than one physical coordinate can independently exceed the span limit.  The
planner creates candidates for each overflowing host dim, generates legal split
candidates for each dim, and searches combinations.

A multi-level result looks like:

```python
SpanOverflowTilePlan(
    levels=(
        SpanOverflowTileLevel(selected_host_dim=0, split_count=2),
        SpanOverflowTileLevel(selected_host_dim=1, split_count=2),
    ),
)
```

Code flow:

```text
_output_span_candidates_from_op -> candidates for dim 0 and dim 1
_candidate_host_dims            -> order by span pressure
_split_candidates_for_host_dim  -> legal divisors per dim
_iter_split_combos              -> cheapest combos first
_remaining_span_candidates_after_tile -> accept only if all spans are safe
```

### 7. Input Stick-Alignment Rejection

Example pattern:

```text
output dim m is safe to split in the output layout,
but the same m symbol maps to the within-stick dim of a transposed BMM input.
```

The planner rejects the split if it would cut through physical sticks in any
input dependency.  This prevents a plan that is legal for the output tensor but
unsafe for an input tensor.

Code flow:

```text
_split_candidates_for_host_dim
  -> _post_tile_stick_alignment_error(op.layout, host_dim, split)
  -> _input_stick_alignment_error(op, host_dim, split)
       -> map output host dim to output symbol
       -> find same symbol in each input dep
       -> check that input layout's stick dim is not cut
```

### 8. Reduction-Controlled Span: Known Unsupported Case

Example operation:

```python
def fn(x):
    return x.sum(dim=-1)
```

If the only overflowing input coordinate is controlled by the reduction symbol
`k`, output-range tiling cannot fix it.  This pass intentionally skips that
candidate.

Code flow:

```text
_input_span_infos_controlled_by_output_dims
  -> coordinate free symbols include k
  -> k is not in output_symbol_to_dim
  -> reduction_syms is non-empty
  -> continue
```

This is not solved by the current pass.  It needs reduction-range tiling and
partial accumulation.

## Entry Point

The planner entry point is:

```python
plan = plan_span_overflow_tile(op, max_cores=config.sencores)
```

It returns:

```python
SpanOverflowTilePlan(
    levels=(
        SpanOverflowTileLevel(selected_host_dim=..., split_count=...),
        ...,
    ),
    chunking_infos=(...),
    reason="...",
)
```

or `None` if no automatic span-overflow tiling is needed or supported for that
op.

`span_overflow_groups(graph)` consumes the plan, attaches synthetic `DimHint`s,
and returns groups shaped like user hint groups:

```python
[([op], [(hint_id, split_count, is_reduction_level), ...])]
```

`is_reduction_level` is currently `False` for automatic span-overflow hints;
they tile output ranges, not reduction ranges.

## Scope

The production planner handles:

- `ComputedBuffer` ops with `Pointwise` data;
- `ComputedBuffer` ops with non-scalar `Reduction` data;
- output layout span overflow;
- reduction/BMM input span overflow when the span is controlled by output
  symbols;
- one or more output host dimensions per op, bounded by `_MAX_TILE_DIMS`;
- static `FixedTiledLayout` metadata only.

The planner skips:

- non-`ComputedBuffer` ops;
- non-`FixedTiledLayout` ops;
- symbolic/dynamic layout metadata;
- scalar/full reductions where `op.data.ranges` is empty;
- Pointwise or Reduction ops with indirect/gather/scatter-style reads;
- input spans controlled only by reduction symbols.

### Support Matrix

| Case | Supported? | Planner behavior |
|---|---:|---|
| Pointwise output span overflow | Yes | Emit one or more output-range tile levels |
| Pointwise indirect/gather read | No | Return `None`; indirect path owns it |
| Reduction output span overflow | Yes | Emit output-range tile levels if post-tile layout validates |
| Reduction input span controlled by output dim | Yes | Emit tile for the matching output dim |
| Reduction input span controlled by reduction dim | No | Skip candidate; reduction-range tiling is future work |
| BMM input span controlled by `b`, `m`, or `n` | Yes | Emit tile for matching output dim |
| BMM input span controlled by `k` | No | Skip candidate; `k` is reduction-only |
| BMM with ambiguous reduction symbols | No | Return no BMM input candidates |
| Scalar/full reduction | No | Return `None` |
| Symbolic layout metadata | No | Return `None` |
| More than `_MAX_TILE_DIMS` overflowing host dims | No | Raise `Unsupported` |
| No legal exact divisor / stick-safe split | No | Raise `Unsupported` |
| User already supplied `spyre_hint` for op | Manual wins | Auto hint is not added for that op |

## Pointwise Flow

For `Pointwise` ops, `plan_span_overflow_tile` does:

```text
check op/layout eligibility
check _has_indirect_reads(op); skip if true
collect output span candidates
search for cheapest valid tile combination
```

The main output collector is `_output_span_candidates_from_op`.  It prefers
physical address-coordinate analysis:

1. read the output `MemoryDep`;
2. map output iteration symbols to output dimensions;
3. compute physical device coordinates for the output layout;
4. walk non-stick device coordinates;
5. create one `SpanOverflowCandidate` for each output-controlled coordinate
   whose span exceeds `MAX_SPAN_BYTES`.

If output write-dep or symbol-coordinate analysis is unavailable, it falls back
to `_output_span_candidates`, which uses `device_size`, `stride_map`, host
strides, and stick size to recover span-driving host dimensions.  The fallback
is less precise, but still feeds the same candidate/combo search.

## Reduction Flow

For `Reduction` ops, `plan_span_overflow_tile` does:

```text
check op/layout eligibility
check _has_indirect_reads(op); skip if true
skip scalar reductions
collect output span candidates
collect input span candidates controlled by output dims
search for cheapest valid tile combination
```

A reduction can overflow in two ways:

1. the output layout itself is too large; or
2. the output is small, but reading an input spans too much physical memory.

For input reads, `_input_span_infos_controlled_by_output_dims` walks the input
physical device coordinates.  For each non-stick coordinate it separates free
symbols into:

- output symbols: symbols present in `_output_symbol_to_dim(op)`;
- reduction symbols: symbols not present in that output map.

If a coordinate is controlled by exactly one output symbol, output-range coarse
tiling can reduce that input span, so the planner creates an input-derived
candidate.  If a coordinate is controlled by a reduction symbol, the planner
skips that coordinate and continues scanning later device coordinates.  That
`continue` behavior is important: a reduction-controlled outer coordinate must
not prevent discovery of a more-inner output-controlled overflowing coordinate.

Reduction-only input span overflow remains a known limitation.  It requires
reduction-range tiling and partial accumulation, which this pass does not
implement.

## BMM-Specific Symbol Mapping

Batch matmul reductions use `BATCH_MATMUL_OP` and are treated specially when
mapping output symbols:

```text
C[b, m, n] = reduce_k A[b, m, k] * B[b, k, n]
```

`_bmm_output_symbol_to_dim` starts from `_output_symbol_to_dim`, then checks
input dependencies to identify exactly one reduction-only symbol (`k`).  If
there is not exactly one such symbol, it returns `{}` and the BMM input-span
path produces no candidates.

BMM input spans controlled by `b`, `m`, or `n` can be fixed by output-range
tiling.  Spans controlled by `k` are skipped as reduction-range-only.

## Span Calculation

For a physical device coordinate at `device_dim`, span is computed as:

```python
per_core_span = (
    coord_span_elems
    * math.prod(device_size[device_dim + 1:])
    * dtype.itemsize
)
```

This includes the selected physical coordinate and all inner physical
dimensions, including the stick dimension.  It does not include outer physical
dimensions because those choose separate base regions; each outer coordinate is
checked as its own candidate when it varies.

Example:

```text
device_size = [A, B, C, stick]
```

- checking `A`: span is `A * B * C * stick * itemsize`;
- checking `B`: span is `B * C * stick * itemsize`;
- checking `C`: span is `C * stick * itemsize`.

The current planner uses `core_split_estimate=1` for candidate detection and
post-tile validation.  This models the worst case where Work Division provides
no additional help for that coordinate.

## Candidate Collection

Each overflow becomes a `SpanOverflowCandidate` containing a `ChunkingInfo`:

```python
ChunkingInfo(
    total_bytes=...,
    per_core_span=...,
    core_split_estimate=1,
    selected_device_dim_size=...,
    selected_device_span_stride_elems=...,
    selected_host_dim=...,
    stick_elems=...,
    reason="output span overflow" or "input span overflow for arg0",
)
```

Candidates can come from:

- output layout span analysis;
- reduction input span analysis;
- BMM input span analysis.

`_candidate_host_dims` merges candidates by host dim and orders dims by
decreasing span pressure.  This ordering affects the bounded combo search when
costs tie.

## Split Candidates and Cost Search

For each candidate host dim, `_split_candidates_for_host_dim` enumerates legal
split counts from exact divisors of `op.data.ranges[host_dim]`.

A split candidate is legal only if:

- it divides the selected host range exactly;
- it does not cut through physical sticks in the output layout;
- it does not cut through physical sticks in any fixed-layout input dependency
  that is addressed by the same output symbol.

The input-layout stick check matters for BMM/transposed operands.  A split can
look safe on the output layout but misalign an input tensor whose physical stick
dimension maps differently.

The search is bounded:

```python
_MAX_TILE_DIMS = 3
_MAX_SPLITS_PER_DIM = 16
_MAX_TILE_COMBOS = 512
```

If more than `_MAX_TILE_DIMS` host dimensions overflow, the planner raises
`Unsupported`.  Split candidates per dim are capped before Cartesian product
construction, and combinations are tried in increasing cost order:

```python
cost = (
    product(split_counts),
    number_of_tiled_dims,
    max(split_counts),
    split_counts,
)
```

So the planner prefers fewer total tiles, fewer tiled dimensions, and smaller
maximum split counts.

## Post-Tile Validation

A combo is accepted only after rebuilding the hypothetical per-tile Spyre
layout and re-running span analysis.

```python
tiled_layout = _post_tile_layout_for_splits(
    op.layout,
    split_by_host_dim,
    op.get_name(),
)
```

`_post_tile_layout_for_splits` changes one or more host sizes, recomputes
contiguous host strides, and calls `_resize_device_layout` to rebuild the real
`SpyreTensorLayout` for that tile.

Then `_remaining_span_candidates_after_tile` checks:

- output spans using the rebuilt output layout;
- reduction/BMM input spans using the candidate split map.

Only if no output or relevant input spans remain above `MAX_SPAN_BYTES` does the
planner return a `SpanOverflowTilePlan`.

This is the key safety property: the accepted tile combination is validated
against the same kind of physical layout that coarse tiling will create.

## Multi-Level Plans

The planner can emit multiple output-range tile levels for one op:

```python
SpanOverflowTilePlan(
    levels=(
        SpanOverflowTileLevel(selected_host_dim=0, split_count=2),
        SpanOverflowTileLevel(selected_host_dim=1, split_count=4),
    ),
    ...,
)
```

Levels are emitted outer-to-inner by host dimension.  The adapter creates one
synthetic `DimHint` per level.  `coarse_tile` then stamps a multi-level
`CoarseTileInfo` and scheduler/codegen lower it as nested counted loops.

## Adapter and Coarse Tiling

`coarse_tile.span_overflow_groups(graph)`:

1. skips auto groups if `config.ignore_wsr_hints` or
   `config.ignore_span_overflow_hints` is enabled;
2. skips ops that already have user dim hints;
3. calls `plan_span_overflow_tile`;
4. maps each `selected_host_dim` to a concrete output loop symbol via
   `op_out_coords(op)`;
5. creates synthetic `DimHint`s with ids starting at
   `_SPAN_OVERFLOW_HINT_ID = 10000`;
6. returns coarse-tile groups in the same format as user hints.

From `coarse_tile` onward, automatic and manual hints share the same path:

- rewrite `op.data.ranges` and `op.layout.size` per tile;
- stamp `CoarseTileInfo`;
- wrap scheduled nodes in `CountedLoopSchedulerNode`;
- emit `LoopSpec` in codegen.

## Configuration

Automatic span-overflow hints are suppressed when:

```python
config.ignore_span_overflow_hints == True
```

or when the broader working-set-reduction hint path is disabled:

```python
config.ignore_wsr_hints == True
```

User-authored `spyre_hint` groups take precedence per op.  Automatic hints are
not added to ops that already carry user dim hints.

## Failure Policy

The planner returns `None` for unsupported-but-safe-to-ignore cases such as
ineligible op types, symbolic layouts, indirect reads, scalar reductions, or no
span overflow.

It raises `Unsupported` when it detects overflow but cannot represent a safe
automatic output-range tile plan.  Common reasons:

- too many overflowing host dims for the bounded search;
- selected host dim is not in `op.data.ranges`;
- selected range is size 1 or non-integral;
- no legal exact divisor exists;
- stick alignment rejects all candidates;
- `_resize_device_layout` cannot reconstruct the post-tile layout;
- every tried combination still leaves output/input spans above the limit.

These failures are deliberate.  They avoid silently emitting a plan that still
violates the hardware span limit.

## Known Limitations

- Reduction-range tiling is not implemented.  Input spans controlled only by
  reduction symbols are skipped.  If such a skipped span still exceeds the
  hardware limit, downstream Work Division currently logs a critical overflow
  diagnostic but does not raise before backend compilation.
- Scalar/full reductions are skipped.
- Indirect/gather/scatter-style Pointwise and Reduction ops are skipped because
  they require the indirect-access SDSC path.
- The combo search is bounded by `_MAX_TILE_DIMS`, `_MAX_SPLITS_PER_DIM`, and
  `_MAX_TILE_COMBOS`; very high-rank overflow cases can raise `Unsupported`.
- Symbolic layout metadata is skipped because exact divisibility and post-tile
  Spyre layout validation require concrete sizes.
- Automatic groups are per-op.  Loop fusion/coalescing of similar automatic
  groups is future work.

## Validation

Main unit/codegen coverage lives in:

```text
tests/inductor/test_span_overflow_hint_analysis.py
```

Run:

```bash
PYTHONPATH=torch-spyre /home/hemaprasanna/dt-inductor/.venv/bin/python \
  -m pytest -q torch-spyre/tests/inductor/test_span_overflow_hint_analysis.py
```

Current coverage includes:

- no-op behavior under span limit;
- Pointwise output span tiling;
- Reduction output span tiling;
- Reduction/BMM input span tiling;
- scalar reduction skip;
- indirect-read guards for Pointwise and Reduction;
- reduction-controlled span known limitation;
- BMM reduction-symbol validation;
- input-layout stick alignment;
- multi-level plan generation;
- adapter and `coarse_tile` stamping;
- codegen `LoopSpec` tests for Pointwise, Reduction, and BMM/LM-head shapes.

Runtime before/after repro scripts live under `span_overflow_hints/`, including:

```text
repro_lm_head_bmm_span_issue.py
repro_6d_reduction_span_issue.py
repro_workdivision_needs_span_overflow_pass.py
```

Use `SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS=1` to simulate before-pass
behavior and `SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS=0` to enable automatic
span-overflow hints.

## Key Files

| File | Role |
|---|---|
| `torch_spyre/_inductor/span_overflow_hint_analysis.py` | Candidate collection, combo search, post-tile validation, tile-plan dataclasses |
| `torch_spyre/_inductor/coarse_tile.py` | Adapter from tile plans to synthetic `DimHint`s; coarse-tile IR stamping |
| `torch_spyre/_inductor/passes.py` | Combines user hint groups and automatic span-overflow groups |
| `torch_spyre/_inductor/ir.py` | Spyre layout resize/reconstruction helpers |
| `torch_spyre/_inductor/work_division.py` | Span limit constants and downstream per-core span diagnostics |
| `tests/inductor/test_span_overflow_hint_analysis.py` | Main unit/codegen coverage |
