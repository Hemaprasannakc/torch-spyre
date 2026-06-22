# Span-Overflow Hint Analysis

## Background

Spyre work division must keep each core's memory span within the hardware
addressing limit.  For large tensors, a pointwise operation can have a physical
device layout whose per-core span is still too large after normal
`work_division` splitting.  User-authored `spyre_hint` scopes can fix this by
asking coarse tiling to run the operation in multiple outer loop iterations.

`span_overflow_hint_analysis.py` is the compiler-generated version of that
decision for pointwise operations.  It does not transform the graph directly.
Instead, it answers one question:

> Does this pointwise `ComputedBuffer` need coarse tiling, and if yes, which
> output dimension and split count should coarse tiling use?

The result is adapted into the same coarse-tiling group format used by
`spyre_hint`, so the downstream pipeline stays shared:

```
span_overflow_hint_analysis
  -> span_overflow_groups adapter
  -> coarse_tile
  -> CountedLoopSchedulerNode
  -> LoopSpec codegen
```

## Scope

The initial pass is deliberately conservative:

- supports `ComputedBuffer` operations whose `data` is `Pointwise`;
- requires a `FixedTiledLayout`, because decisions are based on Spyre physical
  device layout;
- produces one coarse-tile level over one output dimension;
- raises `Unsupported` if the selected dimension cannot be tiled enough to make
  the post-tile layout safe;
- does not yet auto-tile reductions or reduction ranges.

This keeps the pass as a planner.  Coarse tiling owns mutation of
`op.data.ranges`, layout sizes, `CoarseTileInfo`, and scheduler/codegen loop
structure.

## Planner Output

The planner returns a `SpanOverflowTilePlan`:

```python
SpanOverflowTilePlan(
    selected_host_dim=1,
    split_count=5,
    is_reduction=False,
    chunking_info=...,
    reason="output span overflow",
)
```

For example, a pointwise add over shape `[1, 8195, 256, 64]` can produce:

```text
selected_host_dim = 1
split_count       = 5
```

The adapter converts that into a synthetic `DimHint` and group equivalent to a
manual user hint:

```python
with spyre_hint(num_tiles_per_dim={"H": 5}):
    return x + y
```

The coarse-tile group has the usual shape:

```python
[([op], [(hint_id, 5, False)])]
```

where `False` means this is an output-dimension tile, not a reduction-dimension
tile.

## Choosing the Controlling Dimension

The pass walks physical device dimensions from outer to inner, skipping the
final stick dimension.  For each device dimension, it uses `stride_map` to find
the corresponding logical host/output dimension.

For a shape like `[1, 8195, 256, 64]`, the physical layout may expose:

```text
host size   = [1, 8195, 256, 64]
host stride = [134266880, 16384, 64, 1]
device size = [8195, 256, 1, 1, 64]
stride_map  = [16384, 64, 64, -1, 1]
```

The first useful physical dimension has `stride_map=16384`, matching host dim
1, so the selected dimension is the `H`-like dimension of size `8195`.

Skipped outer device dimensions are kept only for debug context.  They do not
multiply the span calculation, because constant outer coordinates do not
increase the per-core address span seen by `work_division`.

## Span Estimate

The pass estimates the same quantity that `work_division` cares about:

```python
per_core_span = (
    ceil(selected_device_dim_size / core_split_estimate)
    * selected_device_span_stride_elems
    * itemsize
)
```

where:

- `selected_device_dim_size` is the physical extent of the selected device dim;
- `core_split_estimate` is the largest divisor of that extent no larger than
  `SENCORES`;
- `selected_device_span_stride_elems` is the product of all inner physical dims,
  including the stick dim;
- `itemsize` is the dtype size in bytes.

There is also a whole-tensor safety check:

```python
total_bytes > MAX_SPAN_BYTES * SENCORES
```

The required coarse-tile count is:

```python
required_count = max(
    ceil(per_core_span / MAX_SPAN_BYTES),
    ceil(total_bytes / (MAX_SPAN_BYTES * SENCORES)),
)
```

If `required_count <= 1`, the op is safe and no automatic hint is emitted.

## Split Count Selection

`coarse_tile` currently divides `op.data.ranges` by the loop count, so the
chosen split count must divide the selected host dimension exactly.  The planner
therefore rounds the required count up to the smallest divisor of the selected
dimension.

Example:

```text
selected host dim size = 8195
required_count         = 5
chosen split_count     = 5
per-tile host size     = 1639
```

If the required count is not itself a divisor, the pass tries the next larger
divisor.  If no divisor on the selected dimension can satisfy the constraints,
the pass raises `Unsupported` rather than emitting a plan that still overflows.

## Post-Tile Validation

The initial span estimate is not the final check.  Before returning a plan, the
pass rebuilds the per-tile `FixedTiledLayout` that coarse tiling will create and
runs the span check again on that layout.

For the `[1, 8195, 256, 64]` example:

```text
before tiling: [1, 8195, 256, 64]
split_count:  5
after tiling: [1, 1639, 256, 64]
```

Only if the post-tile physical layout passes the same span/total checks does the
planner return the split count.  This prevents the analysis from approving a
tile count that looks safe in the original layout but still violates the span
limit after Spyre layout reconstruction.

## Adapter and Coarse-Tile Consumption

The adapter lives in `coarse_tile.span_overflow_groups()`.

When user `spyre_hint` groups exist, they take precedence.  Automatic
span-overflow groups are only generated when no user coarse-tiling groups were
found.

For every returned plan, the adapter:

1. maps `selected_host_dim` to the concrete output loop symbol via
   `op_out_coords(op)`;
2. creates a synthetic `DimHint` with a private hint id;
3. attaches that hint to the op as `op.dim_hints`;
4. returns a group shaped like user-hint output:

```python
([op], [(hint_id, split_count, False)])
```

`coarse_tile()` then uses its normal path:

- resolves `hint_id` back to the op's tiled output dimension;
- divides `op.data.ranges` and `op.layout.size` by `split_count`;
- stamps `CoarseTileInfo` on the op.

For the example above, coarse tiling stamps:

```python
CoarseTileInfo(
    loop_group_id=(0,),
    loop_count=[5],
    loop_tiled_dims=[[1]],
    loop_tiled_reduction_dims=[[]],
)
```

and rewrites the per-tile shape to:

```text
ranges      = [1, 1639, 256, 64]
layout.size = [1, 1639, 256, 64]
```

## Scheduler and Codegen

After `coarse_tile`, the downstream layers do not know whether the loop came
from a user hint or automatic span-overflow analysis.

`build_loop_scheduler_nodes()` sees `CoarseTileInfo` and wraps the scheduled op
run in a `CountedLoopSchedulerNode(count=5)`.

`SpyreKernel` then emits a `LoopSpec`:

```python
LoopSpec(
    count=sympify("5"),
    body=[
        OpSpec(
            op="add",
            iteration_space={
                c0: (1639, 1),
                c1: (256, 4),
                c2: (64, 1),
            },
            tiled_symbols=[c1],
            ...
        )
    ],
    tiled_symbols=[c0],
)
```

This is intentionally the same loop structure produced by the equivalent
manual `spyre_hint`.

## Validation Strategy

Most validation should use small mock or patched-limit tests because true span
overflow usually requires large tensors.

Recommended coverage:

1. **Planner-level:** verify selected dim and split count.
2. **Adapter-level:** verify synthetic `DimHint` and coarse-tile group format.
3. **Coarse-tile IR-level:** verify rewritten ranges/layout and `CoarseTileInfo`.
4. **Scheduler/codegen-level:** verify generated source contains the expected
   `LoopSpec` count.
5. **One E2E smoke test:** use a real large pointwise tensor and compare Spyre
   output against CPU.



## Key Files

| File | Role |
|---|---|
| `torch_spyre/_inductor/span_overflow_hint_analysis.py` | Pointwise planner: choose selected dim and split count |
| `torch_spyre/_inductor/coarse_tile.py` | Adapter (`span_overflow_groups`) and coarse-tile IR stamping |
| `torch_spyre/_inductor/passes.py` | Invokes automatic span-overflow groups when no user hint groups exist |
| `torch_spyre/_inductor/scheduler.py` | Wraps stamped ops in `CountedLoopSchedulerNode` |
| `torch_spyre/_inductor/spyre_kernel.py` | Emits `LoopSpec` around generated `OpSpec` objects |
| `tests/inductor/test_span_overflow_hint_analysis.py` | Unit/codegen coverage for the planner-to-LoopSpec path |

## Current Limitations

- Only pointwise ops are planned automatically.
- Only one output dimension is tiled.
- Exact divisibility is required by the current `coarse_tile` range rewrite.
- If one selected dimension is insufficient, the pass raises `Unsupported`; it
  does not yet emit nested multi-dimensional tile plans.
- Reduction output-range tiling and reduction-range tiling are future work.

