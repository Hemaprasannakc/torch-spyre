# `_insert_read_copy_ops` fix attempt — what worked, what didn't

Reverted, not committed. Recorded so it isn't re-derived from scratch.

## 1. Shape calculation — WORKED

Replaces the positional walk. Goes in `_insert_read_copy_ops`, before
`tile_ranges` is built. Needs `full_layout = full_buf.layout` moved up.

```python
full_layout = full_buf.layout
full_size_ints = [int(s) for s in full_layout.size]
var_extent = dict(zip(dep.var_names, dep.size))

# coefficient in dep.index -> loop var
coeff_to_var = {}
for v in dep.var_names:
    coeff = dep.index.coeff(v)
    if coeff != 0:
        coeff_to_var.setdefault(sympy.sympify(coeff), v)

# per buffer dim: which loop var indexes it, and this tile's extent
dim_vars, tile_size_ints = [], []
for host_dim, full_size in enumerate(full_size_ints):
    var = (None if full_size == 1
           else coeff_to_var.get(sympy.sympify(full_layout.stride[host_dim])))
    dim_vars.append(var)
    tile_size_ints.append(full_size if var is None else int(var_extent[var]))

# copy buffer holds only indexed dims, in FULL_BUF's order (not var order)
kept_dims  = [d for d, v in enumerate(dim_vars) if v is not None]
copy_vars  = [dim_vars[d] for d in kept_dims]
tile_ranges = [sympy.Integer(tile_size_ints[d]) for d in kept_dims]
full_strides = [dep.index.coeff(v) for v in copy_vars]

def _copy_inner_fn(idx, _vars=tuple(copy_vars), _dep=dep,
                   _full_name=full_buf.get_name()):
    return V.ops.load(_full_name, sympy_subs(_dep.index, dict(zip(_vars, idx))))
```

Verified correct on both matmul operands, including the transposed one
(`b` is [B,H,K,N] against a b/m/n/k loop — positional walking swaps K and N,
this does not).

## 2. Per-iteration advance — ALSO NEEDED, still not sufficient

`read_level_extents` (~line 2531) keys by position in the *loop's* squeezed
order via `squeeze_pos[d]`. Once `copy_ranges` is in buffer order that is
wrong. Change to go through the loop var:

```python
var_to_copy_pos = {v: i for i, v in enumerate(copy_vars)}
...
copy_var = dep.var_names[squeeze_pos[d]]
copy_dim = var_to_copy_pos.get(copy_var)
if copy_dim is None:
    continue        # this input has no dim for that loop var
```

Same for the reduction-dim loop below it (`it_idx + reduction_squeeze_pos[d]`).

## 3. Result after BOTH changes

- assert cleared, compiles
- `LoopSpec count=1` — the join is real
- **numbers still wrong**: LM-head tiled mismatch 48727/49152 (99.1%)
  vs untiled 781/49152 (1.6%)

So a THIRD positional assumption remains. Not located. Prime suspects:
`_tiled_dims_for_dep`, `SpyreKernel._general_tile_advance` (positional
dep-index lookup), or the copy buffer's host size vs device_layout order.

## 4. The guard problem

Stride matching cannot identify dims when `dep.index` uses `Mod`/`FloorDiv`.
Failing closed with `Unsupported` there caused, in `test_coarse_tile_e2e.py`:

- 134 passed / 47 skipped  ->  75 passed / 105 skipped / 1 failed
- `test_outside_consumer_copy_then_read_512x256_A4_B4` failed
- 6 skip reasons quoted the new guard

I did not investigate why all 59 flipped — reverted first.

## 5. Suggested direction

The manual path never computes this layout: it leaves it blank pre-stickify
and stickification fills it in, correctly, for matmul. Getting the layout the
way stickification would may sidestep 1-4 entirely rather than fixing each.
