# Design Proposal: Extending Non-Stick View Support

**Status:** Draft for architecture review; not an implemented general solution.

**Issue:** [torch-spyre #1353](https://github.com/torch-spyre/torch-spyre/issues/1353)

**Source baseline:** `8fce19f7bab0f68ccbc9139bd23923b9b7ece9ef`

**Date:** 2026-10-01

## 1. Proposal

Identify unsupported non-stick view patterns and extend the existing handling in
`views.py`, preserving correct access to the original allocation without requiring
a physical layout conversion. Some patterns already work; the first task is to
separate those from failures and identify the exact restriction for each failure.
This follows the direction discussed with Yohei: begin with the existing view
machinery rather than assuming a new execution mechanism is required.

The target includes non-divisible stepped slices and genuine non-unit-fraction
coordinate expressions. These are related but distinct. Fixing the first does
not establish support for the second or for every operation in issue #1353.

The preferred solution extends coordinate conversion, normalization, and alignment
while reusing supported backend descriptors. Compiled loops and materialization
are alternatives investigated, not selected implementation requirements. They
should be proposed separately only after a concrete case demonstrates that the
existing view/descriptor mechanism cannot express the required addresses.

This document proposes an investigation and implementation sequence. It does not
yet establish that all non-stick patterns can be solved entirely in `views.py`.

## 2. Problem and terminology

A **view** describes how to access existing tensor data without necessarily
moving it. A **stick** is the innermost physical storage unit; our FP16 examples
use 64 elements per stick. A **non-stick axis** is a physical device axis outside
that innermost unit. Its position may be outermost or inside other physical axes.
A logical PyTorch axis may split across physical axes, so logical axis position
alone cannot determine whether an access is stick-preserving.

A **stick-preserving access**, for this proposal, reads a complete contiguous
stick at a correctly aligned address. The initial path does not support arbitrary
reordering, stepping, or partial validity inside that stick. Broadcast reads need
separate validation and are not implied by this definition.

A **non-unit-fraction coordinate** has a ratio whose reduced numerator and
denominator are both greater than one, for example `floor(3*t / 2)`. It is an
integer coordinate calculation, not a fractional memory address. The current
normalization can reject such expressions.

### Running example: a non-divisible stepped slice

```python
def f(x):
    return x[:, ::3, :] + 1

# Input: [2, 97, 256], FP16
# Output: [2, 33, 256]
```

Each batch selects input rows `0, 3, 6, ..., 96`. The output row counter `r`
ranges from 0 to 32; it is not the constant 33. The original input still has
97 physical rows per batch.

For a contiguous logical input, the read index in element units is:

```text
24832*b + 768*r + c
24832 = 97*256; 768 = 3*256
```

For the example device layout `[row, column_block, batch, lane]`, with sizes
`[97, 4, 2, 64]`, the correct device element offset is:

```text
(3*r)*512 + (c//64)*128 + b*64 + c%64
```

The stride map used to interpret the original index is distinct from these
physical device strides. Neither should be replaced by strides inferred from
the smaller output shape.

The current conversion produces `Mod(3*r, 97)` and a batch carry term for this
case. The modular expression is rejected before gap-dimension realization.
Here, bounds prove that `3*r <= 96`, so neither wraparound nor a carry is needed.
After that simplification, the existing gap realization still rejects the
non-divisible physical extent `97 % 3 != 0`.

### Different example: a real boundary crossing

```python
y = x.reshape(194, 256)[::3] + 1
```

Now the selected flattened row is `3*t`, and its original coordinates are:

```text
batch = (3*t)//97
row   = (3*t)%97
```

These divisions and remainders are necessary. In the layout above, consecutive
selected stick bases normally advance by 1536 elements, but crossing from batch
0, row 96 to batch 1, row 2 changes the base from 49152 to 1088 for column block 0.
One constant physical increment cannot express the whole sequence.

The crossing alone does not require extra loop regions for every layout. With
physical layout `[batch, row, column_block, lane]`, this flattened selection has
a constant physical increment. Decisions must use the final address mapping.

## 3. Existing implementation and limitations

| Stage | Existing mechanism | Limitation relevant to this proposal |
|---|---|---|
| Coordinate conversion | `compute_coordinates` in [views.py](../../../torch_spyre/_inductor/views.py) | Range reasoning can introduce unnecessary wrap/carry for stepped accesses. Genuine crossings must remain exact. |
| Normalization | `Term.from_coordinate` and `normalize_coordinates` in [views.py](../../../torch_spyre/_inductor/views.py) | Restricted coefficient/modulo forms; floor removal relies on those restrictions; gap realization requires divisibility in relevant cases. |
| Tensor alignment | `align_tensors_pure` in [views.py](../../../torch_spyre/_inductor/views.py) | Division and modulo determine split boundaries. Relaxing parser checks alone does not make reconstruction correct. |
| Direct descriptors | `_create_sdsc_tensors` in [superdsc.py](../../../torch_spyre/_inductor/codegen/superdsc.py) | Offsets and `backGap` represent specific storage relationships, not arbitrary access formulas. |
| Compiled loops | `TensorArg.device_tile_advance_expr`, `LoopSpec`, [compute_ops.py](../../../torch_spyre/_inductor/codegen/compute_ops.py), [bundle.py](../../../torch_spyre/_inductor/codegen/bundle.py) | Existing emission uses per-operand constant byte strides for each loop level. It is not an arbitrary division/modulo address evaluator. |

Much of the required information already exists in tensor arguments and symbolic
coordinates. This proposal extends those mechanisms. It does not assume a new,
parallel layout representation is necessary.

## 4. Scope and correctness requirements

### Initial investigation and delivery scope

- Static, positive-step non-stick accesses, including inner and outer physical axes.
- Divisible and non-divisible extents and nonzero starts.
- Minimal pointwise consumers to force actual input reads and isolate view failures.
- Separate tests for genuine fractional expressions and required boundary crossings.
- Original allocation extents and independent input/output addresses preserved.

Whole-stick FP16 cases provide the first controlled reproducers. This is a test
starting point, not a claim that all non-stick support should be permanently
restricted to FP16 pointwise operations. Broaden consumer, dtype, and layout
coverage before claiming general support. The HBM-only restriction belonged to
the earlier loop prototype; it is not an inherent restriction on view handling.

Dynamic shapes, data-dependent indexing, reverse traversal, and within-stick
rearrangement are outside the initial delivery. Partial sticks require explicit
validation of existing tail handling or a separate extension. Classify accesses
from device coordinates, not merely from the logical axis position.

### Invariants

1. Every generated read selects the same logical element as PyTorch.
2. Every address uses the original allocation's physical layout and stays valid.
3. Input and output strides are independent; selected extents cannot redefine
   the original allocation's outer strides.
4. Required integer division, remainder, offsets, and split structure survive
   until an equivalent lowering has been established.
5. Regions cover the intended output exactly once unless the operation's
   semantics explicitly require otherwise; tails and empty domains are explicit.
6. Unsupported combinations reject clearly rather than silently ignoring terms.
7. Reordering or materializing accesses must preserve dependencies and observable
   aliasing/mutation behavior. Read-only support does not imply write support.

## 5. Proposed changes and investigation

### A. Establish the working/failing pattern matrix

Record the source layout, logical operation, generated coordinates, first failing
stage, and expected addresses for each minimal case. A slice must have a real
consumer, such as addition, so returning an alias alone cannot hide the failure.

| Pattern | Current evidence | Next action |
|---|---|---|
| 96 rows, step 3, example outer layout | Coordinate conversion produces accepted `3*r`. This alone is not an end-to-end pass. | Retain as a control and rerun on device. |
| 97 rows, step 3, same layout | Conversion produces rejected `Mod(3*r,97)`. | Prove the valid bounds and test an exact simplification. |
| Coordinate `3*r` on physical extent 97 | Gap realization has an explicit non-divisibility rejection. | Determine an exact representation preserving outer strides. |
| Genuine non-unit ratio, such as `floor(3*t/2)` | The normalization contract excludes general non-unit ratios. | Add a minimal graph-level reproducer and verify the exact rejection path. |
| Flattened rows crossing original batches | Correct mapping contains division/remainder. | Verify support or failure for each layout; never remove required carry. |
| Nonzero starts and inner-axis stepping | Prior restricted prototype covered examples. | Establish current direct-path behavior independently. |

### B. Correct coordinate bounds without changing required semantics

In `compute_coordinates`, reason about the last executed iteration rather than an
imaginary next iteration. For the running example, `r <= 32` proves `3*r <= 96`.
The complete expression, including offsets and other counters, must be checked;
changing only `step*range` is not a general proof. Unknown bounds do not authorize
simplification. Preserve necessary quotient and remainder terms for genuine
crossings.

### C. Extend normalization and alignment together

Use existing tensor sizes, coordinate expressions, and `Term` information as the
starting point. Investigate the non-unit coefficient restriction, supported modulo
forms, and gap-dimension realization as separate limitations. Do not just remove
the checks: the current floor removal and dimension reconstruction rely on them.

For each proposed extension, show that the normalized representation reconstructs
the same physical addresses, including original outer strides and offsets.
Preserve division/modulo structure used by `align_tensors_pure` to choose split
boundaries. If the current `Term` cannot carry a required expression, document
that precise representational gap before adding fields or another representation.

For non-divisible steps, replacing `97` with `ceil(97/3)*3` would alter the apparent
physical extent and can corrupt outer strides. It is not an acceptable generic
fix. The open design question is how to preserve the physical extent and selected
iteration count independently through alignment and descriptor generation.

### D. Validate the existing descriptor path

Retain dimension splitting, offsets, and gap handling where they preserve all
addresses exactly. Confirm generated descriptors against an independent address
oracle and device results, especially for inner axes.

`backGap` is a mechanism for particular storage relationships, not proof that an
arbitrary stepped access is representable. Host-copy slice calculations in
[spyre_mem.cpp](../../../torch_spyre/csrc/spyre_mem.cpp) provide reference logic but have a
different contract. Any reuse needs validation for device consumers.

The outcome for each pattern must be one of: supported by an exact extension;
blocked by an identified descriptor limitation; or still under investigation.
Do not interpret a failing current normalization as proof of a backend limitation.

### E. Alternatives investigated, if view handling is insufficient

These alternatives are not part of the initial selected implementation:

- **Compiled loops:** existing `device_tile_advance_expr` and `LoopSpec` can express
  independent constant input/output advances. Boundary regions can have different
  starting addresses. A prior restricted prototype demonstrated this approach.
- **Repeating-pattern decomposition:** `floor(3*t/2)` can be split using `t=2*q`
  and `t=2*q+1`, giving `3*q` and `3*q+1`. This requires preserving output positions,
  proving safe execution ordering, coordinating all operands, and limiting region
  growth. Arithmetic equivalence is not backend validation.
- **Richer address emission:** retain full integer address calculations if the
  downstream compiler supports them. Current loop emission extracts constant
  strides; general division/modulo support has not been established.
- **Internal materialization:** copy selected data into supported storage for an
  incompatible consumer. This still requires a correct read path and must preserve
  aliasing, lifetimes, and mutation behavior. It is not a pure-view solution.

Any fallback needs separate review of coarse-loop composition, work ownership,
memory planning, code size, and performance. The earlier late loop rewrite avoided
several of these interactions by rejecting them and should not be adopted unchanged.

## 6. Evidence and its limits

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Current-source inspection and isolated coordinate execution | The 97-row example generates a rejected modular coordinate; later normalization has additional restrictions. | A complete fix or broad device support. |
| Historical opt-in Torch-Spyre prototype, before the reclone | Twelve numerical device comparisons matched CPU for selected FP16 pointwise cases, including inner/outer axes and nonzero starts. Existing loop machinery can execute this restricted class without deeptools changes. | General fractional views, good performance, or clean process completion. Processes crashed during shutdown; a control reproduced the crash. The prototype is absent from the current checkout. |
| Standalone address experiment | 648 slice/layout combinations and 5,139,840 element addresses matched an independent reference; 14 fractional/offset cases were checked at 257 indices each. | Compiler integration, arbitrary symbolic mappings, or backend execution of the proposed regions. |

The standalone checks must become reproducible repository tests before being
used as PR acceptance evidence. Re-establish device results on the current
compatible dependency stack; historical logs are supporting context only.

A column-offset regression is already documented as fixed in
[test_copy_from_d2d_offsets.py](../../../tests/inductor/test_copy_from_d2d_offsets.py), while
some lowering comments still describe it as broken. Retain and rerun the test;
do not treat an old comment as proof of a current silent miscompile.

## 7. Implementation sequence and review gates

| Phase | Deliverable | Gate |
|---|---|---|
| 1. Characterize | Minimal graph and coordinate tests with supported controls; classify each rejection. | Actual failing stage and expected addresses documented. |
| 2. Fix bounds | Proven elimination of unnecessary wrap/carry in `compute_coordinates`. | Required crossings remain intact; subsequent failures are exposed rather than hidden. |
| 3. Extend view representation | Proposed normalization/alignment changes for non-divisible extents and fractional patterns. | Original physical addresses and split structure remain exact; no check is relaxed without downstream validation. |
| 4. Validate descriptors and consumers | Device tests for inner/outer axes, offsets, and composed views; regression and performance checks. | Explicit support matrix and clean end-to-end results. |
| 5. Review remaining gaps | Evidence of any access the descriptor path cannot represent, with a concrete fallback proposal if needed. | Maintainer agreement before selecting loops, richer emission, or materialization. |

The phases may be separate PRs under one design. The first PR must not claim all
of #1353. Backend changes are outside this proposal; any required interface
extension needs a concrete example and separate review.

## 8. Validation and acceptance

- Compare generated addresses with an independent logical-to-physical oracle,
  including first/last elements, boundary transitions, padding, and bounds.
- Test divisible and non-divisible extents, nonzero starts, steps larger than an
  extent, empty/single-element selections, inner/outer axes, and layout permutations.
- Test reshape/slice composition, genuine ratios, multiple operands with different
  layouts, and multiple stepped axes as their support is introduced.
- Run end-to-end CPU/device comparisons using appropriate dtype tolerances;
  use exact address tests so FP16 rounding cannot conceal wrong-element reads.
- Preserve existing offset-copy, coordinate, descriptor, and coarse-loop tests.
- Include mutation/alias tests: selected writes update the intended locations and
  unselected values remain unchanged. Explicit rejection is required until a
  combination is supported; a read-path pass is not evidence for write support.
- Verify unsupported dynamic, reverse, within-stick, and consumer combinations
  fail clearly at the intended layer.
- Require clean execution and shutdown. Track unrelated runtime failures
  separately; numerical success alone is not a full end-to-end pass.
- Measure compile time, emitted region/code count, kernel/loop overhead, device
  latency, and temporary memory. Compare direct, loop, and copy paths where
  available. Do not enable an alternative fallback by default without performance data.

## 9. Decisions requested from reviewers

1. Does the pattern matrix capture the intended non-stick scope, and which
   additional failing view examples should be included before implementation?
2. Should `Term` be extended for the remaining expressions, or should original
   coordinates be retained through a distinct path within existing alignment?
3. What existing descriptor mechanism, if any, can represent non-divisible inner
   steps while preserving the original outer physical strides?
4. Which consumers, dtypes, offsets, and mutation cases are required before
   claiming the first extension is supported?
5. If a descriptor limitation is demonstrated, which alternative should be
   evaluated first, and what performance criteria should determine acceptance?

## 10. Relationship to stick-axis support

Yohei's suggested direction is to extend `insert_restickify` so that a stepped
stick-axis selection can be moved to a non-stick position, then handled as a view.
See the [issue discussion](https://github.com/torch-spyre/torch-spyre/issues/1353#issuecomment-5722215952).

That is a separate layout-conversion extension. The non-stick path should be
reusable after the conversion, but implementing non-stick access alone does not
implement stick-axis slicing. Layout selection can also avoid some difficult
access patterns initially; neither approach replaces the need to represent views
correctly on storage that already exists.
