---
name: generalize-view
description: "Diagnosing and fixing non-stick/strided view support on Spyre: why x[:, ::3, :]-style slices get rejected or silently compute wrong results, the three root causes, and the three fix strategies (pad+reorder, loop, deeptools modulo-wiring)."
---

# Generalizing Non-Stick View Support

This skill covers one specific, deeply-investigated class of bug: **strided
slices on a non-stick dimension** (e.g. `x[:, ::3, :]`) either get rejected
outright at compile time, or — worse — compile cleanly and **silently
compute wrong results**. It documents the confirmed root causes, the fix
strategies that were actually built and tested, and exactly what state the
code is in.

## Reproducing it

```bash
# From the workspace root (one level above torch-spyre/)
.venv/bin/python -u debug_nonstick_view.py --rows 97 --step 3 --trace-only
```

`--rows 97 --step 3` is the originally-rejected case (`x[:, ::3, :]` on a
97-row dim). `--rows 96 --step 3` is a sibling case that **compiles but
silently computes wrong numbers** — do not treat a clean compile as proof
of correctness for this bug class. The `SPYRE_EXPERIMENTAL_NONSTICK_LOOPS=1`
env var enables the opt-in loop-based prototype described below (must be
`export`ed, not prefixed on a piped command, or it silently won't apply to
the python process — `VAR=1 yes c | python3 ...` only sets it for `yes`).

## The three distinct root causes — don't conflate them

| # | Symptom | Where | Status |
|---|---|---|---|
| 1 | **Silent wrong answers** on a compiling case | `views.py`, `normalize_coordinates`: a stride's leftover "gap" dimension was recorded as a bare, variable-less constant (`Term(None, None, None, None, stride)`). Downstream bookkeeping that tracks axes by variable name silently loses an unnamed axis. | **Fixed locally** (see below) — mint a real synthetic variable (range 1) for the gap term even when `residual_offset == 0`, matching the pattern already used for the nonzero-offset case. |
| 2 | **False-positive rejection**: `Mod(k*var, r)` gets generated (and then rejected by `Term.from_coordinate`) for cases that can never actually overflow | `views.py`, `add_term`'s boundary check used `limit = term.xreplace({var: range_val})` — `range_val` is one past the variable's true max, an off-by-one over-conservatism. | **Fixed locally** — use `range_val - 1` (+1 to keep the exclusive-bound convention the `<` comparisons already assume). |
| 3 | **Genuine hardware limit**, not fixable in software: one descriptor requires the axis size to divide evenly by the step (`97 % 3 != 0`) | `views.py:~726`, `"...which does not divide it"`; independently confirmed at the hardware-descriptor level in `deeptools`'s `AgenToSentient`/`AccessDetails.cpp` ("hyper-rectangular form" requirement, constant-stride-only `AccessDetailsAffine`). | **Not a bug** — arithmetic fact. Fixes #1 and #2 only ever get you to this *correct* rejection when the axis genuinely doesn't divide evenly; they don't remove it. |

**Current local state of fixes #1 and #2:** both are live, uncommitted edits
in `torch_spyre/_inductor/views.py`, marked with `claude-check` comments.
Verified by real compile-and-run (`debug_nonstick_view.py`) to fix the
96-row silent-wrong-answer case and to correctly surface cause #3 (rather
than a wrong `Mod` rejection) for the 97-row case. Not run through the full
`pytest` suite — do that before trusting these broadly.

## Why `Term` can't just be extended to accept `Mod`/`FloorDiv` directly

`Term(num, den, var, mod, dim_size)` → `num * ((var % mod) // den)` is not
an abstract convenience — it **is** the one instruction the hardware
address generator supports: *start here, step by a fixed amount, repeat a
fixed number of times.* Plain add/multiply terms map onto this directly.
`Mod`/`FloorDiv` don't — wrapping needs an occasional backward jump,
floor-division needs long stretches of no movement followed by a sudden
jump; neither is a constant per-tick step. That's the physical reason none
of the fixes below touch `Term`'s shape — they all route *around* it.

## The three fix strategies

### (a) Pad + reorder the physical layout — proven, works today with the local patches

Allocate the strided axis at the next multiple of the step (`97 → 99`), and
place it outermost relative to whatever axis it could otherwise carry into
(e.g. row outside batch, not nested inside it). Once the axis size divides
evenly, `add_term` never generates a `Mod` at all — the coordinate comes
out as a plain `num*var` term, which `Term` already accepts natively.

- Verified end-to-end (real compile + run, correct numbers) by manually
  pre-transposing+padding the input tensor at the PyTorch level before
  `.to("spyre")`, letting the *existing*, unmodified layout-assignment
  machinery produce the right `device_size`/`stride_map` automatically.
- No masking needed: the padding rows are never addressed by the compiled
  kernel at all (the grouping math never generates a reference to them) —
  this is pure allocation padding, not a pad-and-mask pattern.
- Cost: at most `step - 1` extra rows; a relayout/copy only if the tensor's
  layout isn't otherwise free to choose (shared with another consumer
  expecting the original shape).
- Real integration point: the existing pluggable layout-assignment
  machinery (`LAYOUT_SOLVER` env var, `SpyreTensorLayout`/device-layout
  choice during upload) — teach it this one extra candidate rule.

### (b) Loop over blocks — proven for the Mod-wrapped case, single-core only

Resolve the problematic variable (e.g. `c1`) to one concrete number via
ordinary loop-counter arithmetic *before* anything reaches `Term`, so
`Mod(3*c1, 97)` collapses into a plain number per iteration rather than
ever being handed to `Term` as a formula. `Term` only ever sees the one
genuinely simple leftover piece (one 64-element stick, plain stride 1).

- Already exists as an opt-in prototype: `torch_spyre/_inductor/nonstick_loops.py`
  (`lower_nonstick_add`, gated by `SPYRE_EXPERIMENTAL_NONSTICK_LOOPS=1`).
  Verified correct, end-to-end, for the originally-rejected 97-row case.
- Current gaps to generalize: trigger condition is narrow (only fires on
  "Mod wrapping a Mul" — should instead fire on the real condition, the
  point where `normalize_coordinates` raises the divisibility
  `Unsupported`); scoped to `op=="add"` only (the mechanism itself is
  op-agnostic for pointwise ops; reductions need an extra
  combine-across-blocks step); requires exactly 3 axes and full output
  sticks (prototype guardrails, not fundamental limits).
- **The real remaining cost**: runs on a single core today (explicitly
  documented in the prototype's own design notes). No multi-core
  work-division across the block loops yet — a genuine performance gap to
  close before this is competitive as a default path, separate from
  correctness.
- Known separate, pre-existing issue: processes don't exit cleanly after a
  passing run (SIGSEGV during Python finalization in
  `JobPlanStepHostCompute` destruction) — documented as unrelated to
  numerical correctness; don't mistake it for a regression.

### (c) Wire deeptools' existing modulo arithmetic into the address path — partially built, not proven end-to-end

`deeptools` already has a general arithmetic engine (`VariableDefinition`,
with real `MOD`/`DIV`/`DIV_CEIL` operators) capable of this math directly —
confirmed by writing and running real unit tests against the exact
`(3*c1) mod 97` scenario. The gap: the MLIR-expression translator that
would carry a formula into that engine
(`deeptools/dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp`, `processExpr`)
only had branches for `Add`/`Mul`, not `Mod`/`FloorDiv`.

- We added `Mod`/`FloorDiv` handling there (a `resolveModOrFloorDiv`
  helper, reused both for a bare top-level case and for the shape MLIR's
  own canonicalizer actually produces — a `FloorDiv` nested inside a
  `Mul`'s operand, from the identity `x mod m = x - m*floordiv(x,m)`).
  Live, uncommitted, marked with `claude-check` comments in
  `deeptools/dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp`.
- Confirmed: builds clean, fixed a real crash (wrong `if/else-if` nesting
  calling `.getKind()` on a null cast result), fixed a real silently-dropped
  term (the orphaned-constant shape), produces a well-formed symbol graph
  for a synthetic test case.
- **Not confirmed**: full numeric correctness at actual runtime — needs
  live instrumentation of the real per-iteration values feeding the
  dynamic-loop-symbol resolution (`DataConvertInfoGenerate.cpp`,
  `DCIVarSubs::run`), or native debugging, neither completed.
- **Not started at all**: the torch-spyre side. `Term`/coordinate emission
  still rejects `Mod(k*var, r)` outright rather than emitting a real MLIR
  `affine.apply` with a `mod`/`floordiv` operator for deeptools to consume.
- If finished, this is the only strategy needing **no padding and no loop
  overhead** — one descriptor, full multi-core work-division preserved —
  but it is also the least mature of the three.
- Watch the C++ integer division/remainder semantics for negative operands
  (truncating, not floor) before trusting this generally — flagged, not
  yet resolved.

## Decision rule

Prefer (a) when the tensor's layout is free to choose and padding is
cheap. Fall back to (b) when the layout is shared/fixed elsewhere or
padding isn't practical — accepting the single-core cost until its
work-division gap is closed. Treat (c) as a future investigation, not a
currently-available option.

## Where to look next, by file

- `torch_spyre/_inductor/views.py` — `Term`, `Term.from_coordinate`,
  `add_term`, `normalize_coordinates`, `align_tensors_pure` (the gap-realize
  logic, rank-equalization/padding).
- `torch_spyre/_inductor/pass_utils.py` — `compute_coordinates`,
  `alignment_coordinates`, `device_coordinates`.
- `torch_spyre/_inductor/nonstick_loops.py` — the loop-based prototype.
- `torch_spyre/_inductor/codegen/superdsc.py`,
  `torch_spyre/_inductor/codegen/compute_ops.py` — the real,
  symbol-keyed (not positional) consumer of `device_size`/
  `device_coordinates`/`work_division`.
- `torch_spyre/csrc/job_plan.cpp` — the native bridge into
  `deeptools::processComputeOnHostCommand`.
- `deeptools/dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp` — the
  MLIR-affine-expression → `VariableDefinition` translator.
- `deeptools/util/variabledefinition/VariableDefinition.{h,cpp}` — the
  arithmetic engine (`MOD`/`DIV`/`DIV_CEIL` already implemented here).
- `deeptools/dcc/src/Conversion/AgenToSentient/AccessDetails.cpp` — the
  real hardware-descriptor legality check (root cause #3).

For the full plain-language walkthrough of the reasoning behind all of
this (mixed-radix digits, why a hardware descriptor is "start, step,
repeat," why `Mod`/`FloorDiv` don't fit it), see the published explainer
and slide deck generated alongside this investigation — ask whoever shared
this skill with you for the links, or regenerate equivalents from this
skill's content.
