---
name: epic5238-view-testing
description: Test and debug stick-layout reshape/view failures in torch-spyre (epic #5238). Use when working on unaligned reshapes, the stick_reshape_pivot pass, issues #5227/#5214/#5224/#5320/#1353, or when a reshape compiles but returns wrong values. Covers the shape matrix, the issue repros, and how to bisect a layout bug.
---

# Testing stick-layout reshape failures (epic #5238)

## The problem in one line

A reshape whose new innermost dim is not a multiple of the stick (64 fp16/bf16,
32 fp32) is not a view on Spyre: the destination pads every row up to a whole
stick, so source and destination are different byte layouts. Splits fail to
form a coordinate (`5*c1/2`); merges **silently return wrong values**.

## Stick size is dtype-dependent

`64` for fp16/bf16, `32` for fp32. So `W=160` is unaligned in fp16 but
**aligned** in fp32 (5 x 32). Never assert the same behaviour across dtypes.

## Harness

Run with the pass tree first on `PYTHONPATH`:

    PYTHONPATH=<tree>:$PYTHONPATH python -u harness/matrix.py    # 58 shape cases
    PYTHONPATH=<tree>:$PYTHONPATH python -u harness/retest.py    # 162 epic cases

Each prints one line per case: `OK` / `BAD` (compiled but wrong values) / `ERR`.
**`BAD` is the one that matters** — this family of bugs corrupts silently.
`matrix.py` takes case ids or category letters as argv filters.

Categories in `matrix.py`: A split, B merge, C must-be-left-alone, D no spare
dim (host path), E pivot selection, F dtypes, G consumers, H op spellings,
I must-be-skipped.

## Bisecting a layout bug

This sequence found three bugs in one day:

1. **Is the shape itself supported?** Run the op with no reshape
   (`probes/isolate_c5.py`). If `(8,8)*1.0` works but `(64,)->(8,8)` fails,
   it is a relayout bug, not a shape limitation.
2. **Vary one dimension** and find the boundary (`probes/probe_e1.py`).
   "P<=64 correct, P>64 wrong" is far more useful than "it is broken".
3. **Check the host path** (`spyre_config.patch(stick_reshape_on_device=False)`).
   If the CPU path is correct, the bug is in the on-device sequence.
4. **Isolate the primitive.** A plain `transpose(0,1).clone()` of the same
   shape being correct proves the backend restickify is innocent.
5. **Dump coordinates** (`probes/dump_coords.py`). Needs
   `torch._inductor.config.compile_threads = 1`, or compilation happens in a
   subprocess and the instrumentation never sees it. Patch
   `compute_coordinates` in `views.py` **and** in `pass_utils`/`padding`,
   which import it by name.

## Gotchas that cost time

- One device: never run two compile suites at once, the numbers get unreliable.
- `pgrep -f "[t]estX.sh"` matches the launcher's own command line. Wait on PIDs.
- `setsid nohup script &` reports the wrapper's pid, not the script's.
- `git diff HEAD` omits untracked files, so a new pass file is silently
  left out of a patch.
- Copying whole files between trees on different bases drags unrelated
  upstream changes along. Use a patch with `git apply --3way`.
- An fp16 `sum` over ~1000 elements misses `atol=1e-2` by ~17x on its own.
  Always run a no-reshape control before calling a sum mismatch a layout bug.

## Known limitations of the pass (by design)

Eager mode, non-contiguous inputs, dynamic shapes, `cat` on the stick dim,
a pivot spanning >1 stick with no outer dim, oversized intermediates, and
non-restickifiable dtypes all take `spyre.reshape_via_cpu`.
