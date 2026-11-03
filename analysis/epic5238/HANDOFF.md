# Epic #5238 view ops — state as of 2026-11-03

## What the branch contains

Branch `stick-reshape-pivot`, commit on top of upstream `5268ba95`.
7 files, 594 insertions, 0 deletions:

| file | what |
|---|---|
| `_inductor/stick_reshape_pivot.py` | new post-grad FX pass |
| `_inductor/customops.py` | `spyre::rebase_copy` |
| `_inductor/lowering.py` | its lowering (realized `Pointwise`) |
| `_inductor/config.py` | `SPYRE_STICK_RESHAPE_ON_DEVICE` |
| `_inductor/passes.py` | registration, last in `CustomPostPasses` |
| `_inductor/views.py` | 5 lines: `normalize_coordinates` padded-dim fusion |
| `tests/inductor/test_inductor_fx_passes.py` | 9 tests |

## How the fix works

Permute a dim the reshape leaves alone onto the stick (restickify), do the
split/merge on the now non-stick dims where it is free, `rebase_copy` to
materialise, permute back, and materialise the outer merge if there is one.
Falls back to `spyre.reshape_via_cpu` in five situations (see the SKILL).

## Measured results (tip Python + Oct-15 `_C.so`)

| suite | before | after |
|---|---|---|
| 58-case shape matrix | 13 OK / 40 ERR / 5 BAD | 55 OK / 3 ERR / 0 BAD |
| 162-case epic suite | 50 OK / 107 ERR / 5 BAD | 78 OK / 84 ERR / 0 BAD |
| unit tests | - | 78 passed, 57 subtests |
| `test_restickify.py` | - | 749 passed |

The 3 remaining matrix failures are eager mode (2) and a non-contiguous
input (1) — all out of scope.

## Issues

Fixes **#5227** (the YOLO one: all four error forms, `5*c1/2`, `3*c1/2`,
`5*c1/4`, and the W<64 `no supported output layout`), **#5214** (21/21),
**#5224** (3/3), **#5320** (compiled; eager now raises instead of corrupting).

Refs #1353 (reshape done, `x[...,::3]` not), #5197 (compiled only),
#5321 (compiled only), #5872. Supersedes the host workaround in PR #5222.

## Still to do

1. **Open two issues**: depthwise conv2d rejects non-zero padding (YOLO26s's
   real blocker, no issue exists); and `restickify` restore misreads a
   multi-stick pivot intermediate (our `TODO(issue#TBD)` needs that number).
2. **Comment on #5804** (CaiwuChen): landing both as-is miscompiles
   `(1,2,4,8,4,7)->(1,8,224)`. Their `_wrap_and_project` yields `Mod(d1, 28)`
   on a dim of size 7; tip yields `Mod(d1, 7)`. Our 5 `views.py` lines are
   still needed on top of their fix (their PR alone: 53 OK; with ours: 78).
3. **Comment on PR #5222** that this supersedes its host workaround.
4. Untested gaps to state in the PR body: dynamic shapes, `MAX_SPAN_BYTES`,
   bf16 breadth, and the backend build (`_C.so` is from ~Oct 15 because
   newer `flex`/`libaiupti` are needed past `0798eda1`).

## YOLO26s

Does not compile, with or without this change. Every reshape it performs is
handled (4 on device, 8 via host). Its remaining blocker is
`Depthwise conv2d currently only supports zero padding; got padding=[1,1]`,
unrelated to reshape. The earlier "90 reshapes all on device" figure from
October is NOT reproducible — this run sees 12.
