---
name: stick-pivot-epic-5238
description: "Where the uncommitted stick-pivot fix for torch-spyre view epic #5238 lives, plus its evidence and status"
metadata:
  node_type: memory
  type: project
  originSessionId: 0eea9bcf-770d-4b0b-920d-af1245b6c115
  modified: 2026-11-02T04:02:45.278Z
---

The stick-pivot fix for epic #5238 (issues #5197, #5198, #5214, also #1353) is uncommitted, not in any PR.

- Worktree: /home/hemaprasanna/dt-inductor/analysis/yolo26/stick-pivot-fix/torch-spyre (branch `stick-pivot-fix`, base 42aae465; pr-integration base staged, fix unstaged, plus untracked `stick_reshape_pivot.py`).
- README.md, logs, repro/ and patches/ sit one level up from that worktree. Results as of 2026-10-27 are in the README's Epic #5238 section.
- The real checkout uses HOME=/home/hemaprasanna; Claude sessions run from /home/senuser, so earlier history may be in ~/.codex/sessions rather than Claude transcripts.

- In that checkout `origin` is the user's fork (Hemaprasannakc) and can be stale; compare against the `upstream` remote.
- As of 2026-11-02, draft PR #5804 (CaiwuChen, reviewed by yoheiueda) targets the same bug as the pivot's flatten case, root cause #5872. #5486 (merged) added fp32 restickify, so the pivot's fp16-only limit needs a retest.

- Retest of the issues' own cases (2026-11-02): /home/hemaprasanna/dt-inductor/analysis/epic5238-retest (`retest.py`, logs). Local backend stack dates from about Oct 15-16; upstream past 0798eda1 needs newer flex and libaiupti, so a true upstream-tip build isn't possible without updating it. `tip-overlay/` runs tip Python on the older `_C.so`.

**Why:** the chat that held this context was lost, and the work is invisible on GitHub.
**How to apply:** start from the README rather than re-deriving; verify status against the issues, since CaiwuChen self-assigned #5197 on 2026-10-26.
