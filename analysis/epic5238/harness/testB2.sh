#!/bin/bash
# Test B2: our pivot applied ON TOP of PR #5804's own branch head.
# No conflict resolution: #5804 is taken exactly as CaiwuChen wrote it.
# Control = #5804 alone on the same base (logs/head5804.txt from earlier).
set -u
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
export GIT_AUTHOR_NAME=retest GIT_AUTHOR_EMAIL=retest@localhost
export GIT_COMMITTER_NAME=retest GIT_COMMITTER_EMAIL=retest@localhost
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
R=$DTI_PROJECT_ROOT/torch-spyre
P=$A/pivot-tip

# wait for Test A2 to finish first: one device, no concurrent runs
while pgrep -f "[t]estA2.sh" >/dev/null 2>&1; do sleep 20; done

source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH

W=$A/pivotB2
rm -rf $W; git -C $R worktree prune
git -C $R worktree add -q --detach $W refs/retest/pr5804 || { echo "worktree failed"; exit 1; }
echo "pivotB2 base = $(git -C $W rev-parse --short HEAD) (PR 5804 head, unmodified)"

for f in torch_spyre/_inductor/config.py torch_spyre/_inductor/customops.py \
         torch_spyre/_inductor/lowering.py torch_spyre/_inductor/passes.py \
         torch_spyre/_inductor/stick_reshape_pivot.py \
         tests/inductor/test_inductor_fx_passes.py; do
  cp $P/$f $W/$f || { echo "copy failed: $f"; exit 1; }
done

# the one views.py line Test A proved is load-bearing, applied to THEIR views.py
python3 - "$W/torch_spyre/_inductor/views.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "            and fused_term.den == term.mod\n"
add = ("            # padded inner dim: fusing would read across the gap\n"
       "            and fused_term.den == term.den * term.dim_size\n")
if add in s:
    print("views.py: condition already present")
elif anchor in s:
    open(p, "w").write(s.replace(anchor, anchor + add, 1))
    print("views.py: padded-dim condition applied to PR 5804's views.py")
else:
    print("views.py: ANCHOR NOT FOUND -- running without the condition")
PY

cp $A/torch-spyre/torch_spyre/_C.so $W/torch_spyre/

mkdir -p $A/runs/pivotB2 && cd $A/runs/pivotB2
PYTHONPATH=$W:$BASEPP timeout 1800 python -u $A/retest.py > $A/logs/pivotB2.log 2>&1
echo "pivotB2 rc=$? OK=$(grep -c '^OK ' $A/logs/pivotB2.log) ERR=$(grep -c '^ERR ' $A/logs/pivotB2.log) BAD=$(grep -c '^BAD ' $A/logs/pivotB2.log) done=$(grep -c '^DONE' $A/logs/pivotB2.log)"

echo "=== unit tests on PR5804 + pivot ==="
PYTHONPATH=$W:$BASEPP timeout 900 python -m pytest -q --no-header -p no:cacheprovider \
  --rootdir=$W $W/tests/inductor/test_inductor_fx_passes.py 2>&1 | tail -6 | cut -c1-170

echo "=== CONTROL: #5804 alone on same base (earlier run) ==="
tail -3 $A/logs/head5804.txt 2>/dev/null || echo "(no earlier control log found)"
echo "B2 DONE"
