#!/bin/bash
# Test B3: #5804 + our pivot, but WITHOUT our normalize_coordinates line.
# Answers: does CaiwuChen's _wrap_and_project already make our views.py line
# unnecessary? If the merge unit tests pass here, our views.py footprint is 0.
set -u
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
export GIT_AUTHOR_NAME=retest GIT_AUTHOR_EMAIL=retest@localhost
export GIT_COMMITTER_NAME=retest GIT_COMMITTER_EMAIL=retest@localhost
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
R=$DTI_PROJECT_ROOT/torch-spyre
P=$A/pivot-tip

while pgrep -f "[t]estA2.sh|[t]estB2.sh" >/dev/null 2>&1; do sleep 20; done

source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH

W=$A/pivotB3
rm -rf $W; git -C $R worktree prune
git -C $R worktree add -q --detach $W refs/retest/pr5804 || { echo "worktree failed"; exit 1; }
echo "pivotB3 base = $(git -C $W rev-parse --short HEAD) (PR 5804 head, views.py UNMODIFIED by us)"

for f in torch_spyre/_inductor/config.py torch_spyre/_inductor/customops.py \
         torch_spyre/_inductor/lowering.py torch_spyre/_inductor/passes.py \
         torch_spyre/_inductor/stick_reshape_pivot.py \
         tests/inductor/test_inductor_fx_passes.py; do
  cp $P/$f $W/$f || { echo "copy failed: $f"; exit 1; }
done
echo "views.py untouched by us: $(git -C $W diff --name-only -- torch_spyre/_inductor/views.py | wc -l) lines of our change (expect 0)"

cp $A/torch-spyre/torch_spyre/_C.so $W/torch_spyre/

echo "=== unit tests FIRST (these are what caught the bug) ==="
mkdir -p $A/runs/pivotB3 && cd $A/runs/pivotB3
PYTHONPATH=$W:$BASEPP timeout 900 python -m pytest -q --no-header -p no:cacheprovider \
  --rootdir=$W $W/tests/inductor/test_inductor_fx_passes.py 2>&1 | tail -8 | cut -c1-170

PYTHONPATH=$W:$BASEPP timeout 1800 python -u $A/retest.py > $A/logs/pivotB3.log 2>&1
echo "pivotB3 rc=$? OK=$(grep -c '^OK ' $A/logs/pivotB3.log) ERR=$(grep -c '^ERR ' $A/logs/pivotB3.log) BAD=$(grep -c '^BAD ' $A/logs/pivotB3.log) done=$(grep -c '^DONE' $A/logs/pivotB3.log)"
echo "B3 DONE"
