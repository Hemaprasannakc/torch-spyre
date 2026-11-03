#!/bin/bash
export HOME=/home/hemaprasanna; export DTI_PROJECT_ROOT=$HOME/dt-inductor; A=$DTI_PROJECT_ROOT/analysis/epic5238-retest; R=$DTI_PROJECT_ROOT/torch-spyre; U=$1
source $DTI_PROJECT_ROOT/.venv/bin/activate; source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH; BASEPP=$PYTHONPATH; SO=$A/torch-spyre/torch_spyre/_C.so
mkdir -p $A/logs $A/runs
go() { name=$1; tree=$2; mkdir -p $A/runs/$name; cd $A/runs/$name; PYTHONPATH=$tree:$BASEPP timeout 1500 python -u $A/retest.py > $A/logs/$name.log 2>&1; echo "$name rc=$? OK=$(grep -c '^OK ' $A/logs/$name.log) ERR=$(grep -c '^ERR ' $A/logs/$name.log) BAD=$(grep -c '^BAD ' $A/logs/$name.log) done=$(grep -c '^DONE' $A/logs/$name.log)"; }
go tip $A/tip-overlay
cp $SO $A/pivot-tip/torch_spyre/; go pivot $A/pivot-tip
for n in 5804 5068 5354 5813 5098 5800 5777; do
  W=$A/pr$n; git -C $R worktree add -q --detach $W $U; cd $W
  if git merge -q --no-commit --no-ff refs/retest/pr$n >/dev/null 2>&1; then echo "pr$n merged clean, csrc: $(git diff --cached --name-only | grep -c csrc)"; else echo "pr$n CONFLICT: $(git diff --name-only --diff-filter=U | tr '\n' ' ')"; git merge --abort; git -C $W checkout -q --detach $U; git diff $(git merge-base $U refs/retest/pr$n) refs/retest/pr$n -- torch_spyre > /tmp/x$n.diff 2>/dev/null; continue; fi
  cp $SO $W/torch_spyre/; go pr$n $W
done
cd $A/runs/pivot; echo '--- pivot unit tests'; PYTHONPATH=$A/pivot-tip:$BASEPP timeout 900 python -m pytest -q -x --no-header -p no:cacheprovider --rootdir=$A/pivot-tip $A/pivot-tip/tests/tensor/test_coordinates.py $A/pivot-tip/tests/inductor/test_inductor_fx_passes.py 2>&1 | tail -4 | cut -c1-180
echo '--- hand offset pivot on tip'; cp $DTI_PROJECT_ROOT/analysis/yolo26/stick-pivot-fix/repro/offset_pivot.py $A/offset_pivot.py; PYTHONPATH=$A/pivot-tip:$BASEPP timeout 600 python -u $A/offset_pivot.py 2>&1 | grep -E '^(OK|ERR|BAD)' | cut -c1-150 | tee $A/logs/offset_pivot.log
