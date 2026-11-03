#!/bin/bash
# Test A: pivot pass alone, views.py untouched (upstream).
# Test B: pivot pass alone + PR #5804 merged.
set -u
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
export GIT_AUTHOR_NAME=retest GIT_AUTHOR_EMAIL=retest@localhost
export GIT_COMMITTER_NAME=retest GIT_COMMITTER_EMAIL=retest@localhost
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
R=$DTI_PROJECT_ROOT/torch-spyre
BASE=555dbc5ff7c08d4bfe9e7d1f57b1d23e8c56b833
P=$A/pivot-tip
SO=$A/torch-spyre/torch_spyre/_C.so

source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH
mkdir -p $A/logs $A/runs

# the pivot fix WITHOUT views.py / test_coordinates.py
PIVOT_FILES="torch_spyre/_inductor/config.py
torch_spyre/_inductor/customops.py
torch_spyre/_inductor/lowering.py
torch_spyre/_inductor/passes.py
torch_spyre/_inductor/stick_reshape_pivot.py
tests/inductor/test_inductor_fx_passes.py"

setup() {  # $1 = worktree name, $2 = extra ref to merge ("" for none)
  W=$A/$1
  rm -rf $W
  git -C $R worktree prune
  git -C $R worktree add -q --detach $W $BASE || return 1
  if [ -n "$2" ]; then
    if ! git -C $W merge -q --no-commit --no-ff $2 >/dev/null 2>&1; then
      echo "$1: MERGE CONFLICT: $(git -C $W diff --name-only --diff-filter=U | tr '\n' ' ')"
      return 2
    fi
    echo "$1: merged $2 clean"
  fi
  for f in $PIVOT_FILES; do cp $P/$f $W/$f || return 1; done
  cp $SO $W/torch_spyre/
  return 0
}

go() {  # $1 = name, $2 = worktree
  mkdir -p $A/runs/$1; cd $A/runs/$1
  PYTHONPATH=$A/$2:$BASEPP timeout 1800 python -u $A/retest.py > $A/logs/$1.log 2>&1
  echo "$1 rc=$? OK=$(grep -c '^OK ' $A/logs/$1.log) ERR=$(grep -c '^ERR ' $A/logs/$1.log) BAD=$(grep -c '^BAD ' $A/logs/$1.log) done=$(grep -c '^DONE' $A/logs/$1.log)"
}

echo "===== TEST A: pivot, views.py untouched ====="
if setup pivotA ""; then
  echo "pivotA: views.py diff vs base = $(git -C $A/pivotA diff --name-only $BASE -- torch_spyre/_inductor/views.py | wc -l) (expect 0)"
  go pivotA pivotA
else
  echo "pivotA setup FAILED"
fi

echo "===== TEST B: pivot + PR 5804 ====="
if setup pivotB refs/retest/pr5804; then
  go pivotB pivotB
else
  echo "pivotB setup failed or conflicted"
fi

echo "===== unit tests, Test A tree (fx passes only) ====="
cd $A/runs/pivotA 2>/dev/null && PYTHONPATH=$A/pivotA:$BASEPP timeout 900 python -m pytest -q --no-header \
  -p no:cacheprovider --rootdir=$A/pivotA $A/pivotA/tests/inductor/test_inductor_fx_passes.py 2>&1 | tail -3 | cut -c1-180

echo "ALL DONE"
