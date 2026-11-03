#!/bin/bash
# Test A2: pivot pass + ONLY the normalize_coordinates padded-dim fix.
# No _check_nested_strides, no UnrepresentableAccess -> no collision with #5804.
set -u
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH

echo "=== views.py lines changed vs upstream: $(git -C $A/pivotA2 diff --numstat -- torch_spyre/_inductor/views.py | awk '{print $1"+ "$2"-"}') ==="

mkdir -p $A/runs/pivotA2 && cd $A/runs/pivotA2
PYTHONPATH=$A/pivotA2:$BASEPP timeout 1800 python -u $A/retest.py > $A/logs/pivotA2.log 2>&1
echo "pivotA2 rc=$? OK=$(grep -c '^OK ' $A/logs/pivotA2.log) ERR=$(grep -c '^ERR ' $A/logs/pivotA2.log) BAD=$(grep -c '^BAD ' $A/logs/pivotA2.log) done=$(grep -c '^DONE' $A/logs/pivotA2.log)"

echo "=== unit tests (fx passes) ==="
PYTHONPATH=$A/pivotA2:$BASEPP timeout 900 python -m pytest -q --no-header -p no:cacheprovider \
  --rootdir=$A/pivotA2 $A/pivotA2/tests/inductor/test_inductor_fx_passes.py 2>&1 | tail -6 | cut -c1-170
echo "A2 DONE"
