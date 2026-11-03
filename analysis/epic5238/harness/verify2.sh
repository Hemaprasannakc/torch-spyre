#!/bin/bash
set -u
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
SC=/tmp/claude-1001310000/-home-senuser/0eea9bcf-770d-4b0b-920d-af1245b6c115/scratchpad
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH

echo "### 1. the cat probes (G5)"
mkdir -p $A/runs/v2g5 && cd $A/runs/v2g5
PYTHONPATH=$A/pivotA2:$BASEPP timeout 1200 python -u $SC/probe_g5.py 2>&1 | grep -E "^(OK|ERR|BAD|---)"

echo "### 2. coverage matrix (58 cases)"
mkdir -p $A/runs/v2mx && cd $A/runs/v2mx
PYTHONPATH=$A/pivotA2:$BASEPP timeout 2400 python -u $A/matrix.py > $A/logs/matrix_final.log 2>&1
echo "matrix rc=$? OK=$(grep -c '^OK ' $A/logs/matrix_final.log) ERR=$(grep -c '^ERR ' $A/logs/matrix_final.log) BAD=$(grep -c '^BAD ' $A/logs/matrix_final.log)"
echo "remaining failures:"; grep -E "^(ERR|BAD) " $A/logs/matrix_final.log | cut -c1-120 | sed 's/^/   /'

echo "### 3. 162-case epic suite"
mkdir -p $A/runs/v2_162 && cd $A/runs/v2_162
PYTHONPATH=$A/pivotA2:$BASEPP timeout 2400 python -u $A/retest.py > $A/logs/retest_final.log 2>&1
echo "retest rc=$? OK=$(grep -c '^OK ' $A/logs/retest_final.log) ERR=$(grep -c '^ERR ' $A/logs/retest_final.log) BAD=$(grep -c '^BAD ' $A/logs/retest_final.log)"

echo "### 4. unit tests + coordinates"
PYTHONPATH=$A/pivotA2:$BASEPP timeout 1800 python -m pytest -q --no-header -p no:cacheprovider \
  --rootdir=$A/pivotA2 $A/pivotA2/tests/inductor/test_inductor_fx_passes.py \
  $A/pivotA2/tests/tensor/test_coordinates.py 2>&1 | tail -5 | cut -c1-170

echo "### 5. wider regression: restickify + inductor ops"
for f in tests/inductor/test_restickify.py tests/inductor/test_inductor_ops.py; do
  if [ -f $A/pivotA2/$f ]; then
    echo "-- $f"
    PYTHONPATH=$A/pivotA2:$BASEPP timeout 2400 python -m pytest -q --no-header -p no:cacheprovider \
      --rootdir=$A/pivotA2 $A/pivotA2/$f 2>&1 | tail -3 | cut -c1-165
  fi
done
echo "VERIFY2 DONE"
