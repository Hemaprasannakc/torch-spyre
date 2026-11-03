#!/bin/bash
set -u
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH
SC=/tmp/claude-1001310000/-home-senuser/0eea9bcf-770d-4b0b-920d-af1245b6c115/scratchpad

echo "### 1. coverage matrix (58 cases)"
mkdir -p $A/runs/fix_mx && cd $A/runs/fix_mx
PYTHONPATH=$A/pivotA2:$BASEPP timeout 2400 python -u $A/matrix.py > $A/logs/matrix_fixed.log 2>&1
echo "matrix rc=$? OK=$(grep -c '^OK ' $A/logs/matrix_fixed.log) ERR=$(grep -c '^ERR ' $A/logs/matrix_fixed.log) BAD=$(grep -c '^BAD ' $A/logs/matrix_fixed.log)"

echo "### 2. the E1 probes that were wrong"
mkdir -p $A/runs/fix_probe && cd $A/runs/fix_probe
PYTHONPATH=$A/pivotA2:$BASEPP timeout 1200 python -u $SC/probe_e1.py 2>&1 | grep -E "^(OK|ERR|BAD|---)"

echo "### 3. sub-stick cases that used to be skipped"
mkdir -p $A/runs/fix_sub && cd $A/runs/fix_sub
PYTHONPATH=$A/pivotA2:$BASEPP timeout 1200 python -u $SC/substick.py 2>&1 | grep -E "^(OK|ERR|BAD|---)"

echo "### 4. the 162-case epic suite (regression)"
mkdir -p $A/runs/fix_162 && cd $A/runs/fix_162
PYTHONPATH=$A/pivotA2:$BASEPP timeout 2400 python -u $A/retest.py > $A/logs/retest_fixed.log 2>&1
echo "retest rc=$? OK=$(grep -c '^OK ' $A/logs/retest_fixed.log) ERR=$(grep -c '^ERR ' $A/logs/retest_fixed.log) BAD=$(grep -c '^BAD ' $A/logs/retest_fixed.log)"

echo "### 5. unit tests"
PYTHONPATH=$A/pivotA2:$BASEPP timeout 1800 python -m pytest -q --no-header -p no:cacheprovider \
  --rootdir=$A/pivotA2 $A/pivotA2/tests/inductor/test_inductor_fx_passes.py \
  $A/pivotA2/tests/tensor/test_coordinates.py 2>&1 | tail -6 | cut -c1-170
echo "VERIFY DONE"
