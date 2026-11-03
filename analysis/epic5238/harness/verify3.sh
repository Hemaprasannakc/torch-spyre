#!/bin/bash
set -u
export HOME=/home/hemaprasanna; export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH
echo "### 162-case epic suite"
mkdir -p $A/runs/v3 && cd $A/runs/v3
PYTHONPATH=$A/pivotA2:$BASEPP timeout 2400 python -u $A/retest.py > $A/logs/retest_v3.log 2>&1
echo "retest OK=$(grep -c '^OK ' $A/logs/retest_v3.log) ERR=$(grep -c '^ERR ' $A/logs/retest_v3.log) BAD=$(grep -c '^BAD ' $A/logs/retest_v3.log)"
echo "### unit tests + coordinates"
PYTHONPATH=$A/pivotA2:$BASEPP timeout 1800 python -m pytest -q --no-header -p no:cacheprovider \
  --rootdir=$A/pivotA2 $A/pivotA2/tests/inductor/test_inductor_fx_passes.py \
  $A/pivotA2/tests/tensor/test_coordinates.py 2>&1 | tail -4 | cut -c1-165
echo "VERIFY3 DONE"
