#!/bin/bash
set -u
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH
for t in tip-overlay pivotA2; do
  n=$(basename $t)
  mkdir -p $A/runs/mx_$n && cd $A/runs/mx_$n
  PYTHONPATH=$A/$t:$BASEPP timeout 2400 python -u $A/matrix.py > $A/logs/matrix_$n.log 2>&1
  echo "$n rc=$? OK=$(grep -c '^OK ' $A/logs/matrix_$n.log) ERR=$(grep -c '^ERR ' $A/logs/matrix_$n.log) BAD=$(grep -c '^BAD ' $A/logs/matrix_$n.log) done=$(grep -c 'MATRIX DONE' $A/logs/matrix_$n.log)"
done
echo MATRIX ALL DONE
