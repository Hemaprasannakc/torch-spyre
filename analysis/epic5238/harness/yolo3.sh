#!/bin/bash
set -u
export HOME=/home/hemaprasanna; export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest; Y=$DTI_PROJECT_ROOT/analysis/yolo26
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
cd $Y
echo "=== YOLO with trailing view materialised ==="
PYTHONPATH=$A/pivotA3:$PYTHONPATH timeout 3600 python -u $Y/yolo_validate.py > $A/logs/yolo_A3.log 2>&1
echo "exit=$?"
grep -E "^RESULT|does not support|InductorError" $A/logs/yolo_A3.log | head -5 | cut -c1-185
echo "=== reshape summary ==="
sed -n '/reshapes our pass handled/,$p' $A/logs/yolo_A3.log | head -10
echo "=== matrix on A3 (regression check) ==="
mkdir -p $A/runs/a3mx && cd $A/runs/a3mx
PYTHONPATH=$A/pivotA3:$PYTHONPATH timeout 2400 python -u $A/matrix.py > $A/logs/matrix_A3.log 2>&1
echo "matrix OK=$(grep -c '^OK ' $A/logs/matrix_A3.log) ERR=$(grep -c '^ERR ' $A/logs/matrix_A3.log) BAD=$(grep -c '^BAD ' $A/logs/matrix_A3.log)"
echo "YOLO3 DONE"
