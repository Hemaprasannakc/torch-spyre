#!/bin/bash
set -u
export HOME=/home/hemaprasanna; export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest; Y=$DTI_PROJECT_ROOT/analysis/yolo26
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
cd $Y
PYTHONPATH=$A/pivotA2:$PYTHONPATH timeout 3600 python -u $Y/yolo_validate.py > $A/logs/yolo_full.log 2>&1
echo "exit=$?"
grep -E "Error|Unsupported|AssertionError|NotImplemented|does not support" $A/logs/yolo_full.log | head -8 | cut -c1-190
echo "YOLO2 DONE"
