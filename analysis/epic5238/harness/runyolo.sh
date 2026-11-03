#!/bin/bash
set -u
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
Y=$DTI_PROJECT_ROOT/analysis/yolo26
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH

# one device: wait for the regression suite to finish first
while pgrep -f "[t]est_inductor_ops.py" >/dev/null 2>&1; do sleep 20; done

cd $Y
for t in tip-overlay pivotA2; do
  echo "################################################################"
  echo "########  YOLO26s on $t"
  echo "################################################################"
  PYTHONPATH=$A/$t:$BASEPP timeout 3600 python -u $Y/yolo_validate.py \
    > $A/logs/yolo_$t.log 2>&1
  echo "exit=$?"
  grep -E "^(RESULT|\[probe\]|===|    )" $A/logs/yolo_$t.log | head -40
done
echo "YOLO DONE"
