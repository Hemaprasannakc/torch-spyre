#!/bin/bash
set -u
export HOME=/home/hemaprasanna; export DTI_PROJECT_ROOT=$HOME/dt-inductor
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH
echo "### branch head: $(git -C $A/prbranch log -1 --format='%h') on $(git -C $A/prbranch log -1 --format='%h' HEAD~1)"
echo "### 1. coverage matrix"
mkdir -p $A/runs/br && cd $A/runs/br
PYTHONPATH=$A/prbranch:$BASEPP timeout 2400 python -u $A/matrix.py > $A/logs/matrix_branch.log 2>&1
echo "matrix OK=$(grep -c '^OK ' $A/logs/matrix_branch.log) ERR=$(grep -c '^ERR ' $A/logs/matrix_branch.log) BAD=$(grep -c '^BAD ' $A/logs/matrix_branch.log)"
echo "### 2. 162-case epic suite"
PYTHONPATH=$A/prbranch:$BASEPP timeout 2400 python -u $A/retest.py > $A/logs/retest_branch.log 2>&1
echo "retest OK=$(grep -c '^OK ' $A/logs/retest_branch.log) ERR=$(grep -c '^ERR ' $A/logs/retest_branch.log) BAD=$(grep -c '^BAD ' $A/logs/retest_branch.log)"
echo "### 3. unit tests + coordinates"
PYTHONPATH=$A/prbranch:$BASEPP timeout 1800 python -m pytest -q --no-header -p no:cacheprovider \
  --rootdir=$A/prbranch $A/prbranch/tests/inductor/test_inductor_fx_passes.py \
  $A/prbranch/tests/tensor/test_coordinates.py 2>&1 | tail -4 | cut -c1-160
echo "### 4. issue #5227 verbatim repro"
PYTHONPATH=$A/prbranch:$BASEPP timeout 1800 python -u /tmp/claude-1001310000/-home-senuser/0eea9bcf-770d-4b0b-920d-af1245b6c115/scratchpad/issue5227.py 2>&1 | grep -E "^(OK|ERR|BAD)"
echo "BRANCH VERIFY DONE"
