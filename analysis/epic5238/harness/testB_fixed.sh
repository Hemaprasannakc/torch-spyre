#!/bin/bash
# B2/B3 redone correctly: apply our changes as a PATCH onto PR #5804's head,
# instead of copying whole files from a different base.
#   B2 = #5804 + pivot + our 1 normalize_coordinates line
#   B3 = #5804 + pivot, our line REMOVED (is their fix enough on its own?)
# Control = #5804 alone on the same base.
set -u
export HOME=/home/hemaprasanna/
export HOME=/home/hemaprasanna
export DTI_PROJECT_ROOT=$HOME/dt-inductor
export GIT_AUTHOR_NAME=retest GIT_AUTHOR_EMAIL=retest@localhost
export GIT_COMMITTER_NAME=retest GIT_COMMITTER_EMAIL=retest@localhost
A=$DTI_PROJECT_ROOT/analysis/epic5238-retest
R=$DTI_PROJECT_ROOT/torch-spyre
SC=/tmp/claude-1001310000/-home-senuser/0eea9bcf-770d-4b0b-920d-af1245b6c115/scratchpad
PATCH=$SC/pivot-only.patch

source $DTI_PROJECT_ROOT/.venv/bin/activate
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
export PATH=$DTI_PROJECT_ROOT/build/deeptools/dbo/tools/dbo-opt:$PATH
BASEPP=$PYTHONPATH

build() {  # $1 = name, $2 = apply our views.py line? yes/no
  W=$A/$1
  rm -rf $W; git -C $R worktree prune
  git -C $R worktree add -q --detach $W refs/retest/pr5804 || return 1
  if ! git -C $W apply --3way $PATCH 2>&1; then
    echo "$1: PATCH FAILED"; return 1
  fi
  echo "$1: pivot patch applied onto $(git -C $W rev-parse --short HEAD) (PR5804 head)"
  if [ "$2" = yes ]; then
    python3 - "$W/torch_spyre/_inductor/views.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = "            and fused_term.den == term.mod\n"
add = ("            # padded inner dim: fusing would read across the gap\n"
       "            and fused_term.den == term.den * term.dim_size\n")
if add in s: print("  views.py: already present")
elif anchor in s:
    open(p,"w").write(s.replace(anchor, anchor+add, 1)); print("  views.py: padded-dim line applied")
else: print("  views.py: ANCHOR NOT FOUND")
PY
  else
    echo "  views.py: left exactly as #5804 wrote it"
  fi
  cp $A/torch-spyre/torch_spyre/_C.so $W/torch_spyre/
  # sanity: a trivial compile must work, else the tree is broken
  mkdir -p $A/runs/$1 && cd $A/runs/$1
  PYTHONPATH=$W:$BASEPP timeout 300 python -c "
import torch, torch_spyre
x = torch.randn(4, 64, dtype=torch.float16)
f = torch.compile(lambda t: t * 2 + 1, dynamic=False)
print('  SANITY OK', float((f(x.to('spyre')).cpu().float() - (x.float()*2+1)).abs().max()))
" 2>&1 | tail -2
  return 0
}

run() {  # $1 = name
  mkdir -p $A/runs/$1 && cd $A/runs/$1
  PYTHONPATH=$A/$1:$BASEPP timeout 1800 python -u $A/retest.py > $A/logs/$1.log 2>&1
  echo "$1 rc=$? OK=$(grep -c '^OK ' $A/logs/$1.log) ERR=$(grep -c '^ERR ' $A/logs/$1.log) BAD=$(grep -c '^BAD ' $A/logs/$1.log) done=$(grep -c '^DONE' $A/logs/$1.log)"
  PYTHONPATH=$A/$1:$BASEPP timeout 900 python -m pytest -q --no-header -p no:cacheprovider \
    --rootdir=$A/$1 $A/$1/tests/inductor/test_inductor_fx_passes.py 2>&1 | tail -4 | cut -c1-170
}

echo "########## CONTROL: #5804 alone, same base ##########"
W=$A/ctl5804; rm -rf $W; git -C $R worktree prune
git -C $R worktree add -q --detach $W refs/retest/pr5804 && cp $A/torch-spyre/torch_spyre/_C.so $W/torch_spyre/
mkdir -p $A/runs/ctl5804 && cd $A/runs/ctl5804
PYTHONPATH=$W:$BASEPP timeout 1800 python -u $A/retest.py > $A/logs/ctl5804.log 2>&1
echo "ctl5804 rc=$? OK=$(grep -c '^OK ' $A/logs/ctl5804.log) ERR=$(grep -c '^ERR ' $A/logs/ctl5804.log) BAD=$(grep -c '^BAD ' $A/logs/ctl5804.log) done=$(grep -c '^DONE' $A/logs/ctl5804.log)"

echo "########## B2: #5804 + pivot + our line ##########"
build b2 yes && run b2

echo "########## B3: #5804 + pivot, our line REMOVED ##########"
build b3 no && run b3

echo "ALL B DONE"
