#!/bin/bash
# Bit-for-bit check of the final change (adapted from docs/omnicast/round5/verify.sh): factor hash, builds with
# factor_hash.patch applied (not committed): base = parent ffcf1cb, new = ACC_DP_PANEL 256 (MKL).
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
export HICMA_DEBUG_FACTOR_HASH=1
X=${X:-/home/qcao3/hicma-x/.claude/worktrees/optimistic-shaw-05ec60/build_cpu/tests/testing_potrf_tlr}   # the change
B=${B:-/home/qcao3/hicma-x-acc-base/build_cpu/tests/testing_potrf_tlr}
MPIEXEC=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/openmpi-5.0.5-24tp3ez2epnadfl3x57bpvg5epgjshqh/bin/mpiexec
OUT=${OUT:-/home/qcao3/omc-runs/accloop/verify_final};  mkdir -p $OUT
echo "# $(hostname -s) $(date -Is) X=$X B=$B"
summ() { sed 's/\x1b\[[0-9;]*m//g' $1 | grep -oE 'FACTOR_HASH.*|DENSE_DP= [0-9]+ DENSE_SP= [0-9]+|\|\|L-L0\|\|_F/\|\|L\|\|_F = [0-9.e+-]+|Check (passed|FAILED)[^,]*, \|\|LL\^T-A\|\|_F/\|\|A\|\|_F = [0-9.e+-]+' | tail -${2:-8} | tr '\n' ' '; }
run() { # name np arm args...
  local name=$1 np=$2 arm=$3; shift 3
  local log=$OUT/$name.np$np.$arm.log bin=$X
  [ ${arm%%.*} = base ] && bin=$B
  if [ $np = 1 ]; then $bin "$@" > $log 2>&1 < /dev/null
  else $MPIEXEC -n $np --bind-to none -x HICMA_DEBUG_FACTOR_HASH $bin "$@" --cores $((12 / np)) > $log 2>&1 < /dev/null; fi
  echo "$name np=$np $arm rc=$?: $(summ $log $((5 + np)))"
}
K="--kind_of_problem 2 --NB 512 --verbose 2 --check 1 --nruns 2"
R="--adaptive_decision_runtime 1"; F="--kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3"
for a in base new; do run k10_runtime 1 $a $K --N 16384 $R; run k10_1e-3 1 $a $K --N 16384 $F; done
for np in 2 4; do for a in base new; do run k10_runtime $np $a $K --N 8192 $R; run k10_1e-3 $np $a $K --N 8192 $F; done; done
for a in base.1 new.1 base.2 new.2; do run k10_runtime_N16000 1 $a $K --N 16000 $R; run k10_1e-3_N16000 1 $a $K --N 16000 $F; done
