#!/bin/bash
# Bit-for-bit check of the host operand cache: hash of the factor (build with factor_hash.patch applied,
# not committed) for several --operand_cache settings, one process and 2 or 4 MPI processes. B is an
# unmodified build of the parent revision, which prints residuals but no hash.
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
export HICMA_DEBUG_FACTOR_HASH=1
X=${X:-/home/qcao3/hicma-x/.claude/worktrees/peaceful-bhabha-f8c6db/build_cpu/tests/testing_potrf_tlr}   # build with the cache
B=${B:-/home/qcao3/hicma-x/build_cpu/tests/testing_potrf_tlr}   # build of the parent revision
MPIEXEC=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/openmpi-5.0.5-24tp3ez2epnadfl3x57bpvg5epgjshqh/bin/mpiexec
OUT=${OUT:-/home/qcao3/omc-runs/hostcache/verify};  mkdir -p $OUT
summ() { sed 's/\x1b\[[0-9;]*m//g' $1 | grep -oE 'FACTOR_HASH.*|HOC_KINDS.*|DENSE_DP= [0-9]+ DENSE_SP= [0-9]+ DENSE_HP= [0-9]+ DENSE_FP8= [0-9]+|Number_of_Datatype_Conversions= [0-9]+|CPU_operand_cache.*|\|\|L-L0\|\|_F/\|\|L\|\|_F = [0-9.e+-]+|Check (passed|FAILED)[^,]*, \|\|LL\^T-A\|\|_F/\|\|A\|\|_F = [0-9.e+-]+' | tail -${2:-8} | tr '\n' ' '; }
run() { # name np oc args...
  local name=$1 np=$2 oc=$3; shift 3
  local log=$OUT/$name.np$np.oc$oc.log
  if [ $np = 1 ]; then
    if [ $oc = base ]; then $B "$@" > $log 2>&1 < /dev/null; else $X "$@" --operand_cache $oc > $log 2>&1 < /dev/null; fi
  else
    if [ $oc = base ]; then $MPIEXEC -n $np --bind-to none $B "$@" --cores $((12 / np)) > $log 2>&1 < /dev/null
    else $MPIEXEC -n $np --bind-to none $X "$@" --cores $((12 / np)) --operand_cache $oc > $log 2>&1 < /dev/null; fi
  fi
  echo "$name np=$np oc=$oc rc=$?: $(summ $log $((7 + np)))"
}
K="--kind_of_problem 2 --NB 512 --verbose 2 --check 1 --nruns 2"
for oc in base 0 -1 64 4 1; do run k10_runtime 1 $oc $K --N 16384 --adaptive_decision_runtime 1; done
for oc in base 0 -1 64 4 1; do run k10_1e-3 1 $oc $K --N 16384 --kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3; done
for np in 2 4; do for oc in base 0 -1 8; do
  run k10_runtime $np $oc $K --N 8192 --adaptive_decision_runtime 1
  run k10_1e-3 $np $oc $K --N 8192 --kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3
done; done
