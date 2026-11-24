#!/bin/bash
# Bit-for-bit check of the accumulation variants (adapted from docs/omnicast/round5/verify.sh): factor hash
# (builds with factor_hash.patch applied, not committed) of the parent build (base, ffcf1cb) and of the new
# build with HICMA_ACC_MODE=0..4, one process at N=16384 and 2 MPI processes at N=8192.
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
export HICMA_DEBUG_FACTOR_HASH=1
X=${X:-/home/qcao3/hicma-x-acc-exp/build_cpu/tests/testing_potrf_tlr}   # parent build with experiment_modes.patch (HICMA_ACC_MODE)
B=${B:-/home/qcao3/hicma-x-acc-base/build_cpu/tests/testing_potrf_tlr}
MPIEXEC=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/openmpi-5.0.5-24tp3ez2epnadfl3x57bpvg5epgjshqh/bin/mpiexec
OUT=${OUT:-/home/qcao3/omc-runs/accloop/verify};  mkdir -p $OUT
echo "# $(hostname -s) $(date -Is) X=$X B=$B"
summ() { sed 's/\x1b\[[0-9;]*m//g' $1 | grep -oE 'FACTOR_HASH.*|DENSE_DP= [0-9]+ DENSE_SP= [0-9]+|\|\|L-L0\|\|_F/\|\|L\|\|_F = [0-9.e+-]+|Check (passed|FAILED)[^,]*, \|\|LL\^T-A\|\|_F/\|\|A\|\|_F = [0-9.e+-]+' | tail -${2:-8} | tr '\n' ' '; }
run() { # name np arm args...
  local name=$1 np=$2 arm=$3; shift 3
  local log=$OUT/$name.np$np.$arm.log bin=$X mode=${arm#m}
  [ $arm = base ] && { bin=$B; mode=0; }
  if [ $np = 1 ]; then HICMA_ACC_MODE=$mode $bin "$@" > $log 2>&1 < /dev/null
  else $MPIEXEC -n $np --bind-to none -x HICMA_ACC_MODE=$mode -x HICMA_DEBUG_FACTOR_HASH $bin "$@" --cores $((12 / np)) > $log 2>&1 < /dev/null; fi
  echo "$name np=$np $arm rc=$?: $(summ $log $((5 + np)))"
}
K="--kind_of_problem 2 --NB 512 --verbose 2 --check 1 --nruns 2"
ARMS=${ARMS:-base m0 m1 m2 m3 m4}
for a in $ARMS; do run k10_runtime 1 $a $K --N 16384 --adaptive_decision_runtime 1; done
for a in $ARMS; do run k10_1e-3 1 $a $K --N 16384 --kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3; done
for a in $ARMS; do run k10_runtime 2 $a $K --N 8192 --adaptive_decision_runtime 1; done
for a in $ARMS; do run k10_1e-3 2 $a $K --N 8192 --kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3; done
