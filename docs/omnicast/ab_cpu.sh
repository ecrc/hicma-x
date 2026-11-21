#!/bin/bash
# Interleaved A/B timing on the CPU (--gpus 0) of the two CUDA builds.
REPS=${1:-3}; N=${2:-8192}; NB=${3:-512}
BASE=/home/qcao3/hicma-x-base/build/tests/testing_potrf_tlr
OMC=/home/qcao3/hicma-x/build_omc/tests/testing_potrf_tlr
COMMON="--N $N --NB $NB --kind_of_problem 2 --gpus 0 --check 0 --verbose 1"
declare -A CFG
CFG[cpu_k3_band]="--kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4"
CFG[cpu_k5_adapt]="--adaptive_decision 1"
CFG[cpu_k5_adapt_sr]="--adaptive_decision 1 --enable_stochastic_rounding 1"
CFG[cpu_k10_runtime]="--adaptive_decision_runtime 1"
for c in cpu_k3_band cpu_k5_adapt cpu_k5_adapt_sr cpu_k10_runtime; do
  for r in $(seq 1 $REPS); do
    for b in base omc; do
      exe=$BASE; [ $b = omc ] && exe=$OMC
      line=$($exe $COMMON ${CFG[$c]} 2>&1 | grep -E 'TIME\(s\).*: hicma_parsec_cholesky')
      t=$(echo "$line" | awk '{print $3}')
      tf=$(echo "$line" | grep -oE '[0-9.]+ Tflops' | awk '{print $1}')
      echo "$c $b $r $t $tf"
    done
  done
done
