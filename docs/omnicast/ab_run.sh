#!/bin/bash
# Interleaved A/B timing of the HiCMA-X Cholesky: baseline (HEAD without
# OmniCast) vs OmniCast build. Usage: ab_run.sh REPS N NB > out.txt
REPS=${1:-3}; N=${2:-32768}; NB=${3:-2048}
# HICMA_BASE: build directory of the commit before OmniCast (4256ff9), e.g. a git worktree built like build_omc
BASE=${HICMA_BASE:?set HICMA_BASE to the baseline build directory}/tests/testing_potrf_tlr
OMC=/home/qcao3/hicma-x/build_omc/tests/testing_potrf_tlr
COMMON="--N $N --NB $NB --kind_of_problem 2 --gpus 1 --check 0 --verbose 1"
declare -A CFG
CFG[k3_hp_band]="--kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4"
CFG[k3_fp16acc]="--kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --tensor_gemm 0x10"
CFG[k5_adapt]="--adaptive_decision 1"
CFG[k5_adapt_sr]="--adaptive_decision 1 --enable_stochastic_rounding 1"
CFG[k5_dtconv2]="--adaptive_decision 1 --datatype_convert 2"
CFG[k10_runtime]="--adaptive_decision_runtime 1"
for c in ${CONFIGS:-k3_hp_band k3_fp16acc k5_adapt k5_adapt_sr k5_dtconv2 k10_runtime}; do
  for r in $(seq 1 $REPS); do
    for b in base omc; do
      exe=$BASE; [ $b = omc ] && exe=$OMC
      line=$($exe $COMMON ${CFG[$c]} 2>&1 | grep -E 'TIME\(s\).*: hicma_parsec_cholesky')
      t=$(echo "$line" | awk '{print $3}')
      tf=$(echo "$line" | grep -oE '[0-9.]+ Tflops' | awk '{print $1}')
      mix=$(echo "$line" | grep -oE 'nb_dense_dp= *[0-9]+ nb_dense_sp= *[0-9]+ nb_dense_hp= *[0-9]+' | tr -s ' ')
      echo "$c $b $r $t $tf | $mix"
    done
  done
done
