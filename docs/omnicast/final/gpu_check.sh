#!/bin/bash
# HiCMA-X correctness on the A100 (methane): residuals without OmniCast (4256ff9) and with OmniCast,
# operand cache off and on; one factorization per process, checked (--check 1).
# HICMA_BASE / HICMA_HEAD: build directories (CUDA builds for gpu*.sh, CPU builds for cpu.sh) of 4256ff9 and of the
# commit with OmniCast, built the same way (here: git worktrees, Release, CUDA architectures 70;80;90a).
# GPU / NUMA: the GPU index and its NUMA node (methane's A100: 0 and 1, the default; hexane's H100: 0 and 0).
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
GPU=${GPU:-0}; NUMA=${NUMA:-1}
export CUDA_VISIBLE_DEVICES=$GPU
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
BASE=${HICMA_BASE:?set HICMA_BASE to the build directory of 4256ff9 (before OmniCast)}/tests/testing_potrf_tlr
HEAD=${HICMA_HEAD:?set HICMA_HEAD to the build directory of the commit with OmniCast}/tests/testing_potrf_tlr
OUT=${OUT:-${OUT_DIR:-.}/gpu_check}; mkdir -p $OUT
NS=${NSIZE:-16384}
C="--N $NS --NB 2048 --kind_of_problem 2 --gpus 1 --check 1 --verbose 1 --nruns 1"
declare -A CFG
CFG[dense_fp64]="--kind_of_cholesky 5"
CFG[k3_hp_band]="--kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4"
CFG[k3_fp16acc]="--kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --tensor_gemm 0x10"
CFG[k5_1e-8]="--adaptive_decision 1"
CFG[k5_1e-4]="--adaptive_decision 1 --fixedacc 1e-4"
CFG[k5_sr]="--adaptive_decision 1 --fixedacc 1e-4 --enable_stochastic_rounding 1"
CFG[k5_dtconv2]="--adaptive_decision 1 --fixedacc 1e-4 --datatype_convert 2"
CFG[k8_1e-3]="--kind_of_cholesky 8 --adaptive_decision 1 --fixedacc 1e-3"
CFG[k10_1e-3]="--kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3"
CFG[runtime]="--adaptive_decision_runtime 1"
for rnd in 1; do
  for cfg in ${CFGS:-dense_fp64 k3_hp_band k3_fp16acc k5_1e-8 k5_1e-4 k5_sr k5_dtconv2 k8_1e-3 k10_1e-3 runtime}; do
    for var in ${VARS:-base nocache cache}; do
      exe=$HEAD; extra=
      case $var in base) exe=$BASE;; nocache) extra="--operand_cache 0";; esac
      log=$OUT/$cfg.$var.N$NS.r$rnd.log
      numactl --cpunodebind=$NUMA --membind=$NUMA $exe $C ${CFG[$cfg]} $extra > $log 2>&1 < /dev/null
      echo "$cfg $var: $(sed 's/\x1b\[[0-9;]*m//g' $log | grep -oE 'Check (passed|FAILED)[^,]*' | head -1)"
    done
  done
done
