#!/bin/bash
# HiCMA-X end to end on the A100 (methane): without OmniCast (4256ff9) vs with OmniCast (f0ec341),
# operand cache off and on. Interleaved rounds; each process runs NRUNS factorizations.
# HICMA_BASE / HICMA_HEAD: build directories (CUDA builds for gpu*.sh, CPU builds for cpu.sh) of 4256ff9 and of the
# commit with OmniCast, built the same way (here: git worktrees, Release, CUDA architectures 70;80;90a).
echo "# host $(hostname): $(lscpu | grep -m1 "Model name" | sed "s/.*: *//"), $(nproc) cpus"
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
BASE=${HICMA_BASE:?set HICMA_BASE to the build directory of 4256ff9 (before OmniCast)}/tests/testing_potrf_tlr
HEAD=${HICMA_HEAD:?set HICMA_HEAD to the build directory of the commit with OmniCast}/tests/testing_potrf_tlr
OUT=${OUT:-${OUT_DIR:-.}/gpu}; mkdir -p $OUT
NS=${NSIZE:-45056}
C="--N $NS --NB 2048 --kind_of_problem 2 --gpus 1 --check 0 --verbose 1 --nruns ${NRUNS:-3}"
# microseconds of software thermal slowdown since boot
thr() { nvidia-smi -q -d PERFORMANCE | grep -m1 'SW Thermal Slowdown.*us' | grep -oE '[0-9]+'; }
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
for rnd in $(seq 1 ${ROUNDS:-3}); do
  for cfg in ${CFGS:-dense_fp64 k3_hp_band k3_fp16acc k5_1e-8 k5_1e-4 k5_sr k5_dtconv2 k8_1e-3 k10_1e-3 runtime}; do
    for var in ${VARS:-base nocache cache}; do
      exe=$HEAD; extra=
      case $var in base) exe=$BASE;; nocache) extra="--operand_cache 0";; esac
      log=$OUT/$cfg.$var.N$NS.r$rnd.log
      # methane's A100 throttles thermally above ~82 C: start every process cool
      for w in $(seq 1 60); do [ $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -le ${TCOOL:-65} ] && break; sleep 5; done
      t0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader); s0=$(thr)
      numactl --cpunodebind=1 --membind=1 $exe $C ${CFG[$cfg]} $extra > $log 2>&1 < /dev/null
      s1=$(thr); t1=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
      echo "$cfg $var r=$rnd: $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | awk '{print $3}' | tr '\n' ' ')| $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | tail -1 | grep -oE 'nb_dense_dp= *[0-9]+ nb_dense_sp= *[0-9]+ nb_dense_hp= *[0-9]+' | tr -s ' ') | temp $t0-$t1 C, thermal slowdown $(( (s1 - s0) / 1000 )) ms"
    done
  done
done
