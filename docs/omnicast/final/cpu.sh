#!/bin/bash
# HiCMA-X end to end on the CPU (caffeine, Broadwell, AVX2): without OmniCast (4256ff9) vs with (f0ec341).
# Interleaved; each process runs NRUNS factorizations.
# HICMA_BASE / HICMA_HEAD: build directories (CUDA builds for gpu*.sh, CPU builds for cpu.sh) of 4256ff9 and of the
# commit with OmniCast, built the same way (here: git worktrees, Release, CUDA architectures 70;80;90a).
echo "# host $(hostname): $(lscpu | grep -m1 "Model name" | sed "s/.*: *//"), $(nproc) cpus"
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
BASE=${HICMA_BASE:?set HICMA_BASE to the build directory of 4256ff9 (before OmniCast)}/tests/testing_potrf_tlr
HEAD=${HICMA_HEAD:?set HICMA_HEAD to the build directory of the commit with OmniCast}/tests/testing_potrf_tlr
OUT=${OUT:-${OUT_DIR:-.}/cpu}; mkdir -p $OUT
NS=${NSIZE:-16384}
C="--N $NS --NB 512 --kind_of_problem 2 --check 0 --verbose 1 --nruns ${NRUNS:-3}"
declare -A CFG
CFG[dense_fp64]="--kind_of_cholesky 5"
CFG[k3_band]="--kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 8"
CFG[k5_adapt]="--adaptive_decision 1 --fixedacc 1e-4"
CFG[k5_sr]="--adaptive_decision 1 --fixedacc 1e-4 --enable_stochastic_rounding 1"
CFG[k10_runtime]="--adaptive_decision_runtime 1"
for rnd in $(seq 1 ${ROUNDS:-6}); do
  for cfg in ${CFGS:-dense_fp64 k3_band k5_adapt k5_sr k10_runtime}; do
    for var in base omc; do
      exe=$HEAD; [ $var = base ] && exe=$BASE
      log=$OUT/$cfg.$var.N$NS.r$rnd.log
      $exe $C ${CFG[$cfg]} > $log 2>&1 < /dev/null
      echo "$cfg $var r=$rnd: $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | awk '{print $3}' | tr '\n' ' ')| $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | tail -1 | grep -oE 'nb_dense_dp= *[0-9]+ nb_dense_sp= *[0-9]+ nb_dense_hp= *[0-9]+' | tr -s ' ')"
    done
  done
done
