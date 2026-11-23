#!/bin/bash
# HiCMA-X CPU build: --flush_subnormals 1 (default) vs 0, interleaved rounds, then one checked run each
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
H=$(hostname -s); OUT=/home/qcao3/omc-runs/ftz/$H; mkdir -p $OUT
echo "# host $H: $(lscpu | grep -m1 'Model name' | sed 's/.*: *//'), $(nproc) cpus, $(date -Is)"
X=/home/qcao3/hicma-x/build_cpu/tests/testing_potrf_tlr
C="--N ${NSIZE:-16384} --NB 512 --kind_of_problem 2 --verbose 1"
declare -A CFG
CFG[dense_fp64]="--kind_of_cholesky 5"
CFG[k3_band]="--kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 8"
CFG[k5_1e-4]="--adaptive_decision 1 --fixedacc 1e-4"
CFG[k5_sr]="--adaptive_decision 1 --fixedacc 1e-4 --enable_stochastic_rounding 1"
CFG[k5_1e-8]="--adaptive_decision 1"
CFG[k10_runtime]="--adaptive_decision_runtime 1"
CFGS=${CFGS:-dense_fp64 k3_band k5_1e-4 k5_sr k5_1e-8 k10_runtime}
for rnd in $(seq 1 ${ROUNDS:-4}); do for cfg in $CFGS; do for f in 1 0; do
  log=$OUT/$cfg.f$f.r$rnd.log
  $X $C --check 0 --nruns 3 ${CFG[$cfg]} --flush_subnormals $f > $log 2>&1 < /dev/null
  echo "$cfg flush=$f r=$rnd: $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | awk '{print $3}' | tr '\n' ' ')| $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | tail -1 | grep -oE 'nb_dense_dp= *[0-9]+ nb_dense_sp= *[0-9]+ nb_dense_hp= *[0-9]+' | tr -s ' ')"
done; done; done
for cfg in $CFGS; do for f in 1 0; do
  log=$OUT/$cfg.f$f.check.log
  $X $C --check 1 --nruns 1 ${CFG[$cfg]} --flush_subnormals $f > $log 2>&1 < /dev/null
  echo "check $cfg flush=$f: $(sed 's/\x1b\[[0-9;]*m//g' $log | grep -oE 'Check (passed|FAILED)[^,]*|\|\|L-L0\|\|_F/\|\|L\|\|_F = [0-9.e+-]+' | tr '\n' ' ')"
done; done
