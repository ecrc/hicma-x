#!/bin/bash
# TLR (kind 1, low-rank off-diagonal tiles, adaptive precision): --flush_subnormals 1 vs 0
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
H=$(hostname -s); OUT=/home/qcao3/omc-runs/ftz/$H; mkdir -p $OUT
echo "# host $H: $(lscpu | grep -m1 'Model name' | sed 's/.*: *//'), $(nproc) cpus, $(date -Is)"
X=/home/qcao3/hicma-x/build_cpu/tests/testing_potrf_tlr
C="--kind_of_problem 2 --N 54000 --NB 1350 --fixedacc 1e-8 --maxrank 300 --verbose 1"
declare -A CFG
CFG[tlr_dp]="--kind_of_cholesky 2 --band_dense 1"
CFG[tlr_mp_adaptive]="--kind_of_cholesky 1 --band_dense 1 --adaptive_decision 1"
for rnd in 1 2 3; do for cfg in tlr_dp tlr_mp_adaptive; do for f in 1 0; do
  log=$OUT/$cfg.f$f.r$rnd.log
  $X $C --check 0 --nruns 2 ${CFG[$cfg]} --flush_subnormals $f > $log 2>&1 < /dev/null
  echo "$cfg flush=$f r=$rnd: $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | awk '{print $3}' | tr '\n' ' ')"
done; done; done
