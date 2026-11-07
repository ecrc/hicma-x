#!/bin/bash
# HiCMA-X validation matrix: every Cholesky kind, the TRMM and the sparse paths,
# with --check 1 and the residual bound each precision mix can reach.
#
# usage: scripts/validation_matrix.sh <build_dir> <label> [cpu|gpu] [out_dir]
#
# Writes <out_dir>/results_<label>.tsv (case, exit code, residual, factorization
# time, GEMM counts) and one log per case under <out_dir>/logs_<label>/.
# <out_dir> defaults to ./validation. "gpu" adds the single-GPU cases and the
# two-rank cases that share one GPU (they pass --mca device_cuda_memory_use 30).
# The run-time libraries (BLAS, MPI) must be on LD_LIBRARY_PATH / PATH.
set -u
BUILD=$(cd "$1" && pwd); LABEL=$2; MODE=${3:-cpu}
OUT=${4:-./validation}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
REPO=$(cd "$(dirname "$0")/.." && pwd)
RES=$OUT/results_${LABEL}.tsv
LOGD=$OUT/logs_${LABEL}; mkdir -p $LOGD
echo -e "case\texit\tresidual\ttime_s\tgemms" > $RES
cd $BUILD
MPIRUN="mpirun --oversubscribe --bind-to none"
run() { # name, np, cmd...
  local name=$1; local np=$2; shift 2
  local log=$LOGD/$name.log
  if [ "$np" -gt 1 ]; then $MPIRUN -np $np "$@" > $log 2>&1; else "$@" > $log 2>&1; fi
  local rc=$?
  local resid=$(grep -oE 'result_accuracy= *[0-9.e+-]+' $log | tail -1 | awk '{print $2}')
  [ -z "$resid" ] && resid=$(grep -oE '\|\|A-LL.?\|\|/\|\|A\|\| *= *[0-9.e+-]+' $log | tail -1 | grep -oE '[0-9.e+-]+$')
  [ -z "$resid" ] && resid=$(grep -oE 'residual[^0-9]*[0-9.]+e[+-][0-9]+' $log | tail -1 | grep -oE '[0-9.]+e[+-][0-9]+')
  local t=$(grep -E 'TIME\(s\).*hicma_parsec_cholesky' $log | head -1 | sed -E 's/.*TIME\(s\) *([0-9.]+) .*/\1/')
  [ -z "$t" ] && t=$(grep -E 'TIME\(s\).*(trmm|TRMM)' $log | head -1 | sed -E 's/.*TIME\(s\) *([0-9.]+) .*/\1/')
  [ -z "$resid" ] && { grep -q 'PASSED' $log && resid=PASSED; grep -q 'FAILED' $log && resid=FAILED; }
  local g=$(grep -oE 'Number_of_GEMMs: .*' $log | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
  echo -e "$name\t$rc\t${resid:-NA}\t${t:-NA}\t${g:-}" >> $RES
  printf "%-40s rc=%s resid=%s t=%s\n" "$name" "$rc" "${resid:-NA}" "${t:-NA}"
}
P=./tests/testing_potrf_tlr
T=./tests/testing_trmm_mp
S=./tests/testing_syrk
# --- CPU cases ---
run tlr_dp_k2_p2            1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --check 1 --verbose 2
run tlr_mp_k1_band1_adapt   1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --check 1 --check_tol 1e-7 --verbose 2 --band_dense 1 --adaptive_decision 1 --kind_of_cholesky 1
run tlr_mp_k1_band1_noadapt 1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --check 1 --verbose 2 --band_dense 1 --adaptive_decision 0 --kind_of_cholesky 1
run tlr_mp_k1_dense_adapt   1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --check 1 --verbose 2 --band_dense 1000 --adaptive_decision 1 --kind_of_cholesky 1
run tlr_mp_k1_dense_admem   1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --check 1 --verbose 2 --band_dense 1000 --adaptive_decision 1 --adaptive_memory 1 --kind_of_cholesky 1
run tlr_kind2_band1         1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --kind_of_cholesky 2 --band_dense 1 --check 1 --verbose 2
run tlr_kind2_band2         1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --kind_of_cholesky 2 --band_dense 2 --check 1 --verbose 2
run tlr_kind2_band2_np2     2 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --kind_of_cholesky 2 --band_dense 2 --check 1 --verbose 2
run tlr_kind1_band2_np2     2 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --kind_of_cholesky 1 --band_dense 2 --adaptive_decision 1 --check 1 --check_tol 1e-7 --verbose 2
run tlr_dp_k2_p4_autoband   1 $P --N 5400 --NB 540 --fixedacc 1e-8 --maxrank 540 --kind_of_problem 4 --auto_band 1 --check 1 --verbose 2 -- -mca runtime_comm_coll_bcast 0
run dense_k5_fp64_cpu       1 $P --N 4096 --NB 512 --kind_of_problem 2 --check 1 --verbose 2
run dense_k5_band2_cpu      1 $P --N 4096 --NB 512 --kind_of_problem 2 --band_dense 2 --maxrank 256 --check 1 --verbose 2
run dense_k5_adapt_cpu      1 $P --N 4096 --NB 512 --kind_of_problem 2 --adaptive_decision 1 --check 1 --verbose 2
run band_k3_cpu             1 $P --N 4096 --NB 512 --kind_of_problem 2 --kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --check 1 --check_tol 1e-6 --verbose 2
run band_k4_cpu             1 $P --N 4096 --NB 512 --kind_of_problem 2 --kind_of_cholesky 4 --band_dense_sp 4 --check 1 --check_tol 1e-6 --verbose 2
run runtime_k10_cpu         1 $P --N 4096 --NB 512 --kind_of_problem 2 --adaptive_decision_runtime 1 --check_tol 1e-7 --check 1 --verbose 2
run runtime_k10_cpu_adapt1  1 $P --N 4096 --NB 512 --kind_of_problem 2 --adaptive_decision_runtime 1 --check_tol 1e-7 --adaptive_decision 1 --check 1 --verbose 2
run sparse_k6_cpu           1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --sparse 1 --check 1 --verbose 2
run sparse_k7_cpu           1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --sparse 2 --check 1 --verbose 2
run sparse_k6_np2           2 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --sparse 1 --check 1 --verbose 2
MESH=$REPO/stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt
if [ -f $MESH ]; then
run sparse_k6_mesh          1 $P --N 10370 --NB 1037 --fixedacc 1e-4 --maxrank 50 --adddiag 0.1 --kind_of_problem 6 --band_dense 1 --send_full_tile 0 --lookahead 1 --auto_band 0 --mesh_file $MESH --numobj 1 --rbf_kernel 0 --radius 4.63e-04 --order 2 --density -1 --sparse 1 --check 1 --verbose 2
run sparse_k7_mesh_np2      2 $P --N 10370 --NB 1037 --fixedacc 1e-4 --maxrank 50 --adddiag 0.1 --kind_of_problem 6 --band_dense 1 --send_full_tile 0 --lookahead 1 --auto_band 0 --mesh_file $MESH --numobj 1 --rbf_kernel 0 --radius 4.63e-04 --order 2 --density -1 --sparse 2 --check 1 --verbose 2
fi
run trmm_fp64               1 $T --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 0
run trmm_adapt              1 $T --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4
run trmm_adapt_sr           1 $T --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4 --enable_stochastic_rounding 1
run tlr_dp_k2_np2           2 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --check 1 --verbose 2
run dense_k5_fp64_np2       2 $P --N 4096 --NB 512 --kind_of_problem 2 --check 1 --verbose 2
run dense_k5_band2_np2      2 $P --N 4096 --NB 512 --kind_of_problem 2 --band_dense 2 --maxrank 256 --check 1 --verbose 2
run runtime_k10_np2         2 $P --N 4096 --NB 512 --kind_of_problem 2 --adaptive_decision_runtime 1 --check_tol 1e-7 --check 1 --verbose 2
run trmm_adapt_np2          2 $T --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4
if [ "$MODE" = gpu ]; then
run gpu_dense_k5_fp64       1 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --check 1 --verbose 2
run gpu_dense_k5_adapt      1 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --adaptive_decision 1 --check 1 --verbose 2
run gpu_dense_k5_band2      1 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --band_dense 2 --maxrank 512 --check 1 --verbose 2
run gpu_dense_k8_adapt      1 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --kind_of_cholesky 8 --adaptive_decision 1 --check 1 --verbose 2
run gpu_dense_k9_adapt      1 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --kind_of_cholesky 9 --adaptive_decision 1 --check 1 --check_tol 1e-6 --verbose 2
run gpu_runtime_k10         1 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --adaptive_decision_runtime 1 --check_tol 1e-7 --check 1 --verbose 2
run gpu_runtime_k10_admem   1 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --adaptive_decision_runtime 1 --check_tol 1e-7 --adaptive_memory 1 --check 1 --verbose 2
run gpu_tlr_kind2_band2     1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --kind_of_cholesky 2 --band_dense 2 --gpus 1 --check 1 --verbose 2
run gpu_tlr_kind1_band2     1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --kind_of_cholesky 1 --band_dense 2 --adaptive_decision 1 --gpus 1 --check 1 --check_tol 1e-7 --verbose 2
run gpu_tlr_mp_k1           1 $P --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --gpus 1 --check 1 --verbose 2 --band_dense 1000 --adaptive_decision 1 --kind_of_cholesky 1
run gpu_dense_k5_dtconv2    1 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --adaptive_decision 1 --datatype_convert 2 --check 1 --verbose 2
run gpu_trmm_adapt          1 $T --N 2000 --K 1500 --NB 256 --gpus 1 --check 1 --adaptive_decision 1 --fixedacc 1e-4
run gpu_dense_k5_fp64_np2   2 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --check 1 --verbose 2 -- --mca device_cuda_memory_use 30
run gpu_runtime_k10_np2     2 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --adaptive_decision_runtime 1 --check_tol 1e-7 --check 1 --verbose 2 -- --mca device_cuda_memory_use 30
run gpu_dense_k8_np2        2 $P --N 8192 --NB 1024 --kind_of_problem 2 --gpus 1 --kind_of_cholesky 8 --adaptive_decision 1 --check 1 --verbose 2 -- --mca device_cuda_memory_use 30
fi
echo "results: $RES"
