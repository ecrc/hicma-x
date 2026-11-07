#!/bin/bash
#
# HiCMA-X regression script: runs the drivers with their correctness check on
# and fails on the first wrong result. Run it from a build directory:
#
#     cd build && ../scripts/test_CI.sh
#
# Every Cholesky run passes --check 1, which compares the factor against a dense
# reference and L*L^T against the generated matrix and exits non-zero when the
# residual exceeds the bound. The bound is --fixedacc (the compression
# tolerance) unless --check_tol is given; mixed-precision runs that compute
# whole bands in single or half precision cannot reach an FP64-level fixedacc
# and state the bound they are expected to meet instead.
#
# Environment:
#   HICMA_CLIMATE_DATA   directory with the climate emulator input (the test is
#                        skipped when unset or missing)
#   HICMA_MESH_FILE      SARS-CoV-2 mesh (default: the copy in stars-h)
#   HICMA_GPUS           number of GPUs for the GPU cases (default 0: skipped)
#   HICMA_MPIRUN         launcher for the 2-rank cases (default: mpirun
#                        --oversubscribe; set to "skip" to run single rank only)
#
# The same cases are registered with CTest (tests/CMakeLists.txt); this script
# is the stand-alone variant for machines where ctest is inconvenient.

set -u

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[SKIP]${NC} $1"; }
print_error()   { echo -e "${RED}[FAIL]${NC} $1"; }

if [ ! -x tests/testing_potrf_tlr ]; then
    print_error "tests/testing_potrf_tlr not found: run this script from the build directory"
    exit 1
fi

GPUS=${HICMA_GPUS:-0}
MPIRUN=${HICMA_MPIRUN:-"mpirun --oversubscribe"}
MESH=${HICMA_MESH_FILE:-$(dirname "$0")/../stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt}
failed=0
passed=0
skipped=0

# run <name> <ranks> <command...>
run() {
    local name=$1 ranks=$2; shift 2
    if [ "$ranks" -gt 1 ]; then
        if [ "$MPIRUN" = "skip" ]; then print_warning "$name (needs $ranks ranks)"; skipped=$((skipped+1)); return; fi
        set -- $MPIRUN -np "$ranks" "$@"
    fi
    print_status "$name: $*"
    if "$@"; then
        print_success "$name"; passed=$((passed+1))
    else
        print_error "$name (exit code $?)"; failed=$((failed+1))
    fi
}

P=./tests/testing_potrf_tlr
T=./tests/testing_trmm_mp
COMMON="--kind_of_problem 2 --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --check 1"

# Cholesky, every kind that runs on CPUs (kind_of_problem 2: 2-D square exponential)
run potrf_dense_fp64        1 $P $COMMON
run potrf_tlr_dp_band1      1 $P $COMMON --kind_of_cholesky 2 --band_dense 1
run potrf_tlr_dp_band2      1 $P $COMMON --kind_of_cholesky 2 --band_dense 2
run potrf_tlr_mp_band1      1 $P $COMMON --kind_of_cholesky 1 --band_dense 1 --adaptive_decision 0
run potrf_tlr_mp_adaptive   1 $P $COMMON --kind_of_cholesky 1 --band_dense 1 --adaptive_decision 1 --check_tol 1e-7
run potrf_mp_dense_adaptive 1 $P $COMMON --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1 --check_tol 1e-7
run potrf_mp_dense_admem    1 $P $COMMON --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1 --adaptive_memory 1 --check_tol 1e-7
run potrf_band_dp_sp        1 $P $COMMON --kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --check_tol 1e-6
run potrf_runtime_decision  1 $P $COMMON --adaptive_decision_runtime 1 --check_tol 1e-7
run potrf_sparse_general    1 $P $COMMON --sparse 1
run potrf_sparse_balance    1 $P $COMMON --sparse 2
run potrf_autoband_3d       1 $P --N 5400 --NB 540 --fixedacc 1e-8 --maxrank 540 --kind_of_problem 4 --auto_band 1 --check 1 -- -mca runtime_comm_coll_bcast 0

# Two ranks
run potrf_dense_fp64_np2    2 $P $COMMON
run potrf_tlr_dp_band2_np2  2 $P $COMMON --kind_of_cholesky 2 --band_dense 2
run potrf_runtime_np2       2 $P $COMMON --adaptive_decision_runtime 1 --check_tol 1e-7
run potrf_sparse_np2        2 $P $COMMON --sparse 1

# Mixed-precision TRMM against DPLASMA's FP64 TRMM, with partial tiles
run trmm_fp64               1 $T --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 0
run trmm_adaptive           1 $T --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4
run trmm_adaptive_sr        1 $T --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4 --enable_stochastic_rounding 1
run trmm_adaptive_np2       2 $T --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4

# 3-D RBF mesh (needs the SARS-CoV-2 mesh shipped with stars-h)
if [ -f "$MESH" ]; then
    run potrf_mesh_sparse   1 $P --N 10370 --NB 1037 --fixedacc 1e-4 --maxrank 50 --adddiag 0.1 --kind_of_problem 6 --band_dense 1 --send_full_tile 0 --lookahead 1 --auto_band 0 --mesh_file "$MESH" --numobj 1 --rbf_kernel 0 --radius 4.63e-04 --order 2 --density -1 --sparse 1 --check 1
else
    print_warning "mesh file $MESH not found"; skipped=$((skipped+1))
fi

# GPU cases
if [ "$GPUS" -gt 0 ]; then
    G="--kind_of_problem 2 --N 8192 --NB 1024 --gpus $GPUS --check 1"
    run gpu_dense_fp64          1 $P $G
    run gpu_dense_adaptive      1 $P $G --adaptive_decision 1
    run gpu_dense_band2         1 $P $G --band_dense 2 --maxrank 512
    run gpu_fp8_static          1 $P $G --kind_of_cholesky 8 --adaptive_decision 1
    run gpu_fp8_sp              1 $P $G --kind_of_cholesky 9 --adaptive_decision 1 --check_tol 1e-6
    run gpu_runtime_decision    1 $P $G --adaptive_decision_runtime 1 --check_tol 1e-7
    run gpu_runtime_admem       1 $P $G --adaptive_decision_runtime 1 --adaptive_memory 1 --check_tol 1e-7
    run gpu_tlr_mp              1 $P $COMMON --gpus $GPUS --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1 --check_tol 1e-7
    run gpu_tlr_dp_band2        1 $P $COMMON --gpus $GPUS --kind_of_cholesky 2 --band_dense 2
    run gpu_trmm_adaptive       1 $T --N 2000 --K 1500 --NB 256 --gpus $GPUS --check 1 --adaptive_decision 1 --fixedacc 1e-4
else
    print_warning "GPU cases (set HICMA_GPUS=1)"; skipped=$((skipped+1))
fi

# Climate emulator (needs the external data set)
if [ -n "${HICMA_CLIMATE_DATA:-}" ] && [ -d "$HICMA_CLIMATE_DATA" ]; then
    run climate_emulator        1 ./tests/testing_climate_emulator --latitude 120 --NB 1440 --N 14400 --gpus "$GPUS" --adaptive_decision 1 --mesh_file "$HICMA_CLIMATE_DATA"
else
    print_warning "climate emulator (set HICMA_CLIMATE_DATA to the data directory)"; skipped=$((skipped+1))
fi

echo "----------------------------------------"
echo "passed: $passed  failed: $failed  skipped: $skipped"
[ "$failed" -eq 0 ]
