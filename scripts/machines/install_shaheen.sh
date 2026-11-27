#!/bin/bash -l
# Build HiCMA-X (CPU only) on Shaheen (KAUST; Cray), with the Intel
# programming environment and sequential MKL.
#   scripts/machines/install_shaheen.sh [build-dir] [extra CMake options]
# The build directory defaults to build/ in the repository. CMake also builds
# DPLASMA, PaRSEC, STARS-H and HCORE from the submodules (BUILD.md);
# "cmake --install <build-dir>" then installs into <build-dir>/installdir.
# DPLASMA's code generator needs Python 3 (the system python3 will do).
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# the first argument is the build directory unless it is a CMake option
BUILD=$ROOT/build
case "${1:-}" in ""|-*) ;; *) BUILD=$1; shift ;; esac
BUILD=$(realpath -m "$BUILD")

module swap PrgEnv-cray PrgEnv-intel
module load cmake gsl
module list
set -e

export CRAYPE_LINK_TYPE=dynamic

cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD/installdir" \
    -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_Fortran_FLAGS="-nofor-main" \
    -DDPLASMA_PRECISIONS="s;d" \
    -DBLA_VENDOR=Intel10_64lp_seq \
    -DGSL_LIBRARY="$GSL_HOME/lib/libgsl.so" -DGSL_CBLAS_LIBRARY="$GSL_HOME/lib/libgslcblas.so" \
    -DPARSEC_ATOMIC_USE_GCC_32_BUILTINS_EXITCODE=0 \
    -DPARSEC_ATOMIC_USE_GCC_64_BUILTINS_EXITCODE=0 \
    -DPARSEC_ATOMIC_USE_GCC_128_BUILTINS_EXITCODE=0 \
    "$@"
cmake --build "$BUILD" -j "${JOBS:-8}"
