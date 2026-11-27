#!/bin/bash -l
# Build HiCMA-X on Polaris (ALCF; NVIDIA A100).
#   scripts/machines/install_polaris.sh [build-dir] [extra CMake options]
# The build directory defaults to build/ in the repository. CMake also builds
# DPLASMA, PaRSEC, STARS-H and HCORE from the submodules (BUILD.md);
# "cmake --install <build-dir>" then installs into <build-dir>/installdir.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# the first argument is the build directory unless it is a CMake option
BUILD=$ROOT/build
case "${1:-}" in ""|-*) ;; *) BUILD=$1; shift ;; esac
BUILD=$(realpath -m "$BUILD")

module list
set -e

# A LAPACKE installation (scripts/build_hicma_with_lapack.sh builds one)
export LAPACKE_ROOT=${LAPACKE_ROOT:-/home/cql0536/software/lapack/build/installdir}

cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD/installdir" \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
    -DDPLASMA_PRECISIONS="s;d" \
    -DHICMA_PARSEC_HAVE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${CRAY_NVIDIA_PREFIX}/math_libs/lib64" \
    "$@"
cmake --build "$BUILD" -j "${JOBS:-16}"
