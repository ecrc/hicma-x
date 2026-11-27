#!/bin/bash -l
# Build HiCMA-X on Vista (TACC; NVIDIA GH200, aarch64).
#   scripts/machines/install_tacc_vista.sh [build-dir] [extra CMake options]
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

cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD/installdir" \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
    -DDPLASMA_PRECISIONS="s;d" \
    -DHICMA_PARSEC_HAVE_CUDA=ON \
    "$@"
cmake --build "$BUILD" -j "${JOBS:-16}"
