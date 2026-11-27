#!/bin/bash -l
# Build HiCMA-X on the ICL machines (University of Tennessee; guyot, hexane,
# methane, ...), with CUDA and sequential MKL.
#   scripts/machines/install_icl.sh [build-dir] [extra CMake options]
# The build directory defaults to build/ in the repository. CMake also builds
# DPLASMA, PaRSEC, STARS-H and HCORE from the submodules (BUILD.md);
# "cmake --install <build-dir>" then installs into <build-dir>/installdir.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# the first argument is the build directory unless it is a CMake option
BUILD=$ROOT/build
case "${1:-}" in ""|-*) ;; *) BUILD=$1; shift ;; esac
BUILD=$(realpath -m "$BUILD")

module load env/gcc12-cuda12 cmake ninja openmpi intel-oneapi-mkl cuda python
module list
set -e

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD/installdir" \
    -DDPLASMA_PRECISIONS="s;d" \
    -DBLA_VENDOR=Intel10_64lp_seq \
    -DHICMA_PARSEC_HAVE_CUDA=ON \
    "$@"
cmake --build "$BUILD"
