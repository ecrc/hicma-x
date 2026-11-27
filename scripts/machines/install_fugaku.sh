#!/bin/bash -l
# Build HiCMA-X on Fugaku (RIKEN; A64FX), on a compute node with the native
# Fujitsu compilers, e.g. after
#   pjsub --interact -L "node=1,freq=2200" -L "rscgrp=int" -L "elapse=1:00:00" --sparam "wait-time=600"
#   scripts/machines/install_fugaku.sh [build-dir] [extra CMake options]
# The build directory defaults to build/ in the repository. CMake also builds
# DPLASMA, PaRSEC, STARS-H and HCORE from the submodules (BUILD.md);
# "cmake --install <build-dir>" then installs into <build-dir>/installdir.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# the first argument is the build directory unless it is a CMake option
BUILD=$ROOT/build
case "${1:-}" in ""|-*) ;; *) BUILD=$1; shift ;; esac
BUILD=$(realpath -m "$BUILD")

source /vol0004/apps/oss/spack/share/spack/setup-env.sh
export MODULEPATH=/vol0004/apps/arm-lib/modulefiles:/vol0004/apps/oss/spack/share/spack/modules/linux-rhel8-a64fx:$MODULEPATH
module load gsl ninja cmake
module list
set -e

export PLE_MPI_STD_EMPTYFILE=off
export FLIB_SCCR_CNTL=FALSE
export FLIB_PTHREAD=1

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD/installdir" \
    -DCMAKE_C_COMPILER=mpifcc -DCMAKE_CXX_COMPILER=mpiFCC -DCMAKE_Fortran_COMPILER=mpifrt \
    -DCMAKE_C_FLAGS="-Nclang -fPIC -Ofast" -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" \
    -DBLAS_LIBRARIES="-L$FJSVXTCLANGA/lib64 -lfjlapacksve -SSL2 -SSL2MPI" \
    -DBLAS_INCLUDE_DIRS="$FJSVXTCLANGA/include" \
    -DDPLASMA_PRECISIONS="s;d" \
    -DSUPPORT_FORTRAN=ON \
    -DON_FUGAKU=ON \
    -DHWLOC_DIR=/lib64 \
    -DBUILD_SHARED_LIBS=OFF \
    "$@"
cmake --build "$BUILD"
