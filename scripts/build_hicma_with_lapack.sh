#!/bin/bash
# Build HiCMA-X against a reference LAPACK (BLAS, CBLAS, LAPACK, LAPACKE) built
# here, for systems without a LAPACKE. OpenBLAS or MKL are much faster: use them
# when the system has them (BUILD.md).
#   scripts/build_hicma_with_lapack.sh [build-dir] [extra CMake options]
# e.g. scripts/build_hicma_with_lapack.sh build -DHICMA_PARSEC_HAVE_CUDA=ON
# LAPACK ($LAPACK_VERSION, default v3.12.1) is installed in lapack_build/ of the
# repository once and reused. The build directory defaults to build/ in the
# repository.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# the first argument is the build directory unless it is a CMake option
BUILD=$ROOT/build
case "${1:-}" in ""|-*) ;; *) BUILD=$1; shift ;; esac
BUILD=$(realpath -m "$BUILD")
LAPACK_VERSION=${LAPACK_VERSION:-v3.12.1}
LAPACK_SRC=$ROOT/lapack_build/lapack
LAPACK_PREFIX=$ROOT/lapack_build/installdir
JOBS=${JOBS:-$(nproc)}

if [ ! -f "$LAPACK_PREFIX/include/lapacke.h" ]; then
    [ -d "$LAPACK_SRC" ] || git clone --depth 1 --branch "$LAPACK_VERSION" \
        https://github.com/Reference-LAPACK/lapack.git "$LAPACK_SRC"
    cmake -S "$LAPACK_SRC" -B "$LAPACK_SRC/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$LAPACK_PREFIX" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_SHARED_LIBS=ON -DCBLAS=ON -DLAPACKE=ON -DLAPACKE_WITH_TMG=ON
    cmake --build "$LAPACK_SRC/build" -j "$JOBS"
    cmake --install "$LAPACK_SRC/build"
fi

# DPLASMA searches with CMake's FindBLAS (CMAKE_PREFIX_PATH) and its FindLAPACKE
# (LAPACKE_DIR); STARS-H and HCORE with their own modules, which take the
# BLAS_DIR, CBLAS_DIR, LAPACK_DIR and LAPACKE_DIR hints.
cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD/installdir" \
    -DCMAKE_PREFIX_PATH="$LAPACK_PREFIX" \
    -DBLA_VENDOR=Generic \
    -DBLAS_DIR="$LAPACK_PREFIX" -DCBLAS_DIR="$LAPACK_PREFIX" \
    -DLAPACK_DIR="$LAPACK_PREFIX" -DLAPACKE_DIR="$LAPACK_PREFIX" \
    "$@"
cmake --build "$BUILD" -j "$JOBS"
