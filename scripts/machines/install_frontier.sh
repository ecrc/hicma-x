#!/bin/bash
# Build HiCMA-X on Frontier (OLCF; AMD MI250X).
#   scripts/machines/install_frontier.sh [build-dir] [extra CMake options]
# The build directory defaults to build/ in the repository. CMake also builds
# DPLASMA, PaRSEC, STARS-H and HCORE from the submodules (BUILD.md);
# "cmake --install <build-dir>" then installs into <build-dir>/installdir.
# Needs ROCM_PATH (rocm module), OLCF_OPENBLAS_ROOT (openblas module) and
# CRAY_MPICH_DIR. Without MPICH_GPU_SUPPORT_ENABLED=1 a multi-rank GPU run
# moves remote data through host memory (Cray MPICH cannot read GPU buffers
# then). For GPU-aware MPI, load craype-accel-amd-gfx90a, export
# MPICH_GPU_SUPPORT_ENABLED=1 and pass
# -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib -lamdhip64 -L$CRAY_MPICH_ROOTDIR/gtl/lib -lmpi_gtl_hsa".
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# the first argument is the build directory unless it is a CMake option
BUILD=$ROOT/build
case "${1:-}" in ""|-*) ;; *) BUILD=$1; shift ;; esac
BUILD=$(realpath -m "$BUILD")

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD/installdir" \
    -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_Fortran_COMPILER=mpif90 \
    -DDPLASMA_PRECISIONS="s;d" \
    -DHICMA_PARSEC_HAVE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DROCM_PATH="$ROCM_PATH" \
    -DCMAKE_HIP_FLAGS="-I$CRAY_MPICH_DIR/include -D__HIP_PLATFORM_AMD__" \
    -DCMAKE_C_FLAGS="-I$OLCF_OPENBLAS_ROOT/include -I$ROCM_PATH/include -DLAPACKE_UTILS -D__HIP_PLATFORM_AMD__" \
    -DCMAKE_CXX_FLAGS="-I$OLCF_OPENBLAS_ROOT/include -I$ROCM_PATH/include -DLAPACKE_UTILS -D__HIP_PLATFORM_AMD__" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$ROCM_PATH/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib" \
    "$@"
cmake --build "$BUILD" -j "${JOBS:-8}"
