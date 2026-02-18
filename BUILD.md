# HiCMA-PaRSEC Build Guide

This document provides comprehensive instructions for building HiCMA-PaRSEC, a hierarchical matrix computation library powered by the PaRSEC runtime system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Build Options](#build-options)
4. [Build Methods](#build-methods)
5. [Platform-Specific Instructions](#platform-specific-instructions)

## Prerequisites

### Required Dependencies

- **CMake**
- **C/C++ Compiler**
- **Fortran Compiler**
- **MPI**
- **BLAS**
- **LAPACKE**
- **Git**

### Submodule Dependencies

The following dependencies are automatically built as part of the project:

- **DPLASMA** - Distributed Parallel Linear Algebra Software for Multicore Architectures
  - Provides distributed linear algebra kernels and PaRSEC runtime integration
- **STARS-H** - Software for Testing Accuracy, Reliability and Scalability of Hierarchical computations
  - Hierarchical matrix generation and approximation algorithms
  - **ECRC CMake Modules** - CMake modules for ECRC projects (nested submodule)
- **HCORE** - Hierarchical matrix core library
  - Low-rank matrix operations and BLAS kernels for hierarchical matrices
  - **ECRC CMake Modules** - CMake modules for ECRC projects (nested submodule)

These are managed as Git submodules and will be automatically built when you configure the project.

### Optional Dependencies

- **CUDA Toolkit** 12.0+ (for GPU acceleration)
- **ROCm** 4.0+ (for AMD GPU support)
- **CUTLASS** (for optimized CUDA kernels)
- **GSL** (GNU Scientific Library)
- **Ninja**

### System Requirements

- **Memory**: 4GB minimum, 8GB+ recommended
- **Disk Space**: 2GB for build, 5GB+ for full installation
- **CPU**: Multi-core processor recommended

## Quick Start

### 1. Update submodules 
```bash
git submodule update --init --recursive
```

### 2. Build

#### Basic Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. (-G Ninja)

# Build
make -j$(nproc) or ninja

```
If LAPACKE is not available, use the example script:
```bash
./scripts/build_hicma_with_lapack.sh
```

#### Using the Build Script

```bash
# Basic build
./scripts/build_release.sh

# Build with CUDA support
./scripts/build_release.sh --cuda

```

#### Using CMake Presets

```bash
# Configure and build with preset
cmake --preset=release
cmake --build --preset=release

# Run tests
ctest --preset=release
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `RelWithDebInfo` | Build type (Debug, Release, RelWithDebInfo, MinSizeRel) |
| `HICMA_PARSEC_HAVE_CUDA` | `OFF` | Enable CUDA support |
| `HICMA_PARSEC_HAVE_HIP` | `OFF` | Enable HIP support |
| `ENABLE_SANITIZER` | `OFF` | Enable AddressSanitizer instrumentation |
| `HICMA_ASAN_RUNTIME_LIBRARY` | empty | Optional absolute path to `libasan.so` when sanitizer link probe fails |
| `ENABLE_CUTLASS` | `OFF` | Enable CUTLASS support (requires CUDA) |
| `CMAKE_CUDA_ARCHITECTURES` | `70 80 90 90a` | Target CUDA architectures |

Check CMake for more options.

#### Example Configurations

```bash
# CUDA build
cmake -DHICMA_PARSEC_HAVE_CUDA=ON

# CUDA build with CUTLASS
cmake -DHICMA_PARSEC_HAVE_CUDA=ON -DENABLE_CUTLASS=ON -DCUTLASS_ROOT=/path/to/cutlass ..

# HIP build for AMD GPUs
cmake -DHICMA_PARSEC_HAVE_HIP=ON

```

## Troubleshooting

### Common Issues

#### 1. Submodule Issues

```bash
# Error: submodules not found
git submodule update --init --recursive

# Or use the management script
./scripts/update_submodules.sh init

# Check submodule status
./scripts/update_submodules.sh status
```

#### 2. Dependency Not Found

```bash
# Set environment variables for dependencies
export HCORE_DIR=/path/to/hcore
export STARSH_DIR=/path/to/starsh
export BLAS_LIBRARIES=/path/to/blas
```

#### 3. CUDA Issues

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Set CUDA architectures explicitly
cmake -DCMAKE_CUDA_ARCHITECTURES="70;80;90;90a" ..
```

#### 4. MPI Issues

```bash
# Check MPI installation
mpicc --version
mpirun --version

# Use specific MPI
cmake -DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpicxx ..
```

#### 5. Memory Issues

```bash
# Reduce parallel jobs
make -j2
```

#### 6. System Specific Issues

Some system may have different configurations.
##### Frontier
This is the cmake command used on Frontier supercomtpuer.
```bash
cmake ../ \
  -DCMAKE_INSTALL_PREFIX=`pwd`/installdir \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DHICMA_PARSEC_HAVE_HIP=ON \
  -DCMAKE_HIP_FLAGS="-I$CRAY_MPICH_DIR/include -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_C_FLAGS="-I$OLCF_OPENBLAS_ROOT/include -I$ROCM_PATH/include -DLAPACKE_UTILS -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_CXX_FLAGS="-I$OLCF_OPENBLAS_ROOT/include -I$ROCM_PATH/include -DLAPACKE_UTILS -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$ROCM_PATH/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib" \
  -DROCM_PATH="$ROCM_PATH"
```

GPU-direct:
```bash
module load craype-accel-amd-gfx90a
module load rocm
export MPICH_GPU_SUPPORT_ENABLED=1
cmake ../ \
  -DCMAKE_INSTALL_PREFIX=`pwd`/installdir \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DHICMA_PARSEC_HAVE_HIP=ON \
  -DCMAKE_HIP_FLAGS="-I$CRAY_MPICH_DIR/include -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_C_FLAGS="-I$OLCF_OPENBLAS_ROOT/include -I$ROCM_PATH/include -DLAPACKE_UTILS -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_CXX_FLAGS="-I$OLCF_OPENBLAS_ROOT/include -I$ROCM_PATH/include -DLAPACKE_UTILS -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$ROCM_PATH/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib -lamdhip64 -lmpi_gtl_hsa" \
  -DROCM_PATH="$ROCM_PATH"

``` 
##### Jetstream2 
This is the cmake command using gcc and openmpi.
```bash
cmake .. -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_Fortran_COMPILER=mpif90 -DHICMA_PARSEC_HAVE_CUDA=ON
```

#### 7. AddressSanitizer Link Failures

If you enable `ENABLE_SANITIZER=ON` and CMake reports that `-fsanitize=address` cannot link, provide an explicit `libasan.so` path:

```bash
cmake -S . -B build \
  -DENABLE_SANITIZER=ON \
  -DHICMA_ASAN_RUNTIME_LIBRARY=/absolute/path/to/libasan.so
```

Example on systems that define `ICL_GCC_RUNTIME_ROOT`:

```bash
cmake -S . -B build \
  -DENABLE_SANITIZER=ON \
  -DHICMA_ASAN_RUNTIME_LIBRARY="$ICL_GCC_RUNTIME_ROOT/lib/libasan.so.8"
```

### Debug Build

```bash
# Create debug build
cmake --preset=debug
cmake --build --preset=debug

# Run with debugger
gdb ./your_program
```

### Verbose Output

```bash
# Verbose make
make VERBOSE=1

# Verbose cmake
cmake --debug-output ..

# Verbose ctest
ctest --verbose
```
