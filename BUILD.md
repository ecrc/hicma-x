# HiCMA-X Build Guide

How to build, test and install HiCMA-X (formerly HiCMA-PaRSEC), a library of tile low-rank and mixed-precision linear algebra on the PaRSEC runtime.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Build Options](#build-options)
4. [Build Methods](#build-methods)
5. [Platform-Specific Instructions](#platform-specific-instructions)

## Prerequisites

### Required Dependencies

- **CMake** 3.21 or later
- **C11 / C++17 compilers** (GCC 11+, Clang 14+, or the vendor compilers of the machine)
- **Fortran compiler** (for the BLAS/LAPACK linkage of the test drivers)
- **MPI** with `MPI_THREAD_MULTIPLE` (Open MPI 4/5, MPICH 4, Cray MPICH)
- **BLAS** and **LAPACKE** (OpenBLAS or MKL) that is safe to call from several
  threads at once: MKL sequential, or a threaded OpenBLAS with
  `OPENBLAS_NUM_THREADS=1` (a sequential OpenBLAS needs `USE_LOCKING=1`). CTest
  checks this first (`TESTS.md`)
- **Git** (submodules)

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
- **OmniCast** - Data-type conversion on CPUs and GPUs (FP64, FP32, FP16, BF16,
  FP8, integers): every element conversion of HiCMA-X; built with the CUDA or
  HIP backend that HiCMA-X is configured with

These are managed as Git submodules and will be automatically built when you configure the project.

### Optional Dependencies

- **CUDA Toolkit** 12.0+ (for GPU acceleration)
- **ROCm** 6 (for AMD GPU support; tested with 6.2.4 on gfx90a)
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
cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/hicma-x   # or leave out -G Ninja for make
cmake --build build                                               # all cores by default with Ninja
ctest --test-dir build -L hicma -LE gpu --output-on-failure        # the CPU test cases
```

The configure step finds MPI and the BLAS/LAPACKE stack (DPLASMA's
`FindLAPACKE`: set `LAPACKE_ROOT` or `BLA_VENDOR` to choose one) and records
their directories in the binaries, so the drivers start without
`LD_LIBRARY_PATH` changes. STARS-H and HCORE accept the `BLA_VENDOR` values
their own finder knows (`OpenBLAS`, `Intel10_64lp`, `Intel10_64lp_seq`,
`Generic`, `ATLAS`, `Apple`, `IBMESSL`); others, such as `FlexiBLAS`, `AOCL`
or `NVHPC`, stop their configure step. To take OpenBLAS where MKL is also installed,
give the vendor and its directory to every find module (STARS-H and HCORE
use their own, which take the four hints):

```bash
cmake -S . -B build -G Ninja -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=$OPENBLAS \
      -DBLAS_DIR=$OPENBLAS -DCBLAS_DIR=$OPENBLAS -DLAPACK_DIR=$OPENBLAS -DLAPACKE_DIR=$OPENBLAS
```
If no LAPACKE is available, `scripts/build_hicma_with_lapack.sh` builds the
reference LAPACK 3.12.1 (BLAS, CBLAS, LAPACK, LAPACKE) in `lapack_build/` and
HiCMA-X against it. The reference BLAS is slow; prefer OpenBLAS or MKL.
```bash
./scripts/build_hicma_with_lapack.sh [build-dir] [extra CMake options]
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
| `BUILD_TESTING` | `ON` | Build the drivers in `tests/` and register the CTest cases |
| `HICMA_PARSEC_GENOMICS` | `OFF` | Build the genomics (kernel ridge regression) drivers and compile the library with `GENOMICS=1`; needs CUDA (the genomics kernels have no HIP bodies). The Cholesky drivers refuse to run in such a build, so their CTest cases are not registered |
| `BUILD_SHARED_LIBS` | `ON` | Shared (`ON`) or static libraries |
| `HICMA_PARSEC_HAVE_CUDA` | `OFF` | Enable CUDA support. Also builds the FP8 (E4M3) GEMMs, which kinds 8, 9 and 10 use only when every GPU has compute capability >= 8.9; `--fp8 0` turns them off at runtime |
| `HICMA_PARSEC_HAVE_HIP` | `OFF` | Enable HIP support |
| `ENABLE_SANITIZER` | `OFF` | Enable AddressSanitizer instrumentation |
| `HICMA_ASAN_RUNTIME_LIBRARY` | empty | Optional absolute path to `libasan.so` when sanitizer link probe fails |
| `ENABLE_CUTLASS` | `OFF` | Enable CUTLASS support (requires CUDA) |
| `CMAKE_CUDA_ARCHITECTURES` | `70 80 90 90a` (`80 90 90a` with CUDA 13) | Target CUDA architectures; the `CUDAARCHS` environment variable also sets them |

Check CMake for more options.

#### Testing

```bash
ctest --test-dir build --output-on-failure -L hicma -LE gpu
```

Every registered case runs with its correctness check; see `TESTS.md` for the
labels, the GPU and MPI cases and the stand-alone script.

#### Installing and using the installed library

```bash
cmake --install build --prefix /opt/hicma-x
```

A consumer project finds the package and links the exported target; the
install prefix must also be where the bundled PaRSEC and DPLASMA were
installed (the same `cmake --install` does both):

```cmake
find_package(HICMAParsec REQUIRED)
add_executable(app app.c)
target_link_libraries(app PRIVATE HICMAParsec::hicma_parsec)
```

```bash
cmake -S app -B build-app -DCMAKE_PREFIX_PATH=/opt/hicma-x
```

`tests/install_consumer/` is such a project (it builds the Cholesky driver
and the tile/LAPACK conversion test against the installed package) and CI
runs it. Configure the consumer with the compilers of the HiCMA-X build; on
Frontier that is `-DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++
-DCMAKE_Fortran_COMPILER=mpif90` (CMake's default `ftn` fails its compiler
check there unless a CPU target module such as `craype-x86-trento` is loaded).

#### Calling HiCMA-X from an application

`tests/testing_potrf.c` is the reference sequence: `hicma_parsec_init`,
`hicma_parsec_matrix_generation`, `hicma_parsec_matrix_pre_analysis`,
`hicma_parsec_potrf`, `hicma_parsec_matrix_post_analysis`, `hicma_parsec_fini`.
Points to know when embedding it:

- `hicma_parsec_init(argc, argv, ...)` takes its parameters from the argument
  vector (the options of `TESTS.md`); pass `0, NULL` for the defaults. The
  vector is only read. Every option it does not know is an error, so strip the
  application's own options first; arguments after `--` go to PaRSEC.
- If the application initialized MPI, HiCMA-X leaves it initialized at
  `hicma_parsec_fini`; otherwise it initializes and finalizes MPI itself.
- Zero-initialize the `hicma_parsec_matrix_analysis_t` (`= {0}`):
  `hicma_parsec_fini` prints its counters and frees from it, and the dense
  path of `hicma_parsec_matrix_pre_analysis` leaves the sparse counters unset.
- Invalid parameters terminate the process with a message (`exit(1)` on every
  rank), as in the drivers.
- The matrices are PaRSEC tiled descriptors. `hicma_parsec_lapack_tile_convert`
  (`hicma_parsec_datatype_convert.h`) copies between a descriptor and the
  process's local column-major buffer in either direction, for any element
  type; `tests/testing_lapack_tile.c` shows its use.

#### Example Configurations

```bash
# CUDA build
cmake -S . -B build -G Ninja -DHICMA_PARSEC_HAVE_CUDA=ON

# CUDA build with CUTLASS
cmake -S . -B build -G Ninja -DHICMA_PARSEC_HAVE_CUDA=ON -DENABLE_CUTLASS=ON -DCUTLASS_ROOT=/path/to/cutlass

# HIP build for AMD GPUs
cmake -S . -B build -G Ninja -DHICMA_PARSEC_HAVE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a

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
# HCORE, STARS-H, DPLASMA and PaRSEC are built from the submodules; only
# MPI and BLAS/LAPACKE come from the system. Choose the BLAS/LAPACKE stack:
export LAPACKE_ROOT=/path/to/lapacke   # found by DPLASMA's FindLAPACKE (or cmake -DBLA_VENDOR=...)
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
cmake -S . -B build -DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpicxx
```

#### 5. Memory Issues

```bash
# Reduce parallel jobs
cmake --build build -j 2
```

#### 6. System Specific Issues

`scripts/machines/install_<machine>.sh` hold the modules and CMake options of
the machines HiCMA-X is used on: Alps, Frontier, Fugaku (script kept; not
supported or validated in 0.2.0), the ICL machines, Leonardo, Polaris,
Shaheen, TACC and Vista. Each configures and builds the
whole tree once, STARS-H, HCORE, DPLASMA and PaRSEC included:

```bash
scripts/machines/install_icl.sh [build-dir] [extra CMake options]   # build-dir defaults to build/
cmake --install build                                              # into build/installdir
```

##### Frontier
`install_frontier.sh` needs `ROCM_PATH`, `OLCF_OPENBLAS_ROOT` and
`CRAY_MPICH_DIR` (modules `PrgEnv-amd`, `rocm`, `openblas`). Without
GPU-aware MPI, multi-rank GPU runs move remote data through host memory
(HiCMA-X detects Cray MPICH without `MPICH_GPU_SUPPORT_ENABLED=1`). For
GPU-aware MPI, link the GTL library and set the variable at run time:
```bash
module load craype-accel-amd-gfx90a rocm
scripts/machines/install_frontier.sh build -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib -lamdhip64 -L$CRAY_MPICH_ROOTDIR/gtl/lib -lmpi_gtl_hsa"
export MPICH_GPU_SUPPORT_ENABLED=1
```
On AMD GPUs pass `--tensor_gemm_hp 0x20` (BF16) with adaptive precisions in the
dense layout: their FP16 GEMMs flush small entries to zero (`TESTS.md`).

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
# Verbose build
cmake --build build --verbose

# Verbose cmake
cmake --debug-output ..

# Verbose ctest
ctest --verbose
```
