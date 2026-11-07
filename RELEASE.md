# HiCMA-X release notes

## 0.2.0

HiCMA-X (formerly HiCMA-PaRSEC) is a library of tile low-rank and
mixed-precision linear algebra on the PaRSEC runtime: Cholesky factorizations
with dense, banded, low-rank and adaptively chosen tile precisions (FP64, FP32,
FP16, FP8), a mixed-precision TRMM, and the kernels behind the climate emulator
and the genomics (kernel ridge regression) applications.

### What is new since 0.1.0

- Runtime precision decisions for the dense mixed-precision Cholesky
  (`--adaptive_decision_runtime 1`), FP8 GEMMs with per-tile scaling, and a
  mixed-precision TRMM with stochastic rounding.
- A release-quality pass over the existing code: the correctness check is
  binding and drives the exit code, the diagonal shift is applied once, several
  memory-safety bugs in initialization and the file readers are fixed, the
  installed package is usable, and the ten Cholesky wrappers share one
  implementation. The full list is in `CHANGELOG.md`; the review that led to it
  is in `docs/code-review.md`.

### Supported configurations

| Configuration | Status in 0.2.0 |
|---|---|
| Linux x86_64, CPU only (OpenBLAS or MKL, LAPACKE, Open MPI or MPICH) | tested (CI, 1 and 2 ranks) |
| Linux x86_64 + CUDA 12 (A100; H100 for FP8) | tested on one A100 (1 and 2 ranks sharing it); H100 FP8 results from the development of this release are in `PR` history, not re-run here |
| Linux + ROCm 6 (HIP, gfx90a) | compiles; not run on AMD hardware for this release |
| Fugaku (`ON_FUGAKU`), macOS, Windows | not supported |

Requirements: CMake 3.21, a C11/C++17 compiler, a Fortran compiler (BLAS/LAPACK
linkage), MPI with `MPI_THREAD_MULTIPLE`, BLAS, LAPACKE; CUDA 12.0 or later for
GPU builds (`90a` architecture needs 12.0); GSL optional (Matérn kernels).

### Known limitations

- The Cholesky drivers require `N` to be a multiple of `NB` (rejected at
  start-up). The TRMM supports partial tiles.
- Cholesky kinds 1 to 4 have CUDA bodies only; on HIP they run on the CPU.
- `--enable_stochastic_rounding` is implemented for kinds 1 to 5 and the TRMM,
  not for kinds 8 to 10 and not together with FP8 GEMMs.
- `--tensor_gemm 0x80` is a TRMM-only mode; `--datatype_convert 2` needs GPUs.
- In runtime-decision mode the CPU GEMM keeps FP16 updates into an FP32 tile in
  FP64 while the GPU GEMM keeps FP32; CPU-only and GPU runs of kind 10 therefore
  differ in their storage precisions (results are within tolerance in both).
- The residual of mixed-precision runs is bounded by the precisions used, not by
  `--fixedacc`; pass `--check_tol` with the bound you expect (see `TESTS.md`).
- The genomics drivers need `-DHICMA_PARSEC_GENOMICS=ON` and a GPU build; the
  climate emulator needs its external data set. Neither is covered by CI.
- Several ranks sharing one GPU need `-- --mca device_cuda_memory_use <percent>`.

### Validation

`docs/release-readiness.md` records what was built and run for this release:
CPU, CUDA (A100) and HIP (compile only) configurations, the CTest suite, the
validation matrix (`scripts/validation_matrix.sh`), the install consumer and a
before/after timing comparison.

### Upgrading from 0.1.0

- `libhicma_parsec` has SOVERSION 0 (it was 2, from a stale fallback version).
- Results change for problem kinds 0, 1, 5, 6 and 7 when `--adddiag` is non-zero
  (the shift is no longer applied twice), and for TLR kinds 2, 6, 7 only in
  runs that silently overflowed their rank cap before (they now abort with
  HCORE's message, as kind 1 did).
- `--check 1` can fail now. Runs that previously "passed" with a residual
  above the tolerance fail; use `--check_tol` to state the accepted bound.
- The genomics drivers (`testing_syrk`, `testing_KRR*`, `testing_Kmatrix`,
  `testing_igemm_summa`) are built only with `-DHICMA_PARSEC_GENOMICS=ON`.
- Scripts moved: machine recipes to `scripts/machines/`, genomics data tools
  to `scripts/genomics/`; `scripts/test_run*.sh` are replaced by
  `scripts/test_CI.sh` and CTest.

## 0.1.0

First public release (2025-11-12).
