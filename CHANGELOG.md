# Changelog

## 0.2.0 (unreleased)

Release engineering of the 0.1.0 code base plus the features merged on `main`
since November 2025. `docs/code-review.md` records the review this release is
based on, `docs/release-improvement-plan.md` the task list and its status,
`docs/release-readiness.md` the validation results.

### Features (merged before this release cycle)

- Runtime precision decision for the dense mixed-precision Cholesky
  (`--adaptive_decision_runtime 1`, kind 10): every GEMM picks its compute
  precision from the norms of its operands while the factorization runs.
- FP8 (E4M3) GEMMs in every CUDA build, with per-tile power-of-two scaling
  (`--fp8`, `--fp8_scaling`); used by kinds 8, 9 and 10 on GPUs with compute
  capability 8.9 or higher.
- Mixed-precision TRMM (`testing_trmm_mp`), with stochastic rounding
  (`--enable_stochastic_rounding`) and FP32 accumulation (`--accumulation_fp32`).
- DPLASMA and PaRSEC moved to the ICLDisco master branches.

### Correctness

- The correctness check is binding: `check_dpotrf2` and `check_diff` returned
  0 whatever they measured, so `--check 1` could not fail. They now return their
  verdict, the reference dense Cholesky's `info` is inspected, and every driver
  exits non-zero on a failed check or a failed factorization.
- `--check_tol` sets the residual bound of `--check` (default: `--fixedacc`);
  a failed check prints the measured residual and the accepted bound.
- `--adddiag` was applied twice for problem kinds 0, 1, 5, 6 and 7 (by the
  STARS-H kernel and again by the generation task) while the reference matrix
  of the check had it once. The shift is now applied exactly once and the
  reference matrix carries it too. Results of these kinds change when
  `--adddiag` is non-zero (the virus-mesh example now reports a residual of
  3e-6 instead of 9e-2).
- `N % NB != 0` is rejected by the Cholesky drivers: the band generation, the
  check and the low-rank kernels assume square tiles and produced wrong results
  (dense FP64 residual of 3e+3) or aborted in HCORE. The TRMM keeps supporting
  partial tiles.
- Runtime-decision GEMM (kind 10): a tile promoted from FP32 to FP64 keeps a
  correctly sized device buffer whatever `--adaptive_decision`/`--adaptive_memory`
  are, instead of only when one of them is set.
- HCORE's storage maxrank was passed as `2*maxrank` in kinds 2, 6 and 7,
  disabling the rank cap that every buffer is sized for; `compmaxrank` is kept
  at least `maxrank`; the per-stream GPU buffers and the CPU full-tile pools
  hold an `mb x 2*maxrank` operand.
- The dense FP16 GPU GEMM aborts on an unsupported `--tensor_gemm` mask (0x80)
  instead of skipping the update; the Cholesky path rejects 0x80 up front
  (it is a TRMM-only mode).
- CPU low-rank TRSM solves with the size of the diagonal block, not the row
  count of the C tile; CPU TRSM treats FP8-decided tiles like SP/HP ones.
- Refused instead of silently miscomputed: `--enable_stochastic_rounding` with
  kinds 8, 9, 10; `--datatype_convert 2` without GPUs; kinds 1 to 4 with GPUs
  on HIP; a half-precision band on CPU-only runs of kind 3.
- Memory safety: write through `NULL+1` in `hicma_parsec_params_init` when
  `argv[0]` has no `/`; 8-byte allocation for a 24-byte struct in
  `readLocsFile`; `info_gpu` allocated with the requested and freed with the
  detected GPU count; `pheno_file` and other parameter fields read or freed
  uninitialized (the parameter struct is now zeroed first); `gpus` agreed
  across ranks; the Hamming GPU kernel prototypes match their definitions.
- Resource leaks: memory-pool structs of every Cholesky taskpool, GPU device
  index arrays, the cuBLASLt/cuSOLVER setup failures now reported.

### Architecture and maintenance

- The ten Cholesky wrappers share `src/kernels/potrf_common.{c,h}` (timing
  hooks, chore selection by device type, memory pools, GPU discovery, driver);
  7250 lines of wrapper code became 2500. The JDF bodies are untouched.
- Per-core and per-GPU statistics arrays are allocated once, after device
  discovery.
- Dead code removed: 15 unreferenced JDFs and sources (`copy2double`,
  `isyrkmp_LT`, `gemm_TN_batched`, `gemm_TN_summa`, `gemm_f16_wrapper`,
  `matrix_dsum/ssum`, `cast_double_diag`, `check_read_data`, `sqrt_kernel`,
  `Lapack_to_Tile_sym`, `convert_i2s`, `diagonal_correction_calculation`,
  `reorder_gemm`, `climate_emulator_print_double`), two unbuilt test sources,
  `hamming.jl`, ~40 uncalled functions, 14 declared-but-undefined prototypes,
  and the `#if 0` blocks around them.
- Public headers include the PaRSEC and DPLASMA headers that those projects
  install; DPLASMA's test timing macros are replaced by `hicma_parsec_timing.h`.

### Build, install, tests

- `find_package(HICMAParsec)` works: the package config is generated with
  `configure_package_config_file`, finds PaRSEC, DPLASMA, MPI and CUDA/HIP,
  and the library carries its own link interface. `cmake --install` is
  exercised by CI through `tests/install_consumer`.
- One version: `project(VERSION 0.2.0)`, SOVERSION 0; `VERSION.txt` and the
  version updater script are gone.
- `cmake_minimum_required(3.21)`; configure no longer runs
  `git submodule update --remote --force`.
- One BLAS/LAPACKE stack: `hicma_parsec`, `hcore` and `starsh` link the
  `LAPACKE::LAPACKE` target that DPLASMA finds. The previous build loaded a
  second MKL interface and threading layer (`mkl_intel_lp64` +
  `mkl_gnu_thread` next to DPLASMA's `mkl_gf_lp64` + `mkl_sequential`), and
  `libhcore.so` recorded no dependency on LAPACKE at all, which made every
  driver fail to link with `--as-needed` (Ubuntu). Select the stack with
  `LAPACKE_ROOT` or `BLA_VENDOR`.
- `HICMA_PARSEC_GENOMICS` option: the genomics drivers are built only when it
  is on (they need a GPU build); the default build no longer ships binaries that
  cannot run.
- CTest registers the HiCMA tests (labels `hicma`, `mpi`, `gpu`), all with
  `--check 1`; `scripts/test_CI.sh` runs the same cases and no longer depends
  on a private data path; CI runs them and a CUDA compile job.
- `scripts/` reorganized: generic tools at the top level, site recipes under
  `scripts/machines/`, genomics data tools under `scripts/genomics/`,
  byte-identical and obsolete scripts removed; the stale Dockerfiles removed.
- `.gitignore` covers build trees, logs and editor state; LICENSE placeholder
  removed.

### Documentation

- `TESTS.md` regenerated from the option table; `BUILD.md`, `README.md` and
  `RELEASE.md` updated; `CHANGELOG.md` added.

### Known limitations (see `docs/release-readiness.md`)

- Cholesky drivers need `N % NB == 0`.
- Kinds 1 to 4 have no HIP bodies; HIP is compile-tested only.
- `--enable_stochastic_rounding` is not implemented for kinds 8 to 10.
- The CPU and GPU runtime-decision GEMMs use different storage policies for
  FP16 updates.
- Mixed-precision runs reach the accuracy of the precisions they use, not an
  FP64-level `--fixedacc`; use `--check_tol`.

## 0.1.0 (2025-11-12)

First public release of HiCMA-X (then named HiCMA-PaRSEC): tile low-rank and
mixed-precision Cholesky factorizations on PaRSEC with CUDA and HIP support,
the climate emulator and the genomics (kernel ridge regression) applications.
