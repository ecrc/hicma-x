# HiCMA-X 0.2.0 improvement plan

Companion to `docs/code-review.md` (finding IDs refer to it). Status values:
**done** (implemented and validated on methane), **partial**, **todo**, **deferred**
(recorded in `code-review.md` section 6 with the reason). Validation refers to the
matrix in `docs/release-readiness.md`: 40+ `--check 1` runs over every Cholesky kind,
TRMM, 1 and 2 ranks, CPU and one A100, plus a HIP compile.

## 1. Release scope

Version **0.2.0** (see `code-review.md` G3). Contents: the runtime precision decision,
FP8 scaling, mixed-precision TRMM and PaRSEC/DPLASMA update already on `main`, plus
the correctness, lifecycle, build and release fixes below. No new algorithms, no change
to a numerical method except where a demonstrated bug is fixed (E1, D1, C1, D3, D5),
no TODO-backlog items.

Principles applied: fix root causes in one place; fail loudly on unsupported inputs
rather than compute silently; remove code with no caller; share wrapper code without
touching JDF bodies; keep CUDA/HIP bodies and the runtime GPU GEMM as they are; kinds 8
and 9 keep static decisions.

## 2. P0 — release blockers

| # | Task | Findings | Files | Validation | Status |
|---|---|---|---|---|---|
| P0-1 | Make the correctness check binding: return the pass/fail flag from `check_dpotrf2`/`check_diff`, reduce it across ranks, read the reference `dpotrf` info, make the drivers exit non-zero on `info != 0` | E2, E4, G4 | `hicma_parsec_check.c`, `tests/testing_potrf.c`, `testing_climate_emulator.c`, `testing_KRR_file.c` | `--check 1 --fixedacc 1e-30` exits 1; matrix unchanged | done |
| P0-2 | Apply `--adddiag` once: generation and the check reference add the shift only for problem kinds whose STARS-H data does not already carry it | E1 | `matrix_compress.jdf`, `band_regenerate.jdf`, new helper in `hicma_parsec_internal.h` | virus-mesh run: compression norms agree, residual < 1e-4; kind-2 `--adddiag 1` passes | done |
| P0-3 | Reject `N % NB != 0` for the Cholesky path with a clear message | H1, D5 | `hicma_parsec_internal.c` (`params_check`) | the four partial-tile cases exit with the message; TRMM partial-tile tests still pass | done |
| P0-4 | Memory safety in init: `snprintf` + `else` for `exe_file_path`; `readLocsFile` struct size and bounds | A1, E3 | `hicma_parsec_internal.c`, `hicma_parsec_flat_file.c` | run by bare name through `PATH`; ASan build with a 2-column CSV | done |
| P0-5 | Installable package: `configure_package_config_file` + `find_dependency`, library link interface carries hcore/BLAS/LAPACKE/CUDA/MPI, tests link only the library, installed headers self-contained (installed PaRSEC/DPLASMA paths, local timing macros, missing headers installed) | G1, G2, G8 | `CMakeLists.txt`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `cmake_modules/HICMAParsecConfig.cmake.in`, `hicma_parsec_internal.h` | `cmake --install` then build a consumer against the prefix alone (CPU and CUDA); HIP compile | done |
| P0-6 | One version: `project(VERSION 0.2.0)`, SOVERSION, headers, RELEASE.md | G3 | `CMakeLists.txt`, `src/CMakeLists.txt`, `src/**/*.h`, `RELEASE.md` | `readelf -d` SONAME, `cmake -LA` | done |
| P0-7 | CI tests correctness: `--check 1` everywhere it applies, climate step skipped without data, CTest registration, install smoke test | G4, G5 | `scripts/test_CI.sh`, `.github/workflows/ci.yml`, `tests/CMakeLists.txt` | `ctest -N` lists HiCMA tests; a tampered tolerance fails CI locally | done |

## 3. P1 — architecture, duplication, maintainability

| # | Task | Findings | Files | Validation | Status |
|---|---|---|---|---|---|
| P1-1 | Zero-initialize `hicma_parsec_params_t`; allocate `info_gpu` and the statistics arrays once, after device discovery; agree `gpus` across ranks | A2, A3, A4, A7 | `hicma_parsec_internal.c` | Valgrind on a dense run; `--gpus 1` on CPU build; R-LO line stable | done |
| P1-2 | Shared Cholesky wrapper helpers (`potrf_common.{c,h}`): timing hooks on `parsec_task_t`, task-class lookup and runtime incarnation selection, hook installation, pools init/fini (with `free`), GPU setup/teardown, driver; apply to the ten wrappers | B1, B2, C4, C5, A9 | `src/kernels/*_wrapper.c`, new `src/kernels/potrf_common.{c,h}` | full matrix identical residuals and GEMM counts; LSan shows no pool leak | done |
| P1-3 | Runtime GEMM: reallocate/resize the device tile on every FP64 promotion | D1 | `hicma_parsec_core.c` | kind-10 matrix unchanged; low `device_cuda_memory_use` run | done |
| P1-4 | HCORE storage maxrank `2*maxrank` → `maxrank` (3 sites); enforce `maxrank <= compmaxrank`, `<= genmaxrank`, `<= NB/2` | C1, C7, D4 | TLR JDFs, `hicma_parsec_internal.c` | kind 2 `--band_dense 2 --maxrank 24` aborts cleanly like kind 1 | done (compmaxrank >= maxrank enforced; the NB/2 and genmaxrank bounds are left to HCORE's run-time check) |
| P1-5 | `--tensor_gemm 0x80`: abort on unknown mask in the FP16 GEMM switch and reject for Cholesky kinds | D3 | `hicma_parsec_core.c`, `hicma_parsec_internal.c` | option gives a message instead of a wrong result | done |
| P1-6 | GPU kernel status checks: one abort-on-error macro after every cuBLAS/cuSOLVER/cuBLASLt call in the non-runtime kernels and the FP8 setup | D8, A5 | `hicma_parsec_core.c`, `hicma_parsec_gpu.c` | GPU matrix unchanged | done (`2940b51`, `release-readiness.md` section 5.23: `HICMA_CUBLAS()`/`HICMA_CUSOLVER()` after every call of the Cholesky GPU kernels; left unchecked: the HIP warm-up at start-up and the destroy calls at exit) |
| P1-7 | Refuse unsupported combinations loudly: SR with kinds 8-10, `datatype_convert 2` on CPU, HP band on CPU for kind 3, HIP with kinds 1-4 on GPU | B3, D7, B6, C11, C6 | `hicma_parsec_internal.c`, four TLR/band wrappers | each combination exits with a message | done |
| P1-8 | Genomics/GPU-only drivers not built when they cannot run: `GENOMICS` overridable, CMake option, GPU gating | F4, F2 | `tests/CMakeLists.txt`, `hicma_parsec_internal.h`, `testing_igemm_summa.c` | CPU-only configure builds no genomics drivers | done |
| P1-9 | Climate emulator: correctly spelled switches, unconditional file-read checks, CI skip without data | F3, F5 | `src/climate_emulator/*`, `scripts/test_CI.sh` | compiles CPU and CUDA; a missing file is fatal | done |
| P1-10 | Fix the partial-tile low-rank TRSM dimension (unreachable after P0-3, corrected for the record) | D5 | `hicma_parsec_core.c` | full-tile matrix unchanged | done |
| P1-11 | Build hygiene: `cmake_minimum_required 3.21`, no `--remote --force` submodule update, target-based link/include, one test link loop, warning flags option | G6, G7, G8 | `CMakeLists.txt`, `src/CMakeLists.txt`, `tests/CMakeLists.txt` | CPU, CUDA, HIP configurations build | done except the warning-flags option |
| P1-12 | `.gitignore`, LICENSE placeholder, stray tracked files | G11, G13 | `.gitignore`, `LICENSE`, `tests/` | `git status` clean after a build | done |

## 4. P2 — localized cleanup, consistency, tests, docs

| # | Task | Findings | Status |
|---|---|---|---|
| P2-1 | Dead files and functions removed (section 5 of the review); dead declarations removed from the headers | E8, F9, F11, F12, C9, D14, A11 | done |
| P2-2 | Wrapper leaks and NULL taskpool returns in the GEMM-shaped wrappers; shared device discovery | F1, F6, F11.7 | done |
| P2-3 | Genomics helpers: `matrix_sum_vec` accumulator size, `matrix_add` identifiers | E6 | done |
| P2-4 | Sparse analysis free predicate, `dist_array` free, `NT > 65535` fatal when the analysis is used, drop `<omp.h>` | E7, C3a | done (the NT > 65535 check already existed) |
| P2-5 | Conversion/IO robustness: `convert_s2d` span, `convert_d2s` LOW_RANK_SP capacity, NULL taskpool checks, MPI-IO early returns, `dist_read` memcpy/fopen/fread/offset | E7, C10 | partial: convert_s2d span, dist_read and matrix_norm_get done; convert_d2s LOW_RANK_SP capacity and the MPI-IO early returns remain |
| P2-6 | Check matrices zero-initialized; `matrix_uncompress` output flows `RW` | E5 | deferred to 0.2.1 |
| P2-7 | GPU workspace: placeholder leak, `ws` freed without GPUs, `cudaHostRegister` guarded and checked, `cusolver` Spotrf workspace and `dev_info` offset | A5, A6, D9 | partial: placeholder, ws free and cudaHostRegister done; the cuSOLVER Spotrf workspace / dev_info offset remain |
| P2-8 | Hamming GPU prototypes `int8_t *`; Fugaku `main` rename moved to drivers; `print_decisions*` without `sleep` | F8, A11 | partial: int8_t prototypes done; the print_decisions sleep and the Fugaku main rename remain |
| P2-9 | `convert_{d,s}2i/i2{d,s}` as one JDF | E8 | deferred to 0.2.1 |
| P2-10 | Documentation: `RELEASE.md` as release notes, `TESTS.md` options from `--help`, `BUILD.md` requirements and consumer example, `CHANGELOG.md`, known limitations | G9 | done |
| P2-11 | Scripts: generic tools kept, site recipes moved to `scripts/machines/`, byte-identical and syntactically dead scripts removed, Dockerfiles removed or rewritten | G10 | done |
| P2-12 | Statistics hygiene: communicator consistency, `gflops` statement, op counts in dense-band TLR branch | A8, C10 | deferred to 0.2.1 |

## 5. P3 — optional

| # | Task | Findings | Status |
|---|---|---|---|
| P3-1 | `get_datatype_size` sub-byte entries; `matrix_check_norm_diff` edge tile and pools; `memory_free_tile` NULLing | E9 | deferred |
| P3-2 | Legacy `<cublas.h>` include, redundant `cublasSetStream`, `handle_cublas_tensor`, HIP header leftovers | D15 | deferred |
| P3-3 | Timing instrumentation accuracy | B9, C10 | deferred |
| P3-4 | TERMINATE/sink-task CTL edges | B4, C2 | done (sink tasks removed after the two-rank crash was reproduced) |

## 6. Implementation order and validation

1. Documents (this file, `code-review.md`) committed first.
2. P0-1 to P0-4 (correctness) in one commit; rerun the full matrix; expected changes:
   the partial-tile cases now exit with a message, the virus-mesh residual becomes
   meaningful, everything else identical.
3. P1-1, P1-3 to P1-7, P1-10 (library behaviour) in small commits; rerun the matrix after
   each; GPU matrix after P1-3 and P1-6.
4. P1-2 (wrapper consolidation) in its own commit; full matrix; LSan on a CPU run.
5. P2-1 to P2-9 (dead code, small fixes); every removal compile-checked on CPU, CUDA and
   HIP (compile only).
6. P0-5, P0-6, P1-8, P1-11, P1-12, P0-7 (build, install, version, CI, CTest); install
   consumer test; `ctest`.
7. P2-10, P2-11, release notes, `docs/release-readiness.md`.

Performance guard: the matrix records the factorization time of every case; the
large-case comparison (N=32768, one A100, kinds 5 and 10) is done once before and once
after the library changes.

## 7. Status log

- 2026-11-07: review completed, baseline matrix recorded (all CPU and single-GPU cases
  pass; two-rank GPU cases need `--mca device_cuda_memory_use 30` on a shared GPU;
  partial-tile cases fail silently; see `release-readiness.md`).
- 2026-11-07 (later): P0-1 to P0-7, P1-1 to P1-12 and P2-1 to P2-5, P2-7, P2-8,
  P2-10, P2-11 implemented on `qinglei/release-cleanup` (eleven commits, see
  `release-readiness.md` section 2); the full matrix rerun after each batch
  (labels p0, p1a, p1b, p2, p4, p5, p6), identical to the baseline except the
  intended virus-mesh change; the two-rank kind-2 band case exposed the
  sink-task crash (P3-4, fixed); CPU `ctest` 20/20, CUDA-tree `ctest` 30/30,
  install consumer from both prefixes, HIP compile, before/after timing at
  N=32768 within noise. Open: maintainer review of the behaviour changes, CI
  run on GitHub, tag/release/merge (authorization).
- 2026-11-07 (CI): the first GitHub CI run of PR ecrc/hicma-x#13 failed:
  undefined LAPACKE references from libhcore.so on Ubuntu (--as-needed) and no
  Python/flex/bison in the CUDA container. Fixed by linking hcore, starsh and
  hicma_parsec against DPLASMA's LAPACKE::LAPACKE (one BLAS stack, replacing
  the separate BLAS discovery and the hard-coded MKL line), removing GSL's
  reference CBLAS from the link (it had captured cblas_* calls, 40x slower CPU
  GEMMs), and installing the packages. The dense adaptive mixed-precision
  cases use --check_tol 1e-7 (FP32 bound). Re-validated: ctest 20/20 and
  30/30, install consumers, 75-case matrix identical to the baseline.
- 2026-11-07 (CI, later): the installed consumer of the CI job failed at load
  time (no runtime search path in the installed library under RUNPATH
  semantics); fixed with CMAKE_INSTALL_RPATH ($ORIGIN, prefix lib dir, link
  path). Both CI jobs pass on the final commit of ecrc/hicma-x#13.
- 2026-11-08: second pass on request: tile/LAPACK conversion JDFs merged
  (`TODOs` item 1); external-user walkthrough (fresh clone, BUILD.md verbatim,
  an embedding application) and corner-case sweeps (61 CPU cases on methane,
  83 GPU cases on guyot with 1 process x 8 A100s, 8 ranks x 1 and 4 ranks x 2).
  Eleven problems fixed, among them kind 3 on GPUs (reshape abort, sender
  conversion), FP16-accumulation GEMM scalars, sparse kinds with a single tile
  row, embedding crashes and the BLAS run-time path; see
  `release-readiness.md` section 5.6. Final tree: CTest 23/23 (CPU) and 37/37
  (CUDA, methane and guyot), matrix 31 + 50 cases identical to the baseline
  except the mesh fix, HIP compiles.
- 2026-11-08 (third pass, on request): `TODOs` item 3 done (one
  multi-precision `hicma_parsec_matrix_sum`); macro design of
  `hicma_parsec_internal.h` reviewed (`code-review.md` section 8): switches
  settable with -D, dead ones removed, every value compiled and run,
  `GPU_BUFFER_ONCE` removed (broken and slower), workload-balance strategy 4
  heap corruption fixed; AddressSanitizer runs found the pool-storage leak
  (fixed: kind-10 memory over 8 runs 1238 -> 605 MB). See
  `release-readiness.md` section 5.7.
- 2026-11-08 (fourth pass, on request): build scripts rewritten for the
  single-configure build, `scripts/machines/legacy/` removed; validating them
  found that CUDA builds without an explicit architecture compiled for sm_52
  only (fixed) and that the kind-3 guards made the JDF compiler print fatal
  errors (fixed). See `release-readiness.md` section 5.8.
- 2026-11-08 (on request): GEMM reordering (`reorder_gemm.jdf`,
  `hicma_parsec_reorder_gemm`) restored for future use, not wired in; its
  launch compiles, its sort and pool leak are fixed.
- 2026-11-08 (fifth pass, on request): macros and data structures
  inventoried and reviewed (`code-review.md` section 9): unused macros and
  fields removed, local and climate switches settable with -D, two climate
  debug paths fixed; matrix identical to the third pass. See
  `release-readiness.md` section 5.9.
- 2026-11-08 (on request): the 28 unnamed `#if 0`/`#if 1` blocks unwrapped,
  deleted or named (`GENOMICS_SYRK_ISYRK`); matrix again identical.
- 2026-11-08 (on request): CI hardened after a runner-dependent failure:
  BLAS thread-safety fixture for every HiCMA case, runner CPU/BLAS printed,
  `OPENBLAS_NUM_THREADS=1`. See `release-readiness.md` section 5.10.

- 2026-11-08 (sixth pass, on request): external-user pass (fresh clone,
  README/TESTS walkthrough, DPLASMA comparison). Fixed: `--nruns` crash in
  TLR runs, wrong help defaults, duplicate and uninitialized summary line,
  missing revision, two zero exit codes; notes for `--gpus 0` with visible
  GPUs, fully dense TLR kinds, and launcher bindings. See
  `release-readiness.md` section 5.11.
- 2026-11-08 (seventh pass, on request): `--tensor_gemm` redesigned
  (`--tensor_gemm_hp`, `--tensor_gemm_sp`, `TODOs` item 12) after wrong
  results and crashes with several masks; an independent review and rounds
  of checks closed five more paths; a static FP16 band with low-rank tiles
  is refused on GPUs (cause open). See `release-readiness.md` section 5.12.
- 2026-11-08 (on request): the GPU failure with a static FP16 band and
  low-rank tiles root-caused and fixed: the FP16 branch of the low-rank x
  low-rank GPU GEMM converted its low-rank B as a dense tile. The refusal is
  removed; CTest runs the layout in kinds 5 and 1.
- 2026-11-09 (eighth pass, on request): the H100 run (CTest, matrix, FP8,
  two ranks, performance) found FP8 operands read as FP64 by the low-rank
  GEMMs (fixed); FP8 band numerics documented; a documentation audit's
  findings corrected; `scripts/test_CI.sh` brought up to the CTest cases.
  See `release-readiness.md` section 5.13.
- 2026-11-09 (on request): a complete test on hexane, H100 and V100, one and
  two GPUs, one and two ranks, performance and ease of use: correct
  everywhere; notes added for a capped `--gpus` and for oversubscribed cores.
  See `release-readiness.md` section 5.14.
- 2026-11-10 (on request): PaRSEC rebased on ICLDisco master `2c16c652e`
  (#814 merged upstream; the fork's copies and `5cd761f64` dropped, eight
  fork commits reapplied, one conflict resolved). methane: PaRSEC 205 / 205,
  HiCMA 27 / 27 and 48 / 48, matrix identical, eviction reproducer 10 / 10.
  See `release-readiness.md` section 5.16.
- 2026-11-10 (tenth pass, on request): code structure and several GPUs.
  Kinds 5, 8 and 9 merged into one kernel, the low-rank CPU GEMM of kinds 1,
  5 and 10 moved into two functions (`TODOs` item 2, in part; 6,076 lines
  less in `src/`), one GPU placement rule chosen by an interleaved
  comparison on four V100s, stochastic rounding for kinds 8 and 9, a note
  for GPUs that are not all linked. Results unchanged on methane, hexane and
  leconte; old against new timings within the spread. See
  `release-readiness.md` section 5.17.
- 2026-11-11 (on request): kinds 8 and 9 run their own JDFs again
  (`potrf_L_dense_mp_gpu_fp8{,_sp}.jdf` and wrappers restored, with the
  shared CPU GEMM and the placement rule of the tenth pass), so kinds 8 to
  10 refuse stochastic rounding again. Also removed: the prototype of the
  deleted `parsec_convert_i2s` and commented-out calls to removed
  functions. See `release-readiness.md` section 5.18.
