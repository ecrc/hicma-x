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
- DPLASMA and PaRSEC moved to the ICLDisco master branches (PaRSEC `2c16c652e`,
  which includes ICLDisco/parsec#814, plus the fork's commits).

### Data-type conversions: OmniCast

- Every element conversion now goes through OmniCast (new submodule
  `omnicast/`, https://github.com/QingleiCao/OmniCast, branch `main`): one C API
  for FP64, FP32, FP16, BF16, FP8 (E4M3, E5M2) and integers on the CPU (SIMD
  chosen at run time: AVX-512, AVX2+F16C, NEON) and on CUDA and HIP GPUs.
  HiCMA-X's own conversion kernels and loops are removed: the GPU converters
  of `hicma_parsec.cu`/`hicma_parsec_hip.cu` (`double2half_GPU`,
  `float2fp8_padded_GPU`, `hicma_fp8_scaled_convert_GPU`, `memcpy_*_GPU`, ...),
  the stochastic-rounding sources, the `convert_*_unary/binary_CPU` loops and
  the `LAPACKE_dlag2s`/`slag2d` tile conversions. Call sites use `omc_*`
  through `HICMA_OMC()`/`HICMA_OMC_SR()` (`hicma_parsec_datatype_convert.h`);
  `hicma_parsec_fp8_operand_GPU` builds the scaled, transposed FP8 operand of
  the cuBLASLt matmul.
- Same results: the accuracy of every CTest case is identical to all printed
  digits on an A100, a V100 and an H100 (including the FP8 cases).
  Stochastic rounding keeps its random numbers and probabilities bit for bit.
  FP64 -> FP8 (and FP16 on HIP) is now rounded once, directly, where HiCMA-X
  went through FP32 (double rounding, which can differ in rare ties).
- Faster: converting a tile in place is one kernel instead of a conversion into
  a scratch tile plus a copy back (1.6-3.3x); the transposed FP8 operands are
  3-6x faster (coalesced stores); FP32 -> FP16 and the other 2- and 4-byte
  sources up to 2x; GPU stochastic rounding 1.2-1.45x; CPU stochastic rounding
  3-9x; CPU FP64 <-> FP32 tiles 1.1-8x faster than MKL's `dlag2s`/`slag2d`.
  Conversion kernels take 26 % less GPU time in a mixed-precision Cholesky
  (the factorization is 0-1.6 % faster); CPU factorizations with stochastic
  rounding are 18-19 % faster, the other CPU kinds unchanged. Details in
  `docs/omnicast/README.md`.
- OmniCast update (2026-11-22): every conversion with every option in place
  and out of place on the host, CUDA and HIP; faster FP8 operands (1.04-1.13x),
  in-place tile conversions (1.04-1.18x) and stochastic rounding from FP32
  sources; a fix for scaled FP32 -> FP16 conversions on AMD GPUs (HiCMA-X
  scales by powers of two and is not affected). Report in
  `omnicast/docs/REPORT.md`.
- FP16 conversions are available on every CPU (they were Fugaku-only), and
  float -> integer conversions saturate (NaN -> 0) instead of being undefined
  out of range.
- GPU operand cache (`--operand_cache`, default on). A tile sent in a higher
  precision than a GEMM's was converted again by every GEMM reading it, up to
  NT - 1 times. Each GPU now keeps the converted FP8, FP16/BF16 and FP32
  operands of a factorization and reuses them, with bit-identical results.
  The memory comes from PaRSEC's device zone when first needed: up to 2*NT
  tiles per precision, at most 5 % of the zone each. FP8 GEMMs read their
  scales through cuBLASLt pointer attributes instead of two device copies.
  On an H100 (N = 45056): 1.07-1.08x for kinds 8, 9 and 10, 1.02-1.04x for
  the FP32/FP16 kinds; conversions take 4.9 % instead of 19 % of the GPU time
  when most tiles are FP8. Report in `docs/omnicast/round3/REPORT.md`.
- OmniCast round 3: hardware FP8 converts on sm_89+, 4 loads in flight in the
  amax pass of dynamic scaling, an opt-in fused dynamic-scaling kernel, and,
  for its other users, faster host FP8 encoding (F16C) and streaming stores
  for large in-place widening on the host.
- Subnormals on the CPU (`--flush_subnormals`, default on). The FP32 tiles of
  the covariance problems hold many subnormal numbers, and on Intel CPUs every
  SIMD operation on one takes a microcode assist: MKL's FP32 GEMMs ran at
  0.74x the speed of the FP64 ones, so the mixed-precision kinds were
  1.2-1.4x *slower* than FP64 on CPUs. HiCMA-X now sets flush-to-zero and
  denormals-are-zero on its CPU threads while it runs (the main thread's
  setting is restored at the end). On a Broadwell and a Skylake-SP node
  (N = 16384, NB = 512) the mixed-precision kinds are 1.6-2.6x faster and
  about 2x faster than FP64; FP64 runs are unchanged, and residuals agree
  to 4-5 digits. An AMD Zen 2 CPU has no such penalty and is unchanged.
  `--flush_subnormals 0` keeps IEEE subnormals. OmniCast's host conversions
  now give the same bits whatever the caller's FTZ/DAZ or rounding mode
  (some had changed under FTZ/DAZ; QingleiCao/OmniCast#2). Report in
  `docs/omnicast/round4/REPORT.md`.
- OmniCast 2.1.1 (QingleiCao/OmniCast#3, #4): a failed OmniCast call on the
  GPU names the CUDA or HIP call and its error, and `HICMA_OMC()` prints it,
  for instance "OmniCast: device runtime error: launch of k_cvt:
  cudaErrorInvalidConfiguration (invalid configuration argument)", where it
  printed only "device runtime error". OmniCast's launch check used to
  report an error left pending by earlier code as its own launch failure,
  and clear it. Such an error is now reported as pending, nothing is
  launched, and the error stays for the code that caused it. "Not ready",
  which HIP records after PaRSEC polls an unfinished event, is not an error
  (2.1.0 stopped 11 Radeon VII tests on it;
  `docs/release-readiness.md`, section 5.25).
- Host operand cache for kind 10 on CPUs (`--operand_cache`, default on).
  With runtime decisions, every CPU GEMM that picks FP32 converted both of
  its FP64 panel tiles again: 9342 conversions of 496 tiles at N = 16384. A
  process without GPUs now keeps one FP32 copy per panel tile, converted
  once and freed after the last GEMM of the process that reads it. The
  copies are bounded, by default to a quarter of the process's panel tiles;
  at the bound the copy whose tile has the fewest readers left is evicted.
  Results are bit-identical. On a Broadwell node: 1.13-1.15x at N = 16384
  (2454 conversions left instead of 9342, 130 MB) and 1.16x at N = 32768. Report
  in `docs/omnicast/round5/REPORT.md`.
- Kind 10 runtime CPU GEMMs, `ACC_DP`: with MKL, the FP32 GEMM whose
  product is subtracted from an FP64 tile runs in 256-column panels, each
  subtracted right after it is computed (`ACC_DP_PANEL`; CMake passes
  `HICMA_PARSEC_BLAS_MKL` when the BLAS is MKL). Bit-identical; about 1 %
  faster at N = 16384 on a Broadwell node, no measurable change at
  N = 32768. The loop is bound by the memory bandwidth of the core; AVX,
  prefetching and non-temporal stores did not help enough to keep. Report in
  `docs/omnicast/round6/REPORT.md`.

### Correctness

- The correctness check is binding: `check_dpotrf2` and `check_diff` returned
  0 whatever they measured, so `--check 1` could not fail. They now return their
  verdict, the reference dense Cholesky's `info` is inspected, and every driver
  exits non-zero on a failed check or a failed factorization. The verdict is
  the residual ||LL^T-A||_F/||A||_F; the forward error against the dense
  reference factor, ||L-L0||_F/||L||_F, is printed but does not decide (an
  overwrite of `params->info` had always discarded it, and for mixed-precision
  runs it exceeds the residual 10 to 30 times, so one bound cannot serve both).
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
- Memory leak: `parsec_private_memory_fini` destructs pooled items without
  freeing their storage, so every workspace a taskpool's pools handed out was
  lost when the taskpool was destroyed: 167 MB per kind-2 factorization at
  N=2700 with 36 threads, about 100 MB per kind-10 factorization at N=8192,
  again at every call (`--nruns`, applications that factorize repeatedly).
  `hicma_parsec_memory_pool_fini` frees the storage first and replaces the 22
  calls in HiCMA-X; peak memory of eight kind-10 runs went from 1238 MB to
  605 MB. Found with AddressSanitizer's leak checker.
- Kind 3 (DENSE_MP_BAND) on GPUs: every run with a half-precision band died
  with `MPI_ERR_TRUNCATE` (a `[type = HALF]` dependency made PaRSEC reshape the
  FP32 tile into a buffer half its size), and `--datatype_convert 1`, a
  kind-3-only mode, crashed on CPU and GPU (the send-decision step rewrote it
  to 2, which the kind-3 graph does not implement consistently). Both work;
  kinds 3 and 4 with a half band and the sender mode have GPU tests now.
- `--tensor_gemm 0x10` (FP16 accumulation) computed garbage on GPUs in kind 3
  and in the low-rank GEMMs of kinds 1, 2, 5, 8, 9 and 10: cuBLAS reads alpha
  and beta in the compute precision, and the float values read as half are 0,
  so every update vanished (kind 3 residual 5e+1, now 2.3e-5). The dense GEMM
  and the TRMM already converted them.
- The sparse kinds (6, 7) crashed when the dense band covers the matrix (for
  example a single tile row): the data is then stored in the contiguous
  descriptor, which they did not use. They select it as kind 2 does.
- An invalid `--kind_of_cholesky` printed an error and exited 0; it exits 1.
  Negative sizes are refused; a `--P`/`--Q` that does not cover the
  processes is reported before the grid is chosen.
- Refused instead of silently miscomputed: `--enable_stochastic_rounding` with
  kinds 8, 9 and 10; kinds 1 to 4 with GPUs on HIP. Changed with a
  message instead: `--datatype_convert 2` without GPUs runs in mode 0, and a
  half-precision band in CPU-only runs of kind 3 is widened to FP32. Kind 3 with
  `--datatype_convert 2` runs in the receiver mode: it does not implement the
  sender conversion, and with a one-tile FP64 band its GPU TRSM read the FP64
  diagonal factor as FP32.
- HIP builds register DPLASMA's hipBLAS handles (`DPLASMA::HIP::HANDLES`); the
  DPLASMA routines that run on the GPU with `--gpus > 0`, such as the TRMM in
  `--check`, dereferenced a NULL handle.
- Memory safety: write through `NULL+1` in `hicma_parsec_params_init` when
  `argv[0]` has no `/`; 8-byte allocation for a 24-byte struct in
  `readLocsFile`; `info_gpu` allocated with the requested and freed with the
  detected GPU count; `pheno_file` and other parameter fields read or freed
  uninitialized (the parameter struct is now zeroed first); processes that
  see different numbers of GPUs are refused (they took different branches
  around collectives, and PaRSEC keeps the devices it enabled at start-up);
  the Hamming GPU kernel prototypes match their definitions.
- Resource leaks: memory-pool structs of every Cholesky taskpool, GPU device
  index arrays, the cuBLASLt/cuSOLVER setup failures now reported.
- Kind 10 on AMD GPUs: the FP16 matrix instructions of gfx90a (MI200) flush
  FP16 subnormal inputs (|x| < 6.1e-5) to zero, and the run-time decision picks
  FP16 for a GEMM exactly when its operands are small, so those updates were
  lost: residual 1.3e-6 on an MI210 where the A100 gives 1.0e-8, and 2.5e-7
  against 1.9e-13 on the CPU with `--sigma 1e-4` (`gpu_runtime_decision`
  and `gpu_runtime_admem` failed). On HIP the FP16 operands of the run-time
  GEMM are scaled by powers of two from their Frobenius norms and the GEMM's
  alpha divides the scales out (exact): 1.5e-8 and 9.1e-11. CUDA builds are
  unchanged.
- Kind 10 with GPUs, a runtime CPU GEMM on an FP32 tile: widening the tile
  to FP64 left its span (the bytes PaRSEC copies) at the FP32 size, so the
  next GPU task on it got half the tile; restoring the span alone would make
  PaRSEC copy the FP64 tile into a GPU copy the warmup allocated at the FP32
  size, over the neighbouring tile; and with `--adaptive_memory` the
  widening overflowed the FP32-sized host buffer. The CPU GEMM now restores
  the span and moves C to an FP64-sized host buffer when needed, and the GPU
  GEMM, TRSM and SYRK grow a GPU copy smaller than the span before PaRSEC's
  stage-in (`hicma_gpu_stage_in`). PaRSEC's own placement puts no kind-10
  GEMM on the CPU in a process with GPUs (also with
  `device_load_balance_allow_cpu=1` in the runs made); with GEMMs forced onto
  the CPU and `--adaptive_decision 1` the factor was wrong or the run
  crashed, and is now right (residuals as with every GEMM on the GPU). Other
  runs give the same bits as before. Report in
  `docs/omnicast/round6/REPORT.md`, section 7.
- Cholesky with GPUs (kinds 1, 2, 5, 8, 9, 10): a GEMM that a GPU accepts
  now always runs on a GPU. The CPU and recursive chores of `potrf_dgemm`
  get an evaluate, `hicma_potrf_gemm_cpu_evaluate`, installed by
  `hicma_potrf_install_hooks` in a process with GPUs, that declines such a
  GEMM; a GEMM every GPU evaluate declines (outside the dense band) still
  runs on the CPU. Before, the GPU placement rested on the preferred GPU the
  warmup and `potrf_bind_A` give each tile: without it and with
  `device_load_balance_allow_cpu=1`, PaRSEC put every kind-10 GEMM on the
  CPU (3 to 6 times slower in one process on a V100 node). Default runs give
  the same bits. Report in `docs/omnicast/round6/REPORT.md`, section 8.
- The GPU kernels of the Cholesky check the status of their cuBLAS,
  cuBLASLt and cuSOLVER calls (hipBLAS and rocSOLVER on HIP). Most stored it
  unchecked, and the GPU POTRF checked it only in an `assert`, which Release
  builds drop: a failed call left a tile without its update and the run went
  on, to a wrong factor ("not positive definite", residual 7.6e+01 with one
  TRSM failed on purpose) or a later, unrelated error. A failed call now stops
  the run with the file, the line and the status name (`HICMA_CUBLAS()`,
  `HICMA_CUSOLVER()` in `hicma_parsec_internal.h`, as `HICMA_OMC()` for
  OmniCast). The cuBLAS handle setup checked `cublasCreate`'s status again
  after `cublasSetStream` instead of `cublasSetStream`'s own; the cuBLASLt
  descriptor attributes are checked too. Results do not change
  (`docs/release-readiness.md`, section 5.23).
- `hicma_parsec_core_memory_allocation` (the LL^T of `--check`, and the tiles
  of the climate emulator and the TRMM performance test) ignored the result
  of `cudaMallocHost`/`hipHostMalloc`: a failure crashed later with no word
  about memory. It now returns NULL and clears the runtime's error, which
  would otherwise stop the next OmniCast call on the GPU. The check
  falls back to pageable memory as before, and every caller stops with the
  size when no memory is left.

### Architecture and maintenance

- The eight tile/LAPACK conversion JDFs (`Lapack_to_Tile{,_Single}`,
  `Tile_to_Lapack{,_single,_sym,_sym_single,_int,_int8}`), which differed
  only in element type, direction and lower-triangle-only, are one JDF with
  one entry point, `hicma_parsec_lapack_tile_convert(parsec, dcA, buffer, P,
  Q, type, lower, into_tiles)`. The named functions remain as one-line
  wrappers; `hicma_parsec_Lapack_to_Tile_sym`, declared in 0.1.0 without a
  definition, now exists. The buffer offset uses the buffer's element size
  (it used the descriptor's), and a P x Q that does not match the
  descriptor's distribution (P and Q swapped, a band distribution) is refused
  on every rank instead of overwriting memory. `tests/testing_lapack_tile.c`
  covers every type, both directions, partial tiles and every grid of the
  run (CTest `lapack_tile_convert`, `lapack_tile_convert_np2`).
- The ten Cholesky wrappers share `src/kernels/potrf_common.{c,h}` (timing
  hooks, chore selection by device type, memory pools, GPU discovery, driver);
  7250 lines of wrapper code became 2500. The JDF bodies are untouched.
- The CPU GEMM into a low-rank tile, 665 lines pasted in the kernels of kinds
  1, 5, 8, 9 and 10, is two functions:
  `hicma_parsec_core_gemm_lrC_lrA_{denseB,lrB}_cpu`. Kind 1's copy did not
  count its datatype conversions; `Number_of_Datatype_Conversions` of kind-1
  runs is now complete (larger than before).
- One rule places the tiles on the GPUs for every Cholesky kind and for the
  warm-up, `hicma_parsec_gpu_of_tile`, replacing four per-kind rules and a
  fifth in the warm-up: the `gpu_rows x gpu_cols` grid for a dense matrix
  (1 to 10 % faster than by row or by column on four V100s), by column when
  only a band is dense, both on the tile's position among the tiles of its
  rank. The rules of kinds 1 to 4 used global tile indices: with two ranks
  of four GPUs, two GPUs of each rank ran 61 % of its GPU tasks and the
  other two 10 % (kind 3; 16 to 20 % each now).
- These three changes take 6,076 lines out of `src/` (888 added, 6,964
  removed); its code lines that sit in duplicated runs of 25 lines or more
  went from 8,013 of 30,775 to 4,177 of 26,879 (`docs/code-review.md`,
  section 13). Timings are unchanged (old against new, interleaved, on CPUs,
  an H100 and four V100s: `docs/release-readiness.md`, section 5.17).
- Per-core and per-GPU statistics arrays are allocated once, after device
  discovery.
- Dead code removed: 14 unreferenced JDFs and sources (`copy2double`,
  `isyrkmp_LT`, `gemm_TN_batched`, `gemm_TN_summa`, `gemm_f16_wrapper`,
  `matrix_dsum/ssum`, `cast_double_diag`, `check_read_data`, `sqrt_kernel`,
  `Lapack_to_Tile_sym`, `convert_i2s`, `diagonal_correction_calculation`,
  `climate_emulator_print_double`), two unbuilt test sources,
  `hamming.jl`, ~40 uncalled functions, 14 declared-but-undefined prototypes,
  and the `#if 0` blocks around them.
- Macros and data structures reviewed (`docs/code-review.md`, section 9).
  Unused names removed from the installed headers: 16 datatype identifiers,
  three tile formats, three sparse modes, `DATATYPE_USE`, `EPS_DP`,
  `THRESHOLD_MEMORY_PER_NODE` (its check could not fire), `str`/`xstr`, 29
  HIP aliases, and the fields `HMB`, `start_time_kernel_sqr_sum_vec` and
  `str_problem` of `hicma_parsec_params_t`. `SYNC_TIME_PRINT` is defined once.
  The local switches of the sources (`CALL_HCORE_DSYRK`, `ACC_DP`,
  `OWN_CONVERTOR`, `USE_OPENMP`, ...) and of the climate emulator are
  settable with `-D`, and each value compiles; two climate debug paths did
  not, fixed. The climate header no longer `#undef`s the genomics switches.
  The 28 unnamed `#if 0`/`#if 1` blocks are unwrapped, deleted (debug
  dumps, code that did not compile, superseded alternatives) or, for the
  genomics SYRK, named `GENOMICS_SYRK_ISYRK`. Matrix compression no longer
  scans every tile for non-zeros to feed a dead `1 || flag` test.
- GEMM reordering (`reorder_gemm.jdf`, `hicma_parsec_reorder_gemm`) is kept for
  future use, still not called. Its launch was an `#if 0` block that did not
  compile; it is now real code on the gathered rank array, so the build
  checks it. Its sort swapped the positions but not the keys it compared, and
  its destructor leaked the work pool; both fixed. `reorder_gemm.jdf` lists
  what wiring it into the Cholesky needs.
- Public headers include the PaRSEC and DPLASMA headers that those projects
  install; DPLASMA's test timing macros are replaced by `hicma_parsec_timing.h`.

### Build, install, tests

- `hicma_parsec_matrix_sum(parsec, A, Ar, decisions, lower)`: the sum of a
  tiled matrix whose tiles are stored in different precisions (FP64/FP32 per
  decision, low-rank U and V with their ranks), identical on every process and
  for every process grid; replaces the single-precision `matrix_dsum` and
  `matrix_ssum`, whose tile task never stored its partial sum. Test
  `testing_matrix_sum` (CTest `matrix_sum`, `matrix_sum_np2`), including
  low-rank tiles in a partial last tile row, where V has more rows than U.
- The compile-time switches of `hicma_parsec_internal.h` can be set with
  `-DNAME=value` in `CMAKE_C_FLAGS` (20 had no `#ifndef`); every non-default
  value was compiled on the CPU and CUDA trees; the dead `CHOLESKY_CPU_ONLY`,
  `PERFORMANCE_BAND_BOUND` and `WRITE_A` are gone; the bound-measurement
  switches `TLR_BOUND_WITHOUT_OFFBAND_*` warn at compile time; `_GNU_SOURCE`
  and `_POSIX_C_SOURCE` are no longer redefined. See `docs/code-review.md`,
  section 8.
- Embedding: `hicma_parsec_init` crashed when given `argc = 0, argv = NULL`
  and when the argument strings were read-only (`--opt=value` was cut in
  place); both work now and `argv` is only read. `libhicma_parsec.so` is
  linked with `-Bsymbolic-functions`: it exports generic names
  (`parse_arguments`, and from STARS-H `diff`, `cube`, ...), and an
  application function of the same name replaced the one the library called
  (`hicma_parsec_init` ran the application's `parse_arguments` and crashed).
  Six file-local JDF helpers with generic names are `static`.
- The BLAS/LAPACKE directories are recorded in the run-time path of the
  built and installed binaries and of executables linking the installed
  package; with an MKL module that sets `LIBRARY_PATH` only, every test of a
  fresh build failed to start ("libmkl_gf_lp64.so.2: cannot open shared
  object file") unless `LD_LIBRARY_PATH` was set.
- Integer options accept hex with a `0x` prefix (`--tensor_gemm 0x10`, as the
  masks are documented); they were rejected as invalid values.
- The CMake test presets run the HiCMA cases only (they ran DPLASMA's and
  PaRSEC's suites as well); the install consumer also builds and runs the
  conversion test.
- `find_package(HICMAParsec)` works: the package config is generated with
  `configure_package_config_file`, finds PaRSEC, DPLASMA, MPI and CUDA/HIP,
  and the library carries its own link interface. `cmake --install` is
  exercised by CI through `tests/install_consumer`. On HIP the config also
  finds `hipblas` and `rocsolver`, which DPLASMA's link interface names but
  DPLASMA's own config does not look for. With ROCm's Flang (Frontier),
  `cmake --install` stopped at PaRSEC's Fortran module directory, which PaRSEC
  installs but does not create when the compiler fails CMake's F90 check; the
  configure step creates it.
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
- The installed library and drivers carry a runtime search path (`$ORIGIN`,
  the prefix's lib dir, the directories of MPI/GSL/hwloc): with RUNPATH
  semantics a library's dependencies are searched with its own path, and the
  installed `libhicma_parsec.so` could not find `libdplasma.so.2` without
  `LD_LIBRARY_PATH`. The BLAS/LAPACKE directories are recorded as well (see
  above).
- `HICMA_PARSEC_GENOMICS` option: the genomics drivers are built only when it
  is on (they need a CUDA build: the genomics kernels have no HIP bodies); the
  default build no longer ships binaries that cannot run. The Cholesky CTest
  cases are not registered in such a build, where `testing_potrf` refuses to run.
- Every HiCMA CTest case requires DPLASMA's BLAS thread-safety test as a
  fixture: a BLAS that cannot take concurrent calls now reports "Not Run"
  instead of wrong residuals. The test gets the BLAS run-time path (on an MKL
  module it could not start). The CPU CI job prints the runner's CPU and
  OpenBLAS variant and sets `OPENBLAS_NUM_THREADS=1`; one run had failed
  inside DPLASMA's reference Cholesky on a single runner, while an otherwise
  identical run passed.
- CTest registers the HiCMA tests (labels `hicma`, `mpi`, `gpu`), all with
  `--check 1`; `scripts/test_CI.sh` runs the same cases and no longer depends
  on a private data path; CI runs them and a CUDA compile job. The script and
  `scripts/validation_matrix.sh` launch the two-rank cases with `-n` (mpirun,
  or srun inside a Slurm allocation; skipped without a launcher), skip the
  GPU cases of kinds 1 to 4 on HIP and use `device_hip_memory_use` there.
- `scripts/` reorganized: generic tools at the top level, site recipes under
  `scripts/machines/`, genomics data tools under `scripts/genomics/`,
  byte-identical and obsolete scripts removed; the stale Dockerfiles removed.
- The machine build scripts are rewritten for the current build: one
  configure of the whole tree, `[build-dir] [extra CMake options]`, modules
  loaded in a login shell. They built and installed STARS-H and HCORE
  separately (unused: the build compiles the submodules), passed PaRSEC's GPU
  flags without `HICMA_PARSEC_HAVE_CUDA` (Leonardo without GSL: HiCMA-X built
  without CUDA), ran CMake from the wrong directory (Fugaku), loaded Python 2
  (Shaheen; DPLASMA needs Python 3) or emptied an existing `build/`
  (`build_hicma_with_lapack.sh`). `build_hicma_with_lapack.sh` builds a pinned
  LAPACK (3.12.1) once and passes the search hints STARS-H and HCORE need;
  `build_release.sh` uses `cmake --build`. `scripts/machines/legacy/` (Summit,
  Saturn, ECRC, ExaGeoStat builds) is removed.
- CUDA builds compile for the documented default architectures, 70 80 90 90a
  (80 90 90a with CUDA 13), unless `CMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS`
  is given. The default never applied: `enable_language(CUDA)` cached the
  compiler's default first, so a build with only `-DHICMA_PARSEC_HAVE_CUDA=ON`
  compiled its CUDA code for sm_52 alone, which newer GPUs ran after JIT
  compilation.
- `.gitignore` covers build trees, logs and editor state; LICENSE placeholder
  removed.

### GPU GEMM variants (`--tensor_gemm`)

- The option was a bitmask of mutually exclusive choices that each GEMM body
  read bit by bit, unchecked. On a GPU: no FP16 variant (`0x3`) or two
  (`0x18`) gave wrong results in kind 3 (residual 43) and aborted kind 5
  mid-run; BF16 gave wrong results in kind 3; with `--adaptive_decision`
  the option was ignored, and with `0x10` in a TLR run the dense and the
  low-rank GEMMs of one tile disagreed on the type of C (NaNs, then a
  crash); `0x80` in the Cholesky exited 0 without factorizing.
- `--tensor_gemm_hp` names the variant of the FP16 tiles and
  `--tensor_gemm_sp` FP32 or TF32 for the FP32 tiles (`TODOs` item 12);
  `--tensor_gemm` remains the combined form (one FP16 variant, plus 0x4 for
  TF32; 0x1 and 0x2 accepted, no effect), and the new options override its
  parts. `hicma_parsec_init` refuses malformed values and the variants a
  path does not implement; the GEMM bodies derive operand, C and compute
  types from the one variant (`HICMA_FP16_GEMM`); the adaptive decisions use
  it and TF32. The default computes as before (validation matrix identical).
- An independent review of the change found more paths that disagreed or
  were unchecked, now handled: BF16 with `--datatype_convert 2` read the
  FP16 tiles it sends as BF16 (residual 84); the TRMM drivers skipped the
  value check, so `--tensor_gemm_hp 0x80` was ignored there; the low-rank
  GEMMs now take the variant of their C tile from `decisions_gemm_gpu`, as
  the dense GEMMs do, and keep an FP8 C in FP32; 0x10 is refused when a
  low-rank tile has rank 0 (its skipped update would skip C's conversion);
  kind 10 accepts every variant unless the decisions are made at run time.
- GPU runs with a static FP16 band and low-rank tiles (kinds 1 and 5) gave
  wrong results (residuals 10 to 59) or segfaulted, also with 0.1.0's code.
  In the FP16 branch of `hicma_parsec_core_gemm_denseC_lrA_lrB_gpu` the
  low-rank B (U and V of rank Brank) was converted as a dense mb x nb tile,
  reading a LOW_RANK_DP B's FP64 bytes as FP32 and past its buffer; every
  FP16 tile with a low-rank x low-rank update received garbage. It is now
  converted as U and V, like A; residuals 4.3e-5 (FP32 band: 2.1e-5, CPU:
  3.5e-5), and two ranks agree with one.
- On GPUs with FP8 (H100), kind 8 with an FP8 band and low-rank tiles gave
  wrong results (residual 1.5e-2) or crashed: the low-rank GEMMs, CPU and
  GPU, converted a dense operand from FP32 only when it was SP or HP, so an
  FP8 tile (stored and sent as FP32 like them) was read as FP64. Thirteen
  conditions in five kernels and the core file now include DENSE_FP8;
  residual 1.6e-5.
- New CTest cases: BF16 with FP32 and with BF16 C, TF32, adaptive TLR with
  FP16 C, a static FP16 band with low-rank tiles in kinds 5 and 1, an FP8
  band with low-rank tiles in kind 8 (GPU), and the start-up refusal of
  `0x18` (CPU); `scripts/test_CI.sh` runs them too.

### Usability (external-user pass)

- `--nruns n` crashed in TLR runs: run 2 factorized run 1's factor, whose
  ranks outgrow the tile buffers. Where A cannot be regenerated in place,
  one run is done and a warning says why.
- Notes for three silent performance traps: GPUs visible in a GPU build run
  without `--gpus` (7x slower on an A100); kinds 1 and 2 without
  `--band_dense`, where every tile is dense (6.5x slower at N=30720); a
  launcher binding each process to one core, the Open MPI default (10x
  slower with two processes on a 36-core node).
- `--help` and `TESTS.md` gave wrong defaults: `--band_dense` defaults to
  NT (every tile dense), not 1; `--N` and `--NB` default to 2048, not
  "required"; `--HNB` to 300; `--cores` counts physical cores.
- One `R-LO` line per run, without the uninitialized analysis counters of
  dense runs (the drivers zero-initialize the analysis); the version and the
  `git describe` revision are printed (`GITHASH` was never set).
- Internal set-up notes print with `--verbose 1` only; two invalid-option
  exits returned 0.
- `--gpus` above the usable GPUs, or on a build without GPU support, used
  fewer silently (the note needed `--verbose 2`); and several processes per
  node each taking all its cores (`--bind-to none`) went unnoticed. Rank 0
  now says so in both cases.
- Several GPUs that cannot all read each other directly (8-GPU servers
  whose cross-socket pairs have no NVLink) make PaRSEC copy every GPU task's
  output to host memory: four GPUs across two sockets ran a dense N=40960
  four times slower than four linked ones. Rank 0 says so and points to
  `nvidia-smi topo -m` and `CUDA_VISIBLE_DEVICES`.
- `README.md` has a quick start; `scripts/build_release.sh --help` states
  its default build type (RelWithDebInfo).

- `-DBLA_VENDOR=OpenBLAS` stopped the configure step in STARS-H, whose
  find module names the vendor `Open`; the top-level CMake translates it for
  STARS-H and HCORE. Checked with OpenBLAS 0.3.28 on a node that also has
  MKL: every component links OpenBLAS, CTest 48 / 48.

### Frontier pass (HIP on MI210 / MI250X)

- The machine scripts took their first argument as the build directory even
  when it was a CMake option: `install_icl.sh -DFOO=ON` configured with an
  empty build directory and dropped the option. A first argument starting
  with `-` now goes to CMake and the build directory stays the default.
- `scripts/test_CI.sh` and `scripts/validation_matrix.sh` pass
  `--oversubscribe` (and `--bind-to none`) only to Open MPI's `mpirun`;
  MPICH's refuses them.
- `potrf_autoband_3d` (band auto-tuning, kind 4) is registered with CTest; it
  ran only in `scripts/test_CI.sh`.
- A passing check prints its residual and bound (`Check passed: ...`), as a
  failed one did.
- `--tensor_gemm_sp 0x4` (TF32) on a HIP build warns that AMD GPUs run those
  GEMMs in FP32.
- `--band_size_termination`: the help gave "default 3.0, on GPU"; the default
  is 0.67 and the value is used on CPUs only (with GPUs the factor comes from
  the performance model).
- The installed headers carry `@version 0.2.0` (they said 1.0.0).
- AMD GPUs flush FP16 subnormals in their FP16 GEMMs (see the kind-10 entry
  under Correctness). In the adaptive kinds 5, 8 and 9 the FP16 tiles hold
  small entries, so on an MI210 and an MI250X an adaptive kind-5 run at
  N=30720 gave 1.0e-5 for `--fixedacc 1e-8`, and `gpu_dense_adaptive` failed
  now and then (1.9e-6, 2.8e-5). The same tiles in BF16
  (`--tensor_gemm_hp 0x20`) give 1.2e-8. `--tensor_gemm` keeps choosing and
  the default stays FP16; a HIP run whose decisions produce FP16 tiles says
  how many and points to 0x20, and the HIP builds' adaptive GPU tests use it.
- With a BF16 variant (0x20, 0x40) the adaptive decisions use BF16's epsilon
  (3.91e-3) instead of FP16's: BF16 tiles chosen with FP16's threshold missed
  `--fixedacc` by up to 8x. Static bands are unchanged.
- HIP: every rocBLAS/rocSOLVER handle is set up before the factorization.
  Their first calls happened inside it: the first dense factorization at
  N=8192 took 0.52 s on an MI210 and the next ones 0.055 s. Now the first one
  takes 0.058 s; at N=30720 FP64 matches DPLASMA (0.93 s) and mixed precision
  is faster than FP64 (adaptive 0.70 s, FP16 band 0.55 s), where it was slower.
- The message for a GPU handle that cannot be created names
  `device_hip_memory_use` on HIP.
- Multi-rank GPU runs under Cray MPICH (Frontier) aborted at the first
  transfer (`process_vm_readv: Bad address`): PaRSEC sent from GPU memory, and
  GPU runs allocate host tiles with `hipHostMalloc`, neither of which Cray
  MPICH reads without `MPICH_GPU_SUPPORT_ENABLED=1` and its GTL library.
  Without them HiCMA-X keeps remote transfers in host memory and, when it
  initializes MPI, sets `MPICH_SMP_SINGLE_COPY_MODE=NONE`
  (`PARSEC_MCA_mpi_gpu_aware` and a user-set mode still win). With GTL and
  the variable set, two ranks with one MI250X GCD each pass as they are.

### Radeon VII pass (HIP on gfx906)

- HiCMA-X ran on an AMD Radeon VII (gfx906) with the ROCm 6.3.2 user space:
  CTest 46 / 46 (the 14 GPU cases, OmniCast's 4 tests, the two-rank cases),
  with OmniCast and the GPU operand cache. ROCm 6.4 and later ship no rocBLAS
  kernels for gfx906: build against ROCm 6.3 or earlier there
  (`docs/release-readiness.md`, section 5.20).
- gfx906 keeps FP16 subnormals in rocBLAS's FP16 GEMMs (FP32 and FP16 C) and
  has no BF16 hardware: BF16 GEMMs (`--tensor_gemm_hp 0x20`) made an adaptive
  kind-5 run at N=30720 6x slower (12.7 s against 2.1 s). The start-up note
  that points HIP runs with FP16 tiles to BF16 is no longer printed on gfx906.
- The operand cache gives bit-identical residuals on gfx906 (11
  configurations; cache on, off and at 4 slots). Conversions are 2.6-2.9 % of
  the GPU kernel time there, so halving them leaves the factorization within
  noise (0.997-1.014x).

### Multi-node and multi-GPU pass (two Radeon VII nodes)

- Multi-rank GPU runs aborted at the first tile a GPU sent when the MPI
  library cannot read GPU memory (Open MPI 5.0.5 and UCX 1.15 built without
  ROCm: `ibv_reg_mr ... Bad address`, then the other rank waited forever).
  Two causes:
  - HiCMA-X told PaRSEC to send from GPU memory (`mpi_gpu_aware` 2) whatever
    the library. It now asks it (Open MPI's `MPIX_Query_rocm_support` and
    `MPIX_Query_cuda_support`, MPICH's `MPIX_GPU_query_support`) and keeps
    remote transfers in host memory when the answer is no, with a start-up
    note. Cray MPICH is handled as before, and a user-set
    `PARSEC_MCA_mpi_gpu_aware` still wins. On NVIDIA nodes Open MPI reports
    CUDA support and nothing changes.
  - PaRSEC sent from GPU memory even with `mpi_gpu_aware` 0. A GPU task
    forwards a flow it only reads in the GPU copy it read (the
    `potrf_*_send` tasks forward the POTRF and TRSM results that way), and
    the engine was handed that copy. PaRSEC (fork) now sends the host copy,
    and a task whose output a local GPU task forwards read-only writes the
    host copy back when there are other ranks. Two V100 ranks with
    `mpi_gpu_aware` 0 now send no tile from GPU memory, with the same
    residuals.
- With two GPUs in a process, GPUs faulted on host pages ("Memory access
  fault ... Page not present"): at nearly every run of the runtime decisions
  and of kind 10 on two nodes, and rarely on one. HIP pins pageable host
  memory for each copy from it, and the pageable sources were PaRSEC's arena
  buffers (received tiles, new and temporary tiles, host copies). PaRSEC
  (fork) lets an application choose the allocator of arena memory
  (`parsec_arena_data_allocate`), and HiCMA-X makes it pinned on HIP with
  more than one GPU per process. The matrix tiles are pinned already.
- One of the two nodes still faulted now and then (17 of 132 runs) in the
  `--check`, always in its LL^T: a `malloc` buffer that PaRSEC registers
  (`hipHostRegister`) at its first GPU use. The LL^T is now allocated like
  the tiles (pinned when GPUs are enabled, on CUDA as well) and given a
  registration hook that does nothing; that node then passed 48 of 48.
- Validated on histamine2 and histamine3 (two Radeon VIIs each, InfiniBand),
  one rank per node: the two-rank CTest cases, the 14 GPU cases and the
  operand cache on two nodes with one GPU each, and the GPU cases and the
  cache on two nodes with two GPUs each and on one node with two GPUs
  (`docs/release-readiness.md`, section 5.21).
- Also run on two NVIDIA nodes, a V100 (leconte) and an A100 (methane), one
  rank each, over Ethernet with Open MPI and UCX built with CUDA, so tiles
  leave from GPU memory. The two-rank CTest cases, the 22 GPU cases and the
  operand cache pass, and sending from GPU memory and through host memory
  give the same residuals. One run in 144 gave a wrong factor and was not
  reproduced; other failures were clean aborts while another job used the
  same GPU (`docs/release-readiness.md`, section 5.22).

### Documentation

- `TESTS.md` regenerated from the option table; `BUILD.md`, `README.md` and
  `RELEASE.md` updated; `CHANGELOG.md` added.

### Known limitations (see `docs/release-readiness.md`)

- Cholesky drivers need `N % NB == 0`.
- Kinds 1 to 4 have no HIP bodies. HIP GPU runs were validated on Frontier
  (MI210, MI250X; `docs/release-readiness.md`, section 5.15) and on a Radeon
  VII (gfx906, section 5.20).
- AMD gfx90a GPUs (MI210, MI250X) flush FP16 subnormals in their FP16 GEMMs:
  use `--tensor_gemm_hp 0x20` (BF16) with adaptive precisions in the dense
  layout. gfx906 keeps them and has no BF16 hardware: keep FP16 there.
- `--enable_stochastic_rounding` is not implemented for kinds 8, 9 and 10.
- The CPU and GPU runtime-decision GEMMs use different storage policies for
  FP16 updates.
- Mixed-precision runs reach the accuracy of the precisions they use, not an
  FP64-level `--fixedacc`; use `--check_tol`.

## 0.1.0 (2025-11-12)

First public release of HiCMA-X (then named HiCMA-PaRSEC): tile low-rank and
mixed-precision Cholesky factorizations on PaRSEC with CUDA and HIP support,
the climate emulator and the genomics (kernel ridge regression) applications.
