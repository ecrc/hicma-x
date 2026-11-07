# HiCMA-X code review (release 0.2.0 preparation)

Branch `qinglei/release-cleanup`, reviewed from commit 6e987e3 (`main` == `ecrc/main`
== branch `v0.2.0`, 2026-11-07). Submodules: dplasma 3553b25a, hcore 78028a5,
stars-h 5d98f2c.

## 1. Scope and method

Everything under `src/`, `tests/`, `scripts/`, `docker/`, `cmake_modules/`, the three
CMake lists, the CI workflow and the top-level documents were read. PaRSEC and DPLASMA
semantics were checked against the submodule sources rather than assumed. Findings
were classified only after the code path was followed end to end; where a claim could
not be confirmed by reading alone it is marked *plausible* or *speculative*.

Baseline behaviour was measured on `methane` (72 cores, one A100-PCIE-40GB, CUDA
12.6, Open MPI 5.0.5, MKL 2024.2, ROCm 6.4.3 for HIP compile-only) with two fresh
Release builds of the branch point (`build_rc_cpu`, `build_rc_cuda`) and the matrix of
40 `--check 1` runs in `docs/release-readiness.md` (every Cholesky kind, TRMM, 1 and 2
ranks, CPU and GPU).

Severity:

- **P0** correctness, memory safety, concurrency, broken build or release blocker.
- **P1** architecture, duplication or maintainability problems fixable within this
  release without new features.
- **P2** localized cleanup, API consistency, tests, documentation.
- **P3** optional.

Confidence: *confirmed* (code path read end to end, or reproduced), *plausible*,
*speculative*. "Default" means the committed compile-time flags (`GENOMICS=0`,
`FOR_CLIMATE_EMULATOR=0`, `PRINT_RANK=0`, `DEBUG_INFO=0`, `BAND_MEMORY_CONTIGUOUS=0`,
`GPU_BUFFER_ONCE=1`, `GENERATE_RANDOM_DATA=0`) and the command-line defaults.

Out of scope by instruction: the `TODOs` backlog (merging the Cholesky JDFs, merging
the `Lapack_to_Tile`/`Tile_to_Lapack` families, hiding the tile size, API redesign,
Python/Julia bindings, removing hcore, recursive-device fixes, merging the GENOMICS
macros, making `hicma_parsec_init` usable for external users, restructuring the
parameter struct, redesigning `tensor_gemm`, memory register vs. allocate). Also kept
by maintainer decision: separate CUDA and HIP `BODY` sections in the JDFs, the single
runtime-decision GPU GEMM function, and static (pre-factorization) decisions in kinds
8 and 9. These are recorded in section 6 where they came up.

## 2. Executive summary

The numerical core is in better shape than the infrastructure around it. The ten
Cholesky variants, the mixed-precision TRMM and the conversion kernels produce correct
results in every configuration that was run (section 1), and the recent runtime-decision
and FP8 work holds up under review. The problems that matter for a release are:

1. **The correctness check cannot fail.** `check_dpotrf2()` and `check_diff()`
   compute a pass/fail flag and return 0; the reference dense Cholesky's `info` is never
   read. `--check 1` therefore exits 0 for any residual, and CI (which does not even pass
   `--check`) verifies only that the drivers do not crash. (E2, G4)
2. **The diagonal shift is applied twice** for problem kinds 0, 1, 5, 6 and 7: the
   STARS-H kernel adds `--adddiag` and `matrix_compress.jdf` adds it again, while the
   check's reference matrix has it once. The documented virus-mesh example factorizes
   A + 0.2 I instead of A + 0.1 I and reports a residual of 9.1e-2 against a 1e-4
   threshold, which item 1 hid. (E1)
3. **Memory-safety bugs on ordinary command lines**: a write through `NULL+1` when
   the binary is launched without a `/` in `argv[0]`; the GPU `info` array allocated with
   the requested GPU count but freed with the detected one; `free()` of an uninitialized
   `pheno_file` pointer; an 8-byte allocation for a 24-byte struct in the 2-D location
   reader. (A1, A2, A3, E3)
4. **The installed package is unusable**: the CMake config file is generated with the
   wrong command, the exported target lacks its dependencies and link interface, and the
   public header includes source-tree-relative PaRSEC and DPLASMA test headers. (G1, G2)
5. **Version identity is inconsistent** (tag v0.1.0, CMake fallback 2.0.0, headers
   1.0.0, a branch named v0.2.0, no VERSION file). (G3)
6. **Duplication without benefit**: ten ~900-line Cholesky wrappers that share their
   timing hooks, hook installation, arena and pool management; four integer-conversion
   JDFs that are one template; two byte-identical CUDA/HIP conversion files kept in sync by
   hand; 58 scripts of which most are personal machine recipes.

Nothing found requires changing a numerical method or the PaRSEC execution model.

## 3. Architecture assessment

**What the design gets right.** The split into a PaRSEC-facing wrapper per algorithm
(`*_wrapper.c`), a JDF that owns the task graph, and precision-agnostic kernels in
`hicma_parsec_core.c` is sound and matches DPLASMA's conventions. Per-tile decisions
live in flat arrays owned by `hicma_parsec_params_t` and are read by the JDFs through
`hicma_tile_idx()`, which keeps the JDFs free of allocation logic. The GPU workspace is
allocated once per stream (`GPU_BUFFER_ONCE`) and looked up by stream, which is the
right granularity for cuBLAS handles. The variable-size send path (`hicma_tile_trsm_bytes`,
FP8 tile layout with the scale appended) is explicit and documented in one place.

**Where the design costs more than it returns.**

- *The parameter struct is the whole program state.* `hicma_parsec_params_t` mixes
  user options, derived sizes, per-run statistics, timing accumulators written from
  worker threads, hook pointers and application data for three different applications.
  Every test declares it on the stack and `parse_arguments` initializes a subset of the
  fields, so uninitialized reads are a recurring class of bug (A3). Restructuring it is
  backlog; zero-initializing it and resolving the options in one place is not.
- *Parameter resolution is spread over four functions* (`parse_arguments`,
  `hicma_parsec_params_check`, `hicma_parsec_setup_parsec`, `hicma_parsec_potrf`) with
  later ones silently overriding earlier ones (A8).
- *Compile-time application switches.* `GENOMICS`, `FOR_CLIMATE_EMULATOR`, `PRINT_RANK`,
  `GPU_BUFFER_ONCE`, `BAND_MEMORY_CONTIGUOUS` and a dozen others are `#define`d to a
  constant in the public header. The disabled branches are not compiled, so they rot
  (E6 has a compile error behind one of them). Merging GENOMICS is backlog; the rest can
  be left as they are, with the dead branches removed where they have no user.
- *Macros that capture local variables by name* (`IS_DENSE`, `IS_ALLOCATE_DP`) only
  compile in functions that happen to name their arguments `params_tlr` and `descA`.
- *Ten wrappers for ten JDFs.* The wrappers are 80-95% identical. Merging the JDFs is
  backlog; the wrappers' shared parts (timing hooks, hook installation, kernel-time
  wiring, memory-pool creation and destruction) can be factored without touching a JDF.
- *The public header is the internal header.* `hicma_parsec.h` includes
  `hicma_parsec_internal.h`, which includes DPLASMA's test helpers by source-tree path.
  This is why the install set does not work (G2). Fixing the include paths is enough for
  this release; a real public/internal split is backlog.

The rest of this document lists the findings per area. Section 7 lists things that look
wrong but were verified to be fine, so that the next reader does not re-litigate them.

## 4. Findings

Field key: **Where** files and functions; **Root cause**; **Impact** (who hits it);
**Fix** (simplest robust solution); **Benefit**; **Complexity** (S small: < 1 h,
M medium: hours, L large: a day or more); **Risk** of the fix; **Validate**.

### 4A. Library lifecycle, parameters, GPU workspace

#### A1. `hicma_parsec_params_init` writes through NULL when `argv[0]` has no `/` — P0, confirmed, default

- **Where** `src/core/hicma_parsec_internal.c:1317-1323`: `sprintf(exe_file_path, argv[0])`,
  `slash = strrchr(...)`, `if (!slash) exe_file_path[0] = 0;` then unconditionally
  `*(slash + 1) = '\0'`. The `sprintf` into `char exe_file_path[200]` is unbounded.
- **Root cause** missing `else`; fixed-size buffer.
- **Impact** any launch by bare name (`srun testing_potrf_tlr`, PATH lookup) segfaults in
  init; an `argv[0]` longer than 199 bytes overwrites the following struct fields.
- **Fix** `snprintf` with `sizeof`, `if (slash) slash[1] = 0; else path[0] = 0;`.
- **Benefit** removes a crash. **Complexity** S. **Risk** none.
- **Validate** run the driver through `PATH`.

#### A2. `info_gpu` allocated with the requested GPU count, freed with the detected one — P1, confirmed, default

- **Where** `hicma_parsec_internal.c:1502-1512` allocates with `cudaMallocHost` when
  `--gpus > 0` (CPU-only build: prints "Devices is not supported" and allocates nothing);
  `hicma_parsec_setup_parsec` clamps `params->gpus` to the detected count (`:1120`);
  `hicma_parsec_free_memory` (`:2980-2990`) chooses `cudaFreeHost` vs `free` from the
  clamped value.
- **Root cause** allocation happens before device discovery.
- **Impact** `--gpus 1` on a node without a usable device frees pinned memory with
  `free()` (CUDA build) or frees an uninitialized pointer (CPU build). Shared CPU/GPU job
  scripts hit this.
- **Fix** allocate `info_gpu` in `setup_parsec` after the clamp; NULL the pointer in
  `parse_arguments`; record how it was allocated.
- **Benefit** removes a crash at `fini`. **Complexity** S. **Risk** none.
- **Validate** CUDA build with `CUDA_VISIBLE_DEVICES=` and `--gpus 1`; CPU build with
  `--gpus 1`.

#### A3. `hicma_parsec_params_t` is only partially initialized — P1, confirmed, default

- **Where** every driver declares the struct on the stack; `parse_arguments`
  (`hicma_parsec_internal.c:638-738`) sets a subset of the fields. `pheno_file` is
  never set to NULL but `parse_arg_value` frees the old value (`:203`), so `--pheno_file x`
  frees garbage. `gpu_type`, `nsnp`, `rbf_kernel`, `order`, `radius`, `log_det_dp`,
  `time_syrk_app`, `critical_path_trsm_message` and `result_accuracy` (a local shadows
  the field at `:1460`) are printed uninitialized in the default R-LO line; the
  post-analysis prints `|log_det_dp-log_det_mp|/|log_det_dp|` from garbage.
- **Root cause** no `memset`; the default list drifted from the struct.
- **Impact** UB on a documented option; non-reproducible benchmark output columns.
- **Fix** `memset(params, 0, sizeof *params)` as the first statement of
  `parse_arguments`, then the non-zero defaults; assign the `result_accuracy` field.
- **Benefit** deterministic output, no UB. **Complexity** S. **Risk** none
  (`climate_emulator_init` sets its fields after `hicma_parsec_init` returns).
- **Validate** Valgrind on a dense run; R-LO line stable across runs.

#### A4. Per-core statistics arrays allocated by two duplicated blocks — P1, confirmed, default

- **Where** `hicma_parsec_internal.c:1033-1073` (`setup_parsec`, when `cores <= 0`) and
  `:1337-1339,1451-1481` (`params_init`, three `if (cores > 0)` blocks) allocate the same
  twelve arrays; `free_memory` frees them unconditionally. Both size the GPU arrays with
  the pre-clamp GPU count.
- **Root cause** `--cores` handling was added by copying the block.
- **Impact** any field added to one block and not the other becomes a leak or an
  uninitialized read.
- **Fix** one `hicma_parsec_stats_allocate()` called from `setup_parsec` after
  `cores` and `gpus` are final.
- **Benefit** one allocation site, correct sizes. **Complexity** S. **Risk** low.
- **Validate** runs with and without `--cores`; Valgrind leak check.

#### A5. GPU workspace leaks and unchecked cuBLASLt setup — P2, confirmed, CUDA/HIP

- **Where** `hicma_parsec_gpu.c:316` mallocs a `parsec_device_cuda_module_t` placeholder
  per device that is overwritten (`:599`) and never freed; `gpu_temporay_buffer_fini`
  (`:871-873`) returns without freeing `ws` when no GPU exists although `data_init`
  always allocated it; the FP8 block (`:650-744`) does not check `cublasLtCreate`,
  `cublasLtMatmulDescCreate`, `cublasLtMatrixLayoutCreate`, `cudaMalloc(32 MB)`,
  `cublasLtMatmulPreferenceCreate`; `gpu_handle_init/fini` (`:462-561`) is a reduced,
  divergent copy of the workspace init/free used only by `testing_syrk`.
- **Impact** leaks at exit; with several ranks sharing a GPU the 32 MB workspace
  allocation is the likeliest to fail and does so silently.
- **Fix** NULL-initialize the placeholder, free `ws` unconditionally, wrap the five
  calls in `HICMA_GPU_HANDLE_CHECK`, make `testing_syrk` use the shared workspace init.
- **Complexity** S. **Risk** low. **Validate** `compute-sanitizer --leak-check full`
  on a small run; `testing_syrk`.

#### A6. Whole dense matrix pinned with `cudaHostRegister` even with `--gpus 0` — P2, confirmed, CUDA/HIP

- **Where** `hicma_parsec_internal.c:2327-2331` registers `dcAd.mat` and clears
  PaRSEC's `register_memory` hooks regardless of `params->gpus`; return value ignored;
  `cudaHostUnregister` at `:3008` unconditional.
- **Impact** CUDA builds run CPU-only pin the whole local matrix and create a CUDA
  context per rank; a failed registration is silent and PaRSEC cannot retry.
- **Fix** guard with `gpus > 0`, check the return code.
- **Complexity** S. **Risk** low. **Validate** CUDA build with `--gpus 0`; `nvidia-smi`
  shows no context.

#### A7. `params->gpus` clamped per rank without inter-rank agreement — P2, plausible, default

- **Where** `hicma_parsec_internal.c:1120` vs. the `MPI_Allreduce(MIN)` done for
  `fp8_gpus_ok` right below (`:1137`). `hicma_parsec_get_precision_tile` changes
  decisions when `gpus < 1` and `hicma_parsec_potrf` issues a collective only when
  `gpus > 0` (`hicma_parsec.c:510-512`).
- **Impact** heterogeneous allocations (one rank with no visible GPU) diverge in their
  decision arrays and mismatch a collective.
- **Fix** `MPI_Allreduce(MPI_IN_PLACE, &gpus, MIN)` after the clamp, as for FP8.
- **Complexity** S. **Risk** low. **Validate** two ranks, one with
  `CUDA_VISIBLE_DEVICES=`.

#### A8. Parameter resolution scattered over four functions with silent overrides — P2, confirmed, default

- **Where** `parse_arguments`, `hicma_parsec_params_check`
  (`hicma_parsec_internal.c:1926-2263`), `hicma_parsec_setup_parsec`,
  `hicma_parsec_potrf`. Examples: `--adaptive_decision_runtime` forces kind 10 and
  `band_size_dense = NT` (`:1943-1947`) but `--auto_band` and `--sparse` later overwrite
  both; `lookahead` is resolved in three places; `band_size_dense_hp` is excluded from
  the clamping/assert chain so `--band_dense_hp` larger than `--band_dense` classifies
  off-band tiles as HP; `--help=1` writes through a NULL target (`:308,470`); `--check 5`
  is accepted while the message says 0 or 1; `if (0 == N || 0 == NB) exit(1)` is
  unreachable (both default to 2048); messages still cite removed short options
  (`-w 5`, `-E 1`, `-A 1`, `-d 0`).
- **Impact** unusual flag combinations silently do something else; the dispatcher
  reduces `info` on `parsec->comm_ctx` and everything else on `MPI_COMM_WORLD`
  (`hicma_parsec.c:506` vs `:372,444,511,533`); `gflops` is assigned inside the rank-0
  print macro (`:463`).
- **Fix** reject the incompatible combinations in `params_check`, include
  `band_size_dense_hp` in the clamp chain, give `help` a dummy target, validate 0|1
  flags, use one communicator, compute `gflops` as a statement. Full consolidation is
  backlog (params restructure).
- **Complexity** M. **Risk** low; only invalid combinations change behaviour (they
  error instead of proceeding).
- **Validate** `--help`, `--help=1`, `--check 5`, `--adaptive_decision_runtime 1 --sparse 1`.

#### A9. Init error path and exit-time leaks — P2/P3, confirmed, default

- **Where** `hicma_parsec_init` returns NULL only from `kernel_init` (`hicma_parsec.c:88`)
  after MPI may have been initialized and ~15 arrays allocated; every other failure is an
  `exit()` without `MPI_Finalize`. `kernel_init` kinds 10/12 leak `locations`/`xycoord`.
  At `fini`: `mesh_file`/`pheno_file` never freed; `starsh_ssdata_free` missing for kinds
  8, 12, 13 (`hicma_parsec_internal.c:2937-2952`); global `dist_array` never freed; the
  eight memory-pool structs malloc'd by every Cholesky wrapper are `parsec_private_memory_fini`'d
  but never `free`d (only `trmm_mp_wrapper.c:225-228` frees them).
- **Fix** free and finalize on the NULL path; add the three kinds to the switch; free the
  pool structs in every `Destruct`.
- **Complexity** S. **Risk** none.

#### A10. HIP parity gaps in lifecycle code — P3, plausible, HIP

- **Where** `setup_parsec` registers only `DPLASMA::CUDA::HANDLES`
  (`hicma_parsec_internal.c:1101-1110`); DPLASMA's HIP bodies look up
  `DPLASMA::HIP::HANDLES`, and `--check` calls `dplasma_dtrmm` which has HIP bodies.
  The static `destroy_cuda_handles` (`:112-127`) duplicates the exported
  `dplasma_destroy_cuda_handles`.
- **Fix** register the HIP handles under `PARSEC_HAVE_DEV_HIP_SUPPORT`; use the DPLASMA
  function. **Complexity** S. **Validate** HIP compile; a HIP `--check` run when hardware
  is available.

#### A11. Compile-time flag hygiene — P3, confirmed, non-default paths

- `#if PREDICTION || CHECKSOLVE && GENOMICS` (`:2306`) vs `#if PREDICTION || CHECKSOLVE`
  (`:2336,3018`): precedence bug, `dcAcpy` allocated without `_init` for
  `CHECKSOLVE=1, GENOMICS=0`.
- `hicma_parsec_get_precision_tile` (`hicma_parsec_decision.c:787-821`) builds an
  `if/else if` chain from nested `#if HAVE_FP8/HAVE_HP`; with `HAVE_HP=0` an FP8 decision
  is overwritten by SP. Dormant (`HAVE_HP` is 1).
- `#define main MAIN__` under `ON_FUGAKU` in a header included by every library TU.
- `print_decisions*` call `sleep(1)` on every rank (`hicma_parsec_decision.c:144-237`).
- **Fix** parenthesize; rewrite the chain with runtime conditions; move the Fugaku
  `main` rename to the drivers; replace `sleep` with a barrier.

### 4E. Precision conversion, correctness checking, I/O, sparse analysis, utility JDFs

#### E1. `--adddiag` applied twice for problem kinds 0, 1, 5, 6, 7 — P0, confirmed (reproduced), default

- **Where** `hicma_parsec_kernel_init` passes `add_diag` to STARS-H for kinds 0
  (`starsh_randtlr_generate(..., diag)` adds it on the diagonal), 1 and 5
  (`starsh_eddata_generate(..., diag, ...)` sets the diagonal to it), 6 and 7
  (`starsh_generate_3d_rbf_mesh_coordinates_*(..., isreg=1, reg=add_diag, ...)`, which
  adds it, `stars-h/src/applications/mesh_deformation/virus.c:90`). `matrix_compress.jdf:130-139`
  then adds `params_tlr->add_diag` to every diagonal tile again. The check's reference
  (`band_regenerate.jdf`, kernel only) has the shift once.
- **Reproduced** `testing_potrf_tlr --N 10370 --NB 1037 --kind_of_problem 6 --adddiag 0.1
  --check 1 ...` (the TESTS.md example): `||A||` map 1.120e2 vs compressed 1.222e2,
  difference 1.018e1 = 0.1·sqrt(10370); final residual 9.09e-2 = 0.1/1.1 against a
  1e-4 threshold, exit code 0 because of E2.
- **Root cause** the shift was added to the generation task without removing it from the
  kernel setup (kinds 2, 3, 4, 8-11 pass `noise` separately and are not affected).
- **Impact** these kinds factorize A + 2δI; every reported accuracy for them is against
  the wrong matrix. The fix changes results for these kinds when `--adddiag != 0`.
- **Fix** add the shift in exactly one place: pass 0 / `isreg = 0` to STARS-H for kinds
  0, 6, 7 and keep the addition in `matrix_compress.jdf`; for the electrodynamics kinds
  1 and 5, where `diag` *is* the diagonal value, keep STARS-H's value and skip the
  addition. Alternatively skip the addition in `matrix_compress.jdf` for kinds 0, 1, 5,
  6, 7; either way the reference (`band_regenerate`) and the factorized matrix must agree.
- **Benefit** correct matrix, meaningful residuals. **Complexity** S. **Risk** results
  change for these kinds (documented in the changelog).
- **Validate** the virus-mesh run reports `||LLᵀ−A||/||A||` below 1e-4 and the two norms
  agree.

#### E2. `--check` can never fail — P0, confirmed (reproduced), default

- **Where** `check_dpotrf2` (`src/core/hicma_parsec_check.c:217-235`) and `check_diff`
  (`:289-302`) compute `info_factorization` and `return 0`; `hicma_parsec_check_dpotrf`
  ORs those zeros into `params->info` (`:541,576`) so its `exit(1)` (`:581`) is dead.
  `check_diff` never assigns `result`. The drivers return 0 when `info != 0`
  (`tests/testing_potrf.c:105`, `testing_climate_emulator.c:121`, `testing_KRR_file.c:197`).
- **Root cause** return value lost in a refactor.
- **Impact** the only automated correctness gate is non-binding (E1 went unnoticed; the
  CI log shows `Factorization is suspicious (info = 1)` followed by `[SUCCESS]`).
- **Fix** return `info_factorization`; `MPI_Allreduce(MAX)` the flag before deciding;
  make the drivers return non-zero when `info != 0`.
- **Benefit** tests mean something. **Complexity** S. **Risk** runs that silently
  passed with wrong results start failing, which is the point.
- **Validate** `--check 1 --fixedacc 1e-30` exits non-zero; the E1 reproducer fails
  before the E1 fix and passes after.

#### E3. `readLocsFile` allocates 8 bytes for a 24-byte struct — P0, confirmed, default build (kind 12 input)

- **Where** `src/core/hicma_parsec_flat_file.c:105`
  `malloc(sizeof(location*))` for `location {double *x,*y,*z}`, then writes `->y`.
  No `i < n` bound on the `getline` loop; `atof(strtok(NULL, ","))` without a NULL check;
  `countlines()` undercounts files without a trailing newline. `readLocsFile3d` (`:160`)
  is correct.
- **Impact** heap overflow on every `--kind_of_problem 12 --mesh_file` run.
- **Fix** `sizeof(location)`, bound the loop, check `pch`.
- **Complexity** S. **Risk** none. **Validate** ASan build with a 2-column CSV.

#### E4. Reference dense Cholesky `info` never inspected; `matrix_det` hides breakdown — P1, confirmed, `--check`

- **Where** `hicma_parsec_check.c:447-466` (`dplasma_dpotrf_New(..., &info)`, `info`
  never read or reduced); `matrix_det.jdf:25-26` skips non-positive pivots in the
  log-determinant.
- **Impact** a non-SPD uncompressed matrix (too loose `fixedacc`) gives a garbage
  reference L and a finite `log_det_dp` without any message.
- **Fix** reduce `info`, report it, fold it into `params->info`; count non-positive
  pivots in `parsec_core_dmdet`.
- **Complexity** S. **Risk** none.

#### E5. Check matrices not zero-initialized; `matrix_uncompress` writes only `final_rank > 0` tiles — P2, plausible, `--check` with zero-rank tiles

- **Where** `hicma_parsec_check.c:379` (`parsec_data_allocate` is `malloc`),
  `band_regenerate.jdf:74-83` (`cudaMallocHost` when GPUs), `matrix_uncompress.jdf:86-90,149-164`
  (skips tiles with `final_rank == 0`; output flows `D`/`D2` declared `READ`).
- **Fix** `dplasma_dlaset` the check matrices after allocation or zero the tile when
  `final_rank == 0`; declare the flows `RW`.
- **Complexity** S. **Risk** none.

#### E6. Genomics helpers reachable from the default-built `testing_KRR` are broken — P1, confirmed, non-default workflow

- **Where** `matrix_sum_vec.jdf:149` zeroes `NT` floats of an `N`-float accumulator;
  `:58-79` the last tile row reads `float` while the regular path reads `int8_t`;
  `matrix_add.jdf` has only a CUDA body (CPU body commented out at `:197-217`) and
  references undefined `params`/`storagemaxrank` under `#if !GPU_BUFFER_ONCE` (`:310`).
- **Impact** `testing_KRR` is built everywhere, cannot complete on CPU, and sums garbage.
- **Fix** `memset(vec, 0, N * sizeof(float))`; consistent element type; restore the CPU
  body or make the driver refuse to run without GPUs; fix the identifiers.
- **Complexity** S. **Risk** none.

#### E7. Smaller correctness issues in conversion and analysis — P2, confirmed

- `hicma_parsec_matrix_norm_get` passes a NULL taskpool to `parsec_context_add_taskpool`
  when `params->uplo != Lower` (`matrix_norm_get.jdf:128-131,187-192`); same missing NULL
  check in five other utility JDFs.
- `convert_s2d` in-place path leaves `span` at the FP32 size (`convert_s2d.jdf:57-75`);
  latent because nothing stages the tile to a GPU afterwards.
- `convert_d2s` LOW_RANK_SP branch shrinks the UV buffer to `rank*2` columns although
  `send_full_tile` relies on `maxrank*2` capacity (`convert_d2s.jdf:54-59,127-138`);
  TLR-MP with `--send_full_tile 1` only.
- Sparse analysis: `gemm_local` rows allocated for `A || Dist` ownership but freed only
  for `A` (`hicma_parsec_sparse_analysis.c:361-390,538-543`); `dist_array` never freed;
  `NT > 65535` only warns although every `uint16_t` index wraps (`:53-56`);
  `#include <omp.h>` with `USE_OPENMP 0`.
- I/O: MPI-IO helpers continue after a failed `MPI_File_open`
  (`hicma_parsec_io.c:114-350`, live in the climate emulator); `dist_read.jdf:35-53`
  `memcpy` reads 10 bytes past `path`, unchecked `fopen`/`fread`, `int` offset math;
  `readObsFile`/`readTimeFile` unbounded and uncalled.
- **Fix** each is a few lines; see the item. **Complexity** S each.

#### E8. Dead and duplicated utility JDFs and helpers — P2 (duplication), P3 (dead), confirmed

- No live caller: `Lapack_to_Tile_sym`, `cast_double_diag`, `check_read_data`,
  `matrix_ssum`, `matrix_dsum` (caller in `#if 0`), `sqrt_kernel` (caller commented),
  `reorder_gemm` (caller in `#if 0`; `--reorder_gemm` still allocates `dcReorder` and
  never frees it; the `k_pre/k_next` hooks are commented out in `potrf_L_dense_tlr_dp.jdf:574-575`),
  `convert_i2s`; on disk but not built: `copy2double.jdf`, `isyrkmp_LT.jdf` (empty CPU
  body), `gemm_f16_wrapper.c`.
- One template: `convert_d2i`/`convert_i2d`/`convert_s2i`/`convert_i2s` (133 lines each,
  differ in the `convert_*_unary_CPU` call and a size constant; each declares an
  `extern int *decisions` that nothing defines); `matrix_dsum`/`matrix_ssum`
  (byte-identical modulo type); `dist_allocate`/`_sym`/`_hamming` (iteration space and
  element size).
- Unused statics: `float_to_half`, `half_to_float_fast`, `float32_to_float16`,
  `float2half_CPU` (`hicma_parsec_datatype_convert.c`), `hicma_parsec_get_stat`
  (`hicma_parsec_rank_statistics.c`), `hicma_parsec_decisions_send_analysis` and two
  `print_decisions_*` (`hicma_parsec_decision.c`); declared-but-undefined
  `parsec_dist_allocate_gemmmp_TN`, `parsec_gemmmp_TN`, `hicma_parsec_decisions_gemm_gpu_init`,
  `decision_datatype_tile_gemmmp_TN`, `decision_datatype_tile_convert`,
  `decision_datatype_out`, `decision_size_out`; duplicate prototypes in
  `hicma_parsec_datatype_convert.h:720/731,746/757`.
- The `Lapack_to_Tile`/`Tile_to_Lapack` family (9 JDFs) is pure type duplication:
  **deferred** (backlog).
- **Fix** delete the dead items; merge the four int conversions into one JDF with a type
  global; delete `matrix_*sum`. **Complexity** M total. **Risk** low (nothing calls them).

#### E9. Minor — P3, confirmed

- `get_datatype_size` returns 0 for `fp4`/`int4`/`1bit` (`hicma_parsec_datatype_convert.c:1484-1496`).
- `matrix_check_norm_diff.jdf:50-88` generates the edge tile with `ld = tempmm` and
  reads it with `ld = mb`; its two memory pools are finalized but not freed.
- `memory_free_tile.jdf`/`band_free_memory.jdf` free `device_private` without NULLing it.
- `convert_s2i`/`convert_i2s` skip the last tile row while `isyrk_LT.jdf` reads every
  row as INT8 (`GENOMICS_ALLOCATE_SP` only).

### 4G. Build system, tests, scripts, documentation, CI, release infrastructure

#### G1. Installed CMake package is unusable — P0, confirmed

- **Where** `CMakeLists.txt:478-502` instantiates `cmake_modules/HICMAParsecConfig.cmake.in`
  with `configure_file`, so `@PACKAGE_INIT@` expands to nothing and
  `check_required_components()` is an unknown command in the generated file; no
  `find_dependency(PaRSEC/DPLASMA/MPI)`; `src/CMakeLists.txt:239` links the library to
  `parsec dplasma stdc++ starsh` only, leaving 26 undefined `HCORE_*`/`cblas_*`/`LAPACKE_*`
  symbols in the CPU `.so` (plus the cuBLAS/cuSOLVER set in the CUDA build) that only the
  test executables resolve; the exported `starsh` target carries an absolute GSL path;
  `GenPkgConfig` is included and never used.
- **Impact** `find_package(HICMAParsec)` fails; a consumer linking only the library
  fails to link; the package is not relocatable.
- **Fix** `configure_package_config_file` + `find_dependency`; put hcore, BLAS/LAPACKE,
  CUDA/HIP and MPI on the library's link interface (`PUBLIC`/`PRIVATE`) and drop them
  from the tests; remove the unused pkg-config include.
- **Benefit** an installable library. **Complexity** M. **Risk** low (the tests already
  link all of it).
- **Validate** install, then configure and build a ten-line consumer against the
  prefix alone; `ldd -r` shows no undefined symbols.

#### G2. Installed headers are not self-contained — P0, confirmed

- **Where** `src/core/hicma_parsec_internal.h:74-79,84,107-108,131` includes
  `dplasma/parsec/parsec/runtime.h`, `dplasma/tests/common_timing.h` (a DPLASMA *test*
  helper, never installed), `dplasma/src/dplasmajdf.h`,
  `../dplasma/parsec/parsec/mca/device/hip/device_hip.h`; the install list
  (`CMakeLists.txt:455-469`) omits `hicma_parsec_hip_cuda.h`, `hicma_kernel_time.h`,
  `climate_emulator.h`.
- **Fix** use the paths PaRSEC/DPLASMA install (`<parsec/...>`, `<dplasma/...>`), copy the
  four timing macros into a HiCMA header, install the missing headers.
- **Complexity** S/M. **Risk** low; HIP branch needs a HIP compile.
- **Validate** G1's consumer test for CPU, CUDA and HIP.

#### G3. Version identity inconsistent — P0, confirmed

- **Where** tag `v0.1.0` (lightweight, on the repository's first commit 644bba3,
  2025-11-12); `RELEASE.md` 0.1.0; `CMakeLists.txt:32-54` falls back to **2.0.0** because
  `VERSION.txt` does not exist, so the library is `libhicma_parsec.so.2.0.0` with
  SOVERSION 2; 15 headers say `@version 1.0.0`; a *branch* is named `v0.2.0`;
  `docker/Dockerfile.gene` and `scripts/update_version_and_date.py` read the missing
  file. Naming drifts between `HICMA_PARSEC`, `hicma-x`, "HiCMA-PaRSEC", `HICMAParsec`.
- **Fix** `project(... VERSION 0.2.0)` (or a `VERSION` file that is fatal when
  missing); SOVERSION 0 for the 0.x series; one display name; annotated tag `v0.2.0`
  after merge; delete the `v0.2.0` branch.
- **Complexity** S. **Risk** none (nothing links the installed library today).

#### G4. CI does not test correctness — P0, confirmed

- **Where** `scripts/test_CI.sh:229-247` runs six `testing_potrf_tlr` cases without
  `--check`; the climate step (`:287-293`) hard-codes `--mesh_file /home/qcao3/data`
  and the driver returns 0 after `info = 1`; `run_test()` (`:61-66`) tests `$?` after a
  successful `if` so its expected-exit-code logic is dead; only the three TRMM steps can
  fail; CI builds CPU only and never runs `cmake --install`; `MPIEXEC_PREFLAGS` is set
  and unused.
- **Fix** with E2 in place, pass `--check 1`; drivers return non-zero on `info`; skip
  the climate step when the data is absent; add an install smoke test.
- **Complexity** S. **Risk** CI time roughly doubles for those steps (small sizes).

#### G5. No HiCMA test registered with CTest — P1, confirmed

- **Where** `tests/CMakeLists.txt` has no `add_test`; `ctest` runs 242 PaRSEC/DPLASMA
  tests and zero HiCMA tests while `BUILD.md`, `CMakePresets.json`, `scripts/build_release.sh`
  and `scripts/test_build.sh` advertise `ctest`/`make test`.
- **Fix** register the existing drivers with `--check 1` (CPU, MPI via `MPIEXEC`, GPU
  behind a label); CI runs `ctest -LE gpu`.
- **Complexity** S. **Risk** none once E2/G4 make the exit codes meaningful.

#### G6. Configure-time `git submodule update --remote --force` — P1, confirmed

- **Where** `CMakeLists.txt:153-170`, run when a submodule's `CMakeLists.txt` is missing.
  `--remote` checks out the branch tip instead of the recorded SHA; `--force` discards
  local submodule edits.
- **Impact** a tagged release does not build what the tag records; a developer loses
  submodule work by re-running cmake. `.gitmodules` uses relative URLs that resolve from
  any `github.com/<org>/hicma-x` clone but not from local-path or mirror clones.
- **Fix** `git submodule update --init --recursive` without `--remote --force`, or a
  `FATAL_ERROR` telling the user to run it.
- **Complexity** S. **Risk** none.

#### G7. `cmake_minimum_required(VERSION 3.16)` understates the requirement — P1, confirmed

- PaRSEC requires 3.21 (`dplasma/parsec/CMakeLists.txt:7`), `enable_language(HIP)` 3.21,
  `CMakePresets.json` version 3 needs 3.21. **Fix** 3.21 and drop the two redundant
  `cmake_policy` lines. **Complexity** S.

#### G8. Build hygiene — P2, confirmed

- Global `include_directories`/`link_directories`/`add_definitions` (one include dir
  listed twice, a *source* dir in `link_directories`); definitions duplicated by
  `target_compile_definitions`; MKL detected by `string(FIND ... "mkl")` and three
  `mkl_*` libraries prepended to every test link although the libraries actually found
  are `libmkl_gf_lp64`; MPI C++ libraries appended twice; `C_STANDARD 11` hard-coded so
  `SUPPORT_C11=OFF` is a no-op; 16 test targets repeat the same link list;
  `LINKER_LANGUAGE Fortran` on 13 tests; ECRC `FindCBLAS/FindLAPACKE` placed before
  DPLASMA's own; no warning flags unless the sanitizer is on.
- **Fix** after G1, delete the `LD_LIB_*` lists and the MKL block, one `foreach` for the
  tests, `target_include_directories`. **Complexity** M. **Risk** moderate churn;
  verify CPU, CUDA and HIP configurations.

#### G9. Documentation claims and option drift — P1, confirmed

- `RELEASE.md` advertises macOS/Windows, a comprehensive test suite, API documentation
  and Docker support, none of which exist. `TESTS.md` documents `--band_dense_fp8`
  (not accepted) and omits 19 accepted options (`--M --K --MB --KB --P --Q --KP --KQ --rhs
  --pheno_file --band_dense_hp --adaptive_decision_runtime --fp8 --fp8_scaling
  --enable_stochastic_rounding --accumulation_fp32 --trmm_window --arena_pinned_memory
  --help`); its first instruction is the stale `test_run.sh`; `/home/qcao3/data` is the
  climate example's data path. `BUILD.md` documents `ctest` (G5), `HCORE_DIR`/`STARSH_DIR`
  (unused) and no minimum versions. `LICENSE:17` contains an unexpanded `$COPYRIGHT$`.
  No changelog.
- **Fix** rewrite `RELEASE.md` as release notes, regenerate the option table from
  `--help`, fix BUILD.md, add `CHANGELOG.md`. **Complexity** S/M.

#### G10. Scripts and Docker — P2, confirmed

- 58 scripts: hard-coded home directories of four people; five byte-identical `_dash`
  copies; obsolete binary names (`hicma_parsec/testing_dpotrf_tlr`) and single-letter
  options the parser rejects (`-N -t -e`, `--t --v`); three near-identical 200-350-line
  test scripts, one of which edits a tracked header with `sed`; `build_release.sh`
  defaults CUDA on so the documented "basic build" requires CUDA;
  `update_version_and_date.py` reads paths that no longer exist. Both Dockerfiles copy
  directories that do not exist and use a removed `-DGENE=1` macro.
- **Fix** keep the generic tools, move current site recipes to `scripts/machines/`,
  delete the byte-identical and syntactically dead ones, delete or rewrite the
  Dockerfiles. **Complexity** M. **Risk** removing a recipe somebody uses; mitigated by
  moving rather than deleting anything still valid.

#### G11. Untracked artifacts and `.gitignore` gaps — P1, confirmed

- `build*/`, `*.log`, `taq`, `parsec_master/`, `.claude/`, `PR_DESCRIPTION.md`,
  `tests/testing_st_round.cu` are not ignored; `.gitignore` lists the old binary name.
  `tests/testing_gemm_gpu_dp.cu` and `tests/testing_2gpu_gemm.cpp` are tracked but not
  built. `data/era_data_sample.mat` (7.3 MB) appears unreferenced (plausible).
- **Fix** extend `.gitignore`; decide on the three stray test sources. **Complexity** S.

#### G12. `--check` failures terminate inside the library with `exit(1)` — P3, confirmed

- `hicma_parsec_check.c:398,441,583`. Acceptable for the drivers; a library should
  return. Noted; the E2 fix keeps the `exit` and makes the drivers' return meaningful.

### 4F. Non-Cholesky algorithms (TRMM, GEMM/SYRK family, Hamming) and climate emulator

The mixed-precision TRMM (`trmm_mp_LLN.jdf`, `trmm_mp_wrapper.c`,
`hicma_parsec_core_trmm.c`, three drivers) was traced flow by flow and is consistent;
its only items are F11 (driver duplication) and F13 (a documented scheduling caveat).

#### F1. GEMM-family wrappers return an uninitialized taskpool pointer on unsupported transposes — P1, confirmed, API misuse only

- **Where** `src/kernels/gemm_wrapper.c:82,237`, `gemmex_wrapper.c:87,235`,
  `hamming_binary_wrapper.c:75`: `parsec_taskpool_t *gemm_tp;` is assigned only in the TN
  branch; the other branches call `dplasma_error` and `return gemm_tp`. The callers test
  `!= NULL` and would add garbage to the context. `trmm_mp_wrapper.c:103-108` does it right.
- **Fix** initialize to NULL; free `dev_index` on the error paths. **Complexity** S.

#### F2. `testing_igemm_summa` builds a 1×0 process grid and stores int32 into a FLOAT matrix — P1, confirmed, default build

- **Where** `tests/testing_igemm_summa.c:131-133` passes `nodes/NB` (always 0) as Q;
  `gemmex_TN.jdf:280-285` writes `CUDA_R_32I` into a `PARSEC_MATRIX_FLOAT` descriptor
  with no conversion; the GEMM iterates all of C but skips `m < n` in the body.
- **Impact** the executable is built by default and has never run correctly as committed.
- **Fix** `nodes/P`; fix the type or convert; or drop the driver with `gemmex_TN` (F11).
  **Complexity** S.

#### F3. Climate emulator preprocessor switches are spelled two ways; the dead spelling holds all file-read error checks — P1, confirmed, default

- **Where** `src/climate_emulator/climate_emulator.h:67-71` defines
  `CLIMATE_EMULATOR_{WRITE_TO_FILE,ENABLE_INVERSE,READ_FROM_FILE,DEBUG_INFO}`; the code
  tests the unprefixed `READ_FROM_FILE`, `ENABLE_INVERSE`, `WRITE_TO_FILE` and
  `DEBUG_INFO_CLIMATE_EMULATOR` (48 places), which are never defined. Every `fscanf`
  status check in the five CSV readers sits inside `#if DEBUG_INFO_CLIMATE_EMULATOR`
  (`climate_emulator_read_csv_double.jdf:41-48` and the four siblings), so a missing or
  wrongly sized file is reported by a `printf` whose caller ignores it, and SYRK/Cholesky
  run on uninitialized memory. Flipping the `READ_FROM_FILE` switch would make
  `climate_emulator_fini` free ten never-initialized descriptors; the debug paths call a
  `static` function from another translation unit and host code on device pointers.
- **Fix** use the defined names; make read failures fatal regardless of debug level;
  delete the rotted debug blocks. **Complexity** S. **Risk** none.

#### F4. Genomics drivers and GPU-only GEMM JDFs are built unconditionally but cannot run as built — P1, confirmed, default build

- **Where** `gemm_TN.jdf`, `gemmex_TN.jdf`, `gemm_TN_summa.jdf`, `gemmex_TN_summa.jdf`,
  `isyrk_LT.jdf`, `hamming_binary.jdf` have GPU bodies only (CPU bodies commented out);
  `testing_syrk`, `testing_igemm_summa`, `testing_Kmatrix`, `testing_KRR`,
  `testing_KRR_file` are built in CPU-only configurations (`tests/CMakeLists.txt:54-122`)
  and abort at the first GEMM. With `GENOMICS 0` (the committed value) `dcAd` is
  `PARSEC_MATRIX_DOUBLE` while `hicma_parsec_kernel_matrix` writes FP32 tiles through a
  `FULL_SP` arena into it, so GPU builds ship binaries that compute on mistyped storage.
  `TESTS.md` says genomics needs `GENOMICS` enabled; `scripts/test_CI.sh:109-215` does
  that by editing the header with `sed` and rebuilding (commented out).
- **Root cause** application modes are header constants rather than build options, and
  the tests are not gated on the mode they need. Merging the GENOMICS macros is backlog.
- **Fix (within scope)** gate the genomics drivers and the GPU-only wrappers/JDFs on
  `HICMA_PARSEC_HAVE_CUDA OR HICMA_PARSEC_HAVE_HIP`, and additionally on a CMake option
  that defines `GENOMICS=1` for the drivers that require it, so the default build does
  not ship binaries that cannot run. **Complexity** S/M. **Risk** low; nothing in the
  Cholesky path references these.

#### F5. Climate emulator: `FOR_CLIMATE_EMULATOR` gates nothing useful; CI test depends on a private path and mismatched data — P1, confirmed

- **Where** the flag only re-defines `GENERATE_RANDOM_DATA` to its existing value and
  guards one early return in `convert_s2d.jdf:194`; the sources and the test are always
  built (CPU-only compiles to an empty GPU TU, fine). `climate_emulator_reading_data`
  (`climate_emulator.c:200-205`) exits without `--mesh_file`; `scripts/test_CI.sh:273`
  hard-codes `/home/qcao3/data`, so the script fails on any other machine (and passed on
  the author's machine while parsing 720-sized files into `--latitude 120` matrices,
  because every file name uses the constant `CLIMATE_EMULATOR_L_SIZE` = 720, not
  `gb->L`, `climate_emulator.c:218-306`). `climate_emulator_init` (`:58-67`) overrides
  `band_size_dense*`, `kind_of_cholesky`, `band_size_dist`, `auto_band` *after*
  `hicma_parsec_init` has allocated descriptors and decisions with the old values.
- **Fix** skip the CI step when the data directory (from an environment variable) is
  absent; build file names from `L`; apply the overrides before `hicma_parsec_init`
  (or re-run `hicma_parsec_decision_init`); drop `FOR_CLIMATE_EMULATOR` or make it a
  build option. **Complexity** S/M.

#### F6. Leaks and asymmetric destruction in wrappers — P2, confirmed, default

- Memory-pool structs finalized but never freed: `gemm_wrapper.c:159/548,281/570`,
  `hamming_binary_wrapper.c:123-128/361-362`, `reorder_gemm.jdf:260/282`, three climate
  JDFs (one of which creates four pools it never uses). The same pattern in nine of the
  ten Cholesky wrappers (A9). `trmm_mp_wrapper.c:225-227` is the correct model.
- `dev_index` malloc'd even when `nb == 0` and freed only when `nb > 0`
  (`trmm_mp_wrapper.c:95,206-215`, gemm/gemmex/isyrk/hamming wrappers, both climate JDFs).
- `float *vec` leaked in `hicma_parsec_kernel_matrix` (`hicma_parsec.c:938,1192`);
  per-device `parsec_device_cuda_module_t` placeholders leaked in
  `climate_emulator_workspace_memory_allocate` (`climate_emulator_gpu.c:74`).
- **Fix** free them; free `dev_index` unconditionally. **Complexity** S.

#### F7. `writeA`/`writeB` and the file-read paths are single-rank and mix allocators — P2, confirmed, non-default

- `hicma_parsec.c:741-810` ignores its matrix argument, writes the local block as if it
  were the global matrix (wrong on more than one rank); `writeB` does not check `fopen`;
  `parsec_data_free` on `malloc`'d buffers (`:792,870,1133,1401,1547`, harmless with this
  PaRSEC); `fflush(stdin)` (UB); "malloc failed" handlers call `MPI_Finalize` then continue;
  `hicma_parsec_kernel_matrix_file` is identical to `hicma_parsec_kernel_matrix` in the
  default configuration (~140 duplicated lines).
- **Fix** when touched: guard multi-rank writes, check `fopen`, use `free`, merge the
  two kernel-matrix functions. **Complexity** S/M.

#### F8. Hamming GPU kernel prototypes disagree with their definitions — P2, confirmed

- `hicma_parsec_gpu.h:1014,1030` declare `int *` parameters; the definitions in
  `hicma_parsec.cu:2783,2849` and `hicma_parsec_hip.cu:288,354` take `int8_t *`.
  Works today, fails under LTO type checking. **Fix** header to `int8_t *`.

#### F9. Public headers declare 13 functions that do not exist — P2, confirmed

- `perform_gemm` (`hicma_parsec.h:268`), `parsec_gemmmp_TN`, `parsec_dist_allocate_gemmmp_TN`,
  `parsec_gemmmp_TN_generator` (`hicma_parsec_internal.h`), `decision_datatype_tile_gemmmp_TN`,
  `decision_datatype_tile_convert`, `decision_datatype_out`, `decision_size_out`,
  `hicma_parsec_decisions_gemm_gpu_init` (`hicma_parsec_decision.h`),
  `hicma_parsec_core_gemmmp_TN_denseC_denseA_denseB_gpu` (`hicma_parsec_gpu.h:1116`),
  `climate_emulator_print_complex`, `hicma_parsec_matrix_allocate_dense_tile`
  (`climate_emulator.h`). `hicma_kernal_matrix` (typo) is defined and used only under
  GENOMICS. Conversely every gemm/gemmex/hamming/isyrk/climate `_New/_Destruct` is
  exported without a prototype.
- **Fix** delete the dead declarations; add prototypes for the taskpools meant to be
  public (TRMM is the model) and make the rest `static`. **Complexity** S.

#### F10. Inconsistent naming and signatures in the GEMM-shaped wrappers — P2, confirmed

- `gemm_wrapper.c` computes C = αAᵀB from `gemm_TN.jdf` but exports it as
  `hicma_parsec_syrk*`; `isyrk_wrapper.c` uses the `dplasma_` prefix and a `Destruct`
  with two unused parameters; `reorder_gemm`, `genotype_generator`, `dist_allocate*`
  use `parsec_`; scalars are `float` in two wrappers and `double` in `gemmex`, whose JDF
  casts them to `int`; two constructors mutate `params->lookahead` as a side effect.
- **Fix** when F4/F11 settle which wrappers survive: one prefix, names that say what is
  computed, one scalar type. **Complexity** S/M; API change for the genomics drivers only.

#### F11. Duplication in this area (quantified) — P1, confirmed

- `gemm_TN.jdf` (486 lines) vs `gemmex_TN.jdf` (312): same DAG; `gemmex` drops the FP32
  finish and the triangular restriction. `gemm_TN_summa.jdf` (329) vs `gemmex_TN_summa.jdf`
  (344): identical modulo names. `gemm_TN_summa` is reachable only through a commented-out
  call; `gemmex_TN_summa` only with `GEMM_SUMMA=1`.
- `isyrk_LT.jdf` (340, built) vs `isyrkmp_LT.jdf` (364, not built); `dplasma_isyrk` has
  no live caller. `gemm_TN_batched.jdf` (620) not built, no references. `gemm_f16_wrapper.c`
  (76) not built, no callers.
- Five climate CSV-reader JDFs (1255 lines) differ only in the element parser and a
  batch column; `climate_emulator_matrix_norm_get.jdf` duplicates
  `kernels/matrix_norm_get.jdf` and hides a diagonal shift inside a norm computation;
  `climate_emulator_print_double.jdf` has no caller.
- The 25-line device-discovery loop is copied six times although
  `hicma_parsec_find_cuda_devices()` exists; `check_saxmb2` is copied into four drivers;
  `parsec_print_tile*` into three JDFs; `always_here/never_here` into five JDFs, used by
  none; the three TRMM drivers share ~200 lines (`testing_trmm_performance` is
  `testing_trmm`'s `!check` branch).
- **Fix for this release** delete what has no caller (`gemm_TN_batched.jdf`,
  `gemm_f16_wrapper.c`, `isyrkmp_LT.jdf`, `copy2double.jdf`, `gemm_TN_summa.jdf`,
  `climate_emulator_print_double.jdf`, `hamming.jl`), use the shared device discovery,
  hoist the params-free block into one function. Collapsing the CSV readers and merging
  `gemmex_TN` into `gemm_TN` are worthwhile but touch application code that cannot be
  validated here (no climate data, no genomics data); list them as follow-ups.

#### F12. Dead code and debug output — P2, confirmed

- Unconditional prints in production paths: `gemm_wrapper.c:162`,
  `gemmex_wrapper.c:174,286`, `isyrk_wrapper.c:171`, `hicma_parsec.c:786`, two climate
  readers per batch.
- `#if 0` / `#if 1 ... #else` alternates in `hicma_parsec.c`, `isyrk_LT.jdf`,
  `hamming_binary.jdf`, `gemm_TN.jdf` (plus unreachable `printf` after `return`),
  `trmm_mp_LLN.jdf:287-432` (raw cuBLAS/CORE alternates), `gemmex_TN.jdf`.
- Unused statics `print_more` ×4; `evaluate_gpu_*` hooks that always return DONE (what a
  NULL evaluate already means) in the four GEMM-shaped wrappers.
- Orphan tests: `tests/testing_2gpu_gemm.cpp`, `tests/testing_gemm_gpu_dp.cu` (tracked,
  not built), `tests/testing_st_round.cu` (untracked).

#### F13. TRMM: a converted B copy made on the GPU is not pushed to the host for a *local* GEMM that PaRSEC moves to a CPU core — P3, mechanism confirmed, non-default trigger

- `trmm_mp_LLN.jdf:168-181` pushes `Bop` out only for remote readers; PaRSEC adds an
  automatic pushout only for successors without a GPU body. With
  `--mca device_load_balance_allow_cpu 1` (default 0) a local `dgemm` scheduled on a core
  would read a stale host copy. The in-place/copy policy is a recorded design decision;
  document the restriction in `trmm_mp_LLN_tuning.md` (done in this release) or set the
  pushout bit when that MCA parameter is non-zero.

#### F14. Integer-overflow candidates in size arithmetic — P3, plausible

- `tests/testing_hamming.c:35,69` (`int*int` before the `size_t` cast; overflows at
  N = nsnp ≥ 46341), `hicma_parsec.c:1375-1521` (MPIIO only), `gemm_wrapper.c:160,282`
  (fine for NB ≤ 46340). **Fix** cast the first operand.

### 4B. Dense mixed-precision GPU Cholesky (kinds 5, 8, 9, 10) and the GPU warmup

The task graphs of the four kinds were traced pair by pair (sender and receiver
conditions, lookahead and left-looking CTLs, runtime rerouting in kind 10) and are
consistent; receive arenas are at least as large as the bytes sent in every precision
combination; every fix listed in `PR_DESCRIPTION.md` is present in all four kinds. A
GPU body returning `PARSEC_HOOK_RETURN_NEXT` is re-queued on the device
(`device_gpu.c`, `parsec_gpu_stream_push_pending`), so the `evaluate` hooks really are
the only protection against the documented hang.

#### B1. Taskpool `Destruct` leaks the memory-pool structs (kinds 5, 8, 9; also kinds 1, 2, 3, 6, 7 and the GEMM/Hamming wrappers) — P2, confirmed, default

- **Where** `potrf_L_dense_mp_gpu_wrapper.c:821-831`, `_fp8_wrapper.c:821-832`,
  `_fp8_sp_wrapper.c:824-835` (`parsec_private_memory_fini` on 8-9 pools, no `free`);
  kind 10 (`_fp8_adaptive_wrapper.c:875-895`) frees them, so the four `Destruct`s are
  not equivalent. `potrf_L_warmup.jdf:198,217-222` never frees `_g_cuda_device_index`.
  Every `New()` mallocs `dev_index` even when no GPU exists and frees it only when
  `nb > 0`.
- **Fix** one shared pools-fini helper used by every wrapper; free the device index
  unconditionally. **Complexity** S (folded into B2).

#### B2. The four wrappers are 87-98% identical code — P1, confirmed (measured)

- **Measurement** (names normalized, comments stripped): kind 5 vs 8: 430/440 code
  lines identical; 8 vs 9: 427/452; 8 vs 10: 427/490. The six TLR/band wrappers share
  the same ~350-line skeleton (section 4C). Differences: kinds 8/9/10 add an FP8 pool and
  arena, kind 10 adds the NORM arena and `hicma_trim_remote_operand` in two `evaluate`
  hooks, kind 9 reads `A->nt` instead of `params->NT`.
- **Shareable without touching a JDF body**: the eight `wrap_*`/`wrap_*_complete`
  timing hooks (identical except for the cast to the generated task struct; the locals
  are read only as `locals.{k,m,n}.value`), the `evaluate_gpu_*` hooks, task-class
  lookup by name and incarnation selection (which also fixes the hard-coded chore indices
  of the sparse wrappers, C5), hook installation (three near-identical 20-line blocks per
  wrapper), memory-pool init/fini with the HCORE workspace formula, GPU device discovery
  and teardown, and the `New → add → start → wait → Destruct` driver.
- **Fix** a `potrf_common.{c,h}` with those helpers; each wrapper keeps its arena table
  and its kind-specific bits. **Benefit** one place for every future fix (B1, D1, C5, C6).
  **Complexity** M. **Risk** low, mechanical; validated by the full matrix.
- Merging the JDFs themselves (70-85% identical) is backlog and not proposed.

#### B3. Kinds 8, 9, 10 ignore `--enable_stochastic_rounding` in their own conversions — P2, confirmed, non-default flag

- **Where** `potrf_L_dense_mp_gpu.jdf` uses SR in its 14 conversion sites;
  `_fp8.jdf`, `_fp8_sp.jdf`, `_fp8_adaptive.jdf` have none, while the shared cores do
  honor it. `params_check` refuses SR only together with FP8 (`hicma_parsec_internal.c:1147`).
  The runtime kernels (K7) likewise never call the `_round_` variants.
- **Impact** `--enable_stochastic_rounding 1` on kinds 8/9/10 silently mixes SR and
  round-to-nearest.
- **Fix** refuse the flag for kinds 8, 9 and 10 in `params_check` with one message
  (porting the 14 SR branches is a follow-up). **Complexity** S.

#### B4. `TERMINATE` can fire while two leaf tasks are still queued — P2, plausible, default

- **Where** `potrf_dpotrf(k)`: `CTL -> (k == mt-1) ? TERMINATE`; `TERMINATE` calls
  `taskpool_set_nb_tasks(1)`. `potrf_dpotrf_send(mt-1)` and `potrf_dtrsm_send(mt-1, mt-2)`
  are released by or alongside `potrf_dpotrf(mt-1)`, have no successors, and are not
  predecessors of `TERMINATE`. Kind 2's `WRITE_Cr` sink has the same property (C2).
- **Mechanism** under `PARSEC_UNDETERMINED_NB_TASKS`, the first completion after
  `set_nb_tasks(1)` terminates the taskpool; a leaf still queued when every thread
  observes termination is never executed and its task object outlives the `Destruct`.
  Benign in practice (`release_task` skips counting at `nb_tasks == 0`, the context
  barrier waits for in-flight GPU work, `dpotrf_send` has maximum priority); never
  observed.
- **Fix** restrict `potrf_dpotrf_send` to `k <= mt-2` (nothing consumes `k = mt-1`) and
  add CTL edges from the leaves into `TERMINATE`; same recipe for `WRITE_Cr`. Touches
  the task graph of nine JDFs, so it is listed as a follow-up with its validation recipe
  (`PARSEC_DEBUG_PARANOID` build, `--nruns 20`) rather than done blind in this release.

#### B5. CPU TRSM core has no `DENSE_FP8` case — P2, latent, confirmed

- `hicma_parsec_core.c:1016` tests `DENSE_SP || DENSE_HP`; an FP8 tile falls into the
  low-rank-SP branch with `Crank = 0` (a no-op solve). Unreachable today because FP8
  requires GPUs and dense TRSMs always run there. **Fix** add `DENSE_FP8` to the case.

#### B6. CPU-only runs with `--datatype_convert 2` forward an unfilled NEW buffer — P2, plausible, non-default

- The CPU body of `potrf_dtrsm_send` in all four kinds prints "You can not reach here"
  and forwards the unwritten `C` buffer when `convert_in_trsm` is true; `params_check`
  does not gate `datatype_convert == 2` on `gpus > 0`. **Fix** force `datatype_convert = 0`
  on CPU-only runs with a message. **Complexity** S.

#### B7. GPU statistics counters indexed by the CUDA ordinal — P2, plausible, non-default masks

- `hicma_parsec_core.c:27-34` uses `cuda_index * PARSEC_GPU_MAX_STREAMS + j` with
  `counter_stride = cores + gpus * PARSEC_GPU_MAX_STREAMS`; a `--mca device_cuda_mask`
  that selects ordinals ≥ `gpus` writes out of bounds (assert compiled out in Release),
  and `lookup_gpu_workspace_j` returns `max_exec_streams` when the stream is not found.
- **Fix** index by the device's position among enabled GPUs; clamp. **Complexity** S.

#### B8. Dead and obsolete code in the JDFs and wrappers — P3, confirmed

- `#if PREDICTION` and `#if CHOLESKY_CPU_ONLY` blocks (kind 9 only); `#if ENABLE_PROFILING`
  `cudaProfilerStop` in kinds 8/9/10; an unused `gettimeofday` per GPU GEMM on the manager
  thread (kinds 5/8/9); `#if !GPU_BUFFER_ONCE` per-taskpool workspace paths in all four;
  the `size` local of `potrf_dgemm` (three bodies × four kinds) that feeds a
  `count_remote` on a flow whose successors are never remote; a stale comment about
  `runtime_comm_short_limit` in kind 10; kind 5's `FULL_FP8` arena of `mb*mb` bytes
  (kinds 8/9/10 use `hicma_fp8_tile_bytes`; kind 5 cannot select FP8, so inert); kind 9's
  dead `potrf_dpotrf_send` conversion (every decision is capped at SP).

#### B9. Timing instrumentation approximations — P3, confirmed

- `trsm_time_temp`/`syrk_time_temp` are single shared slots written by every
  `TRSM(k+1, k)`; in GPU runs `wrap_*_complete` runs for CPU-chore tasks too and reads a
  stale start. Output-only; the per-task table in `hicma_kernel_time.c` is the right
  mechanism if this is ever needed.

### 4C. Tile-low-rank and CPU band Cholesky (kinds 1, 2, 3, 4, 6, 7)

The six variants are structurally sound: flow conditions agree on both ends for every
`band_size_dense`/`lookahead`/`left_looking` combination traced; the sparse analysis
arrays are consistent with the list-based execution spaces; the CPU TRSM bodies set
`size` before every remote send; kinds 1 and 2 install `evaluate` hooks that agree with
the bodies' NEXT returns.

#### C1. `2*maxrank` passed as HCORE's storage maxrank bypasses the rank cap (kinds 2, 6, 7) — P1, confirmed

- **Where** `potrf_L_dense_tlr_dp.jdf:922`, `potrf_L_sparse_tlr_dp_general.jdf:1310`,
  `potrf_L_sparse_tlr_dp_balance.jdf:919` (`HCORE_dgemm_ormqr(..., fixedrk, 2*maxrank,
  compmaxrank, ...)`), versus `maxrank` in the same branch of the MP kind
  (`potrf_L_dense_tlr_mp.jdf:867,1001`). `storage_maxrank` is HCORE's only rank cap
  (`hcore_dgemm.c:1648`).
- **Impact** a C tile can leave the "A low-rank, B dense" GEMM with rank in
  `(maxrank, 2*maxrank]`; the CPU pools (`mb*compmaxrank`, `compmaxrank²`), the GPU scratch
  and the `send_full_tile` buffers all assume `rank <= maxrank`. Reachable with
  `--kind_of_cholesky 2 --band_dense >= 2` (dead in kinds 6/7 where the band is 1).
- **Fix** pass `maxrank` at the three sites. **Complexity** S. **Risk** none; runs that
  overflow today abort with HCORE's message, as kind 1 does.

#### C2. `WRITE_Cr` is a sink task outside the TERMINATE-based termination (kind 2) — P2, plausible

- `potrf_L_dense_tlr_dp.jdf:166-184,192-211`: nobody depends on `WRITE_Cr(m,k)`; same
  mechanism as B4. The MP JDF avoids the task by writing back through TRSM's `Cr` flow.
  Follow-up with B4.

#### C3. Sparse-analysis memory bookkeeping (kind 7, partly 6) — P1 (a) / P2 (b)

- (a) `hicma_parsec_sparse_analysis.c:361-390` allocates `gemm_local[...]` when the tile
  is owned by `A` **or** by `Dist`; `_free` (`:538-543`) frees only the `A` owner: leak on
  every multi-rank `--sparse 2` run. **Fix** free with the allocation predicate. S.
- (b) `potrf_L_sparse_tlr_dp_balance.jdf:179-202,884-906,984-989,1062-1084` create
  `parsec_data_t`/copies per rank growth with no owner. **Fix** keep an owned pointer
  table in `analysis` released in `_free`. M; follow-up.

#### C4. Memory pools and `dev_index` never freed (all six wrappers) — P2, confirmed, default

- Same as B1 (`parsec_private_memory_fini` does not free the struct; DPLASMA and
  `trmm_mp_wrapper.c` do). Folded into B2.

#### C5. Sparse wrappers hard-code chore indices — P2, confirmed, latent

- `potrf_L_sparse_tlr_dp_general_wrapper.c:427-468` and the balance twin use
  `incarnations[0]` (RECURSIVE) and `[1]` (CPU) literally; without
  `PARSEC_HAVE_DEV_RECURSIVE_SUPPORT` the second is the terminator (NULL hook).
  `potrf_L_sparse_tlr_dp_balance.jdf:14` `#undef`s `PARSEC_HAVE_DEV_CUDA_SUPPORT`.
- **Fix** the shared incarnation selection of B2 (walk `incarnations[].type`).

#### C6. HIP builds: kinds 1-4 have CUDA-only bodies but the wrappers treat chore 0 as the GPU chore — P2, plausible, HIP

- The JDFs guard their GPU bodies with `PARSEC_HAVE_DEV_CUDA_SUPPORT` only; the wrappers
  install GPU hooks under `CUDA || HIP` when devices exist, so a HIP run installs the
  timing hooks and `evaluate_gpu_*` on the RECURSIVE chore and silently runs on the CPU
  with GPU statistics. Kind 5 and the FP8 kinds have HIP bodies.
- **Fix** match the wrapper guards to the JDF bodies; refuse `--gpus > 0` for kinds 1-4
  under HIP with a message. **Complexity** S.

#### C7. `maxrank <= compmaxrank` and `<= genmaxrank` are not enforced — P2, confirmed, non-default flags

- Pools, GPU scratch and UV arenas are sized by `compmaxrank`/`genmaxrank`, HCORE caps
  ranks at `maxrank`; `--maxrank 300 --compmaxrank 200` is accepted and overflows. The
  wrappers also only *warn* when `maxrank > NB/2` although the per-stream GPU buffers
  (`mb*nb` doubles) are written with `mb*2*Arank` (K4).
- **Fix** in `params_check`: `compmaxrank = max(compmaxrank, maxrank)`, reject
  `genmaxrank < maxrank` and `maxrank > NB/2`. **Complexity** S.

#### C8. CI coverage — P2, confirmed

- Kinds 2, 3, 4 and 7 are never run; kind 6 only with the mesh file; nothing on more
  than one rank. The validation matrix in `docs/release-readiness.md` and the CTest set
  (G5) cover them.

#### C9. Dead and unreachable code — P2/P3, confirmed

- `#if 0` SVD kernels in `hicma_parsec_core.c:2819-2983` still declared with a different
  signature in `hicma_parsec_core.h:541-559`; in the sparse JDFs four of five GEMM
  branches, the dense TRSM branch and the dense SYRK path are unreachable with the
  enforced `band_size_dense == 1` (~350 lines per JDF); `diagonal_correction_calculation.jdf`
  has no caller; `copy2double.jdf`, `isyrkmp_LT.jdf`, `gemm_TN_batched.jdf` not built and
  unreferenced; `rank_print.jdf` reachable only under `PRINT_RANK > 1`.

#### C10. Statistics-only inconsistencies — P3

- The dense-band GEMM branch of kinds 2/6/7 does not count operations; kinds 3/4 install
  no timing hooks so their kernel times print as 0; `evaluate_gpu_trsm/gemm` use the band
  distance while the bodies use `IS_DENSE` (equal today).

#### C11. Kind 3 relies on a compiled-out `assert` for CPU-only runs — P2, plausible

- `potrf_L_dense_mp_band_wrapper.c:140-141`: the CPU bodies have no half-precision path;
  an HP band on a CPU-only run is prevented only by an `assert`. **Fix** clamp
  `band_size_dense_sp = NT` when `gpus == 0` in `params_check` (it already does this
  for `band_size_dense_hp`) or return an error. S.

### 4D. Computational kernels and GPU helper code

The runtime-decision GPU GEMM checks every cuBLAS/cuBLASLt status, the FP8 scaling is
stream-ordered, the stochastic-rounding hash is bit-identical on CPU, CUDA and HIP, and
`hicma_parsec_core_trmm.c` is a clean model of a precision-generic kernel.

#### D1. Runtime GEMM: FP64 promotion does not restore `span` unless `adaptive_memory || adaptive_decision` — P1, confirmed mechanism, data-dependent trigger, kind 10 on GPU

- **Where** `hicma_parsec_core.c:4044-4064` (the k == 0 downcast sets
  `original->span` to the FP32 size and the decision to SP); `:4152-4188` (promotion:
  `hicma_reallocate_tile_on_gpu`, which also updates `span` and `flow_span`, is called
  only `if( adaptive_memory || adaptive_decision )`; otherwise only `memcpy_double_GPU`).
  PaRSEC sizes D2H/D2D copies by `min(src->span, dst->span)` and new device buffers by
  `flow_span` (`device_gpu.c:1453,1935,2022,5186`).
- **Reproduced partially** `--adaptive_decision_runtime 1 --adaptive_decision 0
  --verbose 101` at N=8192: 46 tiles were downcast at k == 0 and 13 of them were later
  promoted to FP64 compute ("would change from 2 to 1"); the result was still correct
  because the end-of-chain decision re-downcast them. A promoted tile that *stays* FP64,
  or one evicted and re-staged while FP32-sized, has its host write-back truncated to
  half the tile or its new device buffer under-allocated.
- **Fix** call `hicma_reallocate_tile_on_gpu` on every promotion (its first branch is a
  no-op when `span` already covers the target); keep the `adaptive_memory` gate only on
  the host-side reallocation. **Complexity** S. **Risk** low; one extra zone allocation
  per promotion in this mode. **Validate** kind-10 matrix unchanged; a low
  `device_cuda_memory_use` run to force evictions.

#### D2. Private LRU evictor lacks two of PaRSEC's guards — P1, plausible, kind 10 under memory pressure

- **Where** `hicma_parsec_core.c:223-317` re-implements `parsec_device_data_reserve_space`
  but never checks `data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER` nor the
  copy's coherency/version, and calls `zone_free`/`parsec_data_copy_detach` on fork
  internals that changed recently (#814).
- **Fix** push back copies that are under transfer or whose version differs from the
  host copy; longer term export a reservation helper from the PaRSEC fork. **Complexity**
  S for the two guards. **Risk** medium (memory management); validate with a low
  `device_cuda_memory_use` kind-10 run under `compute-sanitizer`.

#### D3. `--tensor_gemm 0x80` silently skips the dense FP16 GPU GEMM in kinds 5, 8, 9 — P1, confirmed, non-default flag

- **Where** `hicma_parsec_core.c:3691-3845`: the `switch` has cases for 0x8, 0x10, 0x20,
  0x40 and no `default`; `hicma_parsec_decision.c:91-93` ORs 0x80 into
  `decisions_gemm_gpu` (the TRMM honors 0x80, commit 19919eb). Every GEMM into a
  `DENSE_HP` tile returns without touching C and without a message.
- **Fix** a `default:` that aborts with the mask value, and either implement the
  C16/OP32 case or reject 0x80 for the Cholesky kinds in `params_check`. **Complexity** S.

#### D4. `maxrank > NB/2` overflows the per-stream GPU workspaces — P1, plausible, non-default parameters

- `hicma_parsec_gpu.c:808-816` sizes `gpu_buffer_A/B/C` as `mb*nb` doubles; the low-rank
  GPU paths write `mb*2*Arank` (`hicma_parsec_core.c:4504-4885`), and the CPU HP paths do
  the same into `p_work_full_sp`. The wrappers only warn. **Fix** with C7 (reject).

#### D5. CPU low-rank TRSM solves with the C tile's row count instead of the diagonal block size — P1, confirmed by dimension analysis, needs `N % NB != 0`

- `hicma_parsec_core.c:1064-1067,1095-1098`: `V <- T^{-1} V` is a solve of size `nb(k)`
  (rows of V = columns of the tile), but `tempmm` (rows of tile m) is passed; for the
  partial last tile row only the leading `tempmm × tempmm` block of T is applied. See H1:
  partial tiles are unsupported end to end and are now rejected, which makes this
  unreachable; the dimension is corrected anyway.

#### D6. CPU runtime GEMM stores and decides differently from the GPU runtime GEMM — P2, confirmed, kind 10

- `hicma_parsec_core.c:1969-2004` (CPU FP16 path promotes C to FP64 and keeps it) vs
  `:4286-4298` (GPU keeps FP32); no end-of-chain downcast on the CPU; `#if ACC_DP` /
  `#define call_hcore_dsyrk 1` carry permanently dead `#else` branches.
- **Impact** CPU-only and GPU runs of kind 10 use different storage policies; a CPU body
  in a GPU run with `--adaptive_memory 1` would write FP64 into an FP32-sized host buffer
  (prevented today by the default device selection and `hicma_parsec_internal.c:1172`).
- **Fix** align the CPU FP16 path with the GPU one; delete the dead branches. M;
  follow-up (documented difference for 0.2.0).

#### D7. `--enable_stochastic_rounding` ignored in runtime-decision mode — P2, confirmed

- Runtime kernels (`hicma_parsec_core.c:1805-1867,4195-4265`) always round to nearest.
  **Fix** with B3: refuse the flag for kinds 8-10.

#### D8. Non-runtime GPU kernels ignore cuBLAS/cuSOLVER return codes — P2, confirmed, every CUDA/HIP build

- `hicma_parsec_core.c:3156` (`assert` only, compiled out in Release),
  `:3227-3402,3636-3831,4523-4967` (`status =` assigned, never read); HIP `rocsolver_*potrf`
  return discarded; the runtime GEMM's unsupported-decision `else` (`:4369`) prints and
  returns normally; the FP8 setup in `hicma_parsec_gpu.c:653-733` is unchecked.
- **Impact** a failed POTRF/GEMM leaves the tile untouched and the run reports success,
  the failure mode the PR fixed for FP8.
- **Fix** one check macro that prints the call, the tile and the status and aborts,
  applied after every call; the TRMM file is the model. **Complexity** M (mechanical).

#### D9. cuSOLVER workspace for `Spotrf` sized from the `Dpotrf` query; `dev_info` offset depends on precision — P2, plausible, kinds 4/9

- `hicma_parsec_gpu.c:758-773` queries `cusolverDnDpotrf_bufferSize` only;
  `hicma_parsec_core.c:3130-3147` places `dev_info` at `buffer_size*sizeof(float)` or
  `*sizeof(double)` depending on the path. **Fix** query both, allocate the max, fixed
  `dev_info` offset. S.

#### D10. Host synchronizations on the GPU manager thread in runtime mode — P2 (performance), confirmed, kind 10

- One synchronous host-pointer `cublasDnrm2` per tile at the end of its GEMM chain
  (`hicma_parsec_core.c:4400-4404`), a `cudaStreamSynchronize(norm_meta_stream)` per GEMM
  and SYRK (`:41-62`), and a stream drain per FP64 promotion (`:395,410`). All correct;
  they serialize the manager thread with the device. **Fix options** have TRSM write the
  `{norm, precision}` record to the host copy of the NORM flow asynchronously; move the
  end-of-chain decision into TRSM; free-after-event list for promotions. Follow-up with
  measurements (N=32768, one GPU).

#### D11. GPU counter slot computed from the physical CUDA ordinal — P2, plausible

- Same as B7.

#### D12. CUDA and HIP kernel files are hand-maintained copies — P1, confirmed

- `scripts/cuda2hip.sh` performs four `sed` substitutions and is not in the build;
  `hicma_parsec_hip.cu` (1739 lines) vs `hicma_parsec.cu` (3011) have drifted (section
  4D table in the reviewer notes: FP8 and integer converters CUDA-only, `double2half`
  rounds twice on HIP, debug kernels missing on HIP that a `#if 0` block would call).
  **Fix** compile one source with both compilers (the kernels are plain C++ except the
  FP8/BF16 intrinsics, already behind `HAVE_FP8`), or generate the HIP file at build time
  and delete the checked-in copy. **Complexity** M. Cannot be validated on hardware here
  (HIP compile-only), so it is a documented follow-up; the stochastic-rounding pair is
  the natural first step.

#### D13. Duplication inside the CPU kernels — P2, confirmed

- Dense GEMM vs runtime GEMM (~80% identical, `hicma_parsec_core.c:1473-1698` vs
  `:1732-2162`); `lrA_denseB` and `lrA_lrB` each carry DP/SP/HP copies of the same BLAS
  calls; `syrk_cpu` vs `syrk_runtime_decision_cpu`; thirteen copies of one triple loop in
  `hicma_parsec_core_matrix_norm_get`; twenty near-identical 2-D conversion kernels in
  the `.cu`. **Fix** follow `hicma_parsec_core_trmm.c`'s structure. M/L; follow-up.

#### D14. Dead code and never-defined declarations — P2, confirmed

- `hicma_parsec_core.h:541-560` declares two SVD GEMMs whose only definitions are inside
  `#if 0` with different signatures; `parsec_print_tilenew`, `hicma_parsec_convert_2fp8_bit`
  unused; `EPS_HP_GPU = 8.88e-01` consumed only under a disabled macro; six unused
  integer-conversion CUDA kernels, four of which swap their leading dimensions;
  `my_cublasGemmEx`, `copy_float_to_device_double`, `matrix_print_*_GPU` unused in both
  CUDA and HIP; `#if 0` fjcblas and FP8 blocks; the unreachable `cudaMemcpy` fallback in
  `hicma_gpu_read_norm_meta`.

#### D15. Small items — P3

- `hicma_parsec_core_matrix_norm_get` "half"/"fp8" branches square raw bit patterns;
  `hicma_parsec_core.c` includes one generated JDF header to reach `data._f_C.data_out`
  (pass the copy from the body instead); `hicma_parsec.cu` includes legacy `<cublas.h>`;
  `cublasSetStream` on every call although handles are per stream; `handle_cublas_tensor`
  is identical to `handle_cublas`; triple `cudaHostRegister*` definitions and sed
  leftovers in `hicma_parsec_hip_cuda.h`.

### 4H. Cross-cutting

#### H1. Partial last tile (`N % NB != 0`) is unsupported end to end but not rejected — P0, confirmed (reproduced), default

- **Reproduced** on the baseline CPU build with `--check 1`:
  `--N 4000 --NB 512` dense FP64 (kind 5): `||LLᵀ−A||/||A|| = 3.2e+03`, exit code 0;
  `--N 2900 --NB 270 --kind_of_problem 2 --maxrank 130` (kind 5 with low-rank off-band):
  5.0e+02; the same with `--kind_of_cholesky 1 --band_dense 1` aborts inside HCORE
  (`LAPACKE_dgesvd` parameter 9: `ld_U = 200 < T_nrows = 206`).
- **Where** `matrix_compress.jdf:57` and `band_regenerate.jdf:63` generate band tiles with
  `tempnn = tempmm` (square), so the last tile row's off-diagonal dense tiles are only
  partially generated; `matrix_uncompress.jdf:129` copies them the same way; HCORE's
  low-rank storage assumes square tiles (`hcore_dsyrk.c:57` "ASSUMPTION FIXME",
  `potrf_L_dense_tlr_dp.jdf:871`); D5. The dense kernels themselves handle
  `tempmm × nb` tiles correctly. `TESTS.md` already states "`--NB`: tile size,
  `N % NB == 0`"; only kind 10 enforces it (`hicma_parsec_internal.c:1936`).
- **Root cause** square-tile assumptions in generation, check and HCORE, never guarded.
- **Fix** reject `N % NB != 0` for the Cholesky path (matrix generation and `potrf`) with
  a clear message; the TRMM path supports partial tiles and is unaffected. Supporting
  partial tiles properly (generation, uncompress, HCORE leading dimensions) is a
  follow-up. **Complexity** S. **Risk** none: inputs that produced wrong results now
  fail early.

## 5. Dead code inventory (summary)

Files with no reference from the build or any caller: `src/kernels/copy2double.jdf`,
`isyrkmp_LT.jdf`, `gemm_TN_batched.jdf`, `gemm_f16_wrapper.c`, `matrix_dsum.jdf`,
`matrix_ssum.jdf`, `cast_double_diag.jdf`, `check_read_data.jdf`, `sqrt_kernel.jdf`,
`Lapack_to_Tile_sym.jdf`, `convert_i2s.jdf`, `diagonal_correction_calculation.jdf`,
`reorder_gemm.jdf` (plus `hicma_parsec_reorder_gemm` and the `dcReorder` descriptor),
`gemm_TN_summa.jdf`, `climate_emulator_print_double.jdf`, `src/core/hamming.jl`,
`tests/testing_2gpu_gemm.cpp`, `tests/testing_gemm_gpu_dp.cu`.

Functions with no caller: `hicma_parsec_warmup_potrf` (+ `always_local_rank_of*`),
`hicma_parsec_writeA`, `hicma_parsec_decisions_send_analysis`, `print_decisions_send`,
`print_decisions_gemm_gpu`, `disable_CPU`, `gpu_load_dense`, `lookup_gpu_workspace_i`,
`hicma_parsec_gpu_cache_reset`, `parsec_print_cm`, `parsec_print_cm_int8`,
`hicma_parsec_memcpy_int32_CPU`, `hicma_parsec_hamming_merge_CPU`, `parsec_print_tilenew`,
`hicma_parsec_convert_2fp8_bit`, `float_to_half`, `half_to_float_fast`,
`float32_to_float16`, `float2half_CPU`, `hicma_parsec_get_stat`, `readObsFile`,
`readTimeFile`, `my_cublasGemmEx`, `copy_float_to_device_double`,
`hicma_parsec_memcpy_int32_GPU`, `hicma_parsec_hamming_merge_GPU`,
`matrix_print_float_GPU`, `matrix_print_double_GPU`, the six unused integer-conversion
kernels, the static `destroy_cuda_handles` (duplicate of DPLASMA's), 120 lines of
`#if 0` LAPACKE/DLASCL shims in `hicma_parsec_internal.c`.

Declared but never defined: `perform_gemm`, `parsec_gemmmp_TN`, `parsec_gemmmp_TN_generator`,
`parsec_dist_allocate_gemmmp_TN`, `decision_datatype_tile_gemmmp_TN`,
`decision_datatype_tile_convert`, `decision_datatype_out`, `decision_size_out`,
`hicma_parsec_decisions_gemm_gpu_init`, `hicma_parsec_core_gemmmp_TN_denseC_denseA_denseB_gpu`,
`hicma_parsec_core_dgemm_lr_svd`, `hicma_parsec_core_dgemm_lrA_lrB_lrC_svd`,
`climate_emulator_print_complex`, `hicma_parsec_matrix_allocate_dense_tile`.

Compile-time-disabled paths that no longer compile or are internally inconsistent:
`matrix_add.jdf` under `!GPU_BUFFER_ONCE` (undefined identifiers), `#if PREDICTION ||
CHECKSOLVE && GENOMICS` vs `#if PREDICTION || CHECKSOLVE`, the climate emulator's
`DEBUG_INFO_CLIMATE_EMULATOR` blocks, `hicma_parsec_reorder_gemm`'s `#if 0` body.

## 6. Deferred (out of scope for 0.2.0, recorded with the reason)

| Item | Why deferred |
|---|---|
| Merging the Cholesky JDFs (70-85% identical) and the HCORE two-step GEMM block pasted 20× across the TLR JDFs | TODO backlog; task-graph rewrite |
| Merging the `Lapack_to_Tile`/`Tile_to_Lapack` family (9 JDFs) | TODO backlog |
| GENOMICS macro consolidation, params restructure, public/internal header split, API redesign, hcore removal, Python/Julia, tensor_gemm redesign | TODO backlog |
| Single-source CUDA/HIP kernels (D12) | cannot be validated on AMD hardware here; documented with the drift table |
| CPU kernel de-duplication (D13), CPU/GPU runtime policy alignment (D6) | body rewrites with numerical consequences; need GPU and CPU result comparisons at scale |
| TERMINATE/sink-task races (B4, C2) | never observed; task-graph change in nine JDFs with a `PARSEC_DEBUG_PARANOID` validation recipe |
| Runtime-mode manager-thread synchronizations (D10) | performance work needing measurements at N ≥ 32768 |
| Partial-tile support for the Cholesky path (H1) | generation, check and HCORE assumptions; rejected at run time for now |
| Collapsing the five climate CSV readers, merging `gemmex_TN` into `gemm_TN`, climate file names from `L` (F5, F11) | application code without data to validate here |
| Private LRU evictor replaced by a PaRSEC-exported helper (D2) | belongs in the PaRSEC fork |
| Moving `use_scratch` into hcore, `hicma_parsec_memory_flush_choleksy` rename | submodule / API change |

## 7. Verified fine (do not re-litigate)

- Triple `parsec_tiled_matrix_destroy` on band descriptors (`super`, `band`, `off_band`):
  each was initialized separately; destroy frees only per-descriptor members.
- `dcDist`, `dcA0`/`dcA1`, `dcAd` lifetimes under `sparse`/`check`/dense paths are
  consistent; the unconditional `dcAd` destroy (commit 5c52832) is needed.
- `parsec_info_unregister` with an undefined id when no GPU was found: safe, same as
  DPLASMA's tests.
- `IS_ALLOCATE_DP` is always true in the default build, so every SP/HP tile lives in an
  FP64-sized host buffer; this is what makes the in-place conversion paths safe.
- Sender `type_remote = BYTE count_remote = size` with typed receive arenas: PaRSEC takes
  the receive type from the successor flow and allocates `max(arena, wire)`; every
  combination checked has an arena at least as large as the bytes sent.
- `HICMA_PARSEC_GPU_PUSHOUT` bits set in the body are honored (`parsec_gpu_task_update_pushout`
  only ORs). The fork now also marks remote-successor flows itself, so the explicit
  pushout is redundant for the multi-rank case and costs one D2H per converted tile on
  one rank: a possible follow-up, not a correctness issue.
- Overwriting `task_classes_array[i]->incarnations[j].hook/evaluate` in `New()` is legal:
  the generated constructor copies the chore tables per taskpool instance.
- `hicma_gpu_read_norm_meta` on a private non-blocking stream without an event: safe
  because the consumer only reaches its exec stage after the producer completed on the
  host.
- `cudaMemcpyAsync` into `info_gpu` is really asynchronous (pinned).
- The TRMM flows, CTL sets, `window_A` throttling and arena/`type_remote` selections are
  consistent; `Bop <- NEW` on every `read_B` is deliberate.
- `hicma_kernel_time.c`'s lock-free start-time table and locked overflow list are
  consistent.
- `decision_make_*` `MPI_Allreduce(SUM)` on `uint16_t` decisions is correct (each tile
  written by exactly one rank into a zeroed array).
- The symmetric→block-cyclic cast in `check_dpotrf2` to read `grid` is layout-compatible.
- `convert_datatype_adaptive_unary_CPU`'s `exit(1)` on FP8/unknown decisions is a
  deliberate fail-fast.
- Stochastic rounding: CPU and GPU hashes are identical (`splitmix64` over
  `(i, j, seed, value bits)`); the FP16 lower-ulp step cannot underflow.
- `.gitmodules` relative URLs resolve from any `github.com/<org>/hicma-x` clone.
- `GNUMakeRulesOverwrite.cmake` (`-MD` → `-MMD` for Ninja) is harmless and useful on
  module-based systems.
