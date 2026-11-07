# HiCMA-X 0.2.0 release readiness

Branch `qinglei/release-cleanup`, 2026-11-07. Base: `main` at `6e987e3`
(merge of the PaRSEC/DPLASMA update). Review: `docs/code-review.md`; plan and
per-item status: `docs/release-improvement-plan.md`; user-facing notes:
`RELEASE.md`, `CHANGELOG.md`.

Nothing has been tagged, pushed or merged. Those three steps need the
maintainer's authorization (section 7).

## 1. Recommendation

Release as **0.2.0**. The previous tag is `v0.1.0` (2025-11-12). Since then
`main` gained features (runtime precision decision, FP8 GEMMs with per-tile
scaling, the mixed-precision TRMM, the ICLDisco PaRSEC/DPLASMA) and this branch
changes observable behaviour (the correctness check can fail, `--adddiag` is
applied once, `N % NB != 0` is rejected, the genomics drivers are optional, the
library's SONAME changes), which is more than a patch release and, in a 0.x
line, not a major one. `project(VERSION 0.2.0)` in `CMakeLists.txt` is the
single source of the version; the library is `libhicma_parsec.so.0.2.0` with
SONAME `libhicma_parsec.so.0`.

## 2. Commits on the branch

- `be50c60` State the FP32 residual bound of the dense adaptive cases; print the residual of a failed check
- `f07913f` CI: the CUDA job needs python3, flex and bison
- `d6cb516` Link one BLAS/LAPACKE stack and give hcore and stars-h their dependencies
- `8f7cb5e` Release documentation for 0.2.0
- `77edd5d` Add the validation matrix script
- `eeaa72d` CI, scripts and repository hygiene
- `5e7ba95` Make the installed package usable, unify the version, register the tests
- `1dc97a3` Small fixes: leaks, NULL checks, file readers, genomics helpers
- `bac0083` Remove dead code
- `4be5157` Share the Cholesky wrapper machinery in potrf_common
- `0b5e817` Remove the sink tasks that could outlive their taskpool
- `08bf1b8` Initialize the parameters once and allocate statistics after device discovery
- `94e3633` Refuse unsupported configurations instead of miscomputing
- `449c906` Fix kernel-level bugs found in the review
- `4eca4c3` Apply --adddiag once for the STARS-H kernels that already carry it
- `0c76e56` Make the correctness check binding and add --check_tol

(This documentation commit follows.) Each commit is self-contained in intent;
the trees that were built and tested are the one after the documentation
commit `8f7cb5e` (sections 4, 5.1 to 5.3) and the final one (section 5.5).
Intermediate commits were not built individually.

## 3. Validation environment

| Item | Value |
|---|---|
| Machine | 1 node, 2 x Intel Xeon Gold 6140 (72 hardware threads), 1 x NVIDIA A100-PCIE-40GB (compute capability 8.0, driver 580.65.06) |
| Compilers / tools | GCC 11.5.0, CMake 3.30.5, Ninja; CUDA 12.6 (nvcc 12.6.77); ROCm 6.4 `hipcc` (compile only, no AMD GPU) |
| Libraries | Open MPI 5.0.5, Intel MKL 2024.2 (BLAS/LAPACKE), GSL present |
| Submodules | dplasma `3553b25a` (ICLDisco master + stage-in fix), parsec `5cd761f6` (4.0.2411-244), hcore `78028a59`, stars-h `5d98f2c2` |
| CPU tree | `-DCMAKE_BUILD_TYPE=Release -DHICMA_PARSEC_HAVE_CUDA=OFF -DDPLASMA_PRECISIONS="s;d" -DPARSEC_DIST_SHORT_LIMIT=1` |
| CUDA tree | same plus `-DHICMA_PARSEC_HAVE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80` |
| HIP tree | same plus `-DHICMA_PARSEC_HAVE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a`, compile only |

The baseline for every "before" number is the same configuration built from
`6e987e3` in a separate worktree (CUDA tree) and from the branch point before
any source change (CPU tree), with the same submodule commits.

## 4. Build results

| Configuration | Result |
|---|---|
| CPU | builds; the only compiler diagnostics are PaRSEC's deprecation notes for `parsec_add2arena`/`parsec_del2arena` (pre-existing, 265 uses in the wrappers and generated JDF code) and linker notes about the MKL run-time path (environment, not code) |
| CUDA (sm_80) | builds, same diagnostics plus the pre-existing nvcc notes in the `.cu` kernels |
| HIP (gfx90a) | builds; not executed (no AMD GPU) |
| CUDA + `-DHICMA_PARSEC_GENOMICS=ON` | compiles (`-DHICMA_PARSEC_GENOMICS=ON` with CUDA, same flags as the CUDA tree); the drivers `testing_syrk`, `testing_KRR`, `testing_KRR_file`, `testing_Kmatrix`, `testing_igemm_summa` are built. Not run: they need the genomics data files. |
| Installed package | `cmake --install` of the CPU and CUDA trees, then `tests/install_consumer` (`find_package(HICMAParsec)`, links `HICMAParsec::hicma_parsec` only) configures, builds and runs `--N 2048 --NB 512 --check 1` with residual 2.69e-16 from both prefixes |

## 5. Test results

All numbers below come from runs executed on 2026-11-07 against the final tree
of the branch (logs are in the session's scratch directory; the matrix is
reproducible with `scripts/validation_matrix.sh`).

### 5.1 CTest

| Tree | Command | Result |
|---|---|---|
| CPU | `ctest -L hicma -LE gpu --output-on-failure` | 20 / 20 passed (15 single-rank, 5 two-rank) |
| CUDA | `ctest -L hicma --output-on-failure` | 30 / 30 passed (the 20 CPU cases plus `gpu_dense_fp64`, `gpu_dense_adaptive`, `gpu_dense_band2`, `gpu_fp8_static`, `gpu_fp8_sp`, `gpu_runtime_decision`, `gpu_runtime_admem`, `gpu_trmm_adaptive`, `gpu_tlr_mp`, `gpu_tlr_dp_band2`) |

Every registered case runs with `--check 1`; the mixed-precision cases state
their accepted bound with `--check_tol` (1e-6 for the SP/HP band kinds, 1e-7
for the adaptive and runtime-decision kinds). The two-rank GPU cases share the
single A100 and pass `--mca device_cuda_memory_use 30`.

### 5.2 Validation matrix

`scripts/validation_matrix.sh` runs 30 cases on a CPU build and 45 on a GPU
build (the 30 CPU cases against the GPU-enabled library, plus 15 GPU cases).
Every case exited 0 before and after. The residuals are identical to the
baseline except for the virus-mesh cases, whose residual was the doubled
diagonal shift (review finding E1); the GEMM precision counts of every case
are identical. Six cases were added during the work (kind 2 with a dense band,
which exposed the sink-task crash) and have no baseline column. The tables
are from the final run (after the CI fixes of section 5.5); the run before
those fixes gave the same residuals and GEMM counts.

CPU tree:

| Case | Ranks | Residual before | Residual after | Factorization time after (s) |
|---|---|---|---|---|
| `tlr_dp_k2_p2` | 1 | 1.157838e-16 | 1.157838e-16 | 0.02215 |
| `tlr_mp_k1_band1_adapt` | 1 | 1.736223e-08 | 1.736223e-08 | 0.05274 |
| `tlr_mp_k1_band1_noadapt` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05097 |
| `tlr_mp_k1_dense_adapt` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02820 |
| `tlr_mp_k1_dense_admem` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02846 |
| `tlr_kind2_band1` | 1 | not in the baseline matrix | 5.014459e-09 | 0.05299 |
| `tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.03791 |
| `tlr_kind2_band2_np2` | 2 | not in the baseline matrix | 2.741820e-09 | 0.07394 |
| `tlr_kind1_band2_np2` | 2 | not in the baseline matrix | 1.740042e-08 | 0.08373 |
| `tlr_dp_k2_p4_autoband` | 1 | 1.328506e-10 | 1.328506e-10 | 0.10997 |
| `dense_k5_fp64_cpu` | 1 | 1.856962e-16 | 1.856962e-16 | 0.06384 |
| `dense_k5_band2_cpu` | 1 | 3.099008e-09 | 3.099008e-09 | 0.06809 |
| `dense_k5_adapt_cpu` | 1 | 4.028860e-10 | 4.028860e-10 | 0.09336 |
| `band_k3_cpu` | 1 | 5.377707e-08 | 5.377707e-08 | 0.11159 |
| `band_k4_cpu` | 1 | 1.692654e-07 | 1.692654e-07 | 0.10184 |
| `runtime_k10_cpu` | 1 | 3.754839e-09 | 3.754839e-09 | 0.12166 |
| `runtime_k10_cpu_adapt1` | 1 | 3.776331e-09 | 3.776331e-09 | 0.11671 |
| `sparse_k6_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05457 |
| `sparse_k7_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05183 |
| `sparse_k6_np2` | 2 | 5.014459e-09 | 5.014459e-09 | 0.08500 |
| `sparse_k6_mesh` | 1 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.47155 |
| `sparse_k7_mesh_np2` | 2 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.57200 |
| `trmm_fp64` | 1 | PASSED | PASSED | NA |
| `trmm_adapt` | 1 | PASSED | PASSED | NA |
| `trmm_adapt_sr` | 1 | PASSED | PASSED | NA |
| `tlr_dp_k2_np2` | 2 | 1.157838e-16 | 1.157838e-16 | 0.06302 |
| `dense_k5_fp64_np2` | 2 | 1.856962e-16 | 1.856962e-16 | 0.12500 |
| `dense_k5_band2_np2` | 2 | 3.099008e-09 | 3.099008e-09 | 0.13400 |
| `runtime_k10_np2` | 2 | 3.754839e-09 | 3.754839e-09 | 0.18000 |
| `trmm_adapt_np2` | 2 | PASSED | PASSED | NA |

CUDA tree (same library built with CUDA; `gpu_*` cases use the A100):

| Case | Ranks | Residual before | Residual after | Factorization time after (s) |
|---|---|---|---|---|
| `tlr_dp_k2_p2` | 1 | 1.157838e-16 | 1.157838e-16 | 0.01989 |
| `tlr_mp_k1_band1_adapt` | 1 | 1.736223e-08 | 1.736223e-08 | 0.05263 |
| `tlr_mp_k1_band1_noadapt` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05334 |
| `tlr_mp_k1_dense_adapt` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02810 |
| `tlr_mp_k1_dense_admem` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02851 |
| `tlr_kind2_band1` | 1 | not in the baseline matrix | 5.014459e-09 | 0.05471 |
| `tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.03740 |
| `tlr_kind2_band2_np2` | 2 | not in the baseline matrix | 2.741820e-09 | 0.08977 |
| `tlr_kind1_band2_np2` | 2 | not in the baseline matrix | 1.740042e-08 | 0.08474 |
| `tlr_dp_k2_p4_autoband` | 1 | 1.328506e-10 | 1.328506e-10 | 0.11658 |
| `dense_k5_fp64_cpu` | 1 | 1.856962e-16 | 1.856962e-16 | 0.06102 |
| `dense_k5_band2_cpu` | 1 | 3.099008e-09 | 3.099008e-09 | 0.06646 |
| `dense_k5_adapt_cpu` | 1 | 4.028860e-10 | 4.028860e-10 | 0.09376 |
| `band_k3_cpu` | 1 | 5.377707e-08 | 5.377707e-08 | 0.11434 |
| `band_k4_cpu` | 1 | 1.692654e-07 | 1.692654e-07 | 0.10238 |
| `runtime_k10_cpu` | 1 | 3.754839e-09 | 3.754839e-09 | 0.12483 |
| `runtime_k10_cpu_adapt1` | 1 | 3.776331e-09 | 3.776331e-09 | 0.11683 |
| `sparse_k6_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05264 |
| `sparse_k7_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05460 |
| `sparse_k6_np2` | 2 | 5.014459e-09 | 5.014459e-09 | 0.08800 |
| `sparse_k6_mesh` | 1 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.46712 |
| `sparse_k7_mesh_np2` | 2 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.55999 |
| `trmm_fp64` | 1 | PASSED | PASSED | NA |
| `trmm_adapt` | 1 | PASSED | PASSED | NA |
| `trmm_adapt_sr` | 1 | PASSED | PASSED | NA |
| `tlr_dp_k2_np2` | 2 | 1.157838e-16 | 1.157838e-16 | 0.06900 |
| `dense_k5_fp64_np2` | 2 | 1.856962e-16 | 1.856962e-16 | 0.14000 |
| `dense_k5_band2_np2` | 2 | 3.099008e-09 | 3.099008e-09 | 0.11800 |
| `runtime_k10_np2` | 2 | 3.754839e-09 | 3.754839e-09 | 0.19200 |
| `trmm_adapt_np2` | 2 | PASSED | PASSED | NA |
| `gpu_dense_k5_fp64` | 1 | 3.602581e-16 | 3.602581e-16 | 0.11584 |
| `gpu_dense_k5_adapt` | 1 | 2.176706e-09 | 2.176706e-09 | 0.23197 |
| `gpu_dense_k5_band2` | 1 | 2.804213e-09 | 2.804213e-09 | 0.21073 |
| `gpu_dense_k8_adapt` | 1 | 2.176706e-09 | 2.176706e-09 | 0.22955 |
| `gpu_dense_k9_adapt` | 1 | 3.313729e-07 | 3.313729e-07 | 0.12063 |
| `gpu_runtime_k10` | 1 | 1.041065e-08 | 1.041065e-08 | 0.26862 |
| `gpu_runtime_k10_admem` | 1 | 1.041065e-08 | 1.041065e-08 | 0.26798 |
| `gpu_tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.08998 |
| `gpu_tlr_kind1_band2` | 1 | not in the baseline matrix | 1.725825e-08 | 0.09983 |
| `gpu_tlr_mp_k1` | 1 | 5.551907e-09 | 5.551907e-09 | 0.20121 |
| `gpu_dense_k5_dtconv2` | 1 | 2.176706e-09 | 2.176706e-09 | 0.17968 |
| `gpu_trmm_adapt` | 1 | PASSED | PASSED | NA |
| `gpu_dense_k5_fp64_np2` | 2 | 3.602581e-16 | 3.602581e-16 | 0.34400 |
| `gpu_runtime_k10_np2` | 2 | 1.041065e-08 | 1.041065e-08 | 0.47139 |
| `gpu_dense_k8_np2` | 2 | 2.176706e-09 | 2.176706e-09 | 0.41567 |

The kind 8 and 9 cases ran on an A100 (compute capability 8.0), where the FP8
GEMMs fall back to FP16/FP32 (`DENSE_FP8= 0` in their GEMM counts): the FP8
code path is compiled and its dispatch exercised, but no FP8 arithmetic ran in
this validation. The H100 results from the development of the runtime decision
are recorded in the pull-request history, not re-run here.

### 5.3 Performance

Factorization time (`hicma_parsec_cholesky`, without generation and check),
three repetitions each, baseline binary from `6e987e3` and the branch binary,
same node, same GPU, run back to back:

| Case (N=32768, NB=1024, one A100) | Before: min / median / max (s) | After: min / median / max (s) | After/before (median) | GEMM precision counts identical |
|---|---|---|---|---|
| kind 5, FP64 (`--kind_of_problem 2 --gpus 1`) | 1.216 / 1.227 / 1.233 | 1.200 / 1.207 / 1.227 | 0.984 | yes |
| kind 5, static mixed precision (`--adaptive_decision 1`) | 1.155 / 1.176 / 1.180 | 1.119 / 1.140 / 1.147 | 0.969 | yes |
| kind 10, runtime decision (`--adaptive_decision_runtime 1`) | 1.180 / 1.181 / 1.186 | 1.181 / 1.186 / 1.197 | 1.005 | yes |

GEMM precision counts of the "after" runs: kind 5 FP64 `DENSE_DP= 4960 DENSE_SP= 0 DENSE_HP= 0 DENSE_FP8= 0 LOW_RANK_DP= 0 LOW_RANK_SP= 0 Number_of_Datatype_Conversions= 0`;
kind 5 adaptive `DENSE_DP= 2156 DENSE_SP= 2067 DENSE_HP= 737 DENSE_FP8= 0 LOW_RANK_DP= 0 LOW_RANK_SP= 0 Number_of_Datatype_Conversions= 6287`; kind 10 `DENSE_DP= 468 DENSE_SP= 2138 DENSE_HP= 2354 DENSE_FP8= 0 LOW_RANK_DP= 0 LOW_RANK_SP= 0 Number_of_Datatype_Conversions= 6766`.

The differences are within run-to-run noise (3 % or less); no change on the
branch touches the GPU kernels' arithmetic or the task graphs' critical path.
The small-case matrix times (tables above) are likewise unchanged within
noise. The large-case timing was taken before the link changes of section
5.5; after them the matrix times of the GPU cases are unchanged within noise
and those of the CPU cases are identical to the earlier runs. No multi-node or
multi-GPU measurement was possible on this machine.

### 5.4 Not validated

- HIP execution (compile only).
- More than one node, more than one GPU per node, GPUs with compute capability
  8.9 or higher (real FP8 arithmetic).
- The climate emulator driver (`testing_climate_emulator`): compiled in every
  configuration, not run (needs the external data set).
- The genomics drivers: see section 4; not run (data files).
- The Fugaku (`ON_FUGAKU`) build.
- The intermediate commits individually (only the final tree).

### 5.5 First CI run and the follow-up fixes

The first run of the workflow on GitHub (pull request ecrc/hicma-x#13) failed
in both jobs:

- **CPU job**: every test driver failed to link with undefined `LAPACKE_*`
  references from `libhcore.so`. hcore's own CMake records no link
  dependency, and with Ubuntu's `--as-needed` default nothing else on the
  line provided the symbols (locally MKL reached them through DPLASMA). hcore
  and starsh now link `LAPACKE::LAPACKE`, and `hicma_parsec` links the same
  target instead of a separate BLAS discovery plus a hard-coded MKL block,
  which had loaded two MKL interface layers and two threading layers into one
  process.
- **CUDA job**: configure failed because the container has no Python
  (DPLASMA's precision generator); flex and bison were missing as well. They
  are installed now.
- **Found while validating the fix**: with hcore's new dependency the link
  order put `libgslcblas` (GSL's reference CBLAS, linked by STARS-H) ahead of
  MKL, and DPLASMA's `cblas_dgemm` bound to it: the CPU factorizations ran 40x
  slower (dense FP64, N=4096: 0.065 s to 2.64 s), residuals shifted (FP64
  1.9e-16 to 1.4e-15) and the dense adaptive mixed-precision cases crossed
  their 1e-8 bound. `libgslcblas` is removed from the starsh target (any CBLAS
  satisfies GSL) and hcore is linked before starsh; `LD_DEBUG=bindings` then
  shows `cblas_dgemm` bound to `libmkl_gf_lp64`, and every residual and time
  returned to the baseline values.
- `potrf_mp_dense_adaptive`, `potrf_mp_dense_admem` and `gpu_tlr_mp` carry
  `--check_tol 1e-7`: their residual is 6.6e-9 with one BLAS binding and
  1.1e-8 with another, bounded by the FP32 arithmetic of half of their GEMMs
  rather than by the 1e-8 compression accuracy; the banded sibling case
  already used 1e-7. A failed check now prints its residual and bound.

- **Second run, CPU job**: build, CTest and the regression script passed;
  the installed consumer failed at load time, `libdplasma.so.2: cannot open
  shared object file`. The installed `libhicma_parsec.so` had no runtime
  search path, and with RUNPATH semantics (Ubuntu's default) a library's
  dependencies are searched with the library's own path, not the
  executable's (locally the Spack toolchain emits RPATH, which hid this).
  `CMAKE_INSTALL_RPATH` (`$ORIGIN`, the prefix lib dir) and
  `CMAKE_INSTALL_RPATH_USE_LINK_PATH` are set for `hicma_parsec` and the
  drivers; reproduced and verified locally by linking the consumer with
  `--enable-new-dtags` and running it with only the BLAS directory on
  `LD_LIBRARY_PATH` (CPU and CUDA prefixes). CMake does not record MKL's
  directory, so a BLAS outside the system paths still needs `LD_LIBRARY_PATH`.

Re-validation after these commits: CPU, CUDA and HIP trees build; CPU `ctest`
20 / 20; CUDA-tree `ctest` 30 / 30; install consumers from both prefixes
(residual 2.69e-16); the 75-case matrix identical to the baseline except the
intended mesh change (tables in 5.2). Both CI jobs of pull request
ecrc/hicma-x#13 pass on the final commit (CPU: build, CTest, regression
script, install and consumer; CUDA: compile).

## 6. Release checklist

| Item | Status |
|---|---|
| Version set once (`project(VERSION 0.2.0)`, SOVERSION 0) | done |
| Release notes (`RELEASE.md`), changelog (`CHANGELOG.md`) | done |
| Build and install instructions (`BUILD.md`), consumer example (`tests/install_consumer`) | done |
| Test instructions (`TESTS.md`, CTest labels, `scripts/validation_matrix.sh`) | done |
| API / behaviour changes documented (`RELEASE.md` "Upgrading from 0.1.0") | done |
| Known limitations documented (`RELEASE.md`, `TESTS.md`) | done |
| CI covers correctness (`--check 1` in CTest and `scripts/test_CI.sh`), install, CUDA compile | done; the first GitHub run failed on three environment issues, fixed (section 5.5); both jobs pass on the final commit |
| P0 items of the plan | all done and validated |
| P1 items | done except: per-call status checks in the non-FP8 GPU kernels (P1-6, partial), warning-flags option (P1-11) |
| P2 items | done except P2-6, P2-9, P2-12 (deferred) and the remaining parts of P2-5, P2-7, P2-8 |
| Review by the maintainer of the behaviour changes (binding check, `--adddiag`, `N % NB`, genomics option) | **open** |
| CI run on GitHub | **open** (push required) |
| Tag `v0.2.0`, GitHub release, merge to `main` | **open** (authorization required) |

## 7. Unresolved blockers and decisions for the maintainer

1. **Behaviour changes to confirm**: `--check 1` now fails runs it used to
   pass; `--adddiag` results change for problem kinds 0, 1, 5, 6, 7; the
   Cholesky drivers reject `N % NB != 0`; the genomics drivers need
   `-DHICMA_PARSEC_GENOMICS=ON`. Each is documented in `RELEASE.md`.
2. **Mixed-precision residual bounds**: the registered CTest cases use
   `--check_tol` values measured on this machine (1e-6, 1e-7). Other BLAS
   libraries or GPUs may need looser bounds; the cases print the residual.
3. **CI**: passes on the final commit after the fixes of section 5.5.
4. **Hardware not available here**: AMD GPUs, H100 (FP8), multi-node. A run of
   `scripts/validation_matrix.sh <build> <label> gpu` on those systems before
   tagging is recommended.

## 8. Remaining and future work

Deferred from the plan (none blocks the release): P2-6 (zero-initialized check
matrices, `matrix_uncompress` flows), P2-9 (one JDF for the integer
conversions), P2-12 (statistics communicator hygiene), P3-1 to P3-3, the
remaining parts of P1-6, P1-11, P2-5, P2-7, P2-8. The project's own `TODOs`
backlog (API, Python/Julia bindings, merging the potrf JDFs, hcore removal,
tensor_gemm redesign) was out of scope by request and is untouched.
