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

## 2. Code commits on the branch

(Two further commits add `scripts/validation_matrix.sh` and this documentation.)

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

Each commit is self-contained in intent; only the final tree was built and
tested (sections 4 and 5). Intermediate commits were not built individually.

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
which exposed the sink-task crash) and have no baseline column.

CPU tree:

| Case | Ranks | Residual before | Residual after | Factorization time after (s) |
|---|---|---|---|---|
| `tlr_dp_k2_p2` | 1 | 1.157838e-16 | 1.157838e-16 | 0.02162 |
| `tlr_mp_k1_band1_adapt` | 1 | 1.736223e-08 | 1.736223e-08 | 0.05360 |
| `tlr_mp_k1_band1_noadapt` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05316 |
| `tlr_mp_k1_dense_adapt` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02809 |
| `tlr_mp_k1_dense_admem` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02988 |
| `tlr_kind2_band1` | 1 | not in the baseline matrix | 5.014459e-09 | 0.05103 |
| `tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.03997 |
| `tlr_kind2_band2_np2` | 2 | not in the baseline matrix | 2.741820e-09 | 0.07800 |
| `tlr_kind1_band2_np2` | 2 | not in the baseline matrix | 1.740042e-08 | 0.08575 |
| `tlr_dp_k2_p4_autoband` | 1 | 1.328506e-10 | 1.328506e-10 | 0.11323 |
| `dense_k5_fp64_cpu` | 1 | 1.856962e-16 | 1.856962e-16 | 0.06539 |
| `dense_k5_band2_cpu` | 1 | 3.099008e-09 | 3.099008e-09 | 0.06480 |
| `dense_k5_adapt_cpu` | 1 | 4.028860e-10 | 4.028860e-10 | 0.09365 |
| `band_k3_cpu` | 1 | 5.377707e-08 | 5.377707e-08 | 0.11248 |
| `band_k4_cpu` | 1 | 1.692654e-07 | 1.692654e-07 | 0.10175 |
| `runtime_k10_cpu` | 1 | 3.754839e-09 | 3.754839e-09 | 0.12321 |
| `runtime_k10_cpu_adapt1` | 1 | 3.776331e-09 | 3.776331e-09 | 0.11764 |
| `sparse_k6_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05469 |
| `sparse_k7_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05351 |
| `sparse_k6_np2` | 2 | 5.014459e-09 | 5.014459e-09 | 0.10200 |
| `sparse_k6_mesh` | 1 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.47122 |
| `sparse_k7_mesh_np2` | 2 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.56000 |
| `trmm_fp64` | 1 | PASSED | PASSED | NA |
| `trmm_adapt` | 1 | PASSED | PASSED | NA |
| `trmm_adapt_sr` | 1 | PASSED | PASSED | NA |
| `tlr_dp_k2_np2` | 2 | 1.157838e-16 | 1.157838e-16 | 0.07171 |
| `dense_k5_fp64_np2` | 2 | 1.856962e-16 | 1.856962e-16 | 0.12800 |
| `dense_k5_band2_np2` | 2 | 3.099008e-09 | 3.099008e-09 | 0.13100 |
| `runtime_k10_np2` | 2 | 3.754839e-09 | 3.754839e-09 | 0.15900 |
| `trmm_adapt_np2` | 2 | PASSED | PASSED | NA |

CUDA tree (same library built with CUDA; `gpu_*` cases use the A100):

| Case | Ranks | Residual before | Residual after | Factorization time after (s) |
|---|---|---|---|---|
| `tlr_dp_k2_p2` | 1 | 1.157838e-16 | 1.157838e-16 | 0.02249 |
| `tlr_mp_k1_band1_adapt` | 1 | 1.736223e-08 | 1.736223e-08 | 0.05200 |
| `tlr_mp_k1_band1_noadapt` | 1 | 5.014459e-09 | 5.014459e-09 | 0.07647 |
| `tlr_mp_k1_dense_adapt` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02857 |
| `tlr_mp_k1_dense_admem` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02910 |
| `tlr_kind2_band1` | 1 | not in the baseline matrix | 5.014459e-09 | 0.05418 |
| `tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.03932 |
| `tlr_kind2_band2_np2` | 2 | not in the baseline matrix | 2.741820e-09 | 0.08500 |
| `tlr_kind1_band2_np2` | 2 | not in the baseline matrix | 1.740042e-08 | 0.08573 |
| `tlr_dp_k2_p4_autoband` | 1 | 1.328506e-10 | 1.328506e-10 | 0.10466 |
| `dense_k5_fp64_cpu` | 1 | 1.856962e-16 | 1.856962e-16 | 0.06131 |
| `dense_k5_band2_cpu` | 1 | 3.099008e-09 | 3.099008e-09 | 0.06425 |
| `dense_k5_adapt_cpu` | 1 | 4.028860e-10 | 4.028860e-10 | 0.09465 |
| `band_k3_cpu` | 1 | 5.377707e-08 | 5.377707e-08 | 0.11365 |
| `band_k4_cpu` | 1 | 1.692654e-07 | 1.692654e-07 | 0.10185 |
| `runtime_k10_cpu` | 1 | 3.754839e-09 | 3.754839e-09 | 0.12252 |
| `runtime_k10_cpu_adapt1` | 1 | 3.776331e-09 | 3.776331e-09 | 0.11660 |
| `sparse_k6_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05376 |
| `sparse_k7_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05335 |
| `sparse_k6_np2` | 2 | 5.014459e-09 | 5.014459e-09 | 0.09900 |
| `sparse_k6_mesh` | 1 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.47146 |
| `sparse_k7_mesh_np2` | 2 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.57200 |
| `trmm_fp64` | 1 | PASSED | PASSED | NA |
| `trmm_adapt` | 1 | PASSED | PASSED | NA |
| `trmm_adapt_sr` | 1 | PASSED | PASSED | NA |
| `tlr_dp_k2_np2` | 2 | 1.157838e-16 | 1.157838e-16 | 0.07269 |
| `dense_k5_fp64_np2` | 2 | 1.856962e-16 | 1.856962e-16 | 0.11271 |
| `dense_k5_band2_np2` | 2 | 3.099008e-09 | 3.099008e-09 | 0.12100 |
| `runtime_k10_np2` | 2 | 3.754839e-09 | 3.754839e-09 | 0.17267 |
| `trmm_adapt_np2` | 2 | PASSED | PASSED | NA |
| `gpu_dense_k5_fp64` | 1 | 3.602581e-16 | 3.602581e-16 | 0.11396 |
| `gpu_dense_k5_adapt` | 1 | 2.176706e-09 | 2.176706e-09 | 0.18248 |
| `gpu_dense_k5_band2` | 1 | 2.804213e-09 | 2.804213e-09 | 0.20455 |
| `gpu_dense_k8_adapt` | 1 | 2.176706e-09 | 2.176706e-09 | 0.17907 |
| `gpu_dense_k9_adapt` | 1 | 3.313729e-07 | 3.313729e-07 | 0.11935 |
| `gpu_runtime_k10` | 1 | 1.041065e-08 | 1.041065e-08 | 0.28133 |
| `gpu_runtime_k10_admem` | 1 | 1.041065e-08 | 1.041065e-08 | 0.27054 |
| `gpu_tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.09307 |
| `gpu_tlr_kind1_band2` | 1 | not in the baseline matrix | 1.725825e-08 | 0.09755 |
| `gpu_tlr_mp_k1` | 1 | 5.551907e-09 | 5.551907e-09 | 0.22803 |
| `gpu_dense_k5_dtconv2` | 1 | 2.176706e-09 | 2.176706e-09 | 0.17879 |
| `gpu_trmm_adapt` | 1 | PASSED | PASSED | NA |
| `gpu_dense_k5_fp64_np2` | 2 | 3.602581e-16 | 3.602581e-16 | 0.31838 |
| `gpu_runtime_k10_np2` | 2 | 1.041065e-08 | 1.041065e-08 | 0.47134 |
| `gpu_dense_k8_np2` | 2 | 2.176706e-09 | 2.176706e-09 | 0.37229 |

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
noise. No multi-node or multi-GPU measurement was possible on this machine.

### 5.4 Not validated

- HIP execution (compile only).
- More than one node, more than one GPU per node, GPUs with compute capability
  8.9 or higher (real FP8 arithmetic).
- The climate emulator driver (`testing_climate_emulator`): compiled in every
  configuration, not run (needs the external data set).
- The genomics drivers: see section 4; not run (data files).
- The Fugaku (`ON_FUGAKU`) build.
- The intermediate commits individually (only the final tree).

## 6. Release checklist

| Item | Status |
|---|---|
| Version set once (`project(VERSION 0.2.0)`, SOVERSION 0) | done |
| Release notes (`RELEASE.md`), changelog (`CHANGELOG.md`) | done |
| Build and install instructions (`BUILD.md`), consumer example (`tests/install_consumer`) | done |
| Test instructions (`TESTS.md`, CTest labels, `scripts/validation_matrix.sh`) | done |
| API / behaviour changes documented (`RELEASE.md` "Upgrading from 0.1.0") | done |
| Known limitations documented (`RELEASE.md`, `TESTS.md`) | done |
| CI covers correctness (`--check 1` in CTest and `scripts/test_CI.sh`), install, CUDA compile | done in the workflow; the workflow has not run on GitHub from this branch |
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
3. **CI**: the workflow file is rewritten but has not run on GitHub.
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
