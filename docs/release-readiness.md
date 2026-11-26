# HiCMA-X 0.2.0 release readiness

Branch `qinglei/release-cleanup`, 2026-11-07. Base: `main` at `6e987e3`
(merge of the PaRSEC/DPLASMA update). Review: `docs/code-review.md`; plan and
per-item status: `docs/release-improvement-plan.md`; user-facing notes:
`RELEASE.md`, `CHANGELOG.md`.

Nothing has been tagged or merged; the branch is pushed for ecrc/hicma-x#13.
Tagging, the GitHub release and the merge need the maintainer's authorization
(section 6).

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

- `02fa12c` Documentation: kinds 8 and 9 keep their own JDFs in TODOs and the deferred items
- `b817e7a` Kinds 8 and 9 on their own kernels again
- `91a64f0` Remove the prototype of a deleted function and comments naming removed ones
- `a7ecff1` One kernel for kinds 5, 8 and 9; shared low-rank CPU GEMM; one GPU tile placement rule
- `1d74b4c` Update dplasma: parsec rebased on ICLDisco master 2c16c652e
- `c658f96` Cray MPICH without GPU support: no intra-node single copy for HIP runs
- `a67bb1b` Cray MPICH without GPU support: keep remote transfers in host memory
- `931c41b` AMD GPUs: BF16 for the adaptive low-precision tiles, chosen with --tensor_gemm
- `5c0874b` HIP: set every BLAS/solver handle up before the factorization
- `34acdce` Frontier pass: machine scripts, launcher, HIP test coverage, check output
- `cc40906` HIP: scale the FP16 operands of the run-time-decision GEMM
- `94c5b62` Documentation: the complete hexane test (H100, V100, both, two ranks)
- `e9e998f` Notes for a capped --gpus and for oversubscribed cores
- `7ac56e1` Documentation: choosing OpenBLAS next to MKL; the fresh-clone and OpenBLAS checks
- `570ec75` CMake: translate BLA_VENDOR=OpenBLAS for STARS-H and HCORE
- `c2aef96` Documentation: the H100 pass, the FP8 fix, and a documentation audit
- `64a88d1` Low-rank GEMMs: an FP8 operand is stored as FP32, like SP and HP
- `89bcd89` Documentation for the FP16-band fix: root cause, results, limitation removed
- `c1456ff` Fix FP16 tiles with low-rank x low-rank updates on GPUs: convert the low-rank B as U and V
- `348bca8` release-readiness: round 3 (fresh clone, install, CI, performance); variant timings in TESTS.md
- `3fd5cda` Documentation for the GPU GEMM variants and rounds 1 and 2
- `317e171` GPU GEMM variants: --tensor_gemm_hp and --tensor_gemm_sp, one checked variant
- `070c85a` release-readiness: record the decision to keep the --gpus and TLR band defaults
- `d5a519a` Documentation for the external-user pass
- `4560e88` External-user pass: --nruns crash, help defaults, summary line, performance notes
- `c912875` Documentation for the BLAS fixture and the CI hardening
- `35b1007` Check the BLAS for concurrent calls before the HiCMA tests; harden CPU CI
- `2617921` Documentation for the #if block cleanup
- `7d21454` Resolve the 28 unnamed #if 0 / #if 1 blocks
- `f4e8512` code-review.md: number the macro review as the fifth pass
- `00afd86` Documentation for the macro and data-structure review
- `491b44c` Climate emulator: one name per switch, settable with -D
- `a0a8d8e` Make the local compile-time switches settable with -D
- `d28c2e0` Remove unused macros and parameter fields; one SYNC_TIME_PRINT
- `508466d` Release readiness: GEMM reordering kept for future use
- `8d9777d` Keep GEMM reordering for future use
- `a4dacb1` Documentation for the build-script pass
- `f448fd3` Rewrite the build scripts for the single-configure build
- `7841d07` Kind 3: test the sender conversion through a JDF global
- `2de52d2` Apply the documented default CUDA architectures
- `d6bf679` HIP install, genomics gating and test scripts on Frontier
- `093b313` matrix_sum: sum V of a low-rank tile with its own row count
- `0f4e8f5` Look up the trsm/syrk/gemm locals by name in the timing hooks
- `43b52ec` Set-up fixes found on Frontier: HIP handles, GPU count, kind 3 mode 2
- `6e01e6b` Brace the if/else around the tile-to-LAPACK conversions
- `ec39832` Documentation for the third pass: matrix sum, macros, sanitizers
- `e1a381a` Fix the heap corruption of workload-balance strategy 4
- `c55d6a1` Remove GPU_BUFFER_ONCE: keep the GPU workspace allocated once
- `fcf311d` Compile-time switches: settable with -D, dead ones removed
- `68578bc` Free the storage of pooled workspaces when a taskpool is destroyed
- `f6703d3` One matrix sum for tiles stored in different precisions
- `f63e5a3` Release readiness: fresh-clone check and CI of the second pass
- `ffd71fe` Documentation for the second pass: external use, corner cases, guyot
- `5b80e98` Pass half-precision alpha and beta to FP16-compute GEMMs
- `53edbb3` Accept hex integer options; const-correct GPU evaluate hooks
- `6493db1` Fix kind 3 (DENSE_MP_BAND) on GPUs and its sender-conversion mode
- `22ff7b5` Report the forward error of --check; the residual decides
- `88c67e4` Fix corner cases found by the parameter sweep
- `719bd2c` Record the BLAS/LAPACKE directories in the run-time paths
- `5022f9f` Make the library safe to embed: argument vector and symbol binding
- `9ffb772` Merge the tile/LAPACK conversion JDFs into one
- `944ba0a` Release readiness: CI passes on the final commit
- `87f0c23` Give the installed library and drivers a runtime search path
- `1d9b7ad` Release readiness: results after the CI fixes
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

Each commit is self-contained in intent;
the trees that were built and tested are the one after the documentation
commit `8f7cb5e` (section 5.3), the one after the CI fixes (section 5.5), the
one after the third pass (`ec39832`; sections 4, 5.1, 5.2, 5.6 and 5.7) and the
final one, which includes the Frontier commits `6e01e6b` to `d6bf679`
(section 5.8: CTest on CPU and CUDA, kind 3 on 2 ranks); the validation
matrix was rerun after each later pass (sections 5.9 to 5.13, 5.16 and
5.17), identical to the third pass on the A100; `a7ecff1` is the tree of
section 5.17, and `b817e7a` the tree of section 5.18.
Intermediate commits were not built individually.

## 3. Validation environment

| Item | Value |
|---|---|
| Machine | 1 node, 2 x Intel Xeon Gold 6140 (72 hardware threads), 1 x NVIDIA A100-PCIE-40GB (compute capability 8.0, driver 580.65.06) |
| Compilers / tools | GCC 11.5.0, CMake 3.30.5, Ninja; CUDA 12.6 (nvcc 12.6.77); ROCm 6.4 `hipcc` (compile only, no AMD GPU) |
| Libraries | Open MPI 5.0.5, Intel MKL 2024.2 (BLAS/LAPACKE), GSL present |
| Submodules | dplasma `ae43716e` (ICLDisco master + stage-in fix + parsec update), parsec `557a710f4` (4.0.2411-249: ICLDisco master `2c16c652e` + 8 fork commits), hcore `78028a59`, stars-h `5d98f2c2` |
| CPU tree | `-DCMAKE_BUILD_TYPE=Release -DHICMA_PARSEC_HAVE_CUDA=OFF -DDPLASMA_PRECISIONS="s;d" -DPARSEC_DIST_SHORT_LIMIT=1` |
| CUDA tree | same plus `-DHICMA_PARSEC_HAVE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80` |
| HIP tree | same plus `-DHICMA_PARSEC_HAVE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a`, compile only here; built and run on Frontier with `install_frontier.sh` (section 5.15) |

The baseline for every "before" number is the same configuration built from
`6e987e3` in a separate worktree (CUDA tree) and from the branch point before
any source change (CPU tree), with the same submodule commits.

## 4. Build results

| Configuration | Result |
|---|---|
| CPU | builds; the only compiler diagnostics are PaRSEC's deprecation notes for `parsec_add2arena`/`parsec_del2arena` (pre-existing, in the wrappers and generated JDF code) |
| CUDA (sm_80) | builds, same diagnostics plus the pre-existing nvcc notes in the `.cu` kernels (the 28 incompatible-pointer warnings of the GPU evaluate hooks are gone) |
| HIP (gfx90a) | builds; not executed (no AMD GPU) |
| CUDA + `-DHICMA_PARSEC_GENOMICS=ON` | compiles (`-DHICMA_PARSEC_GENOMICS=ON` with CUDA, same flags as the CUDA tree); the drivers `testing_syrk`, `testing_KRR`, `testing_KRR_file`, `testing_Kmatrix`, `testing_igemm_summa` are built. Not run: they need the genomics data files. |
| Installed package | `cmake --install` of the CPU and CUDA trees, then `tests/install_consumer` (`find_package(HICMAParsec)`, links `HICMAParsec::hicma_parsec` only) configures, builds and runs `--N 2048 --NB 512 --check 1` (residual 2.69e-16) and the conversion test from both prefixes, with `LD_LIBRARY_PATH` unset |

## 5. Test results

All numbers below come from runs executed on 2026-11-07 and 2026-11-08 against
the final tree of the branch, with `LD_LIBRARY_PATH` unset (logs are in the session's scratch directory; the matrix is
reproducible with `scripts/validation_matrix.sh`).

### 5.1 CTest

This table is the third pass's; the cases added later and the final counts
are in sections 5.12 and 5.13.

| Tree | Command | Result |
|---|---|---|
| CPU | `ctest -L hicma -LE gpu --output-on-failure` | 25 / 25 passed (18 single-rank, 7 two-rank) on methane, and 25 / 25 in an AddressSanitizer build |
| CUDA | `ctest -L hicma --output-on-failure` | 39 / 39 passed on methane (one A100) and on guyot (A100-SXM4): the 25 CPU cases plus `gpu_dense_fp64`, `gpu_dense_adaptive`, `gpu_dense_band2`, `gpu_fp8_static`, `gpu_fp8_sp`, `gpu_runtime_decision`, `gpu_runtime_admem`, `gpu_trmm_adaptive`, `gpu_tlr_mp`, `gpu_tlr_dp_band2`, `gpu_band_dp_sp_hp`, `gpu_band_dp_sp_hp_sender`, `gpu_band_sp_hp`, `gpu_band_hp_fp16acc` |

Every registered case runs with `--check 1`; the mixed-precision cases state
their accepted bound with `--check_tol` (1e-6 for the SP band on CPU, 1e-5
with a half-precision band on GPU, 1e-4 with FP16 accumulation, 1e-7 for the
adaptive and runtime-decision kinds). The two-rank GPU cases of the
validation matrix share the single A100 and pass `--mca device_cuda_memory_use 30`.

### 5.2 Validation matrix

`scripts/validation_matrix.sh` runs 31 cases on a CPU build and 50 on a GPU
build (the 31 CPU cases against the GPU-enabled library, plus 19 GPU cases).
Every case exited 0 before and after. The residuals are identical to the
baseline except for the virus-mesh cases, whose residual was the doubled
diagonal shift (review finding E1); the GEMM precision counts of every case
are identical. Cases added during the work have no baseline column: kind 2 with a dense band
(which exposed the sink-task crash), and kind 3 on GPU, kind 3 with sender-side
conversion, kind 4 with a half band and kind 3 with FP16 accumulation (which
would have exposed the bugs of section 5.6). The tables are from the final
tree; every earlier run of the release gave the same residuals and GEMM counts
for the cases they share.

CPU tree:

| Case | Ranks | Residual before | Residual after | Factorization time after (s) |
|---|---|---|---|---|
| `tlr_dp_k2_p2` | 1 | 1.157838e-16 | 1.157838e-16 | 0.01968 |
| `tlr_mp_k1_band1_adapt` | 1 | 1.736223e-08 | 1.736223e-08 | 0.05574 |
| `tlr_mp_k1_band1_noadapt` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05596 |
| `tlr_mp_k1_dense_adapt` | 1 | 6.605555e-09 | 6.605555e-09 | 0.03023 |
| `tlr_mp_k1_dense_admem` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02852 |
| `tlr_kind2_band1` | 1 | not in the baseline matrix | 5.014459e-09 | 0.05086 |
| `tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.03936 |
| `tlr_kind2_band2_np2` | 2 | not in the baseline matrix | 2.741820e-09 | 0.08401 |
| `tlr_kind1_band2_np2` | 2 | not in the baseline matrix | 1.740042e-08 | 0.08878 |
| `tlr_dp_k2_p4_autoband` | 1 | 1.328506e-10 | 1.328506e-10 | 0.10783 |
| `dense_k5_fp64_cpu` | 1 | 1.856962e-16 | 1.856962e-16 | 0.06393 |
| `dense_k5_band2_cpu` | 1 | 3.099008e-09 | 3.099008e-09 | 0.06403 |
| `dense_k5_adapt_cpu` | 1 | 4.028860e-10 | 4.028860e-10 | 0.09451 |
| `band_k3_cpu` | 1 | 5.377707e-08 | 5.377707e-08 | 0.11429 |
| `band_k3_sender_cpu` | 1 | not in the baseline matrix | 5.377707e-08 | 0.11135 |
| `band_k4_cpu` | 1 | 1.692654e-07 | 1.692654e-07 | 0.10130 |
| `runtime_k10_cpu` | 1 | 3.754839e-09 | 3.754839e-09 | 0.12459 |
| `runtime_k10_cpu_adapt1` | 1 | 3.776331e-09 | 3.776331e-09 | 0.11826 |
| `sparse_k6_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05310 |
| `sparse_k7_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05225 |
| `sparse_k6_np2` | 2 | 5.014459e-09 | 5.014459e-09 | 0.09400 |
| `sparse_k6_mesh` | 1 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.46801 |
| `sparse_k7_mesh_np2` | 2 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.58500 |
| `trmm_fp64` | 1 | PASSED | PASSED | NA |
| `trmm_adapt` | 1 | PASSED | PASSED | NA |
| `trmm_adapt_sr` | 1 | PASSED | PASSED | NA |
| `tlr_dp_k2_np2` | 2 | 1.157838e-16 | 1.157838e-16 | 0.06400 |
| `dense_k5_fp64_np2` | 2 | 1.856962e-16 | 1.856962e-16 | 0.12500 |
| `dense_k5_band2_np2` | 2 | 3.099008e-09 | 3.099008e-09 | 0.12800 |
| `runtime_k10_np2` | 2 | 3.754839e-09 | 3.754839e-09 | 0.15600 |
| `trmm_adapt_np2` | 2 | PASSED | PASSED | NA |

CUDA tree (same library built with CUDA; `gpu_*` cases use the A100):

| Case | Ranks | Residual before | Residual after | Factorization time after (s) |
|---|---|---|---|---|
| `tlr_dp_k2_p2` | 1 | 1.157838e-16 | 1.157838e-16 | 0.02263 |
| `tlr_mp_k1_band1_adapt` | 1 | 1.736223e-08 | 1.736223e-08 | 0.05222 |
| `tlr_mp_k1_band1_noadapt` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05298 |
| `tlr_mp_k1_dense_adapt` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02823 |
| `tlr_mp_k1_dense_admem` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02801 |
| `tlr_kind2_band1` | 1 | not in the baseline matrix | 5.014459e-09 | 0.05144 |
| `tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.03985 |
| `tlr_kind2_band2_np2` | 2 | not in the baseline matrix | 2.741820e-09 | 0.07000 |
| `tlr_kind1_band2_np2` | 2 | not in the baseline matrix | 1.740042e-08 | 0.08974 |
| `tlr_dp_k2_p4_autoband` | 1 | 1.328506e-10 | 1.328506e-10 | 0.10921 |
| `dense_k5_fp64_cpu` | 1 | 1.856962e-16 | 1.856962e-16 | 0.06234 |
| `dense_k5_band2_cpu` | 1 | 3.099008e-09 | 3.099008e-09 | 0.06970 |
| `dense_k5_adapt_cpu` | 1 | 4.028860e-10 | 4.028860e-10 | 0.09461 |
| `band_k3_cpu` | 1 | 5.377707e-08 | 5.377707e-08 | 0.11190 |
| `band_k3_sender_cpu` | 1 | not in the baseline matrix | 5.377707e-08 | 0.11130 |
| `band_k4_cpu` | 1 | 1.692654e-07 | 1.692654e-07 | 0.10134 |
| `runtime_k10_cpu` | 1 | 3.754839e-09 | 3.754839e-09 | 0.12299 |
| `runtime_k10_cpu_adapt1` | 1 | 3.776331e-09 | 3.776331e-09 | 0.11988 |
| `sparse_k6_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05437 |
| `sparse_k7_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05349 |
| `sparse_k6_np2` | 2 | 5.014459e-09 | 5.014459e-09 | 0.09500 |
| `sparse_k6_mesh` | 1 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.47127 |
| `sparse_k7_mesh_np2` | 2 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.56000 |
| `trmm_fp64` | 1 | PASSED | PASSED | NA |
| `trmm_adapt` | 1 | PASSED | PASSED | NA |
| `trmm_adapt_sr` | 1 | PASSED | PASSED | NA |
| `tlr_dp_k2_np2` | 2 | 1.157838e-16 | 1.157838e-16 | 0.05600 |
| `dense_k5_fp64_np2` | 2 | 1.856962e-16 | 1.856962e-16 | 0.13470 |
| `dense_k5_band2_np2` | 2 | 3.099008e-09 | 3.099008e-09 | 0.11880 |
| `runtime_k10_np2` | 2 | 3.754839e-09 | 3.754839e-09 | 0.20300 |
| `trmm_adapt_np2` | 2 | PASSED | PASSED | NA |
| `gpu_dense_k5_fp64` | 1 | 3.602581e-16 | 3.602581e-16 | 0.13402 |
| `gpu_dense_k5_adapt` | 1 | 2.176706e-09 | 2.176706e-09 | 0.17915 |
| `gpu_dense_k5_band2` | 1 | 2.804213e-09 | 2.804213e-09 | 0.20102 |
| `gpu_dense_k8_adapt` | 1 | 2.176706e-09 | 2.176706e-09 | 0.18067 |
| `gpu_dense_k9_adapt` | 1 | 3.313729e-07 | 3.313729e-07 | 0.15448 |
| `gpu_runtime_k10` | 1 | 1.041065e-08 | 1.041065e-08 | 0.28366 |
| `gpu_runtime_k10_admem` | 1 | 1.041065e-08 | 1.041065e-08 | 0.24876 |
| `gpu_tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.09078 |
| `gpu_tlr_kind1_band2` | 1 | not in the baseline matrix | 1.725825e-08 | 0.09851 |
| `gpu_tlr_mp_k1` | 1 | 5.551907e-09 | 5.551907e-09 | 0.22843 |
| `gpu_dense_k5_dtconv2` | 1 | 2.176706e-09 | 2.176706e-09 | 0.18273 |
| `gpu_trmm_adapt` | 1 | PASSED | PASSED | NA |
| `gpu_band_k3_hp` | 1 | not in the baseline matrix | 8.462557e-07 | 0.24778 |
| `gpu_band_k3_hp_sender` | 1 | not in the baseline matrix | 8.480405e-07 | 0.26568 |
| `gpu_band_k4_hp` | 1 | not in the baseline matrix | 9.075389e-07 | 0.27196 |
| `gpu_band_k3_fp16acc` | 1 | not in the baseline matrix | 2.337890e-05 | 0.28367 |
| `gpu_dense_k5_fp64_np2` | 2 | 3.602581e-16 | 3.602581e-16 | 0.31146 |
| `gpu_runtime_k10_np2` | 2 | 1.041065e-08 | 1.041065e-08 | 0.48400 |
| `gpu_dense_k8_np2` | 2 | 2.176706e-09 | 2.176706e-09 | 0.39605 |

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

- HIP on AMD GPUs other than gfx90a (MI210 and MI250X ran, section 5.15), and
  more than one MI250X node. The changes of sections 5.17 and 5.18 (shared
  low-rank CPU GEMM, placement rule, kinds 8 and 9 on their own kernels
  again) compile for gfx90a but were not run on AMD GPUs. (Later: gfx906,
  one and two Radeon VIIs per node and two nodes, sections 5.20 and 5.21.)
- More than one node. (Several GPUs per node: guyot, section 5.6; FP8 on an
  H100: section 5.13. Later: two nodes with Radeon VIIs, section 5.21, and
  with a V100 and an A100, section 5.22.)
- The climate emulator driver (`testing_climate_emulator`): compiled in every
  configuration, not run (needs the external data set).
- The genomics drivers: see section 4; not run (data files).
- The Fugaku (`ON_FUGAKU`) build.
- The machine build scripts on their machines: only `install_icl.sh`
  (methane, section 5.8) and `install_frontier.sh` (Frontier, section 5.15)
  were run as written; the others were configured on methane with their
  modules stubbed, and the Fugaku and Shaheen scripts not at all (section 5.8).
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
  `LD_LIBRARY_PATH` (CPU and CUDA prefixes). CMake did not record MKL's
  directory then; the second pass records it (`719bd2c`, section 5.6).

Re-validation after these commits: CPU, CUDA and HIP trees build; CPU `ctest`
20 / 20; CUDA-tree `ctest` 30 / 30; install consumers from both prefixes
(residual 2.69e-16); the 75-case matrix identical to the baseline except the
intended mesh change (tables in 5.2). Both CI jobs of pull request
ecrc/hicma-x#13 pass on the final commit (CPU: build, CTest, regression
script, install and consumer; CUDA: compile).

### 5.6 Second pass: external user and corner cases (guyot)

Requested after the first review: merge the tile/LAPACK conversion JDFs (item
1 of `TODOs`), look at the project as an external user, and test on guyot (8 x
A100-SXM4-80GB, 256 cores) looking for corner cases.

**Conversion JDFs.** The eight JDFs are one (`lapack_tile_convert.jdf`, 1433
lines became 280) with `hicma_parsec_lapack_tile_convert` as the entry point
and the named functions as wrappers. A new round-trip test
(`testing_lapack_tile`) passed on 1 to 6 processes on methane and 8 on guyot,
and from the installed package.

**External user.** A fresh HTTPS clone with `--recursive` fetched every
pinned submodule commit (all are public), and BUILD.md followed verbatim built
the default configuration. Then:

- every test failed to start: "libmkl_gf_lp64.so.2: cannot open shared object
  file" (the MKL module sets `LIBRARY_PATH` only, whose directories CMake never
  records in a run-time path; the validation scripts had exported
  `LD_LIBRARY_PATH`). Fixed: the BLAS directories are recorded in the built,
  installed and consumer binaries; every run of this section had
  `LD_LIBRARY_PATH` unset;
- an application linked against the installed package crashed in
  `hicma_parsec_init` with `argc = 0, argv = NULL` and with read-only argument
  strings, and an application function named `parse_arguments` replaced the
  library's (it then died on a division by zero). Fixed (argv handling,
  `-Bsymbolic-functions`);
- the CMake test presets ran DPLASMA's and PaRSEC's suites too; BUILD.md
  pointed at nonexistent `HCORE_DIR`/`STARSH_DIR` variables and had no word on
  calling the library from an application. Fixed.

**Corner-case sweeps.** Each case was classified as passed, refused cleanly
(non-zero exit with a message), crashed or hung (timeout):

| Sweep | Machine | Cases | Passed | Refused as expected | Unexpected |
|---|---|---|---|---|---|
| Sizes, options, kinds, grids, 1 to 6 ranks | methane (CPU) | 61 | 44 | 17 | 0 (7 before the fixes) |
| One process, 8 GPUs | guyot | 33 | 28 | 0 | 5: the kind-3 crash, 4 accuracy-envelope cases |
| 8 ranks x 1 GPU | guyot | 20 | 18 | 0 | 2 accuracy-envelope cases |
| 4 and 3 ranks x 2 GPUs | guyot | 6 | 6 | 0 | 0 |
| Round 2 (kinds 3, 4, envelope cases with bounds), 8 GPUs / 8 ranks / 4 x 2 | guyot | 20 | 19 | 0 | 1: FP16 accumulation, fixed |
| Round 3 (FP16 accumulation), 8 GPUs / 8 ranks | guyot | 4 | 1 | 0 | 3: residual 1.6e-4 against a 1e-4 bound, the FP16-accumulation envelope at N=16384 |

The cases covered every kind, fewer tiles than GPUs or ranks (NT = 1, 2, 3
with 8 GPUs or 8 ranks), more GPUs requested than present, 1 and 2 cores with
8 GPUs, recursive kernels (HNB < NB), repeated runs, GPU memory pressure
(`device_cuda_memory_use` 2 to 5 % with evictions), N = 65536 (kind 10 in
0.91 s, kind 5 in 1.07 s on 8 A100s, one run each), and grids 1 x 8, 8 x 1,
2 x 4, 4 x 2. Results are reproducible across layouts: the same case gives the
same residual on 1 process with 8 GPUs and on 8 ranks.

**Bugs found and fixed** (all present before this release cycle; the
evaluate-hook warnings were 14 then and doubled with the wrapper
consolidation):

| Found by | Problem | Effect before |
|---|---|---|
| guyot | kind 3 with a half-precision band on any GPU: `[type = HALF]` input made PaRSEC reshape the FP32 tile into half its size | `MPI_ERR_TRUNCATE` abort |
| guyot | `--tensor_gemm 0x10`: FP16-compute GEMMs given float alpha/beta (kind 3, low-rank GEMMs of kinds 1, 2, 5, 8 to 10) | wrong results (residual 5e+1) |
| methane | kind 3 `--datatype_convert 1`: rewritten to 2 by the send-decision step, mixed tests in the graph | segmentation fault; after the first fix, wrong results on GPU |
| CPU sweep | sparse kinds with the band covering the matrix (e.g. one tile row) | segmentation fault |
| CPU sweep | invalid `--kind_of_cholesky` | error printed, exit 0 |
| CPU sweep | negative `--N`/`--NB` | silently replaced by defaults |
| embedding | `hicma_parsec_init(0, NULL, ...)`, read-only argv, interposed `parse_arguments` | segmentation faults |
| fresh clone | binaries without the BLAS run-time path | tests could not start |
| check review | the forward-error comparison of `--check` was never part of the verdict (its result was overwritten) | now reported as informational, see `TESTS.md` |
| user | hex option values (`--tensor_gemm 0x10`) | rejected |
| build | 28 incompatible-pointer warnings from the consolidated wrappers' evaluate hooks | warnings |

**Accuracy envelope.** Static mixed precision at N = 16384 gives a residual of
2.3e-8 (above `--fixedacc 1e-8`, inside 1e-7), the half band of kinds 3 and 4
3.6e-6 at N = 16384 (8.5e-7 at N = 8192), FP16 accumulation 1.6e-4 at
N = 16384 (2.3e-5 at N = 8192). These grow with N; the registered cases use
bounds measured at their own sizes.

**Fresh clone, final tree.** The walkthrough was repeated on the pushed
branch: HTTPS clone with `--recursive`, `cmake .. -G Ninja` without options,
`ninja`, CPU CTest 23 / 23, `cmake --install`, and the install consumer
(Cholesky driver and conversion test) from the prefix, all with
`LD_LIBRARY_PATH` unset. Both CI jobs of ecrc/hicma-x#13 pass on the commits
of this section.

**Harness note.** Inside a `srun --overlap` step, an Open MPI 5 singleton (a
one-rank test started without `mpirun`, as CTest does) waits forever on the
step's PMIx server; the guyot runs clear the `SLURM_*`/`PMIX_*` variables. Not
a HiCMA-X issue, but worth knowing when running CTest inside an allocation.

### 5.7 Third pass: matrix sum, compile-time switches, sanitizers (methane, guyot)

Requested: merge the matrix sums for multiple precisions (`TODOs` item 3),
check the design of the macros in `hicma_parsec_internal.h` for correctness
and performance without adding CMake options, keep the code changes minimal,
test on guyot.

**Matrix sum.** `hicma_parsec_matrix_sum` (`matrix_sum.jdf`) replaces the
removed FP64-only and FP32-only sums, whose tile task never stored its partial
sum. It reads each tile in its storage precision (decision map, or the
descriptor's type) including low-rank factors, and returns the same value on
every process and grid. `testing_matrix_sum` passes exactly on 1 to 8
processes; on generated kind-1, 2 and 5 matrices the sums are bitwise equal on
1, 2 and 4 processes.

**Compile-time switches** (`code-review.md`, section 8): switches settable with
`-DNAME=value`; every value compiled and run; dead ones removed; two broken
non-default paths found. `GPU_BUFFER_ONCE=0` crashed (it freed the shared GPU
workspace) and was slower: TRMM N=40960 on 8 A100s 11.88 s against 11.31 s
(3 runs each), so the switch was removed. `WORKLOAD_BALANCE=4` corrupted the
heap; fixed.

**Sanitizers.** An AddressSanitizer build (`-DENABLE_SANITIZER=ON`) ran the 25
CPU CTest cases and the 61-case CPU corner sweep with no report. Its leak
checker found that PaRSEC's `parsec_private_memory_fini` frees no pooled
workspace: every factorization lost its pools (167 MB for kind 2 at N=2700, 24
MB for kind 10). HiCMA-X now frees them itself (`hicma_parsec_memory_pool_fini`);
afterwards nothing leaked is allocated by HiCMA-X code (what remains belongs to
MPI and PaRSEC), and repeated factorizations no longer grow:

| Kind 10, N=8192, CPU | 1 run | 4 runs | 8 runs |
|---|---|---|---|
| Peak memory before | 532 MB | 839 MB | 1238 MB |
| Peak memory after | 525 MB | 568 MB | 605 MB |

Remaining one-off leak: the TRMM driver ends with
`hicma_parsec_cleanup_parsec`, which does not free 3.8 KB of statistics
arrays (the Cholesky path does).

**Validation of the final tree.** methane: CTest 25 / 25 (CPU) and 39 / 39
(CUDA); the 31 CPU and 50 GPU-tree matrix cases identical to the previous run
(residuals and GEMM counts; times within noise); install consumers and the
embedding applications pass. guyot: CTest 39 / 39 (twice); `testing_matrix_sum`
and `testing_lapack_tile` on 8 ranks; after the `GPU_BUFFER_ONCE` removal, the
TRMM (N=40960, 8 GPUs; 8 ranks; memory pressure), kind 5 with adaptive
conversion on 8 GPUs and on 8 ranks (also 3 repeated runs), and kind 10 with 3
repeated runs on 8 ranks all pass. HIP compiles.

### 5.8 Fourth pass: build scripts (methane)

Requested: bring the out-of-date install scripts up to date.

**Scripts.** The machine scripts built and installed STARS-H and HCORE before
the main build, which compiles both from the submodules anyway, and had
drifted further (CHANGELOG lists each case). They now configure the whole tree
once with the site's modules and options and take `[build-dir] [extra CMake
options]`. `scripts/machines/legacy/` is removed. `build_hicma_with_lapack.sh`
builds a pinned reference LAPACK and passes the search hints that STARS-H's
own modules need.

| Run on methane | Result |
|---|---|
| `install_icl.sh`, fresh build (ICL modules, MKL, CUDA 12.6) | builds; CTest 39 / 39 on the A100 |
| `build_hicma_with_lapack.sh`, MKL unloaded | LAPACK 3.12.1 built; HiCMA-X links only it; CTest 25 / 25 (CPU) |
| `build_release.sh --test` | CTest 25 / 25 (CPU) |
| Alps, Frontier (ROCm 6.4), Leonardo, Polaris, TACC, Vista: configure with `module` stubbed | all configure; CMake reports no unused variable |
| `install_icl_debug.sh`, `install_leonardo_no_gsl.sh`, `install_shaheen_slurm.sh` | forward their arguments; a later `-D` wins |

**Found while validating.**

- *CUDA architectures.* A build given only `-DHICMA_PARSEC_HAVE_CUDA=ON`
  compiled its CUDA code for sm_52 alone. `enable_language(CUDA)` cached the
  compiler's default before the documented default `70 80 90 90a` could
  apply; CI and the earlier validation always passed the architecture. Fixed:
  a fresh configure gives `70;80;90;90a` (`80;90;90a` with CUDA 13), and a
  user or `CUDAARCHS` value is kept.
- *Kind 3 guards.* The kind-3 fix of the second pass (`6493db1`) used the C
  macro `SENDER_CONVERT` in 17 JDF dependency guards. The JDF compiler printed
  17 "Fatal Error ... unbound variable" lines per build while the generated
  code was right. A hidden JDF global replaces it. The CTest kind-3 cases pass,
  and so do 2 ranks with `--datatype_convert` 0, 1 and 2, with and without
  the GPU.

**GEMM reordering kept (on request).** `reorder_gemm.jdf` and
`hicma_parsec_reorder_gemm`, removed as dead code in `bac0083`, are back and
still not called. Their launch now compiles, the sort compares the keys it
moves, and the work pool is freed. A scratch driver ran it after the
pre-analysis in 19 configurations (1, 2 and 4 ranks; reorder 1, 2, 4; band
1 and 3; CPU and CUDA trees). In each, all 741 order chains were complete
and sorted, with the same counts on every process count. CTest: 25 / 25
(CPU) and 39 / 39 (CUDA). Wiring it into the Cholesky is future work
(`reorder_gemm.jdf` lists the steps).

### 5.9 Fifth pass: macros and data structures (methane)

Requested: check that the macros and data structures are well designed and
necessary; guyot not used. `code-review.md`, section 9, has the findings.

- **Removed as unused.** 16 datatype identifiers, three tile formats, three
  sparse modes, `DATATYPE_USE`, `EPS_DP`, the generic `str`/`xstr`, 29 HIP
  aliases, the JDF `*_SHAPE` macros, and five struct fields (`HMB`,
  `start_time_kernel_sqr_sum_vec`, `str_problem`, and in the climate
  emulator twelve more). A memory check that could not fire is also gone.
- **Fixed.** `SYNC_TIME_PRINT` was defined twice. The local switches of the
  sources and of the climate emulator could not be set with `-D`. The
  climate header `#undef`d three genomics switches, and its switches had two
  names each. Two climate debug paths did not compile.

| Check | Result |
|---|---|
| CPU, CUDA and HIP trees | build, no new warnings |
| CTest | 25 / 25 (CPU), 39 / 39 (CUDA) |
| Validation matrix against the third pass | 31 CPU and 50 GPU-tree cases: identical exit codes, residuals and GEMM counts |
| Every local switch flipped (CPU and CUDA trees) | CTest 25 / 25 and 39 / 39; flipped prints appear |
| Every climate switch value | compiles in both trees (16 sources) |
| Install consumers, `LD_LIBRARY_PATH` unset | build and pass |

Left as they are, with the reasons in section 9 of the review: the
implicit-capture macros `IS_DENSE`/`IS_ALLOCATE_DP`, the generic object-like
names of the installed header, and the 196-field parameter struct.

**Unnamed `#if 0`/`#if 1` blocks (cleaned on request).** All 28 are gone.
Blocks around live code were unwrapped. Debug dumps, code that could not
compile and superseded alternatives were deleted. The genomics SYRK
alternative is now the switch `GENOMICS_SYRK_ISYRK` (review, section 9,
item 8, lists each block). The compression tasks also lost a scan of every
tile that only fed a dead `1 || flag` test. After the cleanup, the CPU, CUDA
and HIP trees build, both values of `GENOMICS_SYRK_ISYRK` compile, CTest
passes (25 / 25 CPU, 39 / 39 CUDA), and the 31 CPU and 50 GPU-tree matrix
cases are again identical to the third pass.

### 5.10 CI hardening

One CI run (commit `f4e8512`, documentation only) failed 21 of 25 CPU cases
inside DPLASMA's reference Cholesky and the LAPACK compressions. A run on the
same code, image and packages 10 seconds earlier passed, so the cause is the
runner, which the log does not name.

- Every HiCMA CTest case now requires DPLASMA's BLAS thread-safety test as a
  fixture. A BLAS that cannot take concurrent calls reports "Not Run"
  instead of wrong residuals.
- The fixture executable gets the BLAS run-time path. On methane's MKL
  module it could not start, which would have skipped every case.
- The CPU job prints the runner's CPU and OpenBLAS variant and sets
  `OPENBLAS_NUM_THREADS=1`.

On methane, CTest passes 26 / 26 (CPU) and 40 / 40 (CUDA), the fixture
included.

### 5.11 Sixth pass: external user, usability and performance (methane)

**Fresh clone.** The pushed branch (`c912875`) cloned with `--recursive`
(every submodule clones anonymously) and built by following `BUILD.md`'s
Quick Start with `-DHICMA_PARSEC_HAVE_CUDA=ON`: configure 32 s, build 33 s
(24 jobs), `ctest -L hicma` 40 / 40. Only the sequential MKL reaches the
binaries, although one submodule's BLAS search also reports the threaded one.
CI on `c912875` passed (AMD EPYC 7763 runner, pthread OpenBLAS).

**Performance**, one process, factorization time only, DPLASMA's
`testing_dpotrf` as the reference (same PaRSEC, same BLAS):

| Case | DPLASMA | HiCMA-X |
|---|---|---|
| 36 CPU cores, dense FP64, N=30720, NB=512, three interleaved runs | 7.24–7.54 s (1.28–1.34 Tflop/s) | 7.21–7.37 s (1.31–1.34 Tflop/s) |
| One A100, dense FP64, N=30720, NB=1024, two runs | 1.42 s (5.7 Tflop/s) | 1.05 s (9.2 Tflop/s) |
| 36 CPU cores, TLR FP64 (`--kind_of_cholesky 2 --band_dense 1 --fixedacc 1e-8 --maxrank 150`), N=30720, NB=512 | | 1.10 s |

On the A100, mixed precision (`--adaptive_decision 1`) reached 9.6 Tflop/s
and the runtime decision (kind 10) 9.3; the FP64 GEMMs already run on the
tensor cores. A later study of lookahead and tile size (section 5.13) found
their effect within the run-to-run noise; no default changed.

**Found and fixed** (`code-review.md` section 10):

- `--nruns 2` with a TLR kind segfaulted in run 2, which factorized run 1's
  factor. Configurations whose matrix cannot be regenerated now do one run
  and warn.
- Three silent performance traps, each now reported by one note on rank 0:
  - a GPU build without `--gpus` runs on the CPU (1.3 instead of 9.2 Tflop/s
    above);
  - kinds 1 and 2 without `--band_dense` factorize every tile dense (7.2 s
    instead of 1.1 s at N=30720), while `--help` gave the default as 1;
  - Open MPI binds each process to one core, so PaRSEC starts one worker
    each. With two processes on one node, TLR at N=16384 took 11.5 s with
    the default binding, 1.28 s with `--bind-to none` and 1.04 s with
    `--map-by ppr:1:package:pe=18` (one process: 1.09 s). Open MPI 5 refuses
    `ppr:1:node:pe=36` here (a binding across sockets).
- Wrong help defaults (`--band_dense`, `--N`, `--NB`, `--HNB`, `--cores`).
- The summary line printed twice per run, the first copy incomplete; dense
  runs printed uninitialized analysis counters in it; the revision always
  read `N/A`. Eight internal notes printed in colour on every run; they now
  need `--verbose 1`. Two invalid-option exits returned 0.
- The README had no quick start; `build_release.sh --help` named the wrong
  default build type.

**Not changed**: the defaults `--gpus 0` and a dense band of NT for kinds 1
and 2 (kept by the maintainer's decision, section 7); the default build type,
RelWithDebInfo, is within 2 % of Release (`-O3`), same source, three
interleaved runs each:
dense FP64 7.1–7.8 s against 7.1–7.6 s, TLR factorization 1.10–1.11 s
against 1.08–1.10 s, generation equal.

After the changes: CTest 26 / 26 (CPU) and 40 / 40 (CUDA); the validation
matrix (31 CPU and 50 GPU-tree cases) is identical to the third pass in exit
codes, residuals and GEMM precision counts; the HIP tree builds.

### 5.12 Seventh pass: GPU GEMM variants, then three rounds of checks (methane)

`--tensor_gemm` redesigned (`code-review.md` section 11): `--tensor_gemm_hp`
and `--tensor_gemm_sp`, one check at start-up, one variant read by every GEMM
body. Before the change, on one A100: kind 3 gave residual 43 with `0x3`,
`0x18`, `0x20` and `0x40`; kind 5 aborted mid-run with `0x3` and `0x18`;
adaptive TLR with `0x10` segfaulted. After it, those values are refused at
start-up (the adaptive TLR case passes, 2.2e-5), and the variants measure as
in `TESTS.md` ("GPU GEMM variants").

**Round 1** (regression): CTest 27 / 27 (CPU) and 45 / 45 (CUDA), with the
new cases; the validation matrix identical to the third pass, so the default
computes as before. Every option form checked: `--tensor_gemm_hp` and
`--tensor_gemm_sp` override the parts of `--tensor_gemm`, invalid values are
refused with the option named; on the CPU, where the variants do not apply,
valid values pass and refused ones are refused alike.

**Round 2** (robustness, and an independent review of the change by a
separate agent, `code-review.md` section 11):

- The review found five more paths where the variant was unchecked or two
  GEMMs disagreed. Confirmed by runs and fixed: BF16 with
  `--datatype_convert 2` (residual 84, now refused); the TRMM drivers skipped
  the value check and ignored `--tensor_gemm_hp 0x80` (the merge moved into
  `parse_arguments`); FP8 C under 0x10 (kind 8, compute capability 8.9 or
  more, not run here); the `ENABLE_TF16_A16_B16_C16_OP16` build switch;
  rank-0 operands under 0x10. Two of its notes did not hold: FP8 is decided
  after kind 10 is set, and `--datatype_convert 2` in kinds 1 and 4 gave the
  residuals of `--datatype_convert 0` (8.5e-7, 9.1e-7). Its note that
  `hicma_parsec_params_init` checks the HIP kinds before `--auto_band`
  changes the kind needs an AMD GPU.
- Two ranks sharing the GPU (`--map-by ppr:1:package:pe=18`, 20 % of the
  device memory each): kinds 3, 4, 5 and 10 with every variant they
  implement, adaptive dense and TLR, and the TRMM with 0x80, 0x8 and 0x10
  plus `--accumulation_fp32` all pass `--check`; the refused values exit 1
  with the message.
- Found: a static FP16 or FP8 band with low-rank tiles fails on GPUs in kinds
  1 and 5 (segfault in cuBLAS, or residual 59), also with 0.1.0's code;
  FP64 allocation does not avoid it, adaptive decisions do. First refused
  before the factorization, then root-caused and fixed (below).
- AddressSanitizer build: CTest 27 / 27 (CPU). CTest 27 / 27 (CPU) and
  46 / 46 (CUDA); the validation matrix identical to the third pass; the HIP
  tree builds.

**Round 3** (the pushed result, from outside): a fresh clone of `3fd5cda`
built as `README.md` says, with CUDA, passes `ctest -L hicma` 46 / 46; the
installed tree builds `tests/install_consumer`, whose drivers pass with
`LD_LIBRARY_PATH` unset (`--tensor_gemm_hp 0x20` included) and print
`HiCMA-X 0.2.0, revision v0.1.0-130-g3fd5cda`. CI passed on `3fd5cda`.
Performance at N=30720, two runs each, against the sixth pass (section
5.11): one A100, FP64 9.3 and 9.1 Tflop/s (9.2), adaptive 9.5 and 9.5
(9.6), kind 10 9.1 and 9.0 (9.3); 36 CPU cores, FP64 1.36 and 1.35
(1.31-1.34), TLR 1.08 s (1.10 s): unchanged within noise, with the node's
load average at 50 to 230 from other jobs. The variants on the A100 (FP64
and FP32 bands of 2 tiles, the rest FP16): 0.70 s with 0x8, 0.66 s with
0x10, 0.66 to 0.69 s with 0x20, 0.77 s with 0x40; an FP32 band instead takes
0.56 s with TF32 and 0.90 s without.

**The FP16-band failure, root-caused** (on request): a debug build with the
refusal disabled crashed inside cuBLAS's host-side heuristics at a different
GEMM from run to run; at N=4096 the result was deterministically wrong
(residual 22 at every `--fixedacc`, ranks not growing), and a per-tile
comparison with the dense reference factor put the first wrong tiles at the
FP16 tiles whose first update is low-rank x low-rank. The FP16 branch of
`hicma_parsec_core_gemm_denseC_lrA_lrB_gpu` converted that low-rank B as a
dense tile (`code-review.md` section 11). With B converted as U and V, every
failing layout passes: residuals 1.8e-5 to 7.9e-5 at `--fixedacc 1e-4`
(FP32 band 2.1e-5, the same layout on the CPU 3.5e-5), 1.7e-4 with 0x10,
two ranks equal to one, kind 1 equal to kind 5. The refusal is removed and
CTest runs the layout in kinds 5 and 1 (`gpu_tlr_hp_band`,
`gpu_tlr_mp_hp_band`). CTest 27 / 27 (CPU) and 47 / 47 (CUDA), the
validation matrix identical to the third pass, AddressSanitizer 27 / 27, the
HIP tree builds.

### 5.13 Eighth pass: the H100, and two more rounds (hexane, methane)

**hexane, H100 PCIe (compute capability 9.0, FP8 on)**, a tree built with the
default architectures (70 80 90 90a), run inside the user's SLURM allocation
with only the H100 visible:

- CTest 48 / 48 and `scripts/test_CI.sh` 48 passed, 1 skipped (47 / 47 before
  the FP8 case was added); the GPU validation matrix (50 cases) passes, twice
  with identical results; the TRMM with 0x80 and 0x8 passes.
- FP8 is used: dense static FP8 band 8.4e-7 with 1 FP8 GEMM (`--fp8 0`: 0),
  kind 10 7.5e-9 with 3; two ranks sharing the H100 give the one-rank
  residuals.
- Found and fixed: FP8 operands read as FP64 by the low-rank GEMMs
  (`code-review.md` section 12): before, residual 1.5e-2, 3.6e-2 or a crash;
  after, 1.6e-5, 4.9e-5, 1.8e-4. The FP16-band fix of section 5.12 holds on
  the H100 (4.4e-5).
- FP8 numerics: a static FP8 band from tile distance 3 to 6 makes the test
  matrix indefinite (`info` > 0, residual 60 to 75), from distance 8 on it
  passes (1.2e-4 to 9e-6), `--adddiag 100` restores distance 3 (3.3e-5).
  Documented, not a defect.

Performance on the H100 (N=30720, NB=1024, bound to NUMA node 0, two runs):

| Case | Time | Rate |
|---|---|---|
| DPLASMA `dpotrf -g 1` | 0.64 s | 11.1 to 12.1 Tflop/s |
| kind 5 FP64 | 0.53 to 0.54 s | 17.8 to 18.1 |
| kind 5 adaptive | 0.58 s | 16.6 to 16.8 |
| kind 8 adaptive | 0.59 s | 16.3 to 16.4 (`--fp8 0`: 15.4 to 16.2) |
| kind 10 run-time decisions | 0.67 to 0.71 s | 13.6 to 14.3 |
| kind 5 FP16 band (0x8) | 0.48 to 0.54 s | 17.8 to 20.3 |
| kind 5 FP32 band with TF32 | 0.40 to 0.41 s | 23.6 to 24.3 |

On both GPUs the FP32 band with TF32 is the fastest; kind 10 pays for its
per-GEMM decisions where FP64 GEMMs are as fast as on the H100.

**Round 1** (checks on methane): CTest 27 / 27 (CPU) and 48 / 48 (CUDA,
with `gpu_tlr_fp8_band`, which runs FP16 there), the validation matrix
identical to the third pass, `scripts/test_CI.sh` 48 passed and 1 skipped
(the climate data set), the HIP tree builds.

**Round 2** (performance and release consistency): no device-wide
synchronization or allocation in per-task GPU code; on the A100, lookahead
(0, 1, 2) and tile size (1024, 2048) moved the dense time within the noise
(up to 10 % between repeats at a load average of 15), so no default changed.
A documentation audit against the code found 21 statements to fix; 20 were
confirmed and corrected (section 12 of `code-review.md`). A fresh clone of
`c2aef96` (MKL, CUDA, README quick start) passes CTest 48 / 48, installs, and
its consumer drivers pass with `LD_LIBRARY_PATH` unset. Against OpenBLAS
0.3.28 on the same node, `-DBLA_VENDOR=OpenBLAS` stopped the configure step
in STARS-H's find module, which names it `Open`; with the translation added
to the top-level CMake, every component links OpenBLAS only and CTest passes
48 / 48 (CUDA) with `OPENBLAS_NUM_THREADS=1`, the thread-safety check
included.

### 5.14 A complete test on hexane (H100 and V100)

`7ac56e1` on hexane (2 x Xeon Silver 4309Y, 16 cores; GPU 0 an H100 PCIe on
NUMA node 0, GPU 1 a V100 16 GB on node 1), inside the user's SLURM
allocation, a tree built with the default architectures.

**Correctness**

| Configuration | CTest | `test_CI.sh` | Validation matrix | Other |
|---|---|---|---|---|
| H100 alone | 48 / 48 | 48 passed, 1 skipped | 50 / 50 | |
| V100 alone (no FP8, no BF16 tensor cores) | 48 / 48 | | 50 / 50 | BF16, TF32, FP16 band with low-rank tiles, kind 8 (FP16 there) pass |
| both GPUs, one process (`--gpus 2`) | | | | kinds 5 (FP64, adaptive), 10, 3, 2 (TLR), 8, FP16 band with low-rank tiles, the TRMM pass; FP8 off with its note |
| two ranks, one GPU and its NUMA node each | | | | kinds 5, 10, FP16 band with low-rank tiles, adaptive TLR with 0x10, the TRMM pass |

**Performance** (two runs each; ranges)

| Case | HiCMA-X | DPLASMA |
|---|---|---|
| H100, dense FP64, N=30720, NB=1024 | 0.53 to 0.55 s (17.5 to 18.2 Tflop/s) | 0.64 to 0.65 s (9.3 to 12.0) |
| H100, same, NB=2048 | 0.50 to 0.52 s (18.5 to 19.4) | |
| H100, FP32 band with TF32 | 0.37 to 0.41 s (23.5 to 26.2) | |
| V100, dense FP64, NB=1024 | 1.64 to 1.65 s (5.9) | 1.77 s (4.9) |
| V100, FP16 band | 0.71 to 0.73 s (13.3 to 13.6) | |
| H100 + V100, two ranks | 2.28 to 2.30 s | 2.44 to 2.49 s |
| H100 + V100, one process | 4.70 to 4.75 s | 3.61 to 3.75 s (`-g 2`) |
| 16 CPU cores, dense FP64, N=20480, NB=512 | 7.77 to 7.79 s (0.37) | 7.74 to 7.77 s (0.37, `-g 0`) |
| 16 CPU cores, TLR, N=30720 | 2.0 s | |

A first CPU comparison was void: DPLASMA's `testing_dpotrf` uses every
visible GPU unless given `-g 0`. Mixed GPUs go at the pace of the V100 in
both libraries (the tiles are split evenly); the H100 alone is four times
faster. On one GPU at N=30720 the dense FP64 Cholesky beats TLR (0.53 s
against 1.64 s): the low-rank updates run on the 8 cores of the NUMA node.

**Ease of use**, as a new user: the no-argument run, `--help` (77 options,
the 10 kinds), the `README.md` example, the notes for `--gpus 0` and for a
TLR kind with every tile dense, the messages for a misspelled option, `N %
NB`, invalid `--tensor_gemm` values and `--left_looking -1` all behave. Two
gaps, fixed: `--gpus 4` with one usable GPU ran on one without a word, and
two processes each taking all 16 cores (`--oversubscribe`) went unnoticed.
Both print a note on rank 0 now; checked on methane and hexane, and CTest
still passes (27 / 27 and 48 / 48 on methane, 48 / 48 on the H100).
### 5.15 Ninth pass: Frontier, HIP on MI210 and MI250X (two rounds)

**Environment.** OLCF Frontier. The login node has one AMD Instinct MI210
(gfx90a, 64 GB); compute nodes have four MI250X (eight GCDs, gfx90a), used
through the debug queue. PrgEnv-amd 8.6.0 (amdclang/amdflang 6.2.4), ROCm
6.2.4, Cray MPICH 8.1.31, OpenBLAS 0.3.30, CMake 3.31.11, Ninja 1.13.2. The
tree was built with `scripts/machines/install_frontier.sh` as written (the
first run of a machine script on its machine after methane's `install_icl.sh`);
a fresh recursive clone of the committed tree builds the same way in 2m48s.

**Round 1: review, first HIP GPU runs.** Two independent reviews of the 27
commits since `d6bf679` found no wrong-result defect in the source and these
build, script and documentation problems, all fixed (`34acdce`): the machine
scripts took a CMake option as the build directory; the
`--band_size_termination` help gave a wrong default; CHANGELOG and two other
places still said a BLAS outside the system paths needs `LD_LIBRARY_PATH`; the
headers said version 1.0.0; the launchers passed Open MPI's `--oversubscribe`
to any `mpirun`; `potrf_autoband_3d` ran only in `test_CI.sh`; `--tensor_gemm_sp
0x4` ran FP32 silently on HIP; the kind-5 and kind-8 GPU cases were registered
for CUDA only. The first HIP GPU CTest run (MI210) passed 26 of 28: kind 10
(`gpu_runtime_decision`, `gpu_runtime_admem`) gave 1.3e-6 where the A100 gives
1.0e-8. A hipBLAS test showed why: `hipblasGemmEx` with FP16 operands returns
exactly 0 for entries around 1e-5 on gfx90a, which flushes FP16 subnormals
(|x| < 6.1e-5) to zero; `HIPBLAS_GEMM_FLAGS_FP16_ALT_IMPL` does not change it.
Kind 10 picks FP16 when ||A||·||B|| is small, so its FP16 operands are scaled
by powers of two from their norms on HIP (`cc40906`): 1.5e-8, and 9.1e-11
with `--sigma 1e-4` (was 2.5e-7; CPU 1.9e-13). After round 1: CTest 35 / 35 on
the MI210 (21 CPU, 14 GPU; the kind-5 and kind-8 cases now run on HIP with the
A100's residuals).

**Round 2: re-review, MI250X node, performance.** A review of the round-1
commits found an undefined cast for norms below 1e-304 (fixed in
`cc40906`). On the MI250X node `gpu_dense_adaptive` failed (2.8e-5); on the
MI210 it failed in 1 of 30 runs (1.9e-6), depending on which GEMMs ran on the
GPU. It is the same flush in the static FP16 tiles of kinds 5, 8 and 9: at
N=30720 the 98 FP16 tiles of an adaptive kind-5 run give 1.0e-5 for
`--fixedacc 1e-8` on both GPUs, deterministically; the same tiles in BF16
(`--tensor_gemm_hp 0x20`) give 1.2e-8. By the maintainer's decision
(section 7, item 8) `--tensor_gemm` keeps choosing and the default stays FP16;
HIP runs with FP16 tiles say so, the adaptive decisions use BF16's epsilon with
a BF16 variant, and the HIP builds' adaptive GPU tests use 0x20 (`931c41b`).
With `--sigma 1e4` the default nugget (0.1) no longer regularizes the matrix
and an adaptive run at N=30720 is indefinite in FP16 and BF16 alike; this is
not specific to AMD GPUs.

The first factorization of a GPU run was slow on AMD GPUs: rocBLAS and
rocSOLVER set a handle up at its first call, and HiCMA-X has four handles per
stream. At N=8192 on the MI210 the first factorization took 0.52 s and the
next ones 0.055 s (DPLASMA 0.126 s, then 0.060 s); the handles are now set up
at start-up (`5c0874b`) and the first factorization takes 0.058 s. On a
MI250X GCD, N=30720, NB=1024, one run per process, two processes per case:

| Case | Before (s) | After (s) | After (Tflop/s) |
|---|---|---|---|
| DPLASMA `testing_dpotrf -g 1` | 0.96 | 0.98 | 9.9 |
| kind 5 FP64 | 1.36 | 0.74 | 13.1 |
| kind 5 adaptive, FP16 tiles | 1.66 to 1.76 | 0.67 | 14.4 |
| kind 5 adaptive, BF16 tiles (0x20) | | 0.68 | 14.2 |
| kind 10 run-time decisions | 1.85 to 1.94 | 0.87 | 11.1 |
| FP64/FP32 bands 2/2, rest FP16 | 1.54 to 1.61 | 0.51 | 18.4 |
| same, rest BF16 (0x20) | | 0.53 | 18.3 |
| FP64 band 2, rest FP32 | 1.36 to 1.42 | 0.61 | 15.9 |

One process over several GCDs (FP64, N=30720): 1.39 s on 1, 1.40 s on 2 and
1.95 s on 8 before; 0.74, 0.42, 0.28 and 0.26 s on 1, 2, 4 and 8 after (BF16
adaptive: 0.69, 0.38, 0.27, 0.24 s). N=61440, NB=2048 on 8 GCDs: 0.91 s,
85 Tflop/s. The residual of the BF16 adaptive run at N=30720 is 1.16e-8 for
`--fixedacc 1e-8` (FP16: 1.0e-5), the growth with N of section 5.6.

Two-rank GPU runs with one GCD per rank aborted in Cray MPICH at the first
transfer (`process_vm_readv: Bad address`): PaRSEC sends from GPU memory, which
Cray MPICH reads only with `MPICH_GPU_SUPPORT_ENABLED=1` and its GTL library.
Under Cray MPICH without GPU support HiCMA-X now keeps remote transfers in host
memory (`a67bb1b`); that was not enough, since GPU runs allocate their host tiles
with `hipHostMalloc`, which Cray MPICH's intra-node single copy cannot read
either, so HiCMA-X also sets `MPICH_SMP_SINGLE_COPY_MODE=NONE` when it
initializes MPI there (`c658f96`). With the GTL library preloaded and
`MPICH_GPU_SUPPORT_ENABLED=1` the same runs passed before these changes.

| Run on an MI250X node, final commit unless noted | Result |
|---|---|
| `scripts/test_CI.sh`, one GCD, two-rank cases through `srun` (`a67bb1b`) | 41 passed, 2 skipped (kinds 1 to 4 on GPUs; climate data) |
| `scripts/validation_matrix.sh ... gpu` (`a67bb1b`) | 40 of 43 pass; the 3 two-rank cases sharing a GCD aborted (single copy), all 3 pass at the final commit (3.2e-16, 1.5e-8, 4.7e-9) |
| Fresh clone, `ctest -L hicma` (MPI cases through `srun`, GPU cases on one GCD) | 42 / 42 |
| Two ranks, one GCD each, no GTL, N=16384: dense, BF16 adaptive, kind 10 | 3.4e-16, 1.1e-8, 8.1e-9 |
| Two ranks, one GCD each, TLR band 2 / adaptive TLR, N=8192 | 3.9e-9 / 2.8e-5 (the band-2 run hung before) |
| Two ranks, one GCD each, TRMM N=16384 | passed |
| Mismatched `--gpus` (rank 0: 1, rank 1: 0) | refused on every rank, no hang |
| CPU-only tree (same sources) | builds; CTest 21 / 21 |


The installed package (`cmake --install`, `tests/install_consumer` built with
the MPI compiler wrappers) runs on the CPU (1.5e-16), on the MI210 with BF16
adaptive tiles (8.7e-9) and the conversion test (13 cases).

### 5.16 PaRSEC rebased on ICLDisco master (2026-11-10)

ICLDisco/parsec#814 was merged upstream (`2c16c652e`), with the
`completed_w2r` fix found here folded in. The fork branch is rebased on it:
the fork's copies of #814 and its own fix `5cd761f64` are dropped, and the
eight other fork commits are reapplied. One conflict, in
`parsec/remote_dep_comm.c`: our "Size eager receives by the declared type"
adds a `wire` argument to `remote_dep_copy_allocate` and #814 a `pinned`
out-argument, so the definition and its four call sites take both; the file
then differs from the old fork tip only by upstream's final #814 lines (an
`assert` and a fatal message). New upstream besides #814: the termdet
release/acquire ordering, #835 (DTD) and #840 (a PTG taskpool freed
without being added to a context, which changes the JDF compiler).
DPLASMA upstream has nothing new; the fork keeps its stage-in fix
(dplasma#170 is not merged).

On methane, with every tree rebuilt (CPU, CUDA with sm_80 and with the
default architectures, HIP):

- PaRSEC's own CTest 205 / 205, including the two new upstream tests.
- HiCMA CTest 27 / 27 (CPU) and 48 / 48 (CUDA) in three runs. A first run
  had nine GPU cases in a row fail or abort; the reruns, and the 70
  GPU-labelled cases (HiCMA and DPLASMA) run on their own, passed. The
  cause is not established; concurrent use of the GPU is the likely one.
- The validation matrix (31 CPU and 50 GPU cases) identical to the third
  pass.
- DPLASMA's one-GPU low-memory tests pass, and the two-rank eviction
  reproducer of #814 (`--N 16384 --NB 1024 --band_dense 2`, 400 device
  blocks) completes in 10 of 10 runs.
- With the Frontier pass (section 5.15) merged on top, rebuilt and rerun:
  CTest 28 / 28 (CPU) and 49 / 49 (CUDA), the validation matrix again
  identical in every case it shares with the third pass, the eviction
  reproducer 3 of 3, the HIP tree builds.
- hexane, same combined tree: CTest 49 / 49 on the H100 and 49 / 49 on the
  V100; DPLASMA's seven low-memory tests pass, the two-GPU ones (`[sd]gemm
  _2gpu_cuda_lowmem_mpi`, both GPUs) included; FP8 with a static band, with
  low-rank tiles and in kind 10 passes with FP8 GEMMs used; two ranks (H100,
  V100) pass. Performance at N=30720 matches section 5.14 within noise:

  | Case | Before the rebase (5.14) | After |
  |---|---|---|
  | H100, kind 5 FP64 | 17.5 to 18.2 Tflop/s | 17.9 and 18.4 |
  | H100, kind 5 adaptive | 16.4 to 16.7 | 16.3 and 16.6 |
  | H100, kind 10 | 13.7 to 14.2 | 14.3 and 13.2 |
  | H100, FP32 band with TF32 | 23.5 to 26.2 | 25.8 and 24.2 |
  | H100, DPLASMA `dpotrf -g 1` | 9.3 to 12.0 | 11.9 and 11.9 |
  | V100, kind 5 FP64 | 5.9 | 5.88 and 5.86 |
  | 16 CPU cores, TLR | 2.0 s | 1.95 and 2.02 s |

### 5.17 Tenth pass: code structure and several GPUs (methane, hexane, leconte)

On request: another round of testing and optimization, with leave to redesign
the code structure. `docs/code-review.md`, section 13, has the findings; this
section has what was run. leconte is a node with eight V100s (GPUs 0 to 3 on
socket 0, 4 to 7 on socket 1, NVLink between the GPUs of a socket and for
some pairs across the sockets).

**What changed.** Kinds 5, 8 and 9 run one kernel (reverted in section 5.18); the CPU GEMM into a
low-rank tile of kinds 1, 5 and 10 is two functions; one rule places the
tiles on the GPUs for every kind and for the warm-up; a note says when the
GPUs in use cannot all read each other directly. `src/` lost 6,076 lines
(888 added, 6,964 removed).

**Results are unchanged.**

- methane (A100), final tree: CTest 28 / 28 (CPU) and 50 / 50 (CUDA, with
  the case added in this pass), `scripts/test_CI.sh` 49 passed and 1
  skipped, the validation matrix 31 / 31
  (CPU) and 50 / 50 (GPU) with the exit code and residual of the third pass
  in every case they share. Three kind-1 cases print a larger
  `Number_of_Datatype_Conversions` (169 where 48 and 89 were printed): kind
  1's copy of the low-rank GEMM did not count its conversions. Kind 5, old
  and new, and kind 1, new, give the same counts on the same command lines
  (169, 169 and, at N=5400, 1335; kind 1 before: 48, 89, 244). The HIP tree
  builds.
- hexane: the validation matrix on the H100 and on the V100 is identical to
  the one recorded before the pass in all 50 cases (three with the counter
  above). Twelve FP8 and mixed layouts on the H100 give the residuals
  recorded before to the last digit (static FP8 band 8.431338e-07, kind 9
  adaptive 3.267401e-07, kind 10 7.459304e-09, FP8 band with low-rank tiles
  1.577594e-05 and 1.769098e-04, ...); two ranks sharing the H100 and one
  rank per GPU pass.
- leconte: eleven layouts (kinds 1, 2, 3, 5, 8, 9, 10; FP64, adaptive,
  static bands, FP16 band with low-rank tiles, TLR) give the same residual
  on 1, 4 and 8 GPUs in one process; the four that were also run with two
  ranks of four GPUs and the two run with eight ranks of one GPU give it
  there too.
  CTest 49 / 49 and the GPU matrix 50 / 50 on one V100; the TRMM on four
  GPUs passes.

**The restructuring costs no time.** The pushed revision (`7d66360`, built in
a second tree) against the working tree, interleaved, same tile placement;
medians, seconds:

| Machine, case | Runs | Before | After |
|---|---|---|---|
| methane, 36 cores, kind 5 TLR (N=30720, NB=512, band 1) | 5 | 0.590 | 0.602 |
| methane, 36 cores, kind 1 TLR | 5 | 0.592 | 0.596 |
| methane, 36 cores, kind 1 adaptive TLR | 5 | 0.485 | 0.471 |
| methane, 36 cores, kind 5 TLR, band 4 | 5 | 0.733 | 0.719 |
| hexane, 16 cores, kind 5 TLR (N=30720, NB=1024) | 6 | 1.607 | 1.530 |
| hexane, 16 cores, kind 1 TLR | 6 | 1.463 | 1.456 |
| hexane, H100, kind 5 TLR, band 8 | 12 | 1.330 | 1.342 |
| hexane, H100, kind 5 TLR, band 1 | 6 | 0.955 | 0.940 |
| hexane, H100, kind 1 TLR, band 4 | 6 | 1.423 | 1.449 |
| hexane, H100, kind 5 adaptive, dense | 6 | 0.624 | 0.593 |
| hexane, H100, kind 8 adaptive, dense | 6 | 0.609 | 0.633 |
| leconte, 4 V100, kind 5 FP64 (N=40960) | 5 | 1.085 | 1.078 |
| leconte, 4 V100, kind 5 adaptive | 5 | 0.876 | 0.870 |
| leconte, 4 V100, kind 8 adaptive | 5 | 0.829 | 0.830 |
| leconte, 4 V100, kind 10 | 5 | 0.635 | 0.639 |
| leconte, 4 V100, kind 5 TLR, band 8 | 5 | 0.682 | 0.693 |

The ranges of the two columns overlap in every row except hexane's 16-core
kind 5 (after is 5 % faster in six of six pairs). A first batch of six pairs
of the H100 band-8 case had medians 7 % apart; the twelve further pairs in
the table did not confirm it. The two builds also differ by 3 % on code the
pass did not touch (kind 10, dense, on methane's CPUs: 7.43 and 7.68 s),
which bounds what such a comparison can resolve.

**Tile placement, measured.** One binary with a temporary switch between the
rules (removed afterwards), five interleaved runs per rule, four linked
V100s, N=40960, NB=1024; median and range in seconds:

| Layout | By row | By column | Grid |
|---|---|---|---|
| kind 5, FP64 | 1.080 (1.071 to 1.084) | 1.133 (1.101 to 1.188) | **1.065** (1.037 to 1.090) |
| kind 5, adaptive | 0.872 (0.864 to 0.880) | 0.912 (0.890 to 0.932) | **0.831** (0.820 to 0.843) |
| kind 3, FP64/FP32 bands of 2/2, rest FP16 | 0.493 (0.490 to 0.498) | 0.488 (0.476 to 0.491) | **0.479** (0.475 to 0.489) |
| kind 10 | 0.698 (0.677 to 0.701) | 0.652 (0.641 to 0.664) | **0.635** (0.628 to 0.641) |
| kind 5, TLR, band 8 | 0.709 (0.614 to 0.812) | 0.753 (0.661 to 0.760) | 0.734 (0.710 to 0.769) |
| kind 1, TLR, band 8 | 0.758 (0.637 to 0.789) | 0.735 (0.637 to 0.781) | 0.678 (0.648 to 0.736) |
| kind 2, TLR, band 8 | 0.723 (0.659 to 0.746) | 0.715 (0.643 to 0.728) | 0.745 (0.675 to 0.796) |
| kind 5, TLR, band 1 | | 0.602 (0.566 to 0.613) | 0.585 (0.571 to 0.620) |
| kind 5, TLR, band 2 | | 0.662 (0.652 to 0.678) | 0.644 (0.636 to 0.685) |
| kind 5, TLR, band 4 | | 0.662 (0.644 to 0.665) | 0.650 (0.636 to 0.702) |
| kind 2, TLR, band 2 | | 0.647 (0.633 to 0.672) | 0.658 (0.643 to 0.670) |

The grid is the fastest rule for every dense layout, by 1 to 10 %; the old
rule of kinds 3 and 4 gave 0.494 s on the third row. With low-rank tiles no
rule wins: each of the three is fastest for one kind at band 8, and the
ranges overlap. On two GPUs kind 5 TLR was faster by row (0.623 against
0.746 s) and kind 1 TLR by column (0.639 against 0.745 s) on the same task
graph, and one binary with one rule gave 0.65 to 0.78 s for the first of
these in three runs: this is the spread of the TLR runs, not an effect of
the rule. By
column is kept for a dense band because it reaches every GPU at any band
width: with a diagonal band on four GPUs the grid gave two GPUs twice the
tasks of the other two (2.6 and 2.8 % against 1.2 and 1.3 % of the rank's
tasks), by column 1.9 to 2.2 % each. An earlier reading in this pass, "by
column is up to 15 % faster for TLR", came from two runs per rule and did
not survive the repetitions.

For kind 5 this changes the dense placement from by row to the grid: 1.4 %
(FP64) and 5 % (adaptive) faster on four GPUs; kind 3 gains 3 %.

**Ranks whose GPUs were left nearly idle.** Two ranks with four GPUs each,
share of a rank's GPU tasks per GPU (PaRSEC's device statistics):

| Case | Rule before | Rule now |
|---|---|---|
| kind 3, 2 x 1 process grid | 27.9, 4.4, 33.1, 5.4 % | 17.1, 15.7, 19.7, 18.3 % |
| kind 1 TLR, 1 x 2 process grid | 1.2, 15.0, 1.2, 15.0 % | 8.1, 8.1, 8.1, 8.1 % |

(The rest of the tasks runs on the CPU.) The run times of these two cases
did not change beyond their spread (2.16 and 2.07 s; 2.05 and 2.14 s): with
several ranks on one node the time is in the transfers between the ranks,
below.

**Several GPUs, one process** (leconte, final tree, dense FP64, N=40960,
NB=1024, three runs): 3.71 to 3.73 s on one GPU (6.1 Tflop/s; DPLASMA
4.09 s), 1.95 to 1.98 s on two (11.7; DPLASMA 2.43 s), 1.05 to 1.07 s on
four linked GPUs (21.5 to 21.8; DPLASMA 1.27 s). Mixed precision on four
GPUs: adaptive 0.83 s, kind 9 adaptive 0.57 s, kind 10 0.63 to 0.64 s,
FP64/FP32 bands with the rest in FP16 0.49 to 0.52 s. All eight GPUs take
2.38 to 2.48 s: some pairs are not linked, and PaRSEC then copies the output of
every GPU task to host memory (`parsec_gpu_task_update_pushout`). Measured
with the device statistics before the pass: 6.4 GB of 89.7 GB copied out on
GPUs 0 to 3, all 89.7 GB on GPUs 0, 1, 4, 5, which take 4.25 s; turning the
direct copies off on GPUs 0 to 3 (`device_cuda_nvlink_mask=0`) gives 4.46 s.
NUMA binding, interleaved memory and fewer cores change nothing. This is the
runtime's policy and is not changed here. Rank 0 now prints a note when the
GPUs in use are not all linked; it appears with eight GPUs and not with
`CUDA_VISIBLE_DEVICES=0,1,2,3`.

**Several ranks on one node** (same problem, before the pass): two ranks of
four GPUs 4.7 s, four of two 6.2 to 6.4 s, eight of one 3.9 to 4.2 s, against
1.06 s for one process with four GPUs. Pinned arenas
(`--arena_pinned_memory 1`: 5.2, 7.1, 4.35 s) and the shared-memory transport
alone do not help, and a CPU-only run does not show it (N=20480: 2.87 s with
two ranks, 2.91 s with one): the time is in the path GPU, host, MPI, host,
GPU.
`TESTS.md` recommends one process per node.

**Stochastic rounding for kinds 8 and 9** (undone in section 5.18; they ran kind 5's kernel, which
implements it): on methane, kind 8 and kind 5 give the same residual with
it (1.484178e-08 on the A100, 3.483492e-09 on the CPU; 1.144828e-08 and
3.444675e-09 without), kind 9 gives 2.007399e-07 (1.989370e-07 without), and
two ranks give the same values. With FP8 GEMMs and in kind 10 it is refused
at start-up, as before.

**Final tree on the GPU nodes.** After the placement rule and the
stochastic-rounding change, rebuilt and rerun:

- hexane: the H100 matrix again identical to the one recorded before the
  pass (50 cases); four FP8 layouts to the last digit; kinds 8 and 9 with
  stochastic rounding (`--fp8 0`) pass, kind 8 with kind 5's residual
  (2.537443e-09 adaptive, 1.279413e-06 with static bands); the H100 and the
  V100 in one process, where the placement rule decides, pass for kinds 1,
  2, 3, 4, 5, 8, 9 and 10 (ten layouts), and one rank per GPU for kinds 1, 3
  and 8. H100 at N=30720, two runs: FP64 17.8 and 18.1 Tflop/s, adaptive
  15.6 and 16.4, kind 8 adaptive 14.3 and 15.7, kind 10 13.9 and 14.2 (before
  the pass: 17.5 to 18.4, 16.3 to 16.8, 16.3 to 16.4, 13.2 to 14.3; six
  further runs of the build before the placement change gave 17.8 to 18.6,
  16.1 to 16.6, 15.1 to 16.5 and 13.1 to 14.4).
- leconte: thirteen layouts, now with kind 4 and kind 8 with stochastic
  rounding, give the same residual on 1, 4 and 8 GPUs; seven of them with
  two ranks of four GPUs (kinds 1, 2 and 3 among them, kind 2 on a 1 x 2
  process grid) and two with eight ranks of one GPU give it too. The GPU
  matrix on one V100 is 50 / 50, the TRMM on four GPUs passes.
- CTest, with the case added in this pass (`gpu_fp8_static_sr`, kind 8 with
  stochastic rounding): 50 / 50 on the H100, on hexane's V100 and on one of
  leconte's V100s; 50 / 50 (CUDA) and 28 / 28 (CPU) on methane. The HIP tree
  builds and registers its 42 cases; it was not run (no AMD GPU here).

### 5.18 Kinds 8 and 9 on their own kernels again (2026-11-11, on request)

**What changed.** `potrf_L_dense_mp_gpu_fp8.jdf`,
`potrf_L_dense_mp_gpu_fp8_sp.jdf` and their wrappers are back (`b817e7a`),
taken from `7d66360` (the tree before section 5.17) with the parts of the
tenth pass that are not the merge: the shared
low-rank CPU GEMM, `hicma_parsec_gpu_of_tile` for the placement, and no
unused FP8 work pool. Kind 5's JDF and wrapper lost the merge hunks
(`DIAG_ADT`, `DIAG_CONVERT`, the FP8 branches, the prediction-mode
submatrix of kind 9), and the datatype functions, prototypes and dispatch of
kinds 8 and 9 are back. The restored kernels have no stochastic rounding, so
kinds 8 to 10 refuse `--enable_stochastic_rounding` again, with one message
(the separate check for FP8 GEMMs, which suggested `--fp8 0`, is gone: FP8
GEMMs run only in these kinds). The CTest and `test_CI.sh` case
`gpu_fp8_static_sr` becomes `gpu_dense_adaptive_sr`, the same stochastic
rounding through kind 5. Each restored JDF differs from its `7d66360`
version only in the pool declaration, the placement call and the CPU GEMM,
whose code is kind 5's.

Also in this round (`91a64f0`): the installed `hicma_parsec_datatype_convert.h`
no longer declares `parsec_convert_i2s`, whose definition was deleted in
`bac0083`, and 15 commented-out calls to removed functions are gone.

**Validation.** `build_fp8r` (the restored tree, CUDA 12.6.2, default
architectures) against `build_rc_h100` (the merged kernel of section 5.17),
same binaries run in the same session; residuals read through
`--check_tol 1e-12`.

- methane: CTest 50 / 50 (CUDA, A100) and 28 / 28 (CPU); the validation
  matrix identical between the two trees in its 50 GPU-mode and 31 CPU-mode
  cases (exit codes, residuals, GEMM counters); 17 direct comparisons of
  kinds 8, 9 and 5 (adaptive, static bands, band with low-rank tiles, A100,
  CPU, two ranks on the A100 and on the CPU) give the same residual and
  precision counts.
- hexane: CTest 50 / 50 on the H100 and on the V100; the H100 matrix
  identical to the merged kernel's (section 5.17) in all 50 cases and, apart
  from the three kind-1 conversion counters of section 5.17, to the one
  before the merge; 23 direct comparisons with FP8 GEMMs on (static FP8
  bands: one FP8 GEMM, residuals 8.431338e-07 and 8.879569e-07; FP8 band
  with low-rank tiles: 34 and 364 FP8 GEMMs, 1.577594e-05 and 3.263227e-05;
  `--fp8 0`, `--fp8_scaling 0`, lookahead, two ranks on the H100, H100 and
  V100 in one process and one rank each) all identical. H100, N=30720,
  five interleaved pairs, medians merged / own: kind 8 adaptive 0.627 /
  0.597 s, kind 9 adaptive 0.574 / 0.572 s, kind 8 static bands 0.576 /
  0.576 s, kind 5 adaptive 0.616 / 0.623 s, kind 5 FP64 0.540 / 0.534 s;
  every pair of ranges overlaps.
- leconte: CTest 50 / 50 on one V100; 24 direct comparisons of kinds 8, 9
  and 5 on 1, 4 and 8 GPUs, two ranks of four GPUs and eight ranks of one
  GPU all identical. Four GPUs, N=40960, five interleaved pairs, medians
  merged / own: kind 8 adaptive 0.834 / 0.830 s, kind 9 adaptive 0.566 /
  0.575 s (1.6 %, under the 3 % by which two builds of identical code
  differed in section 5.17), kind 5 adaptive 0.843 / 0.833 s.
- The HIP tree (gfx90a, methane) builds with the restored kernels; not run.

### 5.19 Complete round on the head (2026-11-11)

**Why.** Both CI jobs on `02fa12c` failed in `apt-get` (exit code 100)
before building anything: the CUDA job's container could not resolve
`archive.ubuntu.com`, and the CPU job gave up after 48 minutes. GitHub
reported degraded service at the time, and a rerun of the failed jobs ended
in `startup_failure`. `02fa12c` changes only documentation; the source is
that of `b817e7a`. The round below replaces those jobs and repeats the
earlier checks on this source.

**methane, fresh HTTPS clone of `02fa12c`** (`--recursive`; DPLASMA
`ae43716e`, PaRSEC `557a710f4`, HCORE `78028a59`, STARS-H `5d98f2c2`):

- CPU build as `BUILD.md` says (`cmake -S . -B build -G Ninja`, only the
  install prefix added): CTest 28 / 28 with `LD_LIBRARY_PATH` unset;
  `test_CI.sh` 27 passed, 0 failed, 2 skipped (climate data; the GPU cases
  without `HICMA_GPUS`); `cmake --install`; the install consumer's Cholesky
  check passes and its conversion test runs 13 cases without failure; the
  CPU validation matrix 31 / 31, exit codes and residuals identical to
  section 5.18.
- CUDA build (`-DHICMA_PARSEC_HAVE_CUDA=ON`, CUDA 12.6.2, the default
  architectures): CTest 50 / 50 on the A100; `test_CI.sh` 49 passed, 1
  skipped (climate data); the GPU validation matrix 50 / 50, identical to
  section 5.18. Another user's job kept the A100 busy, so these runs used
  half its memory (`PARSEC_MCA_device_cuda_memory_use=50`) and no timing was
  taken on methane.
- The HIP tree (gfx90a) is up to date with this source and registers 42
  cases; not run.

**hexane** (`build_fp8r`, same source; H100 = GPU 0, V100 = GPU 1):

- CTest 50 / 50 on the H100 and 50 / 50 on the V100; `test_CI.sh` on the
  H100 49 passed, 1 skipped (climate data).
- The H100 validation matrix 50 / 50, identical to section 5.18 in exit
  codes, residuals and GEMM counters; the V100 matrix 50 / 50, identical to
  section 5.17's in the same three.
- The 13 FP8 and low-precision cases with recorded values give them to the
  last digit: kind 8 static FP8 band 8.431338e-07 (1 FP8 GEMM), with
  `--fp8 0` 8.431506e-07, kind 9 adaptive 3.267401e-07, kind 9 FP8 band
  8.879569e-07, kind 10 7.459304e-09 (3 FP8 GEMMs), FP8 band with low-rank
  tiles 1.577594e-05 (kind 8, 34 FP8 GEMMs) and 3.263227e-05 (kind 9, 364),
  the FP16 bands with low-rank tiles 4.414324e-05 (kind 5) and 1.654633e-04
  (kind 1, `0x10`).
- Stochastic rounding: kind 5 runs it (adaptive 2.537443e-09, bands
  1.279413e-06); kinds 8, 9 and 10 refuse it at start-up with exit code 1,
  with FP8 GEMMs and with `--fp8 0`.
- Two ranks sharing the H100 give the one-rank residuals. Both GPUs in one
  process (eleven layouts, kinds 1 to 5 and 8 to 10) and one rank per GPU
  (four layouts) give, in the thirteen layouts section 5.17 also ran, the
  same residuals and counters.
- The TRMM on the H100 passes with the default variant, `--tensor_gemm_hp
  0x80`, `--tensor_gemm 0xb`, stochastic rounding and `--accumulation_fp32 1`.
- Timings, N=30720, NB=1024, numactl on the GPU's node, four runs (three for
  DPLASMA, whose rate is DPLASMA's own count):

  | Case | Time | Rate |
  |---|---|---|
  | H100, kind 5 FP64 | 0.520–0.535 s | 18.1–18.6 Tflop/s |
  | H100, DPLASMA `dpotrf` | 0.629–0.642 s | 12.1–12.3 Tflop/s |
  | H100, kind 5 adaptive | 0.586–0.616 s | 15.7–16.5 Tflop/s |
  | H100, kind 8 adaptive (FP8) | 0.595–0.625 s | 15.5–16.2 Tflop/s |
  | H100, kind 9 adaptive (FP8) | 0.538–0.576 s | 16.8–17.9 Tflop/s |
  | H100, TLR kind 2 | 1.578–1.589 s | 6.1 Tflop/s |
  | V100, kind 5 FP64 | 1.630–1.644 s | 5.9 Tflop/s |
  | V100, DPLASMA `dpotrf` | 1.771–1.783 s | 5.0 Tflop/s |

  Kind 10 took 0.89 and 1.05 s in two of its four runs (0.71 and 0.72 s in
  the others). Its files have not changed since `a7ecff1`; eight interleaved
  pairs against the tenth-pass build give 0.671–0.728 s on the head and
  0.667–0.727 s on `a7ecff1`, and the slow runs did not recur: run-to-run
  noise.
- Harness notes: the session that ran the first script ended during the
  timings, which were rerun in a second allocation. DPLASMA's
  `testing_dpotrf` carries no BLAS run-time path (the HiCMA-X drivers do),
  so on hexane it needs `LD_LIBRARY_PATH` set to MKL.

**leconte** was not run: it was taken by other jobs, and this round skips it
on request. Section 5.18's leconte results are on this source.

### 5.20 Radeon VII (gfx906): HIP at run time (2026-11-22)

**Why.** Since the ninth pass (section 5.15) the HIP code had been compiled
but not run: the tenth pass's placement rule and shared CPU GEMM, OmniCast,
and the GPU operand cache. No gfx90a GPU is available here; histamine2 and
histamine3 each have two Radeon VIIs (gfx906, 16 GB) and an EPYC 7302.

**Build.** The ROCm 6.4.3 on these nodes (`/opt/rocm`) has no rocBLAS
kernels for gfx906: its Tensile library starts at gfx908. ROCm 6.3.2
(`/apps/rocm/rocm-6.3.2`) still has them, and a rocBLAS `dgemm` plus
rocSOLVER `dpotrf` probe with it gives the reference result on the
Radeon VII. The tree (`27a2a62`) was built against 6.3.2 with
`-DCMAKE_HIP_ARCHITECTURES=gfx906` and every ROCm package directory
(`hip_DIR`, `rocblas_DIR`, ...) set to 6.3.2; otherwise CMake takes HIP 6.4
from `/opt/rocm`.

**Runs.** One Radeon VII and 2 CPU cores (other jobs held the other 30), so
PaRSEC ran one worker thread.

- CTest `-L hicma` 46 / 46: the 14 GPU cases, OmniCast's 4 tests, the 27 CPU
  cases with the two-rank ones, and the fixture.
- GPU operand cache, N=16384, NB=1024, 11 configurations: kind 5 adaptive at
  1e-8 and 1e-4 (FP16, BF16, stochastic rounding), static bands with BF16
  operands and FP32 or BF16 C, an FP32 band, kinds 8, 9 and 10 at 1e-3, and
  runtime decisions. Each ran with the cache on, off, and at 4 slots, and
  all three gave identical residuals. The cache took 67-201 MB with no
  fallbacks, and cut the conversions per factorization 1.3-7.9x.
- Two of the 11 fail their check in all three settings, as they do with the
  baseline (`b72d652`) on the H100:
  - kind 5 at 1e-8 (2.31e-8 for a bound of 1e-8; the CTest case runs N=8192);
  - BF16 operands with BF16 C. The factorization loses positive definiteness
    (info 9265; 9258 on the H100's baseline) and the driver exits non-zero.
- Interleaved timings, N=30720, NB=2048: three rounds of three runs, the
  first run of each process dropped. The cache gives 0.997-1.014x, within
  the run-to-run spread (1-9 %):

  | configuration | no cache (s) | cache (s) | speedup |
  |---|---|---|---|
  | kind 5, 1e-8 | 3.224 | 3.219 | 1.002 |
  | kind 5, 1e-4 | 2.089 | 2.060 | 1.014 |
  | kind 5, 1e-4, BF16 | 12.739 | 12.731 | 1.001 |
  | kind 5, 1e-4, stochastic rounding | 2.098 | 2.074 | 1.011 |
  | kind 8, 1e-3 | 1.939 | 1.927 | 1.006 |
  | kind 9, 1e-3 | 1.493 | 1.480 | 1.009 |
  | kind 10, 1e-3 | 2.036 | 2.017 | 1.009 |
  | runtime decisions | 2.693 | 2.702 | 0.997 |

- rocprofv3 kernel traces of the factorization phase show why the gain is
  small. Conversions take 2.6 % (kind 8) and 2.9 % (kind 5 with stochastic
  rounding) of the GPU kernel time without the cache, and 1.1 % and 1.5 %
  with it (1120 -> 334 and 658 -> 315 launches). GEMMs take most of the rest.
  On the H100 with FP8 tiles, conversions were 17-19 %.
- **FP16 on gfx906.** rocBLAS's FP16 GEMMs keep subnormal inputs (2^-20)
  with FP32 C (`rocblas_gemm_ex`, k = 64 and 2048) and with FP16 C
  (`rocblas_hgemm`). gfx906 has no BF16 hardware: BF16 GEMMs made adaptive
  kind 5 at 1e-4 6x slower (12.73 s against 2.06-2.09 s). FP16 accuracy
  matches NVIDIA's (adaptive kind 5 at 1e-4: 2.42e-7 here, 2.07e-7 on the
  H100).
  - The start-up note pointing HIP runs with FP16 tiles to BF16 therefore
    skips gfx906 (`hicma_parsec.c`, from `gcnArchName`).
  - Checked: no note with 13 FP16 tiles on the Radeon VII. The gfx90a HIP
    tree and the CUDA tree still build.
- The start-up warm-up runs 8 BF16 GEMMs of 84 ms each on this GPU: 0.67 s
  per process, outside the timed factorization.

Not covered here: gfx90a and newer AMD GPUs at run time since the ninth
pass. Several AMD GPUs and two nodes: section 5.21.

### 5.21 Two nodes and two GPUs per process (histamine, 2026-11-22)

**Why.** Section 5.20 ran one Radeon VII. histamine2 and histamine3 have two
each (PCIe, peer access between them) and InfiniBand (mlx5), which allows
multi-node runs and several AMD GPUs per process.

**Setup.** Open MPI 5.0.5 over UCX 1.15.0, neither built with ROCm
(`MPIX_Query_rocm_support` returns 0); `srun --mpi=pmix`, one rank per node
with 2 cores (other jobs held the rest), so one PaRSEC worker thread; the
ROCm 6.3.2 user space of section 5.20; the tree `f0ec341` (`fe302b7`, with
the OmniCast checkout of the day, for the final runs) plus the changes
below. Three layouts: two nodes with one GPU each, two nodes with two GPUs
each (P=2, Q=1), and one node with two GPUs. Cases: the 7 two-rank CTest
cases (CPU, two nodes with one GPU each only), the 14 GPU CTest cases with
`--gpus` set to the layout's count, and the operand cache in 6
configurations at N=16384, NB=1024 (kind 5 adaptive at 1e-4 with FP16 and
with stochastic rounding, static bands with BF16 operands, kinds 8 and 10
at 1e-3, runtime decisions), each with the cache on, off and at 4 slots.

**First failure: tiles sent from GPU memory.** The first GPU case on two
nodes aborted at its first transfer (`ibv_reg_mr(..., length=8388608) failed:
Bad address`, then an assertion in UCX's rendezvous), and the other rank
waited until it was killed. An `ibv_reg_mr` probe registers `malloc`,
`hipHostMalloc` and `hipHostRegister` memory but not `hipMalloc` memory, and
an `LD_PRELOAD` wrapper of the MPI sends showed four 8 MB sends from device
memory per run, with `mpi_gpu_aware` set to 2 by HiCMA-X and also with 0.
- HiCMA-X now asks the MPI library whether it reads GPU memory (Open MPI's
  `MPIX_Query_rocm_support` and `MPIX_Query_cuda_support`, MPICH's
  `MPIX_GPU_query_support`; Cray MPICH as before) and sets `mpi_gpu_aware`
  to 0 when it does not, with a start-up note. A user-set
  `PARSEC_MCA_mpi_gpu_aware` still wins.
- PaRSEC sent the GPU copy of a flow that a GPU task only reads and
  forwards: the `potrf_dpotrf_send` and `potrf_dtrsm_send` tasks forward the
  POTRF and TRSM results that way, and `parsec_device_data_copy_pin_value()`
  returned the copy it was given whatever `mpi_gpu_aware` said. It now
  returns the host copy of the same version when the engine may not read
  accelerator memory, and the producer of such a flow writes the host copy
  back when a local GPU successor forwards it read-only and there are other
  ranks (PaRSEC fork, `device_gpu.c`).
- After the fix no tile leaves from device memory (same wrapper), on two
  V100 ranks (leconte) as well with `mpi_gpu_aware` 0, where the residuals
  are those of the default mode (4.869234e-16 dense, 2.231640e-07 adaptive).
  Open MPI reports CUDA support there, so the default stays at sending from
  GPU memory.

**Second failure: GPU faults on host pages with two GPUs per process.**
"Memory access fault by GPU node-2 ... Page not present", then the job
waited until the 300 s timeout: on two nodes with two GPUs 11 of the 32
runs (both runtime-decision CTest cases, kind 10 and runtime decisions in
all three cache settings, one kind-5 run), on one node with two GPUs 2 of 32
(and 0 of 64 repeats of four of the configurations). The kernel log
(`dmesg`) names a DMA client (XDMA) walking consecutive host pages; with
`HSA_ENABLE_SDMA=0` the copy kernels fault instead (TCP and CB clients).
- On the faulting configuration (runtime decisions, two nodes with two
  GPUs): no peer copies (`PARSEC_MCA_device_hip_nvlink_mask=0`) and pinned
  HiCMA-X arenas (`--arena_pinned_memory 1`) both pass 3 of 3; SDMA off
  faults 3 of 3.
- `rocgdb` stopped the faulting rank inside a stage-in: `hipMemcpyAsync`,
  host to device, in HIP's pinning of the pageable source
  (`KernelBlitManager::writeBuffer`, `addPinnedMem`). A wrapper of
  `hipMemcpyAsync` found no copy whose direction disagrees with the memory
  types.
- The pageable sources were PaRSEC's arena buffers: received tiles, new and
  temporary tiles, host copies. The matrix tiles are pinned already
  (`hipHostMalloc`), and PaRSEC registers the other matrices it reads (see
  the third failure). PaRSEC
  (fork, `arena.c`) now lets an application choose the allocator of arena
  memory (`parsec_arena_data_allocate` and `parsec_arena_data_free`), and
  HiCMA-X makes it pinned memory on HIP with more than one GPU per process,
  for its own taskpools and DPLASMA's (the `--check`). The tracer then sees
  no copy from or to pageable memory (1248 copies in a one-node run).
  Pinning the matrices through `parsec_data_allocate` instead was tried and
  dropped: PaRSEC cannot register memory that is pinned already
  (`hipHostRegister ... invalid argument`), and its tasks then found no
  device.

**Third failure: the `--check`'s LL^T on one of the nodes.** After the arena
change one run of the 32 on one node with two GPUs still faulted, in the
`--check` (DPLASMA's LL^T) after the factorization. Repeats of three
configurations (kind 5 with stochastic rounding, BF16 operands, kind 10)
with an `LD_PRELOAD` log of every `hipHostMalloc`, `hipHostRegister` and
pageable copy:
- histamine3: 0 faults in 36 runs; histamine2: 17 in 132 (3 of 36, 3 of
  48, 11 of 48, the last in a burst of 21 runs, then 25 clean ones).
- Every fault was in the 2 GB LL^T: a `malloc` buffer that PaRSEC registers
  (`hipHostRegister`, portable) when DPLASMA's TRMM first uses it, and
  almost always within its first tile, right after that registration.
  `dcAd`, which HiCMA-X registers at allocation after the CPU has written
  it, never faulted; the `hipHostMalloc` buffers never did either.
- Waiting 0.5 s after the registration (3 of 48) or touching the pages
  before it (11 of 48) does not help. histamine2's kernel was compacting
  memory during the runs (83,800 pages migrated in 100 s, `compact_success`
  483; histamine3 0), and a registered `malloc` buffer's pages are the
  kind the kernel moves; the mechanism was not established further.
- The LL^T is now allocated like the tiles (`hipHostMalloc`, or
  `cudaMallocHost` on CUDA, when GPUs are enabled) and given a
  registration hook that does nothing (PaRSEC cannot register pinned
  memory; DPLASMA's LAPACK-layout wrapper calls the hook without checking
  for NULL, which a first attempt with NULL found). histamine2 then passed
  48 of 48 runs of the same three configurations, with the same residuals.

**Results after the three fixes.**

| layout | cases | before the fixes | after |
|---|---|---|---|
| two nodes, one GPU each | 7 two-rank CTest, 14 GPU, 18 cache | first GPU case aborted | 39 / 39 |
| two nodes, two GPUs each | 14 GPU, 18 cache | 21 / 32 (after the first fix) | 32 / 32 |
| one node, two GPUs | 14 GPU, 18 cache | 30 / 32 | 32 / 32 |

The "after" column is one run of the three layouts on the final code;
the 6 repeats of runtime decisions and kind 10 on two nodes with two GPUs
and the 48 repeats on histamine2 above passed too.

- The residuals of the dense and cache cases are the same in every layout
  and are those of one GPU (section 5.20): 2.418978e-07 (kind 5 at 1e-4),
  3.092794e-07 (stochastic rounding), 2.865805e-05 (BF16), 8.343982e-06
  (kind 8), 7.995640e-06 (kind 10), 1.069236e-08 (runtime decisions). The
  cache on, off and at 4 slots gives the same residual in each, and the
  ten dense GPU cases give the same residual in the three layouts. The three
  tile low-rank GPU cases differ in the third digit from run to run
  (3.204e-5 to 3.224e-5 for the adaptive one, 4.27e-5 to 4.29e-5 for the
  half-precision band), since the order of their low-rank updates varies;
  they pass their bound.
- The cache took 67-176 MB per GPU (its counters are summed over the run's
  GPUs: 134-352 MB on one node with two GPUs, 268-608 MB on two nodes), 67 MB
  for kind 5 as with one GPU, with no fallbacks.
- Pinned arenas leave the factorization time unchanged: the 18 cache runs
  on one node with two GPUs are within 1 % of the runs before the change
  (0.70-2.90 s).
- With all the changes: CTest 46 / 46 on one Radeon VII, 55 / 55 on one
  H100 (hexane, CUDA tree) and 30 / 30 on the CPU tree (methane).

Not covered here: more than two nodes, and gfx90a and newer AMD GPUs at run
time since the ninth pass. NVIDIA GPUs on two nodes: section 5.22.

### 5.22 NVIDIA GPUs on two nodes (leconte and methane, 2026-11-24)

**Why.** In section 5.21 the MPI library could not read GPU memory, so every
tile went through host memory. With NVIDIA GPUs Open MPI reports CUDA
support, and HiCMA-X keeps PaRSEC's default of sending tiles from GPU
memory. That path had only run between two ranks on one node.

**Setup.** One rank on a V100 (leconte) and one on the A100 (methane),
`srun --mpi=pmix`, 4 hardware threads per rank. The tree is `8e8fde8`, built
with CUDA 12.6 for sm 70, 80 and 90. Open MPI 5.0.5 runs over UCX 1.15.0,
both built with CUDA (UCX transports tcp, cuda_copy and cuda_ipc).
leconte's InfiniBand ports are down and methane has none, so the two nodes
communicate over Ethernet (TCP). Cases:
- the 7 two-rank CTest cases;
- the 22 GPU CTest cases;
- the operand cache in the 6 configurations of section 5.21 (FP16 operands
  in place of BF16 for the static bands), with the cache on, off and at
  4 slots.

The GPU cases ran with HiCMA-X's default (tiles sent from GPU memory) and
with `PARSEC_MCA_mpi_gpu_aware=0` (host memory).

**Results.**

| set | tiles sent from GPU memory | through host memory |
|---|---|---|
| two-rank CTest cases (CPU) | 7 / 7 | |
| GPU CTest cases | 22 / 22 | 22 / 22 |
| operand cache, first round | 16 / 18 | |
| operand cache, second round | 18 / 18 | 18 / 18 |
| repeats of the two configurations that failed | 10 / 10, then 51 / 60 | 10 / 10 |
| PaRSEC limited to 3 % and 6 % of each GPU | 6 / 6 | 6 / 6 |

- A wrapper of the MPI calls shows where the tiles leave from. In the dense
  case at N=8192, the default sends 28 tiles from GPU memory and 63 from
  host memory, and `mpi_gpu_aware` 0 sends none from GPU memory. Receives
  land in host memory in both modes.
- The residuals are the same with GPU sends and with host sends in every
  case, including the tile low-rank ones, and the same with the cache on,
  off and at 4 slots:
  - 2.184723e-07 (kind 5 at 1e-4);
  - 2.925121e-07 (stochastic rounding);
  - 3.571553e-06 (FP16 bands);
  - 8.469258e-06 (kind 8);
  - 8.148881e-06 (kind 10);
  - 2.580552e-08 (runtime decisions).
- The cache took 67-176 MB per GPU (134-352 MB summed over the two), with
  no fallbacks.

**Failures.**
- **First round.** Two cache runs failed back to back:
  - runtime decisions with the cache on: "OmniCast: device runtime error"
    on the A100's rank;
  - then kind 5 adaptive with the cache off: "not positive definite" at
    row 4609, residual 1.7e+02.

  Neither node logged a GPU driver error (Xid). Other processes used GPUs
  on both nodes at the time: 31 GB on two other V100s, and the A100 at
  100 % load. methane's `docker0` interface also disappeared about two
  minutes before. Neither failure came back in 20 repeats of the two
  configurations (both modes) or in the 36 runs of the second cache round.
- **60 more runs of the two configurations with GPU sends.** 51 passed.
  The other 9 failed back to back, while a process outside the job held
  28-31 GB of the V100 Slurm had given us: Slurm does not confine GPUs on
  these nodes, and another user's job requested none. All 9 stopped
  cleanly when they created a cuBLAS or cuSOLVER handle ("cublasCreate
  failed", DPLASMA's "Unable to create CUBLAS Handle"), with no wrong
  residual.
- **GPU memory filled after the start.** A helper took all but 63 MB of the
  A100's, the V100's or both GPUs' free memory once HiCMA-X had started
  (11 runs). The factor stayed the same as without the helper (identical
  forward error), and each run stopped cleanly in the `--check`, when
  DPLASMA created a handle.
- **A small share of each GPU.** With PaRSEC limited to 3 % and 6 % of each
  GPU (`device_cuda_memory_use`), so that tiles are evicted, the residuals
  and forward errors were those of the full GPU, in both modes.
- **Still unexplained.** The wrong factor of the first round is one run in
  144 with GPU sends. The same configuration then passed 33 times and
  stopped cleanly 5 times while the other process held its GPU.
- **Two weaknesses found along the way**, neither shown to cause it:
  - Most of HiCMA-X's static GPU kernels store cuBLAS's return status
    without checking it, so a failed cuBLAS call would go unnoticed
    (checked since: section 5.23).
  - OmniCast's launch check reads `cudaGetLastError()`, which also returns
    an earlier error of the same thread, and its message does not name the
    CUDA error.

Not covered: two GPUs per process on two NVIDIA nodes, InfiniBand (or
GPUDirect RDMA) between NVIDIA nodes, and more than two nodes.

### 5.23 Status checks of the GPU library calls (2026-11-25)

**Why.** Section 5.22 found that most GPU kernels of the Cholesky stored the
status of their cuBLAS calls without checking it. The GPU POTRF checked
cuSOLVER's status only in an `assert`, which Release builds drop. A failed
call left a tile without its update, and the run went on.

**Change.** `HICMA_CUBLAS()` and `HICMA_CUSOLVER()`
(`hicma_parsec_internal.h`) stop the run with the file, the line and the
name of the status, as `HICMA_OMC()` does for OmniCast; on HIP they check
hipBLAS and rocSOLVER. They wrap every unchecked cuBLAS, cuBLASLt and
cuSOLVER call of the Cholesky kernels:
- 52 call sites in `hicma_parsec_core.c`: POTRF, TRSM, SYRK, the dense and
  low-rank GEMMs, the FP8 scale attributes and the stream settings;
- 20 in the Cholesky JDFs: stream settings, the GEMMs of kind 3 and the
  norms of kind 10.

The kind-10 and FP8 GEMMs already stopped on a failed call and are
unchanged. The handle setup checked `cublasCreate`'s status a second time
after `cublasSetStream` instead of `cublasSetStream`'s own; it checks the
right one now, and the cuBLASLt descriptor attributes are checked too. Not
changed: the TRMM (checked already), the genomics and climate drivers, and
the HIP warm-up calls at start-up.

**Results** (`fa915ce` plus the change, built for sm 70, 80 and 90, and for
gfx906 with ROCm 6.3.2):
- The CUDA, HIP and CPU-only trees build without new warnings.
- CTest 55/55 on the H100 and 55/55 on the V100 (hexane), and 46/46 on a
  Radeon VII (histamine2).
- Rebased onto `19a02ec` (GEMMs stay on the GPU): CTest 55/55 on a V100
  node (leconte) and 30/30 CPU-only (caffeine); the CUDA (sm 70), HIP
  (gfx90a) and CPU-only trees build without new warnings.
- The 22 GPU CTest cases on the H100 with `fa915ce` and with the change,
  built side by side: the residuals are identical in all 22.
- **Injected failures.** A preload library made one call of a GPU
  factorization at N = 8192 return an error: the third `cublasDtrsm` or the
  second `cusolverDnDpotrf`, in the default kind and in kind 2.
  - With the change, each run stops at that call, for example
    `hicma_parsec_core.c:3942: cuBLAS: the function failed to launch on the
    GPU (status 13)` and `hicma_parsec_core.c:3864: cuSOLVER:
    CUSOLVER_STATUS_EXECUTION_FAILED (status 6)`. With the library loaded
    and no failure injected, the run passes.
  - With `fa915ce`, the default kind went on without the TRSM and reported
    "not positive definite" at row 3131 with residual 7.6e+01, a failure
    like the unexplained one of section 5.22. Kind 2 failed later, in an
    HCORE QR (`info=-4`) or with PaRSEC giving up on the GPU, and neither
    message named the failed call.

The unexplained wrong factor of section 5.22 stays unexplained: a failed
cuBLAS call would now stop such a run instead of letting it finish.

**The CPU side.** The CPU kernels have no such silent failure: their BLAS
calls return no status (a wrong argument is a programming error), the CPU
POTRF passes LAPACK's `info` on, and HCORE's QR and SVD exit with a message
when LAPACK fails. Unchecked were allocations, which crash rather than
miscompute:
- `hicma_parsec_core_memory_allocation`, which allocates the LL^T of
  `--check` (and the tiles of the climate emulator and of the TRMM
  performance test), ignored the result of `cudaMallocHost`/`hipHostMalloc`.
  It now returns NULL on failure and clears the CUDA (HIP) runtime's last
  error, which OmniCast's launch check would otherwise report later as its
  own. The check falls back to pageable memory as before, and every caller
  stops with the size when that fails too.
  - On the A100 (methane), with `cudaMallocHost` of the LL^T failed on
    purpose, the run falls back and passes with the same residual
    (3.602581e-16). With the pageable fallback failed as well, it stops with
    `Fatal: allocating the LL^T of the check failed: 536870912 bytes of host
    memory`.
  - CTest 55/55 on the A100, and 46/46 on the Radeon VII again.
- The work buffers of PaRSEC's private memory pools (about 70 pops) and four
  `calloc`s of recompressed low-rank tiles are still unchecked. They crash at
  once if the system runs out of memory; PaRSEC's pool returns a pointer just
  past NULL then, so a NULL check would not catch it.


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
| P1 items | done except the warning-flags option (P1-11); the GPU status checks (P1-6) were completed in section 5.23 |
| P2 items | done except P2-6, P2-9, P2-12 (deferred) and the remaining parts of P2-5, P2-7, P2-8 |
| Second pass: conversion merge, external-user walkthrough, guyot sweeps (section 5.6) | done; 11 problems fixed |
| Third pass: matrix sum, macro review, sanitizers (section 5.7) | done; pool leak, `GPU_BUFFER_ONCE`, strategy 4 fixed |
| Fourth pass: build scripts (section 5.8) | done; CUDA default architectures and kind-3 guard messages fixed |
| Fifth pass: macros and data structures (section 5.9) | done; unused names removed, switches settable, two climate debug paths fixed |
| Sixth pass: external user, usability and performance (section 5.11) | done; `--nruns` crash, help defaults, summary line fixed; notes for three performance traps |
| Seventh pass: GPU GEMM variants and three rounds of checks (section 5.12) | done; `--tensor_gemm_hp`/`_sp`, five wrong-result or crash paths closed; static FP16 band with low-rank tiles on GPUs fixed (its low-rank B is now converted as U and V) |
| Eighth pass: H100 and two more rounds (section 5.13) | done; FP8 operands in the low-rank GEMMs fixed, documentation audited and corrected |
| Complete hexane test, H100 and V100 (section 5.14) | done; correct on every configuration; notes for capped `--gpus` and oversubscribed cores added |
| Ninth pass: Frontier, HIP on MI210 and MI250X, two rounds (section 5.15) | done; FP16 subnormal flush on gfx90a (kind 10 scaled, BF16 via `--tensor_gemm` for the static paths), HIP handle warm-up (FP64 1.84x faster on a GCD, multi-GCD scaling), Cray MPICH multi-rank GPU runs, script and test fixes |
| Tenth pass: code structure and several GPUs (section 5.17) | done; kinds 5, 8 and 9 one kernel, the low-rank CPU GEMM shared, one measured GPU placement rule, a note for GPUs that are not all linked; results unchanged on A100, H100 and V100s, timings unchanged or better |
| Kinds 8 and 9 on their own kernels again, on request (section 5.18) | done; results identical to the merged kernel on A100, H100 (FP8) and V100s, timings within the spread; kinds 8 to 10 refuse stochastic rounding again |
| Complete round on the head (section 5.19) | done; fresh clone on methane (CPU and CUDA builds, install, both matrices) and hexane (H100 with FP8, V100, both GPUs) pass and match the earlier results; leconte not run |
| Review by the maintainer of the behaviour changes (binding check, `--adddiag`, `N % NB`, genomics option) | **open** |
| Tag `v0.2.0`, GitHub release, merge to `main` | **open** (authorization required) |

## 7. Unresolved blockers and decisions for the maintainer

1. **Behaviour changes to confirm**: `--check 1` now fails runs it used to
   pass; `--adddiag` results change for problem kinds 0, 1, 5, 6, 7; the
   Cholesky drivers reject `N % NB != 0`; the genomics drivers need
   `-DHICMA_PARSEC_GENOMICS=ON`. Each is documented in `RELEASE.md`.
2. **Mixed-precision residual bounds**: the registered CTest cases use
   `--check_tol` values measured on methane and guyot (1e-7 to 1e-4 by
   precision mix). They grow with N (section 5.6), and other BLAS libraries or
   GPUs may need looser bounds; the cases print the residual.
3. **What `--check` decides**: the residual ||LL^T-A||/||A||. The forward error
   against the dense reference factor is printed but not checked (it was never
   effectively checked; binding it would need a second, looser bound).
4. **CI**: both jobs (CPU build, tests and install; CUDA compile) pass on every
   push since section 5.10 (one run per push, listed in ecrc/hicma-x#13).
5. **Hardware not available here**: more than two nodes, InfiniBand between
   NVIDIA nodes, and gfx90a and newer AMD GPUs since the ninth pass. The GPU
   validation matrix ran on an H100 (section 5.13) and on an MI250X node
   (section 5.15). The current code ran on one and two Radeon VIIs per
   process and on two nodes (gfx906, sections 5.20 and 5.21), and on two
   NVIDIA nodes over Ethernet, sending tiles from GPU memory (section 5.22).
   One wrong factor in 144 runs there stays unexplained (section 5.22).
6. **Defaults that cost performance** (section 5.11): a GPU build runs on the
   CPU unless `--gpus` is given, and kinds 1 and 2 (tile low-rank) are fully
   dense unless `--band_dense` is given. Both now print a note. Making
   `--gpus` default to the visible devices, or the TLR kinds default to a
   band of 1, would change the results of existing command lines.
   **Decided (2026-11-08): keep both defaults; the notes and the corrected
   `--help` are the change for 0.2.0.**
7. **Static FP16 band with low-rank tiles on GPUs** (section 5.12):
   **resolved.** Kinds 1 and 5 returned wrong results or crashed, as 0.1.0
   did: the FP16 branch of the low-rank x low-rank GPU GEMM converted its
   low-rank B as a dense tile. Fixed; the layout runs and is in CTest.

8. **FP16 on AMD GPUs** (section 5.15): gfx90a flushes FP16 subnormals in
   its FP16 GEMMs, which costs the adaptive FP16 tiles their accuracy.
   **Decided (2026-11-08): keep FP16 and BF16 and let `--tensor_gemm`
   choose; the default stays FP16 (0x8).** HIP runs that produce FP16 tiles
   say so and point to `--tensor_gemm_hp 0x20`; the HIP builds' adaptive GPU
   tests use it. Kind 10 scales its FP16 operands on HIP. A per-tile scale
   for the static FP16 paths (as the FP8 GEMMs have) would make FP16 safe
   on AMD GPUs too; it is not done. gfx906 keeps FP16 subnormals and has no
   BF16 hardware, so the note skips it (section 5.20).

## 8. Remaining and future work

Deferred from the plan (none blocks the release): P2-6 (zero-initialized check
matrices, `matrix_uncompress` flows), P2-9 (one JDF for the integer
conversions), P2-12 (statistics communicator hygiene), P3-1 to P3-3, the
remaining parts of P1-11, P2-5, P2-7, P2-8. The project's own `TODOs`
backlog (API, Python/Julia bindings, merging the potrf JDFs, hcore removal,
tensor_gemm redesign) was out of scope by request; item 12 (tensor_gemm)
was done on request in the seventh pass (section 5.12), item 1 (merge the
tile/LAPACK conversions) was done on request in the second pass, item 10
(external users) partly (section 5.6), and item 2 (merge the potrf JDFs)
partly in the tenth pass (section 5.17): kind 10 and the sparse kinds 6 and
7 still have their own kernels.

Found in the second pass and left for later:

- Parameters can only be set through an argument vector, and invalid
  parameters terminate the process (`TODOs` items 5 and 10).
- The public headers include STARS-H declarations with generic names (`diff`,
  `cube`, `Gaussian`), which an application cannot reuse in files that include
  `hicma_parsec.h` (third-party submodule, unchanged).
- The FP16-compute branch of the two low-rank GPU GEMMs is covered by
  `gpu_tlr_adaptive_fp16acc` (0x10), and FP16 tiles with low-rank x low-rank
  updates by `gpu_tlr_hp_band` and `gpu_tlr_mp_hp_band` (0x8); the FP16-band
  layout under 0x10 was run by hand (1.7e-4).
- Kind 3 has no CPU half-precision path; without GPUs the single-precision
  band is widened to the whole matrix.
