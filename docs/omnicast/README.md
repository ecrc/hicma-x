# OmniCast in HiCMA-X

HiCMA-X converts tiles between FP64, FP32, FP16, BF16 and FP8 all the time:
before mixed-precision GEMMs and TRSMs, when precisions are decided, for the
FP8 operands of cuBLASLt, with stochastic rounding. All of it now goes through
[OmniCast](../../omnicast) (submodule, branch `main`); this page records what
changed, how it was validated and what it bought. OmniCast's own design,
profiling and kernel measurements are in `omnicast/docs/REPORT.md` (with
`DESIGN.md` and `PERFORMANCE.md` next to it).

## With and without OmniCast, final state

The final build (OmniCast round 3 and the operand cache, `f0ec341`) against
the code before OmniCast (`4256ff9`), built side by side:

- Residuals are identical in all 10 configurations on the A100.
- On the A100 (N = 45056), OmniCast alone is 1.01-1.04x faster, most with
  stochastic rounding. With the operand cache it is up to 1.06x faster
  (stochastic rounding, kinds 8 and 10, faster in every round). FP64 dense,
  which converts nothing, moves by under 2 %.
- On a 28-core Broadwell CPU, stochastic rounding is 1.03x faster and the
  other kinds are unchanged.
- Not measured: the H100 with FP8 tiles (node busy).

Tables, figure, method and scripts: [`final/README.md`](final/README.md).

## Subnormals on the CPU (round 4)

On Intel CPUs the mixed-precision kinds were 1.2-1.4x slower than FP64. The
cause was not the conversions (1 % of the time) but MKL's FP32 GEMMs: the
FP32 tiles of the covariance problems hold many subnormal numbers, and each
SIMD operation on one takes a microcode assist (568 million in one
factorization). HiCMA-X now flushes subnormals on its CPU threads
(`--flush_subnormals`, default on). Kinds 3, 5 and 10 are 1.6-2.6x faster on
a Broadwell and a Skylake-SP node, FP64 is unchanged, and residuals agree to
4-5 digits. An AMD Zen 2 CPU has no such penalty and is unchanged. OmniCast's
host conversions now give the same bits whatever the caller's FTZ/DAZ or
rounding mode. OmniCast#1 is merged, and the merged state passes CTest on the
CPU, an A100 and a Radeon VII. Report, data and scripts:
[`round4/REPORT.md`](round4/REPORT.md).

## Converting once: the GPU operand cache (round 3)

GPU GEMMs used to convert their operands from the tile's send precision on
every call, so a panel tile was converted up to NT - 1 times. Each GPU now
keeps the converted copies, with bit-identical results (`--operand_cache`,
default on). With most tiles in FP8, conversions fall from 19 % to 4.9 % of
the GPU kernel time, and the factorization runs 1.07-1.08x faster on an
H100. On a Radeon VII (HIP, no FP8) the results are identical too, and the
time is within noise: conversions are under 3 % of the GPU time there.
Design, validation and measurements, with figures:
[`round3/REPORT.md`](round3/REPORT.md).

## OmniCast update of 2026-11-22

The submodule now points to the second round of OmniCast work
(`omnicast/docs/REPORT.md`):

- Every conversion with every option runs out of place and in place, on the
  host, CUDA and HIP. This includes GPU in place of tiles with ld > m, in place
  with transposition or padding, scaling for every pair, and stochastic
  rounding of scaled values. HiCMA-X's calls are unchanged.
- Faster kernels that HiCMA-X uses (H100, 8192² tiles, before → after):
  - FP8 operands (transposed, padded) from FP32: 1.13x. From FP64: 1.04x.
  - In-place tile conversions: 1.04–1.06x, and 1.08–1.18x on 2048² tiles.
  - GPU stochastic rounding FP32 → FP16: 1.0–1.3x.
  - The FP64-source stochastic rounding that HiCMA-X uses most is unchanged.
- A fix for AMD GPUs: gfx906 fused an FP32 product and its FP16 conversion
  (one rounding). HiCMA-X's HIP build scales its FP16 operands by powers of
  two, whose products are exact, so its results are unchanged.
- Validation with the updated submodule: CTest `-L hicma` 54/54 on the V100
  and 30/30 in the CPU-only build (the extra tests are OmniCast's coverage
  tests). All 39 accuracy figures are identical to the baseline.

## What changed

| Before | After |
|---|---|
| ~60 conversion kernels in `hicma_parsec.cu` / `hicma_parsec_hip.cu` (32×32 thread blocks, one element per thread) | `omc_gpu_convert_2d` (vectorized, striped), `omc_gpu_convert_inplace` |
| convert into a scratch tile + `memcpy_*_GPU` back (13 sites) | one single-pass in-place kernel |
| `double2fp8_GPU`, `hicma_fp8_scaled_convert_GPU`: transposed FP8 operand, one byte per thread at stride ld; memset + amax + scale + convert | `hicma_parsec_fp8_operand_GPU` → OmniCast transposed/padded/dynamic-scale conversion (2 kernels, coalesced) |
| `hicma_parsec_stochastic_rounding.{c,cu}`, `_hip.cu` | `OMC_ROUND_STOCHASTIC` (same random numbers and probabilities, bit for bit) |
| `convert_*_unary/binary_CPU` scalar loops; FP16 only on Fugaku | `omc_convert*` (AVX-512 / AVX2 / NEON), FP16 on every CPU |
| `LAPACKE_dlag2s` / `LAPACKE_slag2d` (43 sites) | `omc_convert_2d` (2.3–8× faster than MKL's `dlag2s` on 256–2048 tiles) |
| SR `if/else` around two converters at every call site | `HICMA_OMC_SR(on, seed)` options |

The HiCMA-level helpers stay in `hicma_parsec_datatype_convert.{h,c}`:
`hicma_parsec_omc_dtype` (tile precision → OmniCast type), `HICMA_OMC`
(abort with the call site on an error), `HICMA_OMC_SR`/`HICMA_OMC_SCALED`,
`convert_datatype_{unary,binary,adaptive_unary}_{CPU,GPU}`,
`hicma_parsec_convert_2h_bit`, `hicma_parsec_fp8_operand_GPU`. Semantics
were kept where they mattered: stochastic rounding applies exactly where it did
(FP64 → FP32 on the CPU; the narrowing conversions on the GPU), the FP8 scale
triple is {0, 1, 1} without `--fp8_scaling`, `convert_datatype_unary_CPU`
reports the new size for `d2s` only. Two deliberate differences:

- FP64 → FP8 (CUDA) and FP64 → FP16 (HIP) are rounded once, directly; HiCMA-X
  went through FP32 (double rounding). The results of the test suite did not
  change; individual elements can differ in rare ties.
- Float → integer saturates and maps NaN to 0 (undefined behaviour before).

## Validation

| Platform | CTest `-L hicma` | Accuracy vs baseline |
|---|---|---|
| A100 (methane), CUDA build sm_70/80/90 | 52/52 (50 + 2 OmniCast tests) | 39 cases with an accuracy figure: all identical |
| V100 (leconte), same binaries | 52/52 | all identical |
| H100 (hexane), same binaries; kinds 8/9 used up to 36 FP8 tiles | 52/52 | all identical |
| CPU-only build (methane) | 29/29 | — |
| HIP build (gfx90a, ROCm 6.4.3) | compiles and links (no AMD GPU for HiCMA-X here) | — |

OmniCast's own tests also ran on a Radeon VII (gfx906, HIP), an EPYC 7302
(AVX2) and a Xeon E5-2698 v4 (AVX2). Baseline = commit 4256ff9 (HEAD of
`qinglei/release-cleanup` before the change) built the same way in a worktree.

## Time spent converting (Nsight Systems, A100, N=32768, NB=2048)

| Run | Baseline conversion kernels | OmniCast | Change |
|---|---|---|---|
| kind 3, FP16 band, FP16 accumulation (`--tensor_gemm 0x10`) | 60.7 ms in 1267 launches (7.6 % of GPU time) | 44.6 ms in 1135 launches (5.6 %) | −26.5 % |
| kind 5, adaptive decisions, stochastic rounding | 64.8 ms in 709 launches (6.0 %) | 47.9 ms (4.5 %) | −26.0 % |

## End to end (interleaved A/B, factorization time)

Baseline and OmniCast runs alternate, so slow phases of the machine hit both.
The GPU cases ran 3 times each; the CPU cases 10 times each, because a run
varies by up to 40 % on the 36-core node (fastest to slowest run of one binary). `ab_stats.py` gives the medians, the
median of the paired ratios with a bootstrap 95 % interval, and how often
OmniCast was faster.

| A100, N=32768, NB=2048 | baseline (s) | OmniCast (s) | change (mean) | OmniCast faster |
|---|---|---|---|---|
| `k3_hp_band` | 0.802 | 0.799 | -0.3 % | 2/3 |
| `k3_fp16acc` | 0.774 | 0.766 | -1.1 % | 3/3 |
| `k5_adapt` | 1.144 | 1.133 | -0.9 % | 3/3 |
| `k5_adapt_sr` | 1.168 | 1.166 | -0.2 % | 2/3 |
| `k5_dtconv2` | 1.162 | 1.145 | -1.5 % | 3/3 |
| `k10_runtime` | 1.160 | 1.157 | -0.3 % | 2/3 |

| A100, N=32768, NB=1024 | baseline (s) | OmniCast (s) | change (mean) | OmniCast faster |
|---|---|---|---|---|
| `k3_fp16acc` | 0.650 | 0.640 | -1.6 % | 2/3 |
| `k5_adapt_sr` | 1.163 | 1.156 | -0.6 % | 1/3 |
| `k5_dtconv2` | 1.151 | 1.135 | -1.4 % | 3/3 |

| CPU (36-core Xeon Gold 6140), NB=512, 10 runs | N | baseline median (s) | OmniCast median (s) | paired change [95 % CI] | OmniCast faster |
|---|---|---|---|---|---|
| `cpu_k3_band` | 16384 | 1.467 | 1.487 | -0.2 % [-8.3, +13.3] | 5/10 |
| `cpu_k5_adapt` | 16384 | 1.265 | 1.371 | -2.0 % [-7.0, +14.5] | 6/10 |
| `cpu_k5_adapt_sr` | 16384 | 1.571 | 1.235 | **-18.8 %** [-25.7, -15.8] | 9/10 |
| `cpu_k10_runtime` | 16384 | 1.965 | 1.821 | -5.8 % [-19.6, +3.1] | 7/10 |
| `cpu_k3_band` | 8192 | 0.314 | 0.309 | -0.6 % [-7.4, +5.6] | 6/10 |
| `cpu_k5_adapt` | 8192 | 0.300 | 0.299 | -1.1 % [-3.8, +2.1] | 6/10 |
| `cpu_k5_adapt_sr` | 8192 | 0.377 | 0.306 | **-18.0 %** [-20.3, -12.2] | 10/10 |
| `cpu_k10_runtime` | 8192 | 0.380 | 0.371 | -3.3 % [-11.2, +4.2] | 6/10 |

- **GPU**: conversions are 5–8 % of the kernel time and overlap with GEMMs on
  other streams, so cutting them by a quarter gives 0–1.6 %. That is about the
  run-to-run spread (1–3 %). Four of the nine cases were faster in every run,
  and none was slower by more than the spread.
- **CPU**: stochastic rounding is 18–19 % faster at both sizes; OmniCast's
  stochastic rounding is 3–9 times faster than the scalar loop it replaces. The
  other kinds show no measurable difference: their confidence intervals contain
  zero. A first 3-run comparison reported −13.6 % for runtime decisions; one
  outlier run caused it. Those runs are kept in `ab_cpu_skylake_N*_NB512.txt`.

## Files

- `ab_run.sh`, `ab_cpu.sh`: the A/B scripts; `ab_*.txt`: their outputs (`*_r10.txt`: the
  10-run CPU comparison); `ab_stats.py`: medians and paired intervals.
- `leconte_run.sh`, `leconte_hicma.sh`, `hexane_run.sh`: the V100 / H100 runs.
- The scripts take the baseline build (commit 4256ff9, before OmniCast) from
  `HICMA_BASE`. The verbose CTest logs of the runs were not kept; the accuracy
  comparison is summarized above.
