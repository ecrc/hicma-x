# OmniCast in HiCMA-X

HiCMA-X converts tiles between FP64, FP32, FP16, BF16 and FP8 all the time:
before mixed-precision GEMMs and TRSMs, when precisions are decided, for the
FP8 operands of cuBLASLt, with stochastic rounding. All of it now goes through
[OmniCast](../../omnicast) (submodule, branch `v2`); this page records what
changed, how it was validated and what it bought. OmniCast's own design and
kernel measurements are in `omnicast/docs/DESIGN.md` and
`omnicast/docs/PERFORMANCE.md`.

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

## End to end (interleaved A/B, 3 runs each, factorization time)

| A100, N=32768, NB=2048 | baseline (s) | OmniCast (s) | change |
|---|---|---|---|
| `k3_hp_band` | 0.802 | 0.799 | -0.3 % |
| `k3_fp16acc` | 0.774 | 0.766 | -1.1 % |
| `k5_adapt` | 1.144 | 1.133 | -0.9 % |
| `k5_adapt_sr` | 1.168 | 1.166 | -0.2 % |
| `k5_dtconv2` | 1.162 | 1.145 | -1.5 % |
| `k10_runtime` | 1.160 | 1.157 | -0.3 % |

| A100, N=32768, NB=1024 | baseline (s) | OmniCast (s) | change |
|---|---|---|---|
| `k3_fp16acc` | 0.650 | 0.640 | -1.6 % |
| `k5_adapt_sr` | 1.163 | 1.156 | -0.6 % |
| `k5_dtconv2` | 1.151 | 1.135 | -1.4 % |

| CPU (36-core Xeon Gold 6140), N=16384, NB=512 | baseline (s) | OmniCast (s) | change |
|---|---|---|---|
| `cpu_k3_band` | 1.486 | 1.444 | -2.8 % |
| `cpu_k5_adapt` | 1.302 | 1.228 | -5.7 % |
| `cpu_k5_adapt_sr` | 1.559 | 1.361 | -12.7 % |
| `cpu_k10_runtime` | 2.003 | 1.730 | -13.6 % |

On the GPU the conversions are 5–8 % of the kernel time and overlap with GEMMs
on other streams, so cutting them by a quarter gives 1–2 %. On the CPU,
OmniCast replaces MKL's `dlag2s`/`slag2d` and the scalar stochastic rounding;
runs with many conversions gain the most (runtime decisions −14 %,
stochastic rounding −13 %). Earlier, smaller CPU runs (N=8192) are in
`ab_cpu_skylake_N8192_NB512.txt`; they are noisy at 0.3 s.

## Files

- `ab_run.sh`, `ab_cpu.sh`: the A/B scripts; `ab_*.txt`: their outputs.
- `leconte_run.sh`, `leconte_hicma.sh`, `hexane_run.sh`: the V100 / H100 runs.
- The verbose CTest logs (baseline and OmniCast, A100/V100/H100) are outside the
  repository, in `/home/qcao3/hicma-x-omnicast-logs/`.
