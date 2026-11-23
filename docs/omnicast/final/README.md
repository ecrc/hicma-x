# HiCMA-X with and without OmniCast (2026-11-22; H100 2026-11-23)

The first round compared HiCMA-X with its own conversion kernels against the
first OmniCast ([`../README.md`](../README.md)). Rounds 2 and 3 were then
measured against the round before them, never again against HiCMA-X without
OmniCast. This page closes that gap: the final state against the code before
OmniCast, on the same machines, with identical builds.

| build | commit | conversions |
|---|---|---|
| without OmniCast | `4256ff9` | HiCMA-X 0.2's own CPU loops and CUDA kernels |
| with OmniCast | `f0ec341` | OmniCast v2 `0c59754` (round 3); the GPU operand cache on (default) or off (`--operand_cache 0`) |

Both trees have the same DPLASMA, PaRSEC, HCORE and STARS-H commits. They were
built side by side in git worktrees with the same options (Release, CUDA
architectures 70;80;90a, MKL, CUDA 12.6).

![speedups](figures/with_without_omnicast.png)

## Results

**Accuracy: identical.** On the A100 and on the H100 (N = 16384, NB = 2048,
`--check 1`), all 10 configurations give the same residual ||LL^T − A||/||A||
to the last printed digit without OmniCast, with OmniCast, and with the
operand cache. On the H100 that includes the FP8 GEMMs of kinds 8 and 10
([`data/a100_correctness_N16384.txt`](data/a100_correctness_N16384.txt),
[`data/h100_correctness_N16384.txt`](data/h100_correctness_N16384.txt)).
Three runs report "FAILED" in all three builds alike, on both GPUs: the two
kind-3 band runs and runtime decisions at 1e-8. They ran with the default
bound of 1e-8 (`--check_tol`), which an FP16 band does not reach and the
runtime run misses at 1.54e-8.

| configuration | A100 residual (all three builds) | H100 residual (all three builds) |
|---|---|---|
| FP64 dense | 5.342370e-16 | 5.462145e-16 |
| kind 3, FP32/FP16 band | 1.237078e-07 | 1.211495e-07 |
| kind 3, band, FP16 accumulation | 1.344776e-05 | 1.272850e-05 |
| kind 5, adaptive, 1e-8 | 2.877597e-09 | 2.876152e-09 |
| kind 5, adaptive, 1e-4 | 1.221633e-07 | 1.115505e-07 |
| kind 5, 1e-4, stochastic rounding | 1.461554e-07 | 1.433681e-07 |
| kind 5, 1e-4, `--datatype_convert 2` | 1.221633e-07 | 1.115505e-07 |
| kind 8, 1e-3 (FP8 tiles use FP16 on the A100) | 4.895639e-06 | 1.300978e-05 |
| kind 10, runtime decisions, 1e-3 | 4.787225e-06 | 6.964380e-06 |
| runtime decisions, 1e-8 (kind 10) | 1.537963e-08 | 1.537627e-08 |

**H100** (hexane, H100 PCIe, FP8 GEMMs on), N = 45056, NB = 2048,
factorization time. "faster" counts the rounds in which the build with
OmniCast was faster.

| configuration | without OmniCast (s) | OmniCast (s) | speedup [95 % CI] | faster | OmniCast + operand cache (s) | speedup [95 % CI] | faster |
|---|---|---|---|---|---|---|---|
| FP64 dense (no conversions) | 1.259 | 1.256 | 1.005 [0.989, 1.017] | 3/5 | 1.261 | 0.997 [0.985, 1.016] | 2/5 |
| kind 3, FP32/FP16 band | 0.659 | 0.643 | 1.024 [1.022, 1.026] | 5/5 | 0.642 | 1.023 [1.017, 1.029] | 5/5 |
| kind 3, band, FP16 accumulation | 0.653 | 0.628 | 1.038 [1.028, 1.043] | 5/5 | 0.631 | 1.034 [1.019, 1.042] | 5/5 |
| kind 5, adaptive, 1e-8 | 1.083 | 1.074 | 1.012 [1.000, 1.014] | 5/5 | 1.061 | 1.023 [1.015, 1.025] | 5/5 |
| kind 5, adaptive, 1e-4 | 0.745 | 0.732 | 1.018 [1.003, 1.023] | 5/5 | 0.715 | 1.037 [1.035, 1.049] | 5/5 |
| kind 5, 1e-4, stochastic rounding | 0.791 | 0.752 | 1.052 [1.045, 1.054] | 5/5 | 0.722 | 1.096 [1.092, 1.102] | 5/5 |
| kind 5, 1e-4, `--datatype_convert 2` | 0.999 | 0.975 | 1.014 [0.961, 1.111] | 3/5 | 0.959 | 1.040 [0.925, 1.080] | 4/5 |
| kind 8, 1e-3 (FP8 GEMMs) | 0.531 | 0.450 | 1.181 [1.152, 1.203] | 5/5 | 0.428 | 1.242 [1.231, 1.277] | 5/5 |
| kind 10, runtime decisions, 1e-3 (FP8 GEMMs) | 0.643 | 0.519 | 1.241 [1.182, 1.254] | 5/5 | 0.483 | 1.329 [1.301, 1.335] | 5/5 |
| runtime decisions, 1e-8 (kind 10) | 1.069 | 1.013 | 1.054 [1.048, 1.066] | 5/5 | 0.997 | 1.072 [1.059, 1.083] | 5/5 |

**A100** (methane, A100-PCIE-40GB), N = 45056, NB = 2048, factorization time.
"faster" counts the rounds in which the build with OmniCast was faster.

| configuration | without OmniCast (s) | OmniCast (s) | speedup [95 % CI] | faster | OmniCast + operand cache (s) | speedup [95 % CI] | faster |
|---|---|---|---|---|---|---|---|
| FP64 dense (no conversions) | 2.679 | 2.651 | 1.011 [0.991, 1.022] | 3/4 | 2.648 | 1.017 [0.981, 1.021] | 3/4 |
| kind 3, FP32/FP16 band | 1.262 | 1.243 | 1.017 [1.013, 1.019] | 4/4 | 1.245 | 1.013 [1.010, 1.016] | 4/4 |
| kind 3, band, FP16 accumulation | 1.206 | 1.175 | 1.024 [1.021, 1.029] | 4/4 | 1.178 | 1.024 [0.858, 1.026] | 3/4 |
| kind 5, adaptive, 1e-8 | 2.238 | 2.224 | 1.015 [0.992, 1.143] | 3/4 | 2.217 | 1.006 [0.999, 1.172] | 3/4 |
| kind 5, adaptive, 1e-4 | 1.505 | 1.488 | 1.014 [0.994, 1.017] | 3/4 | 1.471 | 1.023 [1.006, 1.034] | 4/4 |
| kind 5, 1e-4, stochastic rounding | 1.578 | 1.520 | 1.036 [1.030, 1.041] | 4/4 | 1.490 | 1.059 [1.045, 1.078] | 4/4 |
| kind 5, 1e-4, `--datatype_convert 2` | 1.774 | 1.777 | 0.989 [0.949, 1.018] | 1/4 | 1.769 | 0.994 [0.896, 1.016] | 2/4 |
| kind 8, 1e-3 (FP16 on the A100) | 0.993 | 0.978 | 1.004 [0.992, 1.035] | 2/4 | 0.934 | 1.063 [1.015, 1.076] | 4/4 |
| kind 10, runtime decisions, 1e-3 | 1.125 | 1.110 | 1.017 [1.007, 1.089] | 4/4 | 1.063 | 1.062 [1.049, 1.238] | 4/4 |
| runtime decisions, 1e-8 (kind 10) | 2.134 | 2.116 | 1.006 [1.003, 1.015] | 4/4 | 2.089 | 1.018 [1.008, 1.036] | 4/4 |

**CPU** (caffeine, Xeon E5-2660 v4, Broadwell, 28 cores, AVX2), N = 16384,
NB = 512. The operand cache is GPU only.

| configuration | without OmniCast (s) | OmniCast (s) | speedup [95 % CI] | faster |
|---|---|---|---|---|
| FP64 dense (no conversions) | 3.942 | 3.941 | 1.000 [0.998, 1.002] | 7/10 |
| kind 3, FP32 band | 4.836 | 4.827 | 1.006 [0.997, 1.009] | 7/10 |
| kind 5, adaptive, 1e-4 | 4.847 | 4.860 | 0.998 [0.985, 1.003] | 3/10 |
| kind 5, 1e-4, stochastic rounding | 5.066 | 4.903 | 1.030 [1.020, 1.054] | 9/10 |
| runtime decisions, 1e-8 (kind 10) | 5.569 | 5.422 | 1.029 [0.992, 1.055] | 7/10 |

## Reading

- **H100, where the FP8 GEMMs run: up to 1.33×.**
  - Kind 8 is 1.18× faster with OmniCast alone and 1.24× with the operand
    cache. Kind 10 at 1e-3 is 1.24× and 1.33×. Both were faster in all 5
    rounds.
  - Before OmniCast, each FP8 GEMM operand went through HiCMA-X's own
    transposed FP8 conversion (one byte per thread, strided stores, a memset,
    an amax pass and a scale). OmniCast's version of it is 4.8–5.1× faster in
    isolation (second round, 8192² tiles), and the cache then converts each operand once per
    factorization.
  - Stochastic rounding: 1.05× and 1.10×. The FP32/FP16 kinds: 1.01–1.04×.
    FP64 dense, which converts nothing, is 1.005× and 0.997×.
  - Every mixed-precision configuration was faster in 5 of 5 rounds, except
    `--datatype_convert 2`, whose intervals contain 1.
- **A100, OmniCast alone: 1.01–1.04×.**
  - Stochastic rounding gains most (1.036×, faster in all 4 rounds), then
    the FP16 band kinds (1.017× and 1.024×, 4/4).
  - The others lie within ±2 %. That is also where FP64 dense lands
    (1.011×), and FP64 dense converts nothing, so ±2 % is the noise level here.
  - Conversions are 6–8 % of the GPU time on the A100 (Nsight Systems, first
    round) and overlap with GEMMs, so a faster conversion kernel can only move
    the total by a few percent. In the first
    round, at N = 32768, OmniCast gave 0–1.6 %.
- **A100, with the GPU operand cache: up to 1.06×.** Each GPU now converts a
  GEMM operand once per factorization instead of once per GEMM:
  - stochastic rounding: 1.059×;
  - kind 8 (FP8 tiles become FP16 on the A100): 1.063×;
  - kind 10: 1.062×;
  - all three faster in 4 of 4 rounds.

  The cache is built on OmniCast's conversion calls, so it exists only in the
  build with OmniCast.
- **`--datatype_convert 2`: no measurable change** (0.989× and 0.994×; both
  intervals contain 1, and OmniCast was faster in only 1 and 2 of 4 rounds).
- **CPU: stochastic rounding 1.03× (9/10 rounds), the rest unchanged.**
  These CPU runs predate `--flush_subnormals`. On Intel CPUs the
  mixed-precision kinds were dominated by subnormal arithmetic in MKL's SGEMM;
  with flushing they are 1.6–2.6× faster (`../round4/REPORT.md`). In the
  first round, on methane's 36-core Skylake-SP (AVX-512), stochastic rounding
  was 18–19 % faster. The Broadwell is 2.9× slower per factorization (FP64
  dense 3.94 s against 1.37 s), so its GEMMs take a larger share and the
  conversions a smaller one. Runtime decisions show 1.029× but the interval
  contains 1.
- **The intervals of the A100 runs are coarse.** They come from 4 paired
  rounds; the "faster" column is the more robust reading there.

## Not measured

- **The V100 and the Radeon VII.** HiCMA-X had no HIP FP8 path before
  OmniCast, and the Radeon VII has no FP8.

## Method

- **Interleaved rounds.** Each round runs every configuration with the three
  builds back to back, so a slow phase of the machine hits all three.
  Ratios are paired by round.
- **Timing.** Each process factorizes 3 times (`--nruns 3`); the first run is
  dropped as warm-up and the other two are averaged.
- **Rounds.** 4 rounds on the A100, 5 on the H100, 10 on the CPU. The 95 % intervals are
  bootstrap intervals of the median of the paired ratios.
- **A100 cool-down.** The A100-PCIE in methane throttles thermally: it ran at
  82–83 °C against an 85 °C limit, and its counter showed hours of "SW Thermal
  Slowdown". Before each process the script waits until the GPU is at or
  below 65 °C, and it logs the throttle time of each run: 0 ms in all 120
  runs, at most 72 °C. The process is bound to the GPU's NUMA node
  (`numactl --cpunodebind=1 --membind=1`).
- **H100.** Run as one `sbatch --exclusive` job on hexane (2026-11-23,
  12:35–13:27 UTC), which waited until nobody else was on the node. It used
  the same scripts with `GPU=0 NUMA=0`: the H100 is GPU 0, on NUMA node 0,
  and memory on node 1 copies 2.5× slower. Same cool-down rule: at most
  70 °C, and 0 ms of thermal slowdown in all 150 runs.
- **Exclusive nodes.** Both nodes were allocated with
  `salloc/srun --exclusive`, with nothing else of ours running on them. The
  CPU runs use `srun`; a plain `salloc -w node script` would run the script on
  the login node.
- **Discarded attempts.** Two earlier attempts were thrown away:
  - CPU runs meant for caffeine ran on methane next to the A100 runs;
  - an A100 run overlapped a cancelled one whose script kept going after
    `scancel`, and the overlapped processes failed.

  Their logs are not used anywhere.

## Reproducing

```bash
# the two trees, built the same way (worktree recipe: submodules need protocol.file.allow)
git worktree add --detach ../hicma-x-cmp-base4256 4256ff9
git worktree add --detach ../hicma-x-cmp-head f0ec341
# ... submodule init, then in each: cmake -G Ninja -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release \
#     -DHICMA_PARSEC_HAVE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;80;90a" (and build_cpu with CUDA OFF)
export HICMA_BASE=../hicma-x-cmp-base4256/build_cuda HICMA_HEAD=../hicma-x-cmp-head/build_cuda
salloc -p shared -w methane --exclusive --gres=gpu:nvidia:1 ./gpu_check.sh   # residuals, N=16384
ROUNDS=4 salloc -p shared -w methane --exclusive --gres=gpu:nvidia:1 ./gpu.sh > data/a100_N45056.txt
# H100: GPU 0 on NUMA node 0 (sbatch runs the script on the node)
GPU=0 NUMA=0 OUT_DIR=. sbatch -p shared -w hexane --exclusive --gres=gpu:nvidia:2 --wrap \
    "./gpu_check.sh > data/h100_correctness_N16384.txt; ROUNDS=5 ./gpu.sh > data/h100_N45056.txt"
export HICMA_BASE=../hicma-x-cmp-base4256/build_cpu HICMA_HEAD=../hicma-x-cmp-head/build_cpu
ROUNDS=10 srun -p shared -w caffeine --exclusive ./cpu.sh > data/cpu_broadwell_N16384.txt
python3 make_figure.py                                                       # tables and figure
```

`gpu.sh` and `gpu_check.sh` take the GPU index and its NUMA node as `GPU` and
`NUMA` (defaults 0 and 1, methane's A100).
