# Round 6: the FP64 accumulation of the kind-10 runtime CPU GEMM

HiCMA-X after ffcf1cb (host operand cache, round 5), 2026-11-24. All
timings on caffeine (Xeon E5-2660 v4, Broadwell, 14 cores, 35 MB L3)
through Slurm (`srun -p shared -w caffeine --exclusive`), CPU-only build,
sequential MKL 2024.2, `--flush_subnormals 1`, 14 worker threads, NB = 512.
Scripts and data are next to this file.

## Summary

- **Question.** With `ACC_DP = 1`, a kind-10 CPU GEMM that picks FP32 for an
  FP64 tile C runs `CORE_sgemm` with beta = 0 into a work tile W, then
  `C -= (double)W` element by element. That loop has about 5 % of the
  samples (5.1 % in the round-4 profile, 4.8 % after round 5). Can it be
  made faster with bit-identical results?
- **The loop is bound by the memory bandwidth of one core.** It reads the
  2 MB FP64 C tile and the 1 MB FP32 product, and writes C back: 5 MB per
  GEMM. One idle core does this in 0.25-0.29 ms (about 20 GB/s). Under a full
  load of GEMMs it takes 0.35-0.37 ms, 7-8 % of a GEMM. An AVX loop is no
  faster than the compiler's SSE2 loop, and neither is software prefetch.
- **The change: the SGEMM in panels of 256 columns, each subtracted right
  after it is computed** (`hicma_parsec_sgemm_sub_dp`, `ACC_DP_PANEL`). It
  can at most save the FP32 product's trip through the L3, a fifth of the
  bytes, and MKL then packs A twice instead of once. The C stream, the bulk
  of the cost, stays.
- **Bit-identical with MKL only.** MKL 2024.2 gives the same FP32 sums for
  128- or 256-column panels as for one GEMM, for every M from 1 to 512, on
  Broadwell, Ice Lake, Zen 2 and Zen 4 (narrower panels and row panels
  change them on Broadwell). OpenBLAS 0.3.28 changes them for every panel
  width tested, BLIS 5.0 for 128 columns and fewer. So the panels are on
  only when CMake finds MKL (`HICMA_PARSEC_BLAS_MKL`); any other BLAS runs
  one GEMM, as before.
- **Correctness.** The factor hash equals the parent's on 1, 2 and 4 MPI
  processes, for runtime decisions and for kind 10 at 1e-3. CTest
  `-L hicma` passes 30/30 in the CPU-only build. On a V100 node, with the
  GEMMs forced onto the CPU, the emulated-FP16 branch gives the parent's
  hashes as well.
- **A bug the GPU check found, fixed (section 7).** In a process with GPUs,
  a CPU GEMM that widened an FP32 tile to FP64 handed the next GPU task half
  the tile, or overwrote GPU or host memory, in the parent too. PaRSEC's own
  placement never puts such a GEMM on the CPU, so only forced placements
  showed it. Other runs keep their bits.
- **Time: about 1 % at N = 16384, nothing measurable at N = 32768.**
  Interleaved A/B, parent / new, median of paired per-round ratios with a
  bootstrap 95 % interval:

  | N | configuration | rounds | parent / new | new faster |
  |---|---|---|---|---|
  | 16384 | runtime decisions | 12 | 1.012 [1.005, 1.015] | 10/12 |
  | 16384 | kind 10, 1e-3 | 12 | 1.011 [1.000, 1.015] | 9/12 |
  | 32768 | runtime decisions | 8 | 1.003 [0.989, 1.010] | 5/8 |

## 1. What the loop costs

`bench/accbench.c` repeats the GEMM with 14 threads, each with its own
tiles. C cycles through a pool larger than the L3, so it is cold, as in the
factorization. Per 512^3 GEMM (`data/microbench_caffeine.txt`):

| variant | SGEMM (ms) | loop (ms) | total |
|---|---|---|---|
| reference (SGEMM, then the scalar loop: SSE2 at -O3) | 4.30-4.41 | 0.35-0.37 | 1 |
| AVX loop (`vcvtps2pd ymm` + `vsubpd`) | 4.31-4.41 | 0.35-0.38 | 0.998-1.005 |
| AVX + prefetch of C (T0 0.5-2 KB, NTA 1-4 KB ahead) | 4.31-4.42 | 0.34-0.38 | 0.996-1.006 |
| AVX + non-temporal stores of C | 4.34-4.35 | 0.44-0.45 | 1.001-1.007 |
| SGEMM in 256-column panels, AVX loop per panel | | | 0.996-1.000 |
| SGEMM in 128-column panels | | | 1.01 |
| SGEMM in 64-column panels | 7.0 | | 1.50 |

Total: per-GEMM time over the reference's in the same run.

The loop alone on all 14 cores takes 1.1 ms per tile, about 65 GB/s in
all: the node's memory bandwidth. Non-temporal stores slow the loop by 25 % but
speed up the other cores' SGEMMs by 1.5 %, since C no longer evicts their
data from the L3.

In the factorization (`data/profile_caffeine.txt`, perf, old loop) the loop
has 4.8 % of the samples. MKL's packing of the operands
(`sgemm_scopy_down4/24`) has 7.3 %.

## 2. Panels and the FP32 sums

A panel GEMM must give each element of W the same FP32 sum as one GEMM over
the tile. It does when the library blocks K the same way and uses the same
micro-kernel for every panel. `bench/accbench.c bitcheck`, `bench/mcheck.c`
and `bench/panelcheck.c` compare the bits (`data/blas_panel_bits.txt`;
caffeine in `data/microbench_caffeine.txt`):

| library | CPU | 256 / 128 columns | 64 / 32 columns | row panels |
|---|---|---|---|---|
| MKL 2024.2 | Broadwell (caffeine, leconte) | identical, all M | differ | differ |
| MKL 2024.2 | Ice Lake (hexane), Zen 2 (histamine1), Zen 4 (serotonin) | identical, all M | identical | identical |
| OpenBLAS 0.3.28 | Broadwell, Ice Lake, Zen 2, Zen 4 | differ | differ | |
| BLIS 5.0 (AMD) | Ice Lake, Zen 4 | 256 identical, 128 differ | differ | |

"All M": every M from 1 to 512 (`mcheck.c`); the other checks use
M = N = K = 512. BLIS gave no output on Broadwell and Zen 2. On Broadwell,
MKL takes another path below 128 columns: different bits and 50 % slower.
128-column panels make MKL pack A four times, which costs more than the
panels save (0.99 in the factorization). Hence 256.

MKL's packed-GEMM API (`cblas_sgemm_pack`/`compute`) would avoid packing
the cached operands again. It gives the same bits as `cblas_sgemm` on
Ice Lake and AMD, but not on Broadwell, so it is not an option here.

## 3. Variants in the factorization

`experiment_modes.patch` (not committed) selects the variant at run time
(`HICMA_ACC_MODE`), so all variants run in one binary. Ratios of the old
loop over each variant, N = 16384, 4 rounds (`data/explore_ab*.txt`) or
8 rounds (`data/ab_confirm_caffeine_N16384.txt`):

| variant | runtime decisions | kind 10, 1e-3 |
|---|---|---|
| AVX loop | 1.002 | 0.995 |
| AVX + prefetchnta | 1.006 | 0.994 |
| 256-column panels, scalar loop | 1.017; 1.007 (8 rounds) | 1.005; 1.013 (8) |
| 128-column panels | 0.992 | 0.989 |
| non-temporal stores | 1.009, 1.004; 1.016 (8) | 1.010, 1.015; 1.010 (8) |
| 256-column panels + non-temporal stores, except the last GEMM of C | 1.013, 1.008; 1.011 (8) | 1.023, 1.016; 1.019 (8) |
| the same, over the final build (panels) | 1.001 (8) | 1.005 (8) |

Two builds of the same code differ by 0.996-1.003 (parent build / old loop
in the experiment build). Single 4-round A/Bs of one variant moved by up to
1 % from one batch to the next, so only the 8- and 12-round figures count.
Non-temporal stores on top of the panels add at most 0.5 %. They would need
intrinsics and x86 dispatch, and they would likely cost time where C stays
in cache between its GEMMs. Not adopted.

## 4. The change

- `src/core/hicma_parsec_core.c`: `hicma_parsec_sgemm_sub_dp(m, n, k, A,
  lda, B, ldb, W, C, ldc)` runs the SGEMM in panels of `ACC_DP_PANEL`
  columns of C (0: one GEMM) and subtracts each panel right after it is
  computed. Each element still gets one `C -= (double)W`. Used by both
  `ACC_DP` branches of `hicma_parsec_core_gemm_denseC_denseA_denseB_runtime_decision_cpu`.
  - FP32 GEMMs.
  - Emulated FP16 (`!HAVE_HP_CPU`). Its C conversion for the first GEMM
    now comes before the SGEMM; they touch different memory. The CPU
    picks FP16 only in a process with GPUs, and only for a GEMM that runs
    on the CPU: section 5 checks it on a GPU node.
- `ACC_DP_PANEL` defaults to 256 with MKL and 0 otherwise. The top-level
  `CMakeLists.txt` sets `HICMA_BLAS_IS_MKL` when `mkl_core` or `mkl_rt` is
  among the libraries of the `LAPACKE::` targets; `src/CMakeLists.txt` then
  passes `HICMA_PARSEC_BLAS_MKL` to `hicma_parsec`. Configure prints
  "BLAS is MKL: ON|OFF".
- The `ACC_DP = 0` branches (not the default) keep their own loop.

## 5. Correctness

`verify_final.sh`, builds with `../round5/factor_hash.patch` applied for the
check only (`data/correctness.txt`):

| run | parent ffcf1cb | new |
|---|---|---|
| runtime decisions, 1 process, N = 16384 | f68c2e050b29dfcc | same |
| kind 10 1e-3, 1 process, N = 16384 | 84f913aa96c45f8b | same |
| runtime decisions, 2 processes, N = 8192 | 0ee8ed2aa93f7cc7, c605807fbf03a315 | same |
| kind 10 1e-3, 2 processes, N = 8192 | c565c6b201a7872e, 8ad2d5d6df583d95 | same |
| runtime decisions, 4 processes, N = 8192 | a3e8f27e..., 929fbc2d..., 42a9936a..., 153149e2... | same |
| kind 10 1e-3, 4 processes, N = 8192 | a0c78565..., 2227db90..., 685c440e..., 8b0ec432... | same |

Modes 0-4 of the experiment build (old loop, AVX, AVX + prefetchnta,
panels + AVX + prefetchnta, non-temporal stores) also gave these hashes on
1 and 2 processes (`data/verify_modes.txt`). Runtime decisions need N to be a multiple of
NB, so no shorter last tile row reaches this code.

CTest `-L hicma`: 30/30 in the CPU-only build on caffeine
(`data/ctest_caffeine.txt`). The file also compiles with the flags of the
CUDA build, without warnings.

### The emulated-FP16 branch on a GPU node

The runs above never pick FP16. The CPU takes that branch only in a
process with GPUs, and only for a GEMM that runs on the CPU. In kind 10
every GEMM is inside the dense band, so PaRSEC runs it on a GPU (its
default, `device_load_balance_allow_cpu=0`, takes the CPU only when no GPU
can run the task). With `device_load_balance_allow_cpu=1` the CPU ran none
of them either in these runs: PaRSEC picks the GPU that holds or prefers a
task's C tile before it compares loads.
`cpu_gemm_debug.patch` (debug only) fixes the placement instead
(`HICMA_DEBUG_CPU_GEMM=1`: every GEMM on the CPU; `=2`: the GEMMs of odd
k; `=3`: those of k = 0) and counts the CPU GEMMs by decision. `verify_gpu.sh` ran on leconte
(V100, Broadwell host, MKL, so 256-column panels), both builds with this
patch and the hash patch, each case twice (`data/gpu_fp16_branch_leconte.txt`).
With every GEMM on the CPU:

| run | CPU GEMMs: FP16 / FP32 / FP64 | factor hash, parent and new, both runs | residual (all GEMMs on the GPU) |
|---|---|---|---|
| kind 10 1e-3, 1 process, N = 16384 | 4928 / 32 / 0 | ec414125a43934cc | 1.03e-5 (1.05e-5) |
| runtime decisions, 1 process, N = 16384 | 2832 / 1839 / 289 | fff31805ac879fb3 | 3.79e-8 (3.79e-8) |
| kind 10 1e-3, 2 processes, N = 8192 | 555 / 5 / 0 | 699b4888..., 246d0646... | 1.15e-5 (1.15e-5) |
| runtime decisions, 2 processes, N = 8192 | 174 / 305 / 81 | f7d1e650..., 4bebd964... | 1.58e-8 (1.46e-8) |

The residuals of runtime decisions are above the tester's default bound
(`--fixedacc`, 1e-8) with every GEMM on the GPU too; the CTest case uses
1e-7. The default placement and the load-balanced one also give the same
hashes in both builds.

With the GEMMs of odd k on the CPU (`=2`), the hashes match as well, but
the factor is wrong in both builds (residual 89 to 6e74). That placement is
not a valid test. It hands C from a GPU GEMM to a CPU GEMM, and a CPU task
reads the host copy, which a GPU task updates only when it pushes its
output back. Kind 10's GEMMs never push C back, since PaRSEC sends a GEMM to
the GPU that holds its C, so the CPU GEMM read a stale tile (a trace showed
it). The valid placements, CPU GEMMs and then GPU tasks (`=1`, `=3`), found
a real fault with `--adaptive_decision 1`, fixed in section 7.

## 6. Not done, and follow-ups

- **Fewer passes over C.** C's 4 MB of FP64 traffic per GEMM is the cost.
  A bit-identical way to cut it changes the dataflow: keep a
  GEMM's FP32 product until the next GEMM on the same tile, then do
  `C -= W_k; C -= W_k+1` in one pass. That saves about 2 MB per GEMM but
  holds 1 MB per tile between GEMMs.
- **MKL's packing (7 % of the samples).** The packed-GEMM API changes the
  bits on Broadwell (section 2).
- **A GEMM on the CPU after a GPU GEMM on the same tile** would read a stale
  host copy: the GPU GEMMs do not push C back (section 5). PaRSEC's placement
  never does this; a placement that did would need the push-back.
- **FP16 on the CPU and on the GPU.** Under `ACC_DP` the emulated-FP16 CPU
  GEMM widens an FP32 C to FP64, where the GPU FP16 GEMM accumulates it in
  FP32. Both are correct, but a tile's precision then depends on where its
  GEMMs ran.

## 7. CPU GEMMs on FP32 tiles (fixed)

With `--adaptive_decision 1`, the forced placements that hand C from the
CPU to a GPU (`=1`, `=3`) failed in the parent ffcf1cb (8e8fde8 has the
same code there). All three faults are in the step where a runtime CPU GEMM widens an FP32 C to
FP64: the FP64 GEMM, and the first update of the emulated-FP16 GEMM under
`ACC_DP`.

- **The span.** `convert_d2s` stores the tiles decided FP32, FP16 or FP8 in
  FP32 and sets their span, the byte count PaRSEC uses for every copy of a
  tile, to the FP32 size. The CPU widened C in place but left the span, so
  the next GPU task on C (a GEMM or the TRSM) got half the FP64 tile.
- **GPU copies at the FP32 span.** The warmup stages every tile on its GPU
  after `convert_d2s`, so a GPU holds an FP32-sized copy of these tiles.
  PaRSEC reuses an existing GPU copy without checking its size. With the
  span restored, it copied the FP64 tile into that copy and over the next
  tile in GPU memory. At N = 2048 the stage-in of C(3,2) overwrote the panel
  tile A(3,1) 1 MB above it (`data/trace_widen_N2048.txt`).
- **Host buffers.** With `--adaptive_memory` and without `--check`,
  `convert_d2s` gives these tiles FP32-sized host buffers, and the in-place
  widening wrote past them (segmentation faults).

The fix, in the style of the GPU GEMM's own FP64 promotion:

- `hicma_cpu_promote_tile_fp64` widens C, moves it to an FP64-sized host
  buffer when its buffer is FP32-sized, and restores the span. The runtime
  CPU GEMM gets C's data copy for this.
- `hicma_gpu_stage_in` is the stage-in of the GPU GEMM, TRSM and SYRK of
  kind 10, the tasks that read such tiles: C, or a panel operand staged onto
  another GPU of the process. It grows a GPU copy smaller than the span,
  then runs PaRSEC's stage-in. It reads the copy's size from PaRSEC's zone
  allocator.

In a process with GPUs, default runs never widen a tile on the CPU. Kind
10 puts every GEMM inside the band on a GPU, and with
`device_load_balance_allow_cpu=1` PaRSEC put none on the CPU either, up to
N = 32768. Every GPU stage-in of these tasks pays one size lookup.

`verify_widen.sh` on leconte, parent ffcf1cb against 8e8fde8 with the fix,
both with the hash and debug patches (`data/widen_fix_leconte.txt`):

| runs | parent | fix |
|---|---|---|
| `--adaptive_decision 1`, placements 1 and 3, 1 and 2 processes, kind 10 1e-3 and runtime decisions (8) | wrong (residual 78 to 3e53) or aborted | right: 1.03e-5 to 1.14e-5 for kind 10, 1.46e-8 to 3.79e-8 for runtime decisions, as with every GEMM on the GPU |
| the same, one process with two GPUs (4) | wrong or aborted | right, the same bits as with one GPU |
| `--adaptive_memory 1`, no `--check`, placements 1 and 3 (4) | segmentation fault | completes |
| `--adaptive_decision 0` with placements 1 and 3; every GEMM on the GPU; PaRSEC load balancing; no GPU (29 pairs) | | same hashes |

The CPU-only runs with `--adaptive_decision 0` give the hashes of the caffeine
runs of section 5. CTest
`-L hicma` passes 55/55 on leconte (`data/ctest_gpu_leconte.txt`) and 30/30
in the CPU-only build on caffeine (`data/ctest_caffeine_widen.txt`).

## Files

- `bench/accbench.c`: loop variants, MKL panel and packed-API bit checks.
  `bench/mcheck.c`: MKL panels for M = 1..512. `bench/panelcheck.c`: the
  same with any CBLAS (OpenBLAS, BLIS).
- `experiment_modes.patch`: the variants, `HICMA_ACC_MODE` 0-9, on top of
  ffcf1cb. `ab_acc.sh`, `ab_confirm.sh`: interleaved A/Bs (from
  `../round5/ab.sh`; statistics with `../round5/ab_stats.py`).
  `verify_acc.sh`, `verify_final.sh`: factor hashes. `prof.sh`: perf
  profiles.
- `verify_gpu.sh` (with `gpu_of_rank.sh`, one GPU per rank) and
  `cpu_gemm_debug.patch`: the GPU-node check of section 5, CUDA builds of
  both revisions with that patch and `../round5/factor_hash.patch`.
- `verify_widen.sh` (with `run1.sh`, one run): the runs of section 7.
  `data/trace_widen_N2048.txt` comes from a build with extra prints.
- `data/`: the outputs quoted above. `ab_pooled_caffeine_N16384.txt` joins
  the 4 rounds of `ab_final4_caffeine_N16384.txt` and the parent/new arms of
  the 8 confirmatory rounds.
