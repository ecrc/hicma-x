# Round 5: converting once on the CPU, a host operand cache for kind 10

HiCMA-X `qinglei/release-cleanup` after f5142be (OmniCast v2.0.1),
2026-11-23. All timings on caffeine (2x Xeon E5-2660 v4, Broadwell) through
Slurm (`srun -p shared -w caffeine --exclusive`), CPU-only build, sequential
MKL 2024.2, `--flush_subnormals 1`, 14 worker threads (the tester's default
on this node), NB = 512. Scripts and data are next to this file.

## Summary

- **Problem.** After round 4 made MKL's FP32 GEMMs fast on Intel CPUs, the
  conversions of kind 10 (runtime decisions) showed. Every CPU GEMM that
  picks FP32 converts both of its operands from FP64, so at N = 16384 there
  were 9342 conversions of 496 panel tiles per factorization: 10.0 % of the
  time in OmniCast's `d2s`.
- **Change.** The kind-10 CPU GEMM now takes its FP32 operands from a host
  operand cache:
  - one FP32 copy per panel tile, converted once;
  - freed after the last GEMM of the process that reads the tile;
  - bounded: by default a quarter of the process's panel tiles. At the
    bound, the cache evicts the copy whose tile has the fewest GEMMs left to
    read it.

  It uses the GPU cache's switch, `--operand_cache` (0 turns it off).
- **Results are bit-identical.** The factor's hash is the same with the cache
  off, at its default and at 64, 4 and 1 tiles, on 1, 2 and 4 MPI processes.
  The residuals equal those of the parent build. CTest `-L hicma` passes
  30/30.
- **Conversions** at the default bound (130 MB) fall from 9342 to 2454 per
  factorization, and to 486 with an unbounded cache.
- **Time.** Interleaved A/B, 4 rounds, cache off vs default:
  - **1.134x** for runtime decisions and **1.146x** for kind 10 at 1e-3
    (N = 16384);
  - **1.163x** for runtime decisions at N = 32768.

  The cache was faster in every round. With the cache off, the time matches
  the parent build (0.994x, within noise).
- **Default bound: a quarter of the panel tiles.** It keeps the FP32 copies
  under an eighth of the matrix's memory and gets 75-85 % of the possible
  gain. Half the tiles would get about 95 % for twice the memory (section
  3.3); `--operand_cache N` sets any bound.

## 1. Where the time went

From the round-4 profile (`perf record`, N = 16384, `--flush_subnormals 1`,
`/home/qcao3/omc-runs/cpuprof/k10_ftz.*`):

| share | where |
|---|---|
| 46.9 % | MKL `sgemm` kernels |
| 15.0 % | MKL `dgemm` kernel |
| 10.0 % | OmniCast `d2s_any`, called from `hicma_parsec_core_gemm_denseC_denseA_denseB_runtime_decision_cpu` |
| 5.1 % | that function itself |

- **The conversions.** Every panel tile leaves its TRSM in FP64: kind 10 on
  a CPU keeps all tiles in FP64 (528 of 528 here; `data/coverage.txt` shows
  the same for three problems and six accuracies). 4671 of the 4960 GEMMs
  choose FP32, and each converts A(m,k) and B(n,k): 9342 conversions. A panel
  tile is read by up to NT − 2 GEMMs, so each of the 496 panel tiles was
  converted about 19 times.
- **The 5.1 %.** It is not a norm computation: with `ACC_DP = 1` (the
  default) the FP32 GEMMs do not recompute the norm of C. `perf annotate`
  puts the time in the loop that accumulates the FP32 product into the FP64
  tile, `C[i] -= C_use[i]` (SSE2 `cvtps2pd`/`subpd`, 56 % of the samples on
  the load of C). That loop streams 5 MB per GEMM and is out of scope here
  (section 6).

## 2. The host operand cache

- **Where.** `hicma_parsec_core_gemm_denseC_denseA_denseB_runtime_decision_cpu`
  (kind 10 with `--adaptive_decision_runtime 1`) gets its FP32 operands from
  `hicma_cpu_operand_fp32`. That function asks the cache first, and converts
  into the GEMM's own work buffer only when the cache cannot help. The JDF,
  kinds 8 and 9 and the GPU code are not touched. The cache is in
  `src/core/hicma_parsec_datatype_convert.c` (`hicma_host_operand_cache_*`).
- **Why the results do not change.** A panel tile does not change after its
  TRSM. The FP64 → FP32 conversion is deterministic: round to nearest, and
  kind 10 has no stochastic rounding. The cache and the fallback call the
  same function (`hicma_host_operand_convert`, the same OmniCast call as
  before). So a cached copy holds the bits a per-GEMM conversion would
  produce.
- **Converted once.** The first GEMM that needs a copy takes a buffer, marks
  the copy as being converted and converts it outside the cache's mutex.
  GEMMs that ask for the same copy meanwhile pin it and wait on a condition
  variable.
- **Freed after the last reader.** Each panel tile (m, k) counts the GEMMs
  of this process that read it:
  - as A of GEMM (m, n, k), and as B of GEMM (i, m, k);
  - each GEMM runs on the process that owns its C, and only GEMMs with C, A
    and B dense count.

  Every such GEMM calls `hicma_host_operand_cache_release` once, whatever
  precision it chose. When all of them are done, the copy goes back to the
  free buffers. A GEMM pins the copy from `_get` to `_release`, so a copy is
  never freed or evicted while a GEMM reads it. The statistics count the
  copies still held after the factorization (`held=`). It was 0 in every
  run, on 1, 2 and 4 processes, so the reader counts are exact.
- **Bounded.** Buffers are allocated when first needed, up to the bound. They
  are then reused, also across factorizations. At the bound, a new copy
  takes the buffer of an unpinned copy (section 3). If every copy is pinned,
  the GEMM converts into its own buffer, as before.
- **Per factorization.** `potrf_L_dense_mp_gpu_fp8_adaptive_New` empties the
  cache before each factorization, while no task runs (`--nruns` included).
- **Only without GPUs.** A process with GPUs has no host cache: its GEMMs run
  on the GPUs, which have their own operand caches (round 3).
- **Option.** `--operand_cache` takes -1 (default: automatic bound), 0 (off:
  convert per GEMM, as before), or N (at most N tiles; on the GPUs it stays
  "tiles per precision"). With `--verbose 2` each run prints
  `CPU_operand_cache: conversions= reuses= fallbacks= evictions= tiles= MB= held=`.
  `Number_of_Datatype_Conversions` counts the conversions that remain.

## 3. Eviction and the bound

### 3.1 The live set is most of the matrix

With an unbounded cache, the copies alive at the same time peak at 446 of
the 496 panel tiles at N = 16384 (468 MB), and at 1730 of 2016 at N = 32768
(1.8 GB). On the CPU, PaRSEC's task priorities run the GEMMs roughly row by
row: GEMM (m, n, k) ranks by (NT − m)³ first. Row i reads almost every panel
tile (n, k), n < i, once as B. So from one row to the next, the B operands
form a cyclic scan over nearly all the panel tiles. Few tiles are dead early,
and freeing a copy after its last reader alone does not bound the memory.

### 3.2 Which copy to evict

Mean conversions per factorization at the same bound, N = 16384 (one
process, three factorizations each; `data/policy.txt`):

| policy at the bound | 64 tiles | 128 tiles | 256 tiles |
|---|---|---|---|
| no eviction (convert in the GEMM's buffer) | 6722 | 5779 | 2916 |
| least recently used (LRU) | 5309 | 3273 | 2264 |
| most readers left, then LRU | 6411 | 5634 | 3429 |
| fewest readers left, then LRU | 4065 | 2793 | 1431 |
| **fewest readers left, then most recently used** | 4176 | **2141** | **1133** |

And at N = 32768:

| policy | 128 tiles | 256 tiles | 504 tiles |
|---|---|---|---|
| fewest readers left, then LRU | 39924 | 28726 | 15879 |
| **fewest readers left, then most recently used** | **37481** | **23420** | **15592** |

- On a cyclic scan, LRU evicts exactly the copy that will be read next.
- A tile whose own row is still to come has more readers left, so evicting
  by readers left keeps the tiles that will be read most.
- Among the tiles with the same number of readers left, evicting the most
  recently used copy keeps a stable set across rows. This is the classic
  remedy for scans.

The cache keeps unpinned copies in lists by readers left, each in order of
last use. Pinning, a release and an eviction each cost O(1) under one mutex;
an eviction skips empty lists from the lowest one up.

Run-to-run spread is large (at N = 32768 and 256 tiles, LRU tie-break gave
31858, 29026 and 18368 in one process), so differences under about 10 % in
these tables mean little. Both the ranking and the model above favor the
last row.

### 3.3 The default bound

The fraction of the possible gain depends on the fraction of panel tiles the
cache may hold, about equally at both sizes. Mean of factorizations 2 and 3,
one process (`data/policy.txt`):

| N | off | 1/16 of the tiles | 1/8 | 1/4 | 1/2 | unbounded |
|---|---|---|---|---|---|---|
| 16384 (496 tiles) | 2.674 s | - | 2.416 s (64) | 2.311 s (128) | 2.262 s (256) | 2.250 s |
| 32768 (2016 tiles) | 19.968 s | 18.237 s (128) | 17.936 s (256) | 17.231 s (504) | 16.783 s (1008) | 16.566 s |

The default bound is a quarter of the process's own panel tiles, and at least
2·NT (`HICMA_HOST_OPERAND_CACHE_AUTO`): 124 tiles (130 MB) at N = 16384 and
504 tiles (528 MB) at N = 32768. The FP32 copies then take at most an eighth
of the memory of the FP64 panel tiles, and the cache gets 80-86 % of the
unbounded gain in this table (75-77 % with the A/B times of section 5). Half
the tiles would get about 95 %, with copies up to a quarter of the matrix's
memory. In the N = 32768 row, the 1/2 column ran with the LRU tie-break (the
final policy was not run at that bound); the unbounded column never evicts. `--operand_cache N` sets any bound. In a run
with several processes, the default follows each process's own tiles, so the
memory stays proportional to its share of the matrix.

## 4. Correctness

- **Bit for bit.** `factor_hash.patch` (debug only, not committed) prints a
  64-bit FNV hash of every local tile of the factor after the check
  converts it to FP64. `verify.sh` runs the two kind-10 configurations at
  N = 16384 with the cache off, at its default, and at 64, 4 and 1 tiles
  (the last two evict and fall back all the time). It also runs them at
  N = 8192 on 2 and 4 MPI processes with the cache off, at its default and at
  8 tiles. Within each configuration every process's hash is the same under
  all cache settings (`data/correctness.txt`).
- **Residuals** ‖LLᵀ − A‖_F / ‖A‖_F, the same in every column, the parent
  build (f5142be) included:

  | configuration | processes | parent | cache off | default | small bound |
  |---|---|---|---|---|---|
  | runtime decisions | 1 | 2.672243e-09 | 2.672243e-09 | 2.672243e-09 | 2.672243e-09 (64, 4, 1) |
  | runtime decisions | 2 | 7.547642e-09 | 7.547642e-09 | 7.547642e-09 | 7.547642e-09 (8) |
  | runtime decisions | 4 | 7.547642e-09 | 7.547642e-09 | 7.547642e-09 | 7.547642e-09 (8) |
  | kind 10, 1e-3 | 1 | 4.871434e-08 | 4.871434e-08 | 4.871434e-08 | 4.871434e-08 (64, 4, 1) |
  | kind 10, 1e-3 | 2 | 4.939674e-08 | 4.939674e-08 | 4.939674e-08 | 4.939674e-08 (8) |
  | kind 10, 1e-3 | 4 | 4.939674e-08 | 4.939674e-08 | 4.939674e-08 | 4.939674e-08 (8) |

  ("runtime decisions" is `--adaptive_decision_runtime 1`; "kind 10, 1e-3" is
  `--kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3`.)
- **No copy left behind.** `held= 0` after every factorization above.
- **CTest** `-L hicma`: 30/30 in the CPU-only build on caffeine
  (`data/ctest_caffeine.txt`), including `potrf_runtime_decision` and
  `potrf_runtime_np2`. The changed files are also compiled in GPU builds,
  where a process with GPUs has no host cache: a CUDA build of the same tree
  passes 55/55 on the A100 of methane (`data/ctest_a100_methane.txt`).

## 5. Performance

`ab.sh`: interleaved rounds, with the order of the arms rotating each round.
Each process runs three factorizations; the first is dropped and the other
two are averaged. Speedups are the median of the per-round ratios, with a
bootstrap 95 % interval of that median (`ab_stats.py`). Arms:

- parent: the build of f5142be, without the cache;
- off: this build with `--operand_cache 0`;
- default: this build with `--operand_cache -1`.

| configuration | N | parent (s) | off (s) | default (s) | off / default | faster in |
|---|---|---|---|---|---|---|
| runtime decisions | 16384 | 2.655 | 2.665 | 2.348 | **1.134** [1.122, 1.150] | 4/4 |
| kind 10, 1e-3 | 16384 | 2.564 | 2.577 | 2.251 | **1.146** [1.134, 1.153] | 4/4 |
| runtime decisions | 32768 | - | 20.227 | 17.443 | **1.163** [1.140, 1.174] | 4/4 |

- **Against the parent build** (N = 16384): 1.135x [1.114, 1.139] and
  1.138x [1.130, 1.149]. Parent / off is 0.994 in both configurations
  ([0.981, 1.015] and [0.985, 1.007]). So the cache-off path costs nothing
  measurable.
- **Conversions** per factorization at the default bound (last
  factorization of each A/B process, mean of 4):

  | configuration | N | off | default | unbounded |
  |---|---|---|---|---|
  | runtime decisions | 16384 | 9342 | 2454 | 486 |
  | kind 10, 1e-3 | 16384 | 9920 | 3036 | - |
  | runtime decisions | 32768 | 78770 | 17447 | 2001 |

- **What is left.** The unbounded cache reached 2.250 s at N = 16384 and
  16.566 s at N = 32768 (section 3.3, another session). Against those, the
  default bound gets about three quarters of the possible gain. The
  conversions left are re-conversions of evicted copies.
- **The gain grows with N**, as on the GPUs: per-GEMM conversions grow as
  NT³ and cached ones as NT². The FP32 copies stay bounded relative to the
  matrix, so the default keeps a fixed share of the tiles at any N.

Data: `data/ab_caffeine_N16384.txt`, `data/ab_caffeine_N32768.txt`.

## 6. Not done, and follow-ups

- **The FP64 accumulation loop** (5.1 %, section 1) is the next CPU cost of
  kind 10 outside MKL. The SGEMM writes an FP32 product C_use with beta = 0,
  and a second pass subtracts it from the FP64 tile. Blocking the SGEMM by
  column panels would keep C_use in cache, or a fused AVX2 loop could replace
  the compiler's SSE2 one. The results stay bit-identical either way, since
  each element is still `C -= (double)C_use` once.
- **Other conversions of the kind-10 CPU GEMM** (FP32 tile → FP64 for an
  FP64 GEMM, FP16 rounding for an emulated FP16 GEMM) are still per call.
  Neither happens in a process without GPUs today: its panel tiles are
  FP64, and without GPUs or native FP16 the runtime decision never picks
  FP16 or FP8.
- **Only one CPU was measured** (Broadwell). Skylake-SP and Zen 2 should
  gain in proportion to the conversions' share of the time.
- **N not a multiple of NB** is not covered: runtime decisions refuse it
  ("adaptive_decision_runtime requires N (8000) to be a multiple of NB
  (512)"), on the parent build too.

## Files

- `ab.sh`, `ab_stats.py`: the interleaved A/B and its statistics
  (`data/ab_*.txt`).
- `sweep.sh`: bounds at N = 16384 and 32768.
- `verify.sh` with `factor_hash.patch`: the bit-for-bit check
  (`data/correctness.txt`).
- `data/policy.txt`: every eviction-policy and bound run (per-run times,
  conversions, log names). The policy variants other than the final one
  were temporary code and are not in the tree.
- `data/coverage.txt`: precision of the panel tiles after kind 10 on a CPU.
