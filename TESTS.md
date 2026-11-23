# Testing HiCMA-X

## Correctness checks

Every Cholesky driver accepts `--check 1`. It factorizes the matrix, then

1. compares the computed factor with a dense FP64 Cholesky of the same
   (compressed) matrix, and
2. forms L·Lᵀ and compares it with the generated matrix.

The residual of step 2, ||LL^T-A||_F/||A||_F, must stay below the bound,
otherwise the driver prints it and exits with a non-zero status. Step 1, the
forward error ||L-L0||_F/||L||_F, is printed with `--verbose` but does not
decide: for a mixed-precision run it exceeds the residual by up to the
condition number of the matrix (10 to 30 times on the test problems). The
bound is `--fixedacc`, the compression tolerance, unless `--check_tol` is
given.
Configurations that compute whole bands in single or half precision (kinds 3,
4, 9, and `--adaptive_decision` runs at a tolerance of 1e-8 or tighter) reach
the accuracy of that precision, not an FP64-level `fixedacc`; pass the bound
you expect them to meet, for example `--check_tol 1e-6` for a single-precision
band. The registered CPU cases use 1e-7 for the `--adaptive_decision` runs
(their residual is 6e-9 to 1.1e-8 depending on the BLAS library) and 1e-6 for
the single-precision band; the GPU cases with FP16, BF16 or TF32 GEMMs use
1e-5 to 1e-3, about ten times their measured residuals (`tests/CMakeLists.txt`). `--check 1` roughly doubles the run time (dense reference factorization)
and uses two extra copies of the matrix.

The mixed-precision TRMM driver `testing_trmm_mp --check 1` compares against
DPLASMA's FP64 TRMM and also exits non-zero on failure.

A failed factorization (`info > 0`, a leading minor that is not positive
definite) makes every driver exit non-zero, with or without `--check`.

## Running the tests

With CTest, from the build directory (the GPU cases carry the label `gpu` and
the MPI cases need `MPIEXEC_EXECUTABLE`):

```bash
ctest --output-on-failure -L hicma
```

```bash
ctest --output-on-failure -L hicma -LE gpu
```

Every case requires DPLASMA's `dplasma_blas_threadsafety`, a CTest fixture
that ctest runs first, also under `-L hicma`. PaRSEC calls the BLAS from
every worker thread at once, and the fixture checks that concurrent calls
return identical results. A sequential OpenBLAS built without `USE_LOCKING`
fails it: the cases then show as "Not Run" instead of failing with wrong
residuals. Use MKL sequential, or a threaded OpenBLAS (which locks) with
`OPENBLAS_NUM_THREADS=1`.

Besides the factorization and TRMM cases, `lapack_tile_convert` and
`lapack_tile_convert_np2` run `testing_lapack_tile`, the round-trip check of
the tile/LAPACK-layout conversion (`hicma_parsec_lapack_tile_convert`) over
every element type, both directions and every process grid of the run.
`matrix_sum` and `matrix_sum_np2` run `testing_matrix_sum`, which checks
`hicma_parsec_matrix_sum` exactly over FP64, FP32 and mixed tiles (dense and
low-rank) and a NaN, on every process grid of the run.

The stand-alone script runs the same cases plus the data-dependent ones:

```bash
cd build && HICMA_GPUS=1 ../scripts/test_CI.sh
```

`HICMA_CLIMATE_DATA` points the climate emulator case at its input directory
(skipped when unset), `HICMA_MESH_FILE` at the SARS-CoV-2 mesh (default: the
copy in `stars-h/`). The two-rank cases use `HICMA_MPIRUN -n 2` (default:
`mpirun --oversubscribe` with Open MPI or plain `mpirun` with another MPI,
else `srun` inside a Slurm allocation as on Frontier,
else they are skipped); `HICMA_MPIRUN=skip` runs the single-rank cases only.
On a HIP build the GPU cases of kinds 1 to 4 are skipped (CUDA bodies only).

### The validation matrix

`scripts/validation_matrix.sh <build_dir> <label> [cpu|gpu] [out_dir]` runs
the release validation matrix (31 CPU cases, 50 with `gpu`): every Cholesky
kind, the TRMM, the sparse paths, single- and two-rank runs, all with
`--check 1`. It writes `results_<label>.tsv` (exit code, residual,
factorization time, GEMM precision counts) and one log per case. The results
recorded for 0.2.0 are in `docs/release-readiness.md`.

## Examples

Dense FP64 Cholesky, checked:

```bash
./tests/testing_potrf_tlr --N 4096 --NB 512 --kind_of_problem 2 --check 1
```

Tile low-rank Cholesky in FP64 (statistics 2-D square exponential):

```bash
./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --kind_of_cholesky 2 --band_dense 1 --check 1
```

Mixed-precision TLR Cholesky with adaptive tile precisions:

```bash
./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2 --kind_of_cholesky 1 --band_dense 1 --adaptive_decision 1 --check 1 --check_tol 1e-7
```

Automatic dense band size (statistics 3-D exponential):

```bash
./tests/testing_potrf_tlr --N 27000 --NB 540 --fixedacc 1e-8 --maxrank 540 --kind_of_problem 4 --auto_band 1 --check 1 -- -mca runtime_comm_coll_bcast 0
```

Runtime precision decision on one GPU (kind 10, needs `N % NB == 0`):

```bash
./tests/testing_potrf_tlr --N 16384 --NB 1024 --kind_of_problem 2 --gpus 1 --adaptive_decision_runtime 1 --check 1 --check_tol 1e-7
```

3-D RBF mesh (SARS-CoV-2 virus), sparse TLR Cholesky:

```bash
./tests/testing_potrf_tlr --N 10370 --NB 1037 --fixedacc 1e-4 --maxrank 50 --adddiag 0.1 --kind_of_problem 6 --band_dense 1 --send_full_tile 0 --lookahead 1 --auto_band 0 --mesh_file ../stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt --numobj 1 --rbf_kernel 0 --radius 4.63e-04 --order 2 --density -1 --sparse 1 --check 1
```

The matrix size is determined by the mesh file (`--N` must match it).

Mixed-precision TRMM against DPLASMA:

```bash
./tests/testing_trmm_mp --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4
```

Climate emulator (needs the external data set):

```bash
./tests/testing_climate_emulator --latitude 120 --NB 1440 --N 14400 --gpus 1 --adaptive_decision 1 --mesh_file /path/to/climate/data
```

Genomics (kernel ridge regression) needs `-DHICMA_PARSEC_GENOMICS=ON` and a
CUDA build; see `BUILD.md`.

## Requirements and limitations

- The Cholesky drivers need `N` to be a multiple of `NB`; other values are
  rejected at start-up. The TRMM supports partial tiles.
- `--kind_of_cholesky 10` (runtime decisions) is selected by
  `--adaptive_decision_runtime 1` and forces a dense band.
- Kinds 1 to 4 have CUDA bodies only; on HIP they run on the CPU (`--gpus 0`).
- AMD GPUs (gfx90a) flush FP16 subnormals (|x| < 6.1e-5) to zero in their
  FP16 GEMMs. Adaptive decisions put small entries in FP16, so on AMD GPUs
  pass `--tensor_gemm_hp 0x20` (BF16) when every tile is dense: at N=30720 an
  adaptive kind-5 run gives 1.0e-5 in FP16 and 1.2e-8 in BF16 for
  `--fixedacc 1e-8`. A HIP run that uses FP16 tiles prints how many. Kind 10
  scales its FP16 operands on HIP and is not affected; the HIP builds' CTest
  cases of adaptive kinds 5, 8 and 9 use 0x20. gfx906 (Radeon VII) keeps FP16
  subnormals and has no BF16 hardware, so keep FP16 there (BF16 GEMMs made an
  adaptive kind-5 run 6x slower); its CTest cases still pass with 0x20.
  rocBLAS has no gfx906 kernels from ROCm 6.4 on: build against ROCm 6.3.
- `--enable_stochastic_rounding` is implemented for kinds 1 to 5 and the
  TRMM, not for kinds 8, 9 and 10.
- The GPU GEMM variants other than the default are not implemented
  everywhere ("GPU GEMM variants" below); the drivers refuse the others
  before the factorization.
- A static FP8 band (kind 8, `--band_dense_hp` < `--band_dense`) close to
  the diagonal can make the matrix indefinite: FP8 keeps 3 mantissa bits. On
  an H100 (problem kind 2, N=8192, NB=512), FP8 from tile distance 3 to 6
  fails with `info` > 0, from distance 8 on the residual is 1e-4 to 1e-5,
  and `--adddiag 100` makes distance 3 work (3.3e-5). With low-rank tiles
  the breakdown reaches HCORE as NaNs and aborts the run. `--adaptive_decision`
  places FP8 from the tile norms instead.
- `--datatype_convert 2` needs GPUs.
- On CPUs other than the A64FX (Fugaku), FP16 tiles are emulated: they model
  the accuracy of half precision, without a speed-up.
- One process per node is the fastest layout; several ranks per GPU work but
  need `-- --mca device_cuda_memory_use <percent>` (`device_hip_memory_use`
  on AMD GPUs) so that each rank reserves only a share of the device memory.
- Multi-rank GPU runs send tiles straight from GPU memory only when the MPI
  library says it reads GPU memory (Open MPI's `MPIX_Query_rocm_support` or
  `MPIX_Query_cuda_support`, MPICH's `MPIX_GPU_query_support`); otherwise
  they go through host memory and a start-up note says so. An Open MPI or a
  UCX built without ROCm cannot read HIP memory. `PARSEC_MCA_mpi_gpu_aware`
  (0, 1, 2 or 3) overrides the choice.

## Cholesky kinds (`--kind_of_cholesky`)

| Kind | Name | What it does | Devices |
|---|---|---|---|
| 1 | `DENSE_TLR_MP` | dense band in mixed precision, low-rank off-band tiles in FP64 or FP32 (`--adaptive_decision`) | CPU, CUDA |
| 2 | `DENSE_TLR_DP` | tile low-rank Cholesky in FP64 | CPU, CUDA |
| 3 | `DENSE_MP_BAND` | dense, FP64/FP32/FP16 bands (`--band_dense_dp`, `--band_dense_sp`) | CPU (no FP16 band), CUDA |
| 4 | `DENSE_SP_HP_BAND` | dense, FP32/FP16 bands | CPU (no FP16 band), CUDA |
| 5 | `DENSE_MP_GPU` | default: dense or banded, static or adaptive precisions, low-rank off-band | CPU, CUDA, HIP |
| 6 | `SPARSE_TLR_DP_GENERAL` | TLR Cholesky over the sparsity pattern (`--sparse 1`) | CPU |
| 7 | `SPARSE_TLR_DP_BALANCE` | kind 6 with load-balanced GEMM distribution (`--sparse 2`) | CPU |
| 8 | `DENSE_MP_GPU_FP8` | kind 5 with a static FP8 band (compute capability ≥ 8.9) | CPU, CUDA, HIP (FP8 only on CUDA with compute capability ≥ 8.9; FP16 elsewhere) |
| 9 | `DENSE_MP_GPU_FP8_SP` | kind 8 with every tile at FP32 or lower | CPU, CUDA, HIP (as kind 8) |
| 10 | `DENSE_MP_GPU_FP8_ADAPTIVE` | GEMM precisions decided at run time from the operand norms (`--adaptive_decision_runtime 1`) | CPU, CUDA, HIP |

Kinds 8 and 9 have their own kernels, `src/kernels/potrf_L_dense_mp_gpu_fp8.jdf`
and `potrf_L_dense_mp_gpu_fp8_sp.jdf`: kind 5's (`potrf_L_dense_mp_gpu.jdf`)
with FP8 branches, and for kind 9 FP32 diagonal tiles, since kind 9 has no
FP64 tile.

## Problem kinds (`--kind_of_problem`)

| Kind | Name | Notes |
|---|---|---|
| 0 | randtlr | synthetic matrix with equal ranks |
| 1 | ed-2d-sin | electrodynamics, 2-D |
| 2 | st-2d-sqexp | spatial statistics, 2-D squared exponential (the CI problem) |
| 3 | st-3d-sqexp | spatial statistics, 3-D squared exponential |
| 4 | st-3d-exp | spatial statistics, 3-D exponential |
| 5 | ed-3d-sin | electrodynamics, 3-D |
| 6 | md-3d-virus | RBF mesh deformation, virus mesh (`--mesh_file`) |
| 7 | md-3d-cube | RBF mesh deformation, cube |
| 8 | matern-2d | Matérn (kinds 8 to 13 need GSL) |
| 9 | space-time-matern-2d | space-time Matérn |
| 10 | space-time-matern-real-2d | space-time Matérn on real locations |
| 11 | space-time-matern-real-exageo-2d | ExaGeoStat variant |
| 12 | matern-2d-real | Matérn on real locations (`--mesh_file` with 2-D coordinates) |
| 13 | matern-2d-real-exageo | ExaGeoStat variant |
| 14 | krr_random_gene | kernel ridge regression, random genotypes |
| 15 | krr_gene | kernel ridge regression from files |

For kinds 0, 1, 5, 6 and 7 the STARS-H kernel places `--adddiag` on the
diagonal itself; for the others HiCMA-X adds it after generation. Either way
the shift is applied once and the reference matrix of `--check` carries it too.

## GPU GEMM variants (`--tensor_gemm_hp`, `--tensor_gemm_sp`)

`--tensor_gemm_hp` chooses how the GPU computes the GEMMs of FP16 tiles, and
`--tensor_gemm_sp 0x4` runs those of FP32 tiles in TF32 (default 0x2, FP32).
`--tensor_gemm`, the 0.1.0 option, sets both in one value: one FP16 variant
plus 0x4 for TF32; its 0x1 and 0x2 change nothing, so the default `0xb`
means 0x8 in FP32. The two new options override its parts. Values with no
FP16 variant or several (`--tensor_gemm 0x3`, `0x18`), and variants a path
does not implement, are refused before the factorization (most at start-up).

| `--tensor_gemm_hp` | A, B | C | Compute | Implemented in |
|---|---|---|---|---|
| 0x8 (default) | FP16 | FP32 | FP32 | everywhere |
| 0x10 | FP16 | FP16 | FP16 | every Cholesky kind, not with `--adaptive_decision_runtime`, and on GPUs not when a low-rank tile has rank 0; the TRMM (same as 0x8 there) |
| 0x20 | BF16 | FP32 | FP32 | dense layout (`--band_dense` ≥ NT), kinds other than 3, not with `--datatype_convert 2` or `--adaptive_decision_runtime`; recommended on AMD GPUs; adaptive decisions then use BF16's epsilon |
| 0x40 | BF16 | BF16 | FP32 | as 0x20 |
| 0x80 | FP16 | FP16 | FP32 | the TRMM only |
| `--tensor_gemm_sp 0x4` | FP32 GEMMs in TF32 | | | kinds 1, 2, 4, 5, 8, 9, 10; ignored in kind 3, with `--adaptive_decision_runtime`, in the TRMM and on HIP |

The TRMM keeps C in FP32 with `--accumulation_fp32` and in FP16 otherwise,
and computes in FP16 except with 0x80. Measured on one A100 (kind 5, N=8192,
NB=1024, FP64/FP32/FP16 bands of 2/2/4 tiles), ||LL^T-A||/||A||: 8.5e-7 with
0x8, 2.3e-5 with 0x10, 6.7e-6 with 0x20, 1.0e-4 with 0x40; an FP64/FP32 band
of 2/6 tiles gives 5.1e-8 in FP32 and 5.1e-6 with TF32. Factorization times
at N=30720 on the A100, FP64/FP32 bands of 2/2 tiles and the rest FP16:
0.70 s (0x8), 0.66 s (0x10), 0.66 to 0.69 s (0x20), 0.77 s (0x40); an FP32
band instead: 0.90 s, 0.56 s with TF32 (FP64 throughout: 1.05 s).

## Options

Generated from the option table in `src/core/hicma_parsec_internal.c`.
`--help` prints the same list. Options after a bare `--` go to PaRSEC
(`-- --parsec-help`).

| Option | Type | Description |
|---|---|---|
| `--cores` | int | Number of compute threads per process (default: number of physical cores) |
| `--gpus` | int | Number of GPUs per process (default: 0, CPU only) |
| `--P` | int | Rows (P) in the PxQ process grid (default: the squarest grid, P >= Q) |
| `--Q` | int | Columns (Q) in the PxQ process grid (default: NP/P) |
| `--grid_rows` | int | Same as --P |
| `--grid_cols` | int | Same as --Q |
| `--KP` | int | Block rows (KP) in the PxQ process grid (default: 1) |
| `--KQ` | int | Block columns (KQ) in the PxQ process grid (default: 1) |
| `--M` | int | Row dimension (M) of the matrices; must equal N (default: N) |
| `--N` | int | Column dimension (N) of the matrices (default: 2048) |
| `--K` | int | Columns (K) of B in the TRMM (default: N) |
| `--MB` | int | Row tile size; must equal NB (default: NB) |
| `--NB` | int | Tile size (default: 2048) |
| `--KB` | int | Tile size along K; must equal NB (default: NB) |
| `--check` | flag | Verify the results (--check, or --check 0\|1) |
| `--check_tol` | double | Residual bound \|\|LL^T-A\|\|_F/\|\|A\|\|_F accepted by --check (default: fixedacc) |
| `--verbose` | int | Extra verbose output: 1=normal, >1=more verbose about decisions |
| `--help` | flag | Show this help message |
| `--fixedrank` | int | Fixed rank threshold used in recompression stage of HCORE_GEMM |
| `--fixedacc` | double | Fixed accuracy threshold used in recompression stage of HCORE_GEMM |
| `--maxrank` | int | Maxrank limit used in creation of descriptors |
| `--genmaxrank` | int | Maxrank limit used in generation |
| `--compmaxrank` | int | Maxrank limit used in allocation of buffers for HiCMA_dpotrf operation |
| `--adddiag` | double | Add this number to diagonal elements to make the matrix positive definite |
| `--lookahead` | int | Set lookahead, from -1 to NT-1; default -1, will set to auto_tuned band_size_dense |
| `--trmm_window` | int | TRMM soft A-panel lookahead: -1=auto (default: unbounded on one process, 4*max(P,Q) distributed), 0=unbounded, positive=explicit lookahead |
| `--kind_of_problem` | int | Problem type |
| `--send_full_tile` | int | Send full tile instead of compressed |
| `--auto_band` | int | Auto select the most suitable band size |
| `--sparse` | int | Sparse mode: 0=no sparse, 1=band distribution, 2=diamond-shaped distribution |
| `--band_dense_dp` | int | Dense band double precision |
| `--band_dense_sp` | int | Dense band single precision |
| `--band_dense_hp` | int | Dense band half precision |
| `--band_dense` | int | Dense band size in tiles (default: NT, every tile dense; 1: only the diagonal dense, the other tiles low-rank) |
| `--band_low_rank_dp` | int | Low rank band double precision |
| `--band_dist` | int | If 0: normal two_dim_block_cyclic; if >0: with band distribution |
| `--band_p` | int | Row process grid on band, default 1 |
| `--adaptive_decision` | int | 0: disabled; ~0: adaptive_decision of each tile's format using norm approach |
| `--adaptive_decision_runtime` | int | 0: disabled; ~0: adaptive_decision of each tile's format using norm approach during runtime |
| `--adaptive_memory` | int | 0: memory allocated once; 1: memory reallocated per tile after precision decision |
| `--fp8` | int | FP8 GEMMs in kinds 8, 9 and 10: 1 (default) use them when every GPU has compute capability >= 8.9; 0: never (those tiles use HP) |
| `--fp8_scaling` | int | FP8 GEMMs: 1 (default) scale each operand tile by a power of two from its max entry; 0: unscaled |
| `--operand_cache` | int | GPU GEMMs: operand tiles kept converted to the GEMM precision (FP8, FP16/BF16, FP32) per GPU for reuse; -1 (default): up to 2*NT tiles per precision, each at most 5% of the GPU memory PaRSEC manages, taken when needed; 0: off (convert per GEMM); > 0: up to that many tiles per precision |
| `--flush_subnormals` | int | CPU threads: 1 (default) flush subnormal results to zero and read subnormal inputs as zero (FTZ/DAZ on x86-64, FZ on AArch64) while HiCMA-X runs: low-precision BLAS on tiles with tiny entries no longer pays for subnormal arithmetic (2.0-2.5x for the mixed-precision kinds on a Broadwell CPU); 0: IEEE subnormals. GPU kernels and OmniCast conversions are not affected |
| `--arena_pinned_memory` | flag | Use pinned host memory for taskpool arenas when GPUs are enabled |
| `--adaptive_maxrank` | int | In -D 6, adaptively set the maxrank used in Cholesky based on the generation |
| `--kind_of_cholesky` | int | Cholesky type |
| `--mesh_file` | string | Path to mesh file |
| `--rbf_kernel` | int | Type of RBF basis function (0:Gaussian, 1:Expon, 2:InvQUAD, 3:InvMQUAD, 4:Maternc1, 5:Maternc2, 6:TPS, 7:CTPS, 8:Wendland) |
| `--radius` | double | Radius of influential nodes |
| `--order` | int | No, Morton, or Hilbert ordering (0, 1, or 2, respectively) |
| `--density` | double | Density of sphere packing for rbf application. Use -1 for random distribution |
| `--tensor_gemm_hp` | int | GPU GEMMs of the FP16 tiles, one of: 0x8=FP16 A and B, FP32 C and compute (default); 0x10=FP16 A, B, C and compute; 0x20=BF16 A and B, FP32 C and compute; 0x40=BF16 A, B and C, FP32 compute; 0x80=FP16 A, B and C, FP32 compute (TRMM only) |
| `--tensor_gemm_sp` | int | GPU GEMMs of the FP32 tiles: 0x2=FP32 (default), 0x4=TF32 |
| `--tensor_gemm` | int | Both in one value (0.1.0): a --tensor_gemm_hp variant, plus 0x4 for TF32; 0x1 and 0x2 are accepted and ignored (default: 0xb). --tensor_gemm_hp and --tensor_gemm_sp override its parts |
| `--datatype_convert` | int | Convert datatype: 0=receiver convert always, 1=sender convert always (only in DENSE_MP_BAND), 2=adaptive |
| `--band_size_termination` | double | --auto_band on CPUs: widen the dense band while the TLR operations of the next band times this factor exceed the dense ones (default 0.67); with GPUs the factor comes from the GPU/CPU performance model and this value is ignored |
| `--left_looking` | int | Modified left-looking Cholesky (row) |
| `--enable_stochastic_rounding` | int | Stochastic rounding for the down-conversions (0 or 1); not in kinds 8, 9 and 10 |
| `--nruns` | int | Number of Cholesky runs (default: 1); more than one needs the dense layout without --check, --auto_band, --adaptive_memory or --sparse |
| `--accumulation_fp32` | int | TRMM: store the FP16 tiles of B in FP32 and accumulate their GEMMs in FP32 (0 or 1) |
| `--time_slots` | int | Time slots in kernels |
| `--sigma` | double | Sigma in kernels |
| `--beta` | double | Beta in kernels |
| `--nu` | double | Nu in kernels |
| `--beta_time` | double | Beta time in kernels |
| `--nu_time` | double | Nu time in kernels |
| `--nonsep_param` | double | Nonsep param in kernels |
| `--noise` | double | Noise in kernels |
| `--HNB` | int | Inner NB used for recursive algorithms (default: 300) |
| `--nsnp` | int | Number of SNPs (number of snps for each individual) |
| `--numobj` | int | Number of objects (number of viruses within a population) |
| `--latitude` | int | Latitude parameter for climate modeling |
| `--wavek` | double | Wave number for electrodynamics problem |
| `--pheno_file` | string | Phenotype file path |
| `--rhs` | int | Right-hand side vector |

## Tuning notes

1. With a dense band wider than one tile, `-- -mca runtime_comm_coll_bcast 0`
   is faster.
2. Set `--cores` to the number of cores minus one when a core is needed for
   communication.
3. Choose a process grid as square as possible (`--P`), with P ≤ Q; without
   `--P` the drivers take the squarest grid with P ≥ Q.
4. The tile size has to be tuned per problem and machine; for dense problems a
   quick scan of a few sizes is enough.
5. Suggested `--maxrank`: 150 for `--kind_of_problem 2`, 500 for 3, NB/2 for 4,
   200 for 6 with `--radius 3.7e-03`.
6. One process per node; with one process per NUMA domain or GPU, bind the
   processes explicitly (for example with Slurm's `--cpu_bind` and
   `--gpu-bind`). PaRSEC starts its threads only on the cores a process is
   bound to, and Open MPI binds each process to one core by default (`cores=1`
   in the parameter printout, and rank 0 warns when the processes of its node
   use half of its cores or fewer). With Open MPI, run one process per node
   with `--map-by ppr:1:node --bind-to none`, or one per socket with
   `--map-by ppr:1:package:pe=<cores per socket>` (Open MPI 5 refuses a
   binding that spans sockets). With two processes on a two-socket, 36-core
   node, a TLR factorization at N=16384 takes 11.5 s with the default binding
   and 1.04 s with `--map-by ppr:1:package:pe=18`. Rank 0 also warns when the
   processes of its node start more worker threads than it has cores
   (`--bind-to none` with several processes per node: 1.28 s in that case).
7. A GPU build runs on the CPU unless `--gpus` is given (rank 0 says so when
   GPUs are visible). The dense FP64 Cholesky at N=30720 runs at 9.2 Tflop/s
   on one A100 (`--gpus 1 --NB 1024`) and at 1.3 Tflop/s on the node's 36 CPU
   cores (`--NB 512`).
   On one MI250X GCD (Frontier) the dense FP64 Cholesky at N=30720 runs at
   13.1 Tflop/s (DPLASMA: 9.9), an FP64/FP32 band of 2/2 tiles with the rest in
   BF16 (`--band_dense_dp 2 --band_dense_sp 4 --tensor_gemm_hp 0x20`) at 18.3;
   one process drives several GCDs with `--gpus n` (0.74 s on 1, 0.26 s on 8).
   With Cray MPICH, remote data goes through host memory unless
   `MPICH_GPU_SUPPORT_ENABLED=1` is set and the GTL library is linked
   (`scripts/machines/install_frontier.sh`); bind one rank per GCD with
   `srun -c 7 --gpus-per-task=1 --gpu-bind=closest`.
8. `--nruns n` factorizes the same matrix n times only in the dense layout
   (`--band_dense` ≥ NT) without `--check`, `--auto_band`, `--adaptive_memory`
   or `--sparse`, where A is regenerated between runs. Elsewhere a later run
   would factorize the previous factor, so one run is done and a warning says
   so: launch the driver once per measurement. `--verbose 1` prints the
   decisions taken at set-up (band sizes, lookahead, memory layout).
9. The Tflops of the `hicma_parsec_cholesky` line is the dense FP64 count
   (N³/3) divided by the factorization time. For TLR and mixed-precision runs
   it is an equivalent rate that can exceed the machine's peak;
   `Operation_counts` reports the operations performed.
10. GPUs of different generations in one run go at the pace of the slowest:
   the tiles are split evenly. On a node with an H100 and a V100, the dense
   FP64 Cholesky at N=30720 took 0.53 s on the H100 alone, 2.3 s on both with
   one process per GPU, and 4.7 s on both in one process; keep the fastest
   GPU with `CUDA_VISIBLE_DEVICES`. On the H100, NB=2048 was 5 % faster than
   1024 at that size (18.5 to 19.4 Tflop/s against 17.5 to 18.2).
11. DPLASMA's `testing_dpotrf`, a natural reference, uses every visible GPU
   unless given `-g 0`.
12. Several GPUs in one process (`--gpus n`) scale when the GPUs can read
   each other's memory directly. On a node with eight V100s (NVLink within
   each socket's four GPUs, some pairs across the sockets), the dense FP64
   Cholesky at N=40960, NB=1024 takes 3.7 s on one GPU, 1.95 s on two and
   1.06 s on four linked ones (DPLASMA: 4.1, 2.4 and 1.3 s). When some pair
   of the GPUs in use is not linked, PaRSEC copies the output of every GPU
   task back to host memory (90 GB instead of 6 GB in this run): two GPUs of
   each socket take 4.3 s, all eight 2.4 s. Rank 0 says so at start-up;
   `nvidia-smi topo -m` shows the links, and `CUDA_VISIBLE_DEVICES` selects
   a linked set.
13. On one node, one process with all the GPUs is faster than one rank per
   GPU or per socket: the same factorization takes 4.7 s with two ranks of
   four GPUs and 3.9 s with eight ranks of one, because tiles that move
   between ranks go from the GPU to host memory, through MPI, and back.
   Use several ranks to span nodes, not the GPUs of a node.
