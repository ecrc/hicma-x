# Testing HiCMA-X

## Correctness checks

Every Cholesky driver accepts `--check 1`. It factorizes the matrix, then

1. compares the computed factor with a dense FP64 Cholesky of the same
   (compressed) matrix, and
2. forms L·Lᵀ and compares it with the generated matrix.

Both relative Frobenius residuals must stay below the bound, otherwise the
driver prints the residuals and exits with a non-zero status. The bound is
`--fixedacc`, the compression tolerance, unless `--check_tol` is given.
Configurations that compute whole bands in single or half precision (kinds 3,
4, 9, and `--adaptive_decision` runs at a tolerance of 1e-8 or tighter) reach
the accuracy of that precision, not an FP64-level `fixedacc`; pass the bound
you expect them to meet, for example `--check_tol 1e-6` for a single-precision
band. The registered CTest cases use 1e-7 for the `--adaptive_decision` runs
(their residual is 6e-9 to 1.1e-8 depending on the BLAS library) and 1e-6 for
the single/half-precision band kinds. `--check 1` roughly doubles the run time (dense reference factorization)
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

The stand-alone script runs the same cases plus the data-dependent ones:

```bash
cd build && HICMA_GPUS=1 ../scripts/test_CI.sh
```

`HICMA_CLIMATE_DATA` points the climate emulator case at its input directory
(skipped when unset), `HICMA_MESH_FILE` at the SARS-CoV-2 mesh (default: the
copy in `stars-h/`), `HICMA_MPIRUN=skip` runs the single-rank cases only.

### The validation matrix

`scripts/validation_matrix.sh <build_dir> <label> [cpu|gpu] [out_dir]` runs
the release validation matrix (30 CPU cases, 45 with `gpu`): every Cholesky
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
GPU build; see `BUILD.md`.

## Requirements and limitations

- The Cholesky drivers need `N` to be a multiple of `NB`; other values are
  rejected at start-up. The TRMM supports partial tiles.
- `--kind_of_cholesky 10` (runtime decisions) is selected by
  `--adaptive_decision_runtime 1` and forces a dense band.
- Kinds 1 to 4 have CUDA bodies only; on HIP they run on the CPU (`--gpus 0`).
- `--enable_stochastic_rounding` is implemented for kinds 1 to 5 and the TRMM,
  not for kinds 8 to 10, and not together with FP8 GEMMs.
- `--tensor_gemm 0x80` (FP16 C, FP32 accumulation) is implemented in the TRMM
  only.
- `--datatype_convert 2` needs GPUs.
- One process per node is the fastest layout; several ranks per GPU work but
  need `-- --mca device_cuda_memory_use <percent>` so that each rank reserves
  only a share of the device memory.

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
| 8 | `DENSE_MP_GPU_FP8` | kind 5 with a static FP8 band (compute capability ≥ 8.9) | CUDA, HIP (no FP8) |
| 9 | `DENSE_MP_GPU_FP8_SP` | kind 8 with every tile at FP32 or lower | CUDA, HIP (no FP8) |
| 10 | `DENSE_MP_GPU_FP8_ADAPTIVE` | GEMM precisions decided at run time from the operand norms (`--adaptive_decision_runtime 1`) | CPU, CUDA, HIP |

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
| 8 | matern-2d | Matérn (needs GSL) |
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

## Options

Generated from the option table in `src/core/hicma_parsec_internal.c`.
`--help` prints the same list. Options after a bare `--` go to PaRSEC
(`-- --parsec-help`).

| Option | Type | Description |
|---|---|---|
| `--cores` | int | Number of concurrent threads (default: number of physical hyper-threads) |
| `--gpus` | int | Number of GPUs (default: 0) |
| `--P` | int | Rows (P) in the PxQ process grid (default: NP) |
| `--Q` | int | Columns (Q) in the PxQ process grid (default: NP/P) |
| `--grid_rows` | int | Same as --P |
| `--grid_cols` | int | Same as --Q |
| `--KP` | int | Block rows (KP) in the PxQ process grid (default: 1) |
| `--KQ` | int | Block columns (KQ) in the PxQ process grid (default: 1) |
| `--M` | int | Row dimension (M) of the matrices; must equal N (default: N) |
| `--N` | int | Column dimension (N) of the matrices (required) |
| `--K` | int | Columns (K) of B in the TRMM (default: N) |
| `--MB` | int | Row tile size; must equal NB (default: NB) |
| `--NB` | int | Tile size (required) |
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
| `--band_dense` | int | Dense band size, default 1: only diagonal is dense |
| `--band_low_rank_dp` | int | Low rank band double precision |
| `--band_dist` | int | If 0: normal two_dim_block_cyclic; if >0: with band distribution |
| `--band_p` | int | Row process grid on band, default 1 |
| `--adaptive_decision` | int | 0: disabled; ~0: adaptive_decision of each tile's format using norm approach |
| `--adaptive_decision_runtime` | int | 0: disabled; ~0: adaptive_decision of each tile's format using norm approach during runtime |
| `--adaptive_memory` | int | 0: memory allocated once; 1: memory reallocated per tile after precision decision |
| `--fp8` | int | FP8 GEMMs in kinds 8, 9 and 10: 1 (default) use them when every GPU has compute capability >= 8.9; 0: never (those tiles use HP) |
| `--fp8_scaling` | int | FP8 GEMMs: 1 (default) scale each operand tile by a power of two from its max entry; 0: unscaled |
| `--arena_pinned_memory` | flag | Use pinned host memory for taskpool arenas when GPUs are enabled |
| `--adaptive_maxrank` | int | In -D 6, adaptively set the maxrank used in Cholesky based on the generation |
| `--kind_of_cholesky` | int | Cholesky type |
| `--mesh_file` | string | Path to mesh file |
| `--rbf_kernel` | int | Type of RBF basis function (0:Gaussian, 1:Expon, 2:InvQUAD, 3:InvMQUAD, 4:Maternc1, 5:Maternc2, 6:TPS, 7:CTPS, 8:Wendland) |
| `--radius` | double | Radius of influential nodes |
| `--order` | int | No, Morton, or Hilbert ordering (0, 1, or 2, respectively) |
| `--density` | double | Density of sphere packing for rbf application. Use -1 for random distribution |
| `--tensor_gemm` | int | Bitmask of GEMM compute modes: 0x1=FP64, 0x2=FP32, 0x4=TF32, 0x8=TF16 A16_B16_C32_OP32, 0x10=TF16 A16_B16_C16_OP16, 0x20=BF16 A16_B16_C32_OP32, 0x40=BF16 A16_B16_C16_OP16, 0x80=TF16 A16_B16_C16_OP32 |
| `--datatype_convert` | int | Convert datatype: 0=receiver convert always, 1=sender convert always (only in DENSE_MP_BAND), 2=adaptive |
| `--band_size_termination` | double | Band size auto-tuning termination on GPU, default 3.0 |
| `--left_looking` | int | Modified left-looking Cholesky (row) |
| `--enable_stochastic_rounding` | int | Stochastic rounding for the down-conversions (0 or 1); not with FP8 GEMMs |
| `--nruns` | int | The number of runs of Cholesky |
| `--accumulation_fp32` | int | TRMM: store the FP16 tiles of B in FP32 and accumulate their GEMMs in FP32 (0 or 1) |
| `--time_slots` | int | Time slots in kernels |
| `--sigma` | double | Sigma in kernels |
| `--beta` | double | Beta in kernels |
| `--nu` | double | Nu in kernels |
| `--beta_time` | double | Beta time in kernels |
| `--nu_time` | double | Nu time in kernels |
| `--nonsep_param` | double | Nonsep param in kernels |
| `--noise` | double | Noise in kernels |
| `--HNB` | int | Inner NB used for recursive algorithms (default: MB) |
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
3. Choose a process grid as square as possible (`--P`), with P ≤ Q.
4. The tile size has to be tuned per problem and machine; for dense problems a
   quick scan of a few sizes is enough.
5. Suggested `--maxrank`: 150 for `--kind_of_problem 2`, 500 for 3, NB/2 for 4,
   200 for 6 with `--radius 3.7e-03`.
6. One process per node; with one process per NUMA domain or GPU, bind the
   processes explicitly (for example with Slurm's `--cpu_bind` and
   `--gpu-bind`).
