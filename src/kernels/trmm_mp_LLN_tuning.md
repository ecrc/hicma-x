# Descending LLN TRMM updates

For physical output row `i = descB->mt - 1 - m`, the read-write chain is
`dtrmm(m,n) -> dgemm(m,n,i-1) -> ... -> dgemm(m,n,0) -> descB(i,n)`.
Consuming `B(i-1,n)` first lets the next output row start while the current
row continues its remaining updates.

The `ctl0` fan-in is retained: every lower row must finish reading the
original `B(i,n)` before `dtrmm(m,n)` overwrites it. Each destination still
has exactly one read-write chain. Summation order changes, so bitwise
agreement with the ascending implementation is not expected.

For positive `trmm_window = W`, `read_A(m,k)` waits for
`dgemm(m, k % min(Q, descB->nt), k+W)` when `k+W < i`.
The diagonal and the first `min(W,i)` off-diagonal tiles in descending
order are immediately eligible. Each credit comes from one rotating
column, not a completion barrier across all columns. This remains a soft
per-row window; it does not bound all outstanding A or B data globally.
Zero remains unbounded, and automatic defaults are unchanged.

## CPU measurements, 2026-10-05

Existing Debug build, eight PaRSEC CPU workers, one process, no GPUs,
double precision, `M=N=4096`, `MB=NB=KB=256`. Values are median seconds
from five runs after discarding the first of six runs. The benchmark
repeatedly updates B in place without reinitialization. These are local
measurements, not multi-node or GPU performance claims.

| B columns (`K`) | Window | Ascending | Descending |
| ---: | ---: | ---: | ---: |
| 256 | 0 | 0.071124 | 0.016341 |
| 256 | 1 | 0.070353 | 0.016055 |
| 256 | 2 | 0.072526 | 0.017288 |
| 256 | 4 | 0.074808 | 0.016124 |
| 256 | 8 | 0.073188 | 0.016158 |
| 1024 | 0 | 0.072381 | 0.037321 |
| 1024 | 1 | 0.077119 | 0.037069 |
| 1024 | 2 | 0.072794 | 0.037044 |
| 1024 | 4 | 0.074756 | 0.037173 |
| 1024 | 8 | 0.072874 | 0.037443 |

At window zero this is a 4.35x speedup for one tile column of B and a
1.94x speedup for four tile columns. The small differences between window
settings do not justify changing the single-process default of zero.
Distributed/GPU window tuning still needs measurements on those systems.

From the repository root, reproduce one benchmark configuration with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UCX_NET_DEVICES=all \
  build/tests/testing_trmm_performance \
  --M 4096 --N 4096 --K 256 --MB 256 --NB 256 --KB 256 \
  --cores 8 --gpus 0 --check 0 --adaptive_decision 0 \
  --band_dense_dp 16 --band_dense 16 --P 1 --Q 1 \
  --trmm_window 0 --nruns 6
```

## Validation

The JDF compiler and native test binaries rebuilt successfully. Strict
double-precision checks (`--fixedacc 0`) passed for Unit and NonUnit
diagonals at windows `0,1,2,4,6,-1`:

- One tile; six tile rows with one or three B tile columns; partial edge
  tiles (`M=N=181`, `K=77`, tile size 32).
- Local MPI grids `2x1`, `1x2`, `2x2`, and `1x4`, including `Q > B.nt`.

The distributed reference checker crashed in reference matrix generation
with its LAPACK-layout descriptors. An isolated copy of `testing_trmm.c`
used `PARSEC_MATRIX_TILE` for both reference descriptors and passed all
96 strict checks. This test-only workaround did not change repository
test sources or the production computation. The unmodified checker also
passed all 48 single-process checks at its default tolerance.

Adaptive DP/SP checks (`--adaptive_decision 1 --fixedacc 1e-6`) passed
84 full-tile checks across the same windows and process grids. Printed
decisions confirmed DP diagonal A tiles, SP off-diagonal A tiles, and
SP B tiles. Partial mixed-precision tiles (`M=N=181`, `K=77`, tile size
32, window zero) failed for both diagonal modes. Rebuilding the original
ascending JDF into an isolated library and explicitly preloading it
reproduced the failure (scaled residuals approximately 40.88 and 40.20).
This pre-existing edge-tile issue remains unresolved; the ordering change
does not establish correctness for that case.

Raw logs and the validation runner for this session are under
`/tmp/trmm-lln-*` and `/tmp/trmm_lln_validate.py`; these are temporary
artifacts. GPU execution was unavailable because the NVIDIA driver was
not accessible on this host.
