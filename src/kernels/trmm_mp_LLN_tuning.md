# Mixed-precision LLN TRMM: task graph notes

## Update chains

For physical output row `i = descB->mt - 1 - m`, the read-write chain is
`dtrmm(m,n) -> dgemm(m,n,i-1) -> ... -> dgemm(m,n,0) -> descB(i,n)`.
Consuming `B(i-1,n)` first lets the next output row start while the current
row continues its remaining updates. Each destination has exactly one
read-write chain. The summation order is descending in `k`, so results are
not bitwise identical to an ascending implementation.

The TRMM and GEMM tasks have the remaining length of their chain as
priority (`dtrmm`: `i+1`, `dgemm`: `k`), so the bottom rows go first.

## Overwriting B(i,n)

`B(i,n)` is read by the GEMMs of the rows below before `dtrmm(m,n)`
overwrites it (`read_B` in `trmm_mp_LLN.jdf`, chosen per tile by
`hicma_parsec_trmm_New`):

- The GEMMs on the same rank read `B(i,n)` in place and send `dtrmm` a
  control message.
- The GEMMs on other ranks read an operand copy, made by `read_B` on the CPU
  and sent from there, so `dtrmm` does not wait for remote control messages.
  The same-rank readers are taken as the rows congruent to `i` modulo `P`;
  a tile whose distribution does not match keeps every reader in place.
- With `--accumulation_fp32`, an FP16 tile stored in FP32 is converted once,
  on the GPU, to an FP16 copy read by all its GEMMs.

Making every GEMM read the copy was measured slower on a 2x1 grid of
A100s: each GPU then made and pushed to the host all its copies before any
GEMM could run.

## A-panel window

For positive `trmm_window = W`, `read_A(m,k)` waits for
`dgemm(m, k % min(Q, descB->nt), k+W)` when `k+W < i`. The diagonal and the
first `min(W,i)` off-diagonal tiles in descending order are immediately
eligible. Each credit comes from one rotating column, not a completion
barrier across all columns, so this is a soft per-row window; it does not
bound all outstanding A or B data globally. Zero is unbounded. The default
(`-1`) is unbounded on one process and `min(MT, 4 * max(P,Q))` when
distributed.

## CPU measurements of the descending order, 2026-10-05

Debug build, eight PaRSEC CPU workers, one process, no GPUs, double
precision, `M=N=4096`, `MB=NB=KB=256`. Median seconds of five runs after
discarding the first of six; B is updated in place between runs. The
"Ascending" column is the earlier ascending-`k` JDF, no longer in the
repository.

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

The differences between window settings did not justify changing the
single-process default of zero.

Reproduce one configuration from the build directory with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  tests/testing_trmm_performance \
  --N 4096 --K 256 --NB 256 \
  --cores 8 --gpus 0 --adaptive_decision 0 \
  --P 1 --Q 1 --trmm_window 0 --nruns 6
```
