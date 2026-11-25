#!/bin/bash
# Kind 10 runtime CPU GEMM: restore span (and an FP64-sized host buffer) when it widens an FP32 C, and grow a GPU
# copy allocated at the FP32 span (warmup) before its stage-in. On leconte.
# base = ffcf1cb (parent of 8e8fde8; same bits as 8e8fde8 wherever both are right), fix = 8e8fde8 + mixed_fix.patch;
# both with factor_hash.patch and cpu_gemm_debug.patch (not committed). GPUs: the ones Slurm assigned.
#   gpu: default placement; lb: PaRSEC load balancing with the CPU allowed; cpu: no CUDA device (CPU-only path);
#   1: every GEMM on the CPU (then TRSM etc. on the GPU); 3: the GEMMs of k == 0 on the CPU, the rest on the GPU.
#   Both 1 and 3 only hand C from the CPU to a GPU. (2, odd k on the CPU, hands C from a GPU to the CPU without the
#   push-back PaRSEC requires, so it is not a valid test.)
#   --adaptive_decision 1: convert_d2s stores the low-norm tiles in FP32 (span at the FP32 size) before the factorization.
R1=$(dirname "$0")/run1.sh
B=/home/qcao3/hicma-x-fp16-base/build_cuda; X=/home/qcao3/hicma-x-fp16-new/build_cuda
echo "# $(hostname -s) $(date -Is) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
K="--kind_of_problem 2 --NB 512 --verbose 2 --check 1 --gpus 1"
F="--kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3"
R="--adaptive_decision_runtime 1 --check_tol 1e-7"
N1="--N 16384"; N2="--N 8192"
c() { # label np config-args...: base then fix
  local lab=$1 np=$2; shift 2
  for b in $B $X; do echo -n "$lab "; $R1 $b $MODE $np "$@"; done
}
# A. CPU GEMMs, then GPU tasks, on C: with --adaptive_decision 1 the CPU widens FP32 tiles
for MODE in 1 3; do for ad in 1 0; do
  c "A k10_1e-3 ad=$ad" 1 $K $N1 $F --adaptive_decision $ad
  c "A runtime  ad=$ad" 1 $K $N1 $R --adaptive_decision $ad
  c "A k10_1e-3 ad=$ad" 2 $K $N2 $F --adaptive_decision $ad
  c "A runtime  ad=$ad" 2 $K $N2 $R --adaptive_decision $ad
done; done
# B. placements without CPU widening: same hashes
for MODE in gpu lb cpu; do for ad in 0 1; do
  c "B k10_1e-3 ad=$ad" 1 $K $N1 $F --adaptive_decision $ad
  c "B runtime  ad=$ad" 1 $K $N1 $R --adaptive_decision $ad
  c "B k10_1e-3 ad=$ad" 2 $K $N2 $F --adaptive_decision $ad
done; done
# C. adaptive_memory without --check: convert_d2s gives FP32-sized host tiles; no residual, only info and crashes
K0="--kind_of_problem 2 --NB 512 --verbose 2 --check 0 --gpus 1 --adaptive_memory 1"
for MODE in gpu 1 3; do
  c "C admem k10_1e-3 ad=1" 1 $K0 $N1 $F --adaptive_decision 1
  c "C admem runtime  ad=1" 1 $K0 $N1 --adaptive_decision_runtime 1 --adaptive_decision 1
done
# E. one process, two GPUs: panel tiles widened on the CPU are staged onto the other GPU too
K2="--kind_of_problem 2 --NB 512 --verbose 2 --check 1 --gpus 2"
export NGPU2=1
for MODE in 3 1 gpu; do
  c "E 2gpu k10_1e-3 ad=1" 1 $K2 $N1 $F --adaptive_decision 1
  c "E 2gpu runtime  ad=1" 1 $K2 $N1 $R --adaptive_decision 1
done
unset NGPU2
# D. PaRSEC's own placement with the CPU allowed, larger N: does it hand any GEMM to the CPU?
MODE=lb; c "D lb k10_1e-3 ad=1 N=32768" 1 $K --N 32768 $F --adaptive_decision 1
echo "# done $(date -Is)"
