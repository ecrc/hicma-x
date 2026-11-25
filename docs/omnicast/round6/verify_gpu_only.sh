#!/bin/bash
# Section 8: GEMMs a GPU accepts never run on the CPU (hicma_potrf_gemm_cpu_evaluate). On leconte, one CUDA build
# of fa915ce + the change + ../round5/factor_hash.patch + cpu_gemm_debug.patch + gpu_only_debug.patch; each config
# runs from that binary with the guard off (HICMA_DEBUG_NO_GUARD=1, the code before) and then on.
#   gpu: default placement; lb: PaRSEC load balancing may use the CPU; lbna: lb with no preferred GPU for any tile;
#   1, 3: GEMMs forced onto the CPU (the GPU evaluate declines them); cpu: no CUDA device.
# data/gpu_only_leconte.txt: sections A, C, D gpu and E ran as one job, B (with D lbna) as a second one.
R1=$(dirname "$0")/run1.sh
B=${BUILD:-/home/qcao3/hicma-x-gpuonly/build_cuda}
echo "# $(hostname -s) $(date -Is) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
K="--kind_of_problem 2 --NB 512 --verbose 2 --check 1 --gpus 1"
F="--kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3"
R="--adaptive_decision_runtime 1 --check_tol 1e-7"
N1="--N 16384"; N2="--N 8192"
c() { # label np config-args...: guard off, then on
  local lab=$1 np=$2; shift 2
  for g in 1 0; do echo -n "$lab "; HICMA_DEBUG_NO_GUARD=$g $R1 $B $MODE $np "$@"; done
}
# A. placements where PaRSEC kept every GEMM on the GPU already: same hashes, no CPU GEMM
for MODE in gpu lb; do for ad in 0 1; do
  c "A k10_1e-3 ad=$ad" 1 $K $N1 $F --adaptive_decision $ad
  c "A runtime  ad=$ad" 1 $K $N1 $R --adaptive_decision $ad
  c "A k10_1e-3 ad=$ad" 2 $K $N2 $F --adaptive_decision $ad
done; done
# B. no preferred GPU and the CPU allowed: without the guard PaRSEC gives GEMMs to the CPU
MODE=lbna
for ad in 0 1; do
  c "B2 k10_1e-3 ad=$ad" 1 $K $N1 $F --adaptive_decision $ad
  c "B2 runtime  ad=$ad" 1 $K $N1 $R --adaptive_decision $ad
  c "B2 k10_1e-3 ad=$ad" 2 $K $N2 $F --adaptive_decision $ad
done
c "B2 k10_1e-3 ad=1 N=32768" 1 $K --N 32768 $F --adaptive_decision 1
K2="--kind_of_problem 2 --NB 512 --verbose 2 --check 1 --gpus 2"
export NGPU2=1; c "B2 2gpu k10_1e-3 ad=1" 1 $K2 $N1 $F --adaptive_decision 1; unset NGPU2
# C. the GPU evaluate declines (forced placements): the CPU still takes those GEMMs
for MODE in 1 3; do
  c "C k10_1e-3 ad=1" 1 $K $N1 $F --adaptive_decision 1
  c "C runtime  ad=1" 1 $K $N1 $R --adaptive_decision 1
done
# D. one process, two GPUs
export NGPU2=1; MODE=gpu; c "D 2gpu k10_1e-3 ad=1" 1 $K2 $N1 $F --adaptive_decision 1; unset NGPU2
# E. no GPU: the guard is not installed
MODE=cpu; c "E k10_1e-3 ad=0" 1 $K $N1 $F --adaptive_decision 0
echo "# done $(date -Is)"
