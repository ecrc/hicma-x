#!/bin/bash
# NGPU2=1: one process keeps both assigned GPUs (pass --gpus 2).
# One run: run1.sh <build dir> <mode: gpu|lb|1|2> <np> <tester args...>; prints hashes, CPU GEMM counts, residual.
# Builds carry factor_hash.patch and cpu_gemm_debug.patch (not committed); for section 8 also gpu_only_debug.patch:
# HICMA_DEBUG_NO_GUARD=1 from the caller turns the GEMM guard off, mode lbna = lb + HICMA_DEBUG_NO_ADVICE=1.
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
export HICMA_DEBUG_FACTOR_HASH=1
MPIEXEC=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/openmpi-5.0.5-24tp3ez2epnadfl3x57bpvg5epgjshqh/bin/mpiexec
W=$(dirname "$0")/gpu_of_rank.sh
bdir=$1; bin=$1/tests/testing_potrf_tlr mode=$2 np=$3; shift 3
xs="-x HICMA_DEBUG_FACTOR_HASH -x HICMA_DEBUG_CPU_GEMM"
[ -n "$HICMA_DEBUG_NO_GUARD" ] && { export HICMA_DEBUG_NO_GUARD; xs="$xs -x HICMA_DEBUG_NO_GUARD"; }
case $mode in
  gpu) export HICMA_DEBUG_CPU_GEMM=0 ;;
  cpu) export HICMA_DEBUG_CPU_GEMM=0 CUDA_VISIBLE_DEVICES= ;;  # no GPU visible: the CPU-only path (--gpus is lowered to 0)
  lb)  export HICMA_DEBUG_CPU_GEMM=0 PARSEC_MCA_device_load_balance_allow_cpu=1; xs="$xs -x PARSEC_MCA_device_load_balance_allow_cpu" ;;
  lbna) export HICMA_DEBUG_CPU_GEMM=0 PARSEC_MCA_device_load_balance_allow_cpu=1 HICMA_DEBUG_NO_ADVICE=1; xs="$xs -x PARSEC_MCA_device_load_balance_allow_cpu -x HICMA_DEBUG_NO_ADVICE" ;;
  *)   export HICMA_DEBUG_CPU_GEMM=$mode ;;
esac
log=$(mktemp /home/qcao3/omc-runs/fp16gpu/logs/run1.XXXXXX)
t0=$(date +%s)
if [ $np = 1 ]; then CUDA_VISIBLE_DEVICES=${NGPU2:+$CUDA_VISIBLE_DEVICES}${NGPU2:-${CUDA_VISIBLE_DEVICES%%,*}} timeout 600 $bin "$@" --cores 18 > $log 2>&1 < /dev/null
else timeout 600 $MPIEXEC -n $np --bind-to none -x CUDA_VISIBLE_DEVICES $xs $W $bin "$@" --cores $((18 / np)) > $log 2>&1 < /dev/null; fi
rc=$?
s=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -oE 'FACTOR_HASH rank [0-9]+ [0-9a-f]+|CPU_RUNTIME_GEMMS pid [0-9]+ DP [0-9]+ SP [0-9]+ HP [0-9]+ FP8 [0-9]+|Check (passed|FAILED): \|\|LL\^T-A\|\|_F/\|\|A\|\|_F = [0-9.e+-]+|Factorization failed \(info = [0-9]+\)|TIME\(s\) +[0-9.]+ : hicma_parsec_cholesky' \
    | sed -E 's/FACTOR_HASH rank ([0-9]+) /h\1 /; s/CPU_RUNTIME_GEMMS pid [0-9]+/cpu/; s/Check (passed|FAILED): .* = /\1 /; s/TIME\(s\) +([0-9.]+) : hicma_parsec_cholesky/t=\1/' | tr '\n' ' ')
err=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -m2 -iE "segmentation fault|aborted|illegal|cuda error|page fault|signal [0-9]|core dumped" | tr '\n' ' ' | cut -c1-160)
lab=$(basename $(dirname $bdir)); [ -n "$HICMA_DEBUG_NO_GUARD" ] && lab="guard_off=$HICMA_DEBUG_NO_GUARD"
echo "$lab $mode np=$np rc=$rc $(( $(date +%s) - t0 ))s: $s ${err:+ERR: $err} [$(basename $log)]"
