#!/bin/bash
# Emulated-FP16 branch of the runtime CPU GEMM (ACC_DP, !HAVE_HP_CPU) on a GPU node:
# parent ffcf1cb (base) vs 8e8fde8 (new), both with factor_hash.patch and cpu_gemm_debug.patch
# (neither committed). Run on leconte through srun (V100, sm_70 builds).
#   HICMA_DEBUG_CPU_GEMM=1: every GEMM on the CPU (fixed placement, bits comparable)
#   HICMA_DEBUG_CPU_GEMM=2: GEMMs of odd k on the CPU, the rest on the GPU (fixed placement)
#     (not a valid test: it hands C from a GPU to the CPU without the push-back PaRSEC needs; REPORT.md section 5)
#   lb: no forced placement, PaRSEC load balancing with the CPU allowed (device_load_balance_allow_cpu=1);
#       placement varies, so the bits are not comparable: counts and residuals only
#   gpu: default placement (every GEMM on the GPU), the control
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
export HICMA_DEBUG_FACTOR_HASH=1
B=/home/qcao3/hicma-x-fp16-base/build_cuda/tests/testing_potrf_tlr
X=/home/qcao3/hicma-x-fp16-new/build_cuda/tests/testing_potrf_tlr
MPIEXEC=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/openmpi-5.0.5-24tp3ez2epnadfl3x57bpvg5epgjshqh/bin/mpiexec
W=/home/qcao3/omc-runs/fp16gpu/gpu_of_rank.sh
OUT=${OUT:-/home/qcao3/omc-runs/fp16gpu/logs}; mkdir -p $OUT
echo "# $(hostname -s) $(date -Is) $(lscpu | sed -n 's/^Model name: *//p')"
nvidia-smi -L | head -2
summ() { sed 's/\x1b\[[0-9;]*m//g' $1 | grep -oE 'FACTOR_HASH rank [0-9]+ [0-9a-f]+|CPU_RUNTIME_GEMMS pid [0-9]+ DP [0-9]+ SP [0-9]+ HP [0-9]+ FP8 [0-9]+|Check (passed|FAILED)[^,]*, \|\|LL\^T-A\|\|_F/\|\|A\|\|_F = [0-9.e+-]+' \
        | sed -E 's/CPU_RUNTIME_GEMMS pid [0-9]+/CPU_GEMMS/; s/\|\|LL\^T-A\|\|_F\/\|\|A\|\|_F = /res /' | tr '\n' ' '; }
run() { # name np mode arm args...
  local name=$1 np=$2 mode=$3 arm=$4; shift 4
  local log=$OUT/$name.np$np.$mode.$arm.log bin=$X xs="-x HICMA_DEBUG_FACTOR_HASH -x HICMA_DEBUG_CPU_GEMM"
  [ ${arm%%.*} = base ] && bin=$B
  unset PARSEC_MCA_device_load_balance_allow_cpu
  case $mode in
    gpu) export HICMA_DEBUG_CPU_GEMM=0 ;;
    lb)  export HICMA_DEBUG_CPU_GEMM=0 PARSEC_MCA_device_load_balance_allow_cpu=1; xs="$xs -x PARSEC_MCA_device_load_balance_allow_cpu" ;;
    *)   export HICMA_DEBUG_CPU_GEMM=$mode ;;
  esac
  local t0=$(date +%s)
  if [ $np = 1 ]; then CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} timeout 900 $bin "$@" --cores 18 > $log 2>&1 < /dev/null
  else timeout 900 $MPIEXEC -n $np --bind-to none -x CUDA_VISIBLE_DEVICES $xs $W $bin "$@" --cores $((18 / np)) > $log 2>&1 < /dev/null; fi
  echo "$name np=$np $mode $arm rc=$? $(( $(date +%s) - t0 ))s: $(summ $log)"
}
K="--kind_of_problem 2 --NB 512 --verbose 2 --check 1 --nruns 2 --gpus 1"
R="--adaptive_decision_runtime 1"; F="--kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3"
CASES=${CASES:-"1 2 lb gpu"}
for mode in $CASES; do
  arms="base.1 new.1 base.2 new.2"; [ $mode = gpu ] && arms="base.1 new.1"
  for a in $arms; do run k10_1e-3 1 $mode $a $K --N 16384 $F; run k10_runtime 1 $mode $a $K --N 16384 $R; done
  for a in $arms; do run k10_1e-3 2 $mode $a $K --N 8192 $F; run k10_runtime 2 $mode $a $K --N 8192 $R; done
done
echo "# done $(date -Is)"
