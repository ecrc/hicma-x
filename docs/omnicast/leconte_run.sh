#!/bin/bash
# V100 (leconte): OmniCast tests and GPU bench, HiCMA-X GPU CTest for both builds, CPU bench on Broadwell.
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export CUDA_VISIBLE_DEVICES=0
R=/home/qcao3/hicma-x/omnicast/results
cd /home/qcao3/hicma-x/omnicast
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; lscpu | grep 'Model name'
./build_cuda/tests/omnicast_test_gpu > $R/test_gpu_v100.txt 2>&1; tail -1 $R/test_gpu_v100.txt
./build_cuda/tests/omnicast_test_cpu > $R/test_cpu_broadwell.txt 2>&1; tail -1 $R/test_cpu_broadwell.txt
./build_cuda/bench/omnicast_bench_gpu 1024 2048 4096 8192 > $R/gpu_v100.txt 2>&1
./build_cuda/bench/omnicast_bench_cpu 16384 262144 33554432 > $R/cpu_broadwell_avx2.txt 2>&1
cd /home/qcao3/hicma-x/build_omc && ctest -L gpu -V -j 1 > $R/hicma/ctest_omc_v100.log 2>&1; grep 'tests passed' $R/hicma/ctest_omc_v100.log
cd /home/qcao3/hicma-x-base/build && ctest -L gpu -V -j 1 > $R/hicma/ctest_base_v100.log 2>&1; grep 'tests passed' $R/hicma/ctest_base_v100.log
