#!/bin/bash
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
R=/home/qcao3/hicma-x/omnicast/results
N0="numactl --cpunodebind=0 --membind=0"
cd /home/qcao3/hicma-x/omnicast
$N0 ./build_cuda/tests/omnicast_test_gpu > $R/test_gpu_h100.txt 2>&1; tail -1 $R/test_gpu_h100.txt
$N0 ./build_cuda/bench/omnicast_bench_gpu 1024 2048 4096 8192 > $R/gpu_h100.txt 2>&1; echo bench done
cd /home/qcao3/hicma-x/build_omc && ctest -L hicma -V -j 1 > $R/hicma/ctest_omc_h100.log 2>&1; grep 'tests passed' $R/hicma/ctest_omc_h100.log
cd ${HICMA_BASE:?set HICMA_BASE to the baseline build directory (4256ff9)} && ctest -L hicma -V -j 1 > $R/hicma/ctest_base_h100.log 2>&1; grep 'tests passed' $R/hicma/ctest_base_h100.log
