#!/bin/bash
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
R=/home/qcao3/hicma-x/omnicast/results/hicma
cd /home/qcao3/hicma-x/build_omc && ctest -L hicma -V -j 1 > $R/ctest_omc_v100.log 2>&1; grep 'tests passed' $R/ctest_omc_v100.log
cd /home/qcao3/hicma-x-base/build && ctest -L hicma -V -j 1 > $R/ctest_base_v100.log 2>&1; grep 'tests passed' $R/ctest_base_v100.log
