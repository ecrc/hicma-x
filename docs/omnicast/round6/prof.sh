#!/bin/bash
# perf profiles of the new build, old loop (HICMA_ACC_MODE=0) vs 256-column panels (8), runtime decisions, N=16384
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
X=${X:-/home/qcao3/hicma-x-acc-exp/build_cpu/tests/testing_potrf_tlr}   # parent build with experiment_modes.patch (HICMA_ACC_MODE)
mkdir -p /home/qcao3/omc-runs/accloop/prof
for m in ${MODES:-0 8}; do
  HICMA_ACC_MODE=$m perf record -F 299 -o /home/qcao3/omc-runs/accloop/prof/k10_m$m.data -- $X --N 16384 --NB 512 --kind_of_problem 2 --check 0 --verbose 2 --nruns 3 --adaptive_decision_runtime 1 > /home/qcao3/omc-runs/accloop/prof/k10_m$m.log 2>&1 < /dev/null
  echo "mode $m: $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' /home/qcao3/omc-runs/accloop/prof/k10_m$m.log | awk '{print $3}' | tr '\n' ' ')"
done
