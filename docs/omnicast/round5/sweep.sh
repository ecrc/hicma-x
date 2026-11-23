#!/bin/bash
# sweep.sh: kind 10 runtime decisions, --operand_cache over $OCS at N in $NSIZES (one process, --nruns 3, one round)
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
X=${X:-/home/qcao3/hicma-x/.claude/worktrees/peaceful-bhabha-f8c6db/build_cpu/tests/testing_potrf_tlr}   # build with the cache
OUT=${OUT:-/home/qcao3/omc-runs/hostcache/sweep}; mkdir -p $OUT
for N in ${NSIZES:-16384}; do
C="--N $N --NB 512 --kind_of_problem 2 --verbose 2 --check 0 --nruns 3"
for oc in ${OCS:-0 64 128 256 100000}; do
  log=$OUT/k10_runtime.N$N.oc$oc.log
  $X $C --adaptive_decision_runtime 1 --operand_cache $oc > $log 2>&1 < /dev/null
  echo "N=$N oc=$oc: $(sed 's/\x1b\[[0-9;]*m//g' $log | grep -E 'TIME\(s\).*: hicma_parsec_cholesky' | awk '{print $3}' | tr '\n' ' ') | $(sed 's/\x1b\[[0-9;]*m//g' $log | grep -oE 'Number_of_Datatype_Conversions= [0-9]+|CPU_operand_cache.*' | tail -2 | tr '\n' ' ')"
done; done
