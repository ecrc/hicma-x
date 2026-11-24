#!/bin/bash
# Confirmatory accumulation-loop A/B (adapted from docs/omnicast/round5/ab.sh): arms base (parent build ffcf1cb),
# new (final build: 256-column panels with MKL), m<k> (experiment build, HICMA_ACC_MODE=k: 0 old loop, 8 panels,
# 4 non-temporal stores of C, 7 panels + non-temporal stores except the last GEMM of C); interleaved rounds
# (arm order rotates per round), --nruns 3 per process.
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
H=$(hostname -s); N=${NSIZE:-16384}; OUT=${OUT:-/home/qcao3/omc-runs/accloop/abconfirm/$H.N$N}; mkdir -p $OUT
echo "# host $H: $(lscpu | grep -m1 'Model name' | sed 's/.*: *//'), $(nproc) cpus, $(date -Is), N=$N"
B=${B:-/home/qcao3/hicma-x-acc-base/build_cpu/tests/testing_potrf_tlr}
NW=${NW:-/home/qcao3/hicma-x/.claude/worktrees/optimistic-shaw-05ec60/build_cpu/tests/testing_potrf_tlr}
X=${X:-/home/qcao3/hicma-x-acc-exp/build_cpu/tests/testing_potrf_tlr}
C="--N $N --NB 512 --kind_of_problem 2 --verbose 2 --check 0 --nruns 3"
declare -A CFG
CFG[k10_runtime]="--adaptive_decision_runtime 1"
CFG[k10_1e-3]="--kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3"
CFGS=${CFGS:-k10_runtime k10_1e-3}
ARMS=(${ARMS:-base new m0 m8 m4 m7})
for rnd in $(seq 1 ${ROUNDS:-8}); do for cfg in $CFGS; do
  n=${#ARMS[@]}
  for i in $(seq 0 $((n-1))); do arm=${ARMS[$(( (i + rnd - 1) % n ))]}
    log=$OUT/$cfg.$arm.r$rnd.log
    case $arm in
      base) $B $C ${CFG[$cfg]} > $log 2>&1 < /dev/null ;;
      new)  $NW $C ${CFG[$cfg]} > $log 2>&1 < /dev/null ;;
      m*)   HICMA_ACC_MODE=${arm#m} $X $C ${CFG[$cfg]} > $log 2>&1 < /dev/null ;;
    esac
    echo "$cfg $arm r$rnd: $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | awk '{print $3}' | tr '\n' ' ')| $(sed 's/\x1b\[[0-9;]*m//g' $log | grep -oE 'Number_of_Datatype_Conversions= [0-9]+|CPU_operand_cache.*' | tail -2 | tr '\n' ' ')"
  done
done; done
