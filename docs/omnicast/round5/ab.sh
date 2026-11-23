#!/bin/bash
# Host operand cache A/B on one CPU node: arms base (unmodified build at f5142be), off (--operand_cache 0),
# on (--operand_cache -1, default), interleaved rounds (arm order rotates per round), --nruns 3 per process.
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
export LD_LIBRARY_PATH=/apps/spacks/2025-04-15/opt/spack/linux-rocky9-x86_64/gcc-12.4.0/intel-oneapi-mkl-2024.2.2-gmswvhnqpsibivkjbt5nuay46ydpn6n4/mkl/2024.2/lib:$LD_LIBRARY_PATH
H=$(hostname -s); N=${NSIZE:-16384}; OUT=/home/qcao3/omc-runs/hostcache/ab/$H.N$N; mkdir -p $OUT
echo "# host $H: $(lscpu | grep -m1 'Model name' | sed 's/.*: *//'), $(nproc) cpus, $(date -Is), N=$N"
X=${X:-/home/qcao3/hicma-x/.claude/worktrees/peaceful-bhabha-f8c6db/build_cpu/tests/testing_potrf_tlr}   # build with the cache
B=${B:-/home/qcao3/hicma-x/build_cpu/tests/testing_potrf_tlr}   # build of the parent revision
C="--N $N --NB 512 --kind_of_problem 2 --verbose 2 --check 0 --nruns 3"
declare -A CFG
CFG[k10_runtime]="--adaptive_decision_runtime 1"
CFG[k10_1e-3]="--kind_of_cholesky 10 --adaptive_decision_runtime 1 --fixedacc 1e-3"
CFGS=${CFGS:-k10_runtime k10_1e-3}
ARMS=(${ARMS:-base off on})
for rnd in $(seq 1 ${ROUNDS:-4}); do for cfg in $CFGS; do
  n=${#ARMS[@]}
  for i in $(seq 0 $((n-1))); do arm=${ARMS[$(( (i + rnd - 1) % n ))]}
    log=$OUT/$cfg.$arm.r$rnd.log
    case $arm in
      base) $B $C ${CFG[$cfg]} > $log 2>&1 < /dev/null ;;
      off)  $X $C ${CFG[$cfg]} --operand_cache 0 > $log 2>&1 < /dev/null ;;
      on)   $X $C ${CFG[$cfg]} --operand_cache -1 > $log 2>&1 < /dev/null ;;
      *)    $X $C ${CFG[$cfg]} --operand_cache $arm > $log 2>&1 < /dev/null ;;
    esac
    echo "$cfg $arm r$rnd: $(grep -E 'TIME\(s\).*: hicma_parsec_cholesky' $log | awk '{print $3}' | tr '\n' ' ')| $(sed 's/\x1b\[[0-9;]*m//g' $log | grep -oE 'Number_of_Datatype_Conversions= [0-9]+|CPU_operand_cache.*' | tail -2 | tr '\n' ' ')"
  done
done; done
