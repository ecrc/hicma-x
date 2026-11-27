#!/bin/bash
#SBATCH -A coreai_devtech_gwas
#SBATCH --job-name hicma_x.syrk
#SBATCH -t 00:30:00
#SBATCH -p defq
#SBATCH -N #NODES#
#SBATCH -c 128
#SBATCH --gpus-per-node=8
#SBATCH --comment=lustreclient=nopcc

set -x


#To be adapted
export RUN_DIR=$(realpath ../../../../../runs)
export scratch=$(realpath ../../../../../)
export HICMA_ROOT=$(realpath ../../../hicma-x-dev/)
########################
export DGXNGPU=8
tasks_per_node=2


# pull image
_cont_image="../../../dtcomp_hicma-x-12.5.0_new.sqsh"
_cont_name="hicma_x"
_cont_mounts="${HICMA_ROOT}:/opt/hicma-x-dev:rw,$(pwd):/cdir:rw,${RUN_DIR}:/runs:rw,${scratch}:${scratch}"
srun --gpus-per-node=${DGXNGPU} --ntasks=${SLURM_JOB_NUM_NODES} --container-image="${_cont_image}" --container-name="${_cont_name}" true


runtype="benchmark"
npat=#NPAT#
nsnp=${npat}
lookahead=2
base_tile_size=4096


gpus_per_task=$(( ${DGXNGPU} / ${tasks_per_node} ))
num_gpu=$(( ${SLURM_JOB_NUM_NODES} * ${DGXNGPU} ))
cores_per_task=$(( 64 / ${tasks_per_node} ))

export TOTALTASKS=$(( ${tasks_per_node} * ${SLURM_JOB_NUM_NODES} ))
export TOTALGPUS=$(( ${TOTALTASKS} * ${gpus_per_task} ))
export TOTALCORES=$(( ${TOTALTASKS} * ${cores_per_task} ))

#scale matrix size
suffix="${SLURM_JOB_NUM_NODES}nodes_${TOTALGPUS}gpus_${TOTALTASKS}tasks_${TOTALCORES}cpus_${npat}npat_${nsnp}nsnp_tile${base_tile_size}_${lookahead}lookahead"


# additional arguments
EXTRA_ARGS="-- --mca runtime_bind_threads 0 --mca bind_threads 0 --mca device_show_statistics 0 --mca device_show_capabilities 0"
BIND_CMD="/cdir/bind_eos.sh -m ${runtype}  "

RUN_CMD="./build/hicma_parsec/testing_syrk -N ${npat} --nsnp=${nsnp} -t ${base_tile_size} --band_dense_dp=1000 --band_dense_sp=1000 --band_dense=10000 -v 1 -g ${gpus_per_task} -e 1.0e-5 -D 14 -c $(( ${cores_per_task} - 1 )) --radius=0.00001 --adddiag=1024 -v 3 -a 1 --nruns 2 "

#RUN_CMD="./build/hicma_parsec/testing_KRR -N ${npat} --nsnp=${nsnp} -t ${base_tile_size} \
            #--band_dense_dp=1000 --band_dense_sp=1000 --band_dense=10000 \
			#-v 1 -g ${gpus_per_task} -e 1.0e-5 -D 14 -c $(( ${cores_per_task} - 1 )) -W 8 -I 1 -y 1 -q 1 -Z 10000 -C 2 \
			#--radius=0.00001 --adddiag=1024 -v 2 -a 1 --nruns 2"


# run code:
srun -u --mpi=pmix \
     --gpus-per-node=${DGXNGPU} \
     -N ${SLURM_JOB_NUM_NODES} \
     -c ${cores_per_task} \
     --cpu-bind=sockets,verbose \
     --ntasks=${TOTALTASKS} \
     --ntasks-per-node=${tasks_per_node} \
     --threads-per-core=1 \
     --comment=skip_dcgm \
     --no-container-mount-home \
     --container-mounts="${_cont_mounts}" \
     --container-name=${_cont_name} \
     --container-workdir /opt/hicma-x-dev \
     ${BIND_CMD} ${RUN_CMD} ${EXTRA_ARGS} |& tee log_krr_${suffix}.txt

