#!/bin/bash

#set -x

mode="benchmark"
profile_name="profile"
while getopts ":m:n:" o; do
    case "${o}" in
        n) profile_name=${OPTARG};;
        m)
            mode=${OPTARG}
            if [[ "$mode" != "profile" && "$mode" != "benchmark" && "$mode" != "test" ]]; then 
              echo "Wrong mode: $mode"
              usage
            fi
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))
 
start_mps()
{
    nvidia-cuda-mps-control -d  > /dev/null 2>&1
}

#--------------------------------------------------------
#UCX 
export OMPI_MCA_pml=ucx
#export UCX_MEMTYPE_CACHE=n
#export UCX_MAX_RNDV_RAILS=1
#export UCX_TLS=cma,cuda,cuda_copy,cuda_ipc,mm,posix,self,shm,sm,sysv,tcp
#export UCX_LOG_LEVEL=info


#--------------------------------------------------------

#export NVIDIA_TF32_OVERRIDE=1;
export NSYS_MPI_STORE_TEAMS_PER_RANK=1;
#--------------------------------------------------------


lrank=0
grank=0
if [ -z ${OMPI_COMM_WORLD_LOCAL_RANK+x} ]
then
    let lrank=$SLURM_LOCALID
    let grank=$SLURM_PROCID
else
    let lrank=$OMPI_COMM_WORLD_LOCAL_RANK
    let grank=$OMPI_COMM_WORLD_RANK
fi

export UCX_IB_PREFER_NEAREST_DEVICE=y
if [[ "${lrank}" == "0" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    export CUDA_VISIBLE_DEVICES=4,5,6,7
fi
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

exec="$@"


#--------------------------------------------------------
#Profiling
if [[ "$mode" == "profile" ]]; then
    num=1
    profile_name_numbered=${profile_name}_${num}_rank_${grank}
    while [ -f "${profile_name_numbered}.nsys-rep" ]; do
        let num=num+1
        profile_name_numbered=${profile_name}_${num}_rank_${grank}
    done
fi

export NSYS_MPI_STORE_TEAMS_PER_RANK=1
if [[ "$mode" = "profile" ]]; then
    nsys profile --trace=cuda,cublas,nvtx -f true -o /lustre/fs1/portfolios/coreai/users/bdorschner/projects/hicma-x/build/${profile_name_numbered} $exec
else
    $exec
fi
