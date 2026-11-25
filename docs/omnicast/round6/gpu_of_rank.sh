#!/bin/bash
# one GPU per rank: local rank r takes the r-th GPU of the job step's CUDA_VISIBLE_DEVICES.
# Slurm on leconte sets physical ids (e.g. 1,2) without fencing the other GPUs: never assume GPU 0 is ours.
IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=${gpus[${OMPI_COMM_WORLD_LOCAL_RANK:-0}]}
exec "$@"
