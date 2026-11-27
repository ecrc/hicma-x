#!/bin/bash
#SBATCH --account=k1205
#SBATCH --job-name=hicma_parsec-install
#SBATCH --output=hicma_parsec-install.out
#SBATCH --error=hicma_parsec-install.err
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --partition=debug
# install_shaheen.sh on a compute node. Submit from the repository root:
#   sbatch scripts/machines/install_shaheen_slurm.sh [build-dir] [extra CMake options]
cd "${SLURM_SUBMIT_DIR:-.}"
exec scripts/machines/install_shaheen.sh "$@"
