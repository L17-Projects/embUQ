#!/bin/bash
# Karolina (IT4I) environment setup for Mirheo HPC project
# Source this file in sbatch scripts: source "$(dirname "$0")/../env_karolina.sh"

module purge
module load NVHPC/24.1-CUDA-12.4.0
module load OpenMPI/4.1.6-NVHPC-24.1-CUDA-12.4.0
module load HDF5/1.14.3-NVHPC-24.1-CUDA-12.4.0
module load UCC-CUDA/1.3.0-GCCcore-12.2.0-CUDA-12.4.0

export PARTITION=qgpu
export NGPUS=8
