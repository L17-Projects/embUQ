#!/usr/bin/env bash
# Karolina (IT4I) environment setup for MesoUQ runtime and Slurm wrappers.
# Source this file in sbatch scripts: source scripts/platforms/karolina/env_karolina.sh

source /apps/all/Lmod/8.7.37/lmod/lmod/init/bash
module purge || true
module load CUDA/12.4.0
module load HDF5/1.14.0-gompi-2022b
module load GSL/2.7-GCC-12.3.0
module load Eigen/3.4.0-GCCcore-12.2.0
module load Python/3.10.8-GCCcore-12.2.0
module load CMake/3.24.3-GCCcore-12.2.0

export MESOUQ_SITE=karolina
export HPC_SITE=karolina
export MESOUQ_PROJECT_ID="${MESOUQ_PROJECT_ID:-eu-26-17}"
export MESOUQ_SCRATCH_ROOT="${MESOUQ_SCRATCH_ROOT:-/scratch/project/${MESOUQ_PROJECT_ID}/eubrieucb/mesouq}"
export MESOUQ_SITE_RUNTIME_ROOT="${MESOUQ_SITE_RUNTIME_ROOT:-${MESOUQ_SCRATCH_ROOT}/runtime}"
export MESOUQ_RUNS_ROOT="${MESOUQ_RUNS_ROOT:-${MESOUQ_SCRATCH_ROOT}/runs}"
export MESOUQ_PROVENANCE_ROOT="${MESOUQ_PROVENANCE_ROOT:-${MESOUQ_SCRATCH_ROOT}/provenance}"

export PARTITION="${PARTITION:-qgpu}"
export NGPUS="${NGPUS:-1}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

for _mesouq_env_script in \
  "${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/env.sh" \
  "${MESOUQ_SITE_RUNTIME_ROOT}/mirheoOBMD/env.sh" \
  "${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh" \
  "${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools/env.sh"; do
  if [[ -f "${_mesouq_env_script}" ]]; then
    source "${_mesouq_env_script}"
  fi
done
unset _mesouq_env_script

if [[ -d "${MESOUQ_SITE_RUNTIME_ROOT}/korali/install/lib/python3.10/site-packages" ]]; then
  export KORALI_PREFIX="${MESOUQ_SITE_RUNTIME_ROOT}/korali/install"
  export KORALI_PYTHONPATH="${MESOUQ_SITE_RUNTIME_ROOT}/korali/install/lib/python3.10/site-packages"
  case ":${PYTHONPATH:-}:" in
    *":${KORALI_PYTHONPATH}:"*) ;;
    *) export PYTHONPATH="${KORALI_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
fi
