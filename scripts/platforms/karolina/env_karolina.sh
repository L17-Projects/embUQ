#!/usr/bin/env bash
# Karolina (IT4I) environment setup for MesoUQ runtime and Slurm wrappers.
# Source this file in sbatch scripts: source scripts/platforms/karolina/env_karolina.sh

source /apps/all/Lmod/8.7.37/lmod/lmod/init/bash
module --force purge || module purge || true
module load NVHPC/24.1-CUDA-12.4.0
module load OpenMPI/4.1.6-NVHPC-24.1-CUDA-12.4.0
module load HDF5/1.14.3-NVHPC-24.1-CUDA-12.4.0
module load UCC-CUDA/1.3.0-GCCcore-12.2.0-CUDA-12.4.0
module load Python/3.10.8-GCCcore-12.2.0
module load CMake/3.24.3-GCCcore-12.2.0

export MESOUQ_SITE=karolina
export MESOUQ_PROJECT_ID="${MESOUQ_PROJECT_ID:-eu-26-17}"
export MESOUQ_SCRATCH_ROOT="${MESOUQ_SCRATCH_ROOT:-/scratch/project/${MESOUQ_PROJECT_ID}/eubrieucb/mesouq}"
if [[ -z "${MESOUQ_SITE_RUNTIME_ROOT:-}" ]]; then
  echo "MESOUQ_SITE_RUNTIME_ROOT must be set before sourcing env_karolina.sh; choose an isolated per-project runtime root, then source \${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh." >&2
  return 2 2>/dev/null || exit 2
fi
export MESOUQ_SITE_RUNTIME_ROOT
export MESOUQ_RUNS_ROOT="${MESOUQ_RUNS_ROOT:-${MESOUQ_SCRATCH_ROOT}/runs}"
export MESOUQ_PROVENANCE_ROOT="${MESOUQ_PROVENANCE_ROOT:-${MESOUQ_SCRATCH_ROOT}/provenance}"
export MESOUQ_HDF5_PARALLEL_PREFIX="${MESOUQ_HDF5_PARALLEL_PREFIX:-${MESOUQ_SITE_RUNTIME_ROOT}/hdf5-1.14.3-parallel-nvhpc24.1-ompi4.1.6}"
export HDF5_ROOT="${MESOUQ_HDF5_PARALLEL_PREFIX}"
export HDF5_DIR="${MESOUQ_HDF5_PARALLEL_PREFIX}"
export HDF5_MPI=ON

export PARTITION="${PARTITION:-qgpu}"
export NGPUS="${NGPUS:-1}"
export PATH="${MESOUQ_HDF5_PARALLEL_PREFIX}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${MESOUQ_HDF5_PARALLEL_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for _mesouq_env_script in \
  "${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh" \
  "${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/env.sh" \
  "${MESOUQ_SITE_RUNTIME_ROOT}/mirheoOBMD/env.sh" \
  "${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools/env.sh"; do
  if [[ -f "${_mesouq_env_script}" ]]; then
    source "${_mesouq_env_script}"
  fi
done
unset _mesouq_env_script

_mesouq_python_tag="$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
_mesouq_korali_pythonpath="${MESOUQ_SITE_RUNTIME_ROOT}/korali/install/lib/${_mesouq_python_tag}/site-packages"

if [[ -d "${_mesouq_korali_pythonpath}" ]]; then
  export KORALI_PREFIX="${MESOUQ_SITE_RUNTIME_ROOT}/korali/install"
  export KORALI_PYTHONPATH="${_mesouq_korali_pythonpath}"
  case ":${PYTHONPATH:-}:" in
    *":${KORALI_PYTHONPATH}:"*) ;;
    *) export PYTHONPATH="${KORALI_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
fi
unset _mesouq_python_tag _mesouq_korali_pythonpath
