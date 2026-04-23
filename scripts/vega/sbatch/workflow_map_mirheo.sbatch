#!/bin/bash
# Submit from the repo root or set REPO_ROOT explicitly before sbatch.
# Required selection variables:
#   EXPERIMENT=compression|indentation
#   MODEL_FAMILY=full-model|reduced-model
#   PROFILE=production|validation

#SBATCH --job-name=mesouq-map-mirheo
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi
EXPERIMENT="${EXPERIMENT:-compression}"
MODEL_FAMILY="${MODEL_FAMILY:-full-model}"
PROFILE="${PROFILE:-validation}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAP_MIRHEO_N_DISPLACEMENTS="${MAP_MIRHEO_N_DISPLACEMENTS:-15}"
# MAP Mirheo dev canaries intentionally use a small but stable floor.
# Values below this are outside the supported sanity/canary contract.
MAP_MIRHEO_NUMSTEPS="${MAP_MIRHEO_NUMSTEPS:-200}"
MAP_MIRHEO_NUMSTEPS_EQ="${MAP_MIRHEO_NUMSTEPS_EQ:-200}"
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-00:45:00}"

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0 HDF5/1.14.0-gompi-2022b

source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"
if [[ ! -f "${REPO_ROOT}/_vega/mirheo/env.sh" ]]; then
  echo "ERROR: repo-local Mirheo bootstrap missing. Run scripts/hpc/bootstrap_mirheo.sh first." >&2
  exit 1
fi
source "${REPO_ROOT}/_vega/mirheo/env.sh"

cd "${REPO_ROOT}"

EXPECTED_PARTITION="$("${PYTHON_BIN}" scripts/vega/resolve_gpu_partition.py --time-limit "${GPU_TIME_LIMIT}")"
if [[ -n "${SLURM_JOB_PARTITION:-}" && "${SLURM_JOB_PARTITION}" != "${EXPECTED_PARTITION}" ]]; then
  echo "ERROR: strict GPU partition policy violation for MAP Mirheo job: partition=${SLURM_JOB_PARTITION} expected=${EXPECTED_PARTITION} time_limit=${GPU_TIME_LIMIT}" >&2
  exit 1
fi
echo "[workflow_map_mirheo] partition=${SLURM_JOB_PARTITION:-unknown} expected=${EXPECTED_PARTITION} time_limit=${GPU_TIME_LIMIT}"

common_args=(--experiment "${EXPERIMENT}" --model-family "${MODEL_FAMILY}" --profile "${PROFILE}" --python-bin "${PYTHON_BIN}")
if [[ -n "${OUTPUT_DIR}" ]]; then
  common_args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${MAP_MIRHEO_NUMSTEPS}" ]]; then
  common_args+=(--numsteps "${MAP_MIRHEO_NUMSTEPS}")
fi
if [[ -n "${MAP_MIRHEO_NUMSTEPS_EQ}" ]]; then
  common_args+=(--numsteps-eq "${MAP_MIRHEO_NUMSTEPS_EQ}")
fi

python scripts/vega/run_map_mirheo.py "${common_args[@]}" --n-displacements "${MAP_MIRHEO_N_DISPLACEMENTS}"
