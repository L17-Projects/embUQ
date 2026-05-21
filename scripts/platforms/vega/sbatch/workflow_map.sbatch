#!/bin/bash
# Submit from the repo root or set REPO_ROOT explicitly before sbatch.
# Required selection variables:
#   EXPERIMENT=compression|indentation
#   MODEL_FAMILY=full-model|reduced-model
#   PROFILE=production|validation
#   STAGE=phase1|phase3b
# Optional selectors:
#   DATASET=compression_2.1um
#   DIAMETER=2.1

#SBATCH --job-name=mesouq-map
#SBATCH --partition=dev
#SBATCH --time=00:15:00
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
STAGE="${STAGE:-phase3b}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
MAPS_DIR="${MAPS_DIR:-}"
MAP_OUTPUT="${MAP_OUTPUT:-}"
DATASET="${DATASET:-}"
DIAMETER="${DIAMETER:-}"
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-00:15:00}"
PYTHON_BIN="${PYTHON_BIN:-python}"

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0

ENV_ROOT="${MESOUQ_ENV_ROOT:-${REPO_ROOT}/_vega/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment: ${ENV_SCRIPT}" >&2
  exit 2
fi
source "${ENV_SCRIPT}"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

EXPECTED_PARTITION="$("${PYTHON_BIN}" scripts/platforms/hpc/resolve_gpu_partition.py --time-limit "${GPU_TIME_LIMIT}")"
if [[ -n "${SLURM_JOB_PARTITION:-}" && "${SLURM_JOB_PARTITION}" != "${EXPECTED_PARTITION}" ]]; then
  echo "ERROR: strict GPU partition policy violation for MAP job: partition=${SLURM_JOB_PARTITION} expected=${EXPECTED_PARTITION} time_limit=${GPU_TIME_LIMIT}" >&2
  exit 1
fi
echo "[workflow_map] partition=${SLURM_JOB_PARTITION:-unknown} expected=${EXPECTED_PARTITION} time_limit=${GPU_TIME_LIMIT}"

common_args=(--experiment "${EXPERIMENT}" --model-family "${MODEL_FAMILY}" --profile "${PROFILE}" --stage "${STAGE}")
if [[ -n "${OUTPUT_DIR}" ]]; then
  common_args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${CONFIG_PATH}" ]]; then
  common_args+=(--config "${CONFIG_PATH}")
fi
if [[ -n "${MAPS_DIR}" ]]; then
  common_args+=(--maps-dir "${MAPS_DIR}")
fi
if [[ -n "${MAP_OUTPUT}" ]]; then
  common_args+=(--output "${MAP_OUTPUT}")
fi
if [[ -n "${DATASET}" ]]; then
  common_args+=(--dataset "${DATASET}")
fi
if [[ -n "${DIAMETER}" ]]; then
  common_args+=(--diameter "${DIAMETER}")
fi

python scripts/platforms/hpc/extract_map.py "${common_args[@]}"
