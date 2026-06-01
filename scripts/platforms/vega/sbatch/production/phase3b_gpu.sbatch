#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=gn10
#SBATCH --mem=70000
#SBATCH --time=12:00:00

set -eo pipefail

[[ -z "${REPO_ROOT:-}" ]] && { echo "ERROR: REPO_ROOT not set"; exit 1; }

module purge
module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.2.2 \
            GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0
OMPI_BIN_DIR="$(ls -d /opt/mpi/openmpi/4.1.*/bin 2>/dev/null | sort -V | tail -1)"
if [[ -n "${OMPI_BIN_DIR}" ]]; then
  OMPI_ROOT="$(dirname "${OMPI_BIN_DIR}")"
  export PATH="${OMPI_BIN_DIR}:${PATH}"
  export LD_LIBRARY_PATH="${OMPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi
ENV_ROOT="${MESOUQ_ENV_ROOT:-${MESOUQ_SITE_RUNTIME_ROOT:?Set MESOUQ_SITE_RUNTIME_ROOT before using this sbatch script.}/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment: ${ENV_SCRIPT}" >&2
  exit 2
fi
source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"

cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-compression}"
MODEL_FAMILY="${MODEL_FAMILY:-full-model}"
PROFILE="${PROFILE:-production}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-gpu}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-_runs/${EXPERIMENT}/${RUN_TAG}/production/setup}"
DATASET_NAME="${PHASE3B_DATASET_NAME:-}"

if [[ -z "${CONFIG_PATH:-}" ]]; then
  echo "ERROR: CONFIG_PATH not set" >&2
  exit 1
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  DATASET_LINE="$("${PYTHON_BIN}" "${REPO_ROOT}/scripts/shared/config/list_experiment_datasets.py" \
    --config "${CONFIG_PATH}" --format name-diameter | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
  if [[ -z "${DATASET_LINE}" ]]; then
    echo "ERROR: no dataset found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
  fi
  EXP_NAME="$(awk '{print $1}' <<< "${DATASET_LINE}")"
  DATASET_NAME="${EXP_NAME}_$(awk '{print $2}' <<< "${DATASET_LINE}")um"
fi

EXTRA_ARGS=()
if [[ -n "${DATASET_NAME}" ]]; then
  EXTRA_ARGS+=(--dataset-name "${DATASET_NAME}")
fi

python scripts/platforms/hpc/run_inference_stage.py \
  --experiment "${EXPERIMENT}" \
  --model-family "${MODEL_FAMILY}" \
  --profile "${PROFILE}" \
  --stage phase3b \
  --device "${DEVICE}" \
  --python-bin "${PYTHON_BIN}" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
