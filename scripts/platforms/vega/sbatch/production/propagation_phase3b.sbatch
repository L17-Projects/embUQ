#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

set -eo pipefail

[[ -z "${REPO_ROOT:-}" ]] && { echo "ERROR: REPO_ROOT not set"; exit 1; }

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 \
            GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0
source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-compression}"
MODEL_FAMILY="${MODEL_FAMILY:-full-model}"
PROFILE="${PROFILE:-production}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROPAGATION_DEVICE="${PROPAGATION_DEVICE:-gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-_runs/${EXPERIMENT}/${RUN_TAG}/production/setup}"

if [[ -z "${CONFIG_PATH:-}" ]]; then
  echo "ERROR: CONFIG_PATH not set" >&2
  exit 1
fi

python scripts/platforms/vega/run_propagation.py \
  --experiment "${EXPERIMENT}" \
  --model-family "${MODEL_FAMILY}" \
  --profile "${PROFILE}" \
  --stage phase3b \
  --device "${PROPAGATION_DEVICE}" \
  --python-bin "${PYTHON_BIN}" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}"
