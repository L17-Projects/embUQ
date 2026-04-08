#!/bin/bash
# Submit from the repo root or set REPO_ROOT explicitly before sbatch.
# Required selection variables:
#   EXPERIMENT=compression|indentation
#   MODEL_FAMILY=full-model|reduced-model
#   PROFILE=production|validation

#SBATCH --job-name=mesouq-phase123b
#SBATCH --partition=dev
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
EXPERIMENT="${EXPERIMENT:-compression}"
MODEL_FAMILY="${MODEL_FAMILY:-full-model}"
PROFILE="${PROFILE:-validation}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0

source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

common_args=(--experiment "${EXPERIMENT}" --model-family "${MODEL_FAMILY}" --profile "${PROFILE}" --python-bin "${PYTHON_BIN}")
if [[ -n "${OUTPUT_DIR}" ]]; then
  common_args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${CONFIG_PATH}" ]]; then
  common_args+=(--config "${CONFIG_PATH}")
fi

python scripts/vega/run_inference_stage.py "${common_args[@]}" --stage phase1
python scripts/vega/run_inference_stage.py "${common_args[@]}" --stage phase2 --cpu-ranks "${PHASE2_CPU_RANKS}"
python scripts/vega/run_inference_stage.py "${common_args[@]}" --stage phase3b
