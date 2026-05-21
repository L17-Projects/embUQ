#!/usr/bin/env bash
#SBATCH --job-name=mesouq-emb-dnn-metrics
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000
#SBATCH --gpus=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi
cd "${REPO_ROOT}"

source /scratch/project/eu-26-17/eubrieucb/mesouq/load_mesouq_karolina.sh
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-}"
COMPLETED_ROWS="${COMPLETED_ROWS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
ANALYZE_OUTPUT_ROOT="${ANALYZE_OUTPUT_ROOT:-}"
DEVICE="${DEVICE:-cuda}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-10}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ANALYZE="${RUN_ANALYZE:-1}"
ALLOW_BLOCKED="${ALLOW_BLOCKED:-0}"

if [[ -z "${CAMPAIGN_ROOT}" && -z "${COMPLETED_ROWS}" ]]; then
  echo "CAMPAIGN_ROOT or COMPLETED_ROWS is required." >&2
  exit 2
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
  if [[ -n "${CAMPAIGN_ROOT}" ]]; then
    OUTPUT_ROOT="${CAMPAIGN_ROOT}/ingest"
  else
    OUTPUT_ROOT="$(dirname "${COMPLETED_ROWS}")/metrics"
  fi
fi
if [[ -z "${ANALYZE_OUTPUT_ROOT}" && -n "${CAMPAIGN_ROOT}" ]]; then
  ANALYZE_OUTPUT_ROOT="${CAMPAIGN_ROOT}/analyze"
fi

METRIC_CMD=(
  "${PYTHON_BIN}"
  "scripts/workflows/emb/active_learning/run_emb_34um_dnn_causal_validation_metrics.py"
  "--output-root" "${OUTPUT_ROOT}"
  "--device" "${DEVICE}"
  "--ensemble-size" "${ENSEMBLE_SIZE}"
  "--epochs" "${EPOCHS}"
  "--batch-size" "${BATCH_SIZE}"
  "--learning-rate" "${LEARNING_RATE}"
  "--validation-fraction" "${VALIDATION_FRACTION}"
)
if [[ -n "${CAMPAIGN_ROOT}" ]]; then
  METRIC_CMD+=(--campaign-root "${CAMPAIGN_ROOT}")
fi
if [[ -n "${COMPLETED_ROWS}" ]]; then
  METRIC_CMD+=(--completed-rows "${COMPLETED_ROWS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  METRIC_CMD+=(--dry-run)
fi

echo "Running metrics: ${METRIC_CMD[*]}"
"${METRIC_CMD[@]}"

if [[ "${RUN_ANALYZE}" == "1" ]]; then
  ROWS_PATH="${OUTPUT_ROOT}/emb_34um_dnn_causal_validation_rows.json"
  ANALYZE_CMD=(
    "${PYTHON_BIN}"
    "scripts/workflows/emb/active_learning/analyze_emb_34um_dnn_causal_validation.py"
    "--rows" "${ROWS_PATH}"
    "--output-root" "${ANALYZE_OUTPUT_ROOT:-${OUTPUT_ROOT}/analyze}"
  )
  if [[ "${ALLOW_BLOCKED}" == "1" ]]; then
    ANALYZE_CMD+=(--allow-blocked)
  fi
  echo "Running analysis: ${ANALYZE_CMD[*]}"
  "${ANALYZE_CMD[@]}"
fi
