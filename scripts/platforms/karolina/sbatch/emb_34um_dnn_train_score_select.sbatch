#!/usr/bin/env bash
#SBATCH --job-name=mesouq-emb-dnn-select
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
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

if [[ -z "${MESOUQ_SITE_RUNTIME_ROOT:-}" ]]; then
  echo "MESOUQ_SITE_RUNTIME_ROOT must be set before using this Karolina sbatch script." >&2
  exit 2
fi
source scripts/platforms/karolina/env_karolina.sh
source scripts/platforms/hpc/site_env.sh
mesouq_activate_site_env karolina "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
COMPLETED_ROWS="${COMPLETED_ROWS:-}"
CANDIDATE_POOL="${CANDIDATE_POOL:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
TOP_N="${TOP_N:-100}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-0.25}"
SEED_OFFSET="${SEED_OFFSET:-0}"
TIMING_CANARY="${TIMING_CANARY:-1}"
TIMING_CANARY_POINT_COUNT="${TIMING_CANARY_POINT_COUNT:-8}"
TIMING_CANARY_OVERRIDE_REASON="${TIMING_CANARY_OVERRIDE_REASON:-}"
DRY_RUN="${DRY_RUN:-0}"
EXISTING_POINTS_JSON="${EXISTING_POINTS_JSON:-}"
FORCE_GRID_JSON="${FORCE_GRID_JSON:-}"

if [[ -z "${COMPLETED_ROWS}" || -z "${CANDIDATE_POOL}" || -z "${OUTPUT_ROOT}" ]]; then
  echo "COMPLETED_ROWS, CANDIDATE_POOL, and OUTPUT_ROOT are required." >&2
  exit 2
fi

CMD=(
  "${PYTHON_BIN}"
  "scripts/workflows/emb/active_learning/run_emb_34um_dnn_train_score_select.py"
  "--completed-rows" "${COMPLETED_ROWS}"
  "--candidate-pool" "${CANDIDATE_POOL}"
  "--output-root" "${OUTPUT_ROOT}"
  "--device" "${DEVICE}"
  "--epochs" "${EPOCHS}"
  "--batch-size" "${BATCH_SIZE}"
  "--learning-rate" "${LEARNING_RATE}"
  "--validation-fraction" "${VALIDATION_FRACTION}"
  "--top-n" "${TOP_N}"
  "--diversity-weight" "${DIVERSITY_WEIGHT}"
  "--seed-offset" "${SEED_OFFSET}"
)

if [[ "${TIMING_CANARY}" == "1" ]]; then
  CMD+=(--timing-canary --timing-canary-point-count "${TIMING_CANARY_POINT_COUNT}")
  if [[ -n "${TIMING_CANARY_OVERRIDE_REASON}" ]]; then
    CMD+=(--timing-canary-override-reason "${TIMING_CANARY_OVERRIDE_REASON}")
  fi
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi
if [[ -n "${EXISTING_POINTS_JSON}" ]]; then
  CMD+=(--existing-points "${EXISTING_POINTS_JSON}")
fi
if [[ -n "${FORCE_GRID_JSON}" ]]; then
  CMD+=(--force-grid "${FORCE_GRID_JSON}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
