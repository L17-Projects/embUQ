#!/usr/bin/env bash
#SBATCH --job-name=mesouq-dnn-train
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/vega/dnn_training/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_EPOCH="${MAX_EPOCH:-100}"
COMPRESSION_BATCH_SIZE="${COMPRESSION_BATCH_SIZE:-128}"
COMPRESSION_LR="${COMPRESSION_LR:-5e-4}"
INDENTATION_BATCH_SIZE="${INDENTATION_BATCH_SIZE:-1024}"
INDENTATION_LR="${INDENTATION_LR:-1e-3}"
ONLY_SPECS="${ONLY_SPECS:-}"
START_FROM="${START_FROM:-}"
STATE_PATH="${STATE_PATH:-}"
RESUME="${RESUME:-true}"

ONLY_ARGS=""
for sel in ${ONLY_SPECS}; do ONLY_ARGS="${ONLY_ARGS} --only ${sel}"; done
START_FROM_ARGS=""
if [[ -n "${START_FROM}" ]]; then START_FROM_ARGS="--start-from ${START_FROM}"; fi
STATE_PATH_ARGS=""
if [[ -n "${STATE_PATH}" ]]; then STATE_PATH_ARGS="--state-path ${STATE_PATH}"; fi
RESUME_ARG="--resume"
if [[ "${RESUME}" == "false" ]]; then RESUME_ARG="--no-resume"; fi

module purge
module load Python/3.10.8-GCCcore-12.2.0

source "${REPO_ROOT}/_vega/venv/bin/activate"

cd "${REPO_ROOT}"

# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/platforms/vega/run_dnn_surrogate_training.py \
  --site vega \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --python-bin "${PYTHON_BIN}" \
  --max-epoch "${MAX_EPOCH}" \
  --compression-batch-size "${COMPRESSION_BATCH_SIZE}" \
  --compression-lr "${COMPRESSION_LR}" \
  --indentation-batch-size "${INDENTATION_BATCH_SIZE}" \
  --indentation-lr "${INDENTATION_LR}" \
  ${RESUME_ARG} \
  ${START_FROM_ARGS} \
  ${STATE_PATH_ARGS} \
  ${ONLY_ARGS}
