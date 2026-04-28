#!/bin/bash -l
#SBATCH --job-name=mesouq-dnn-rebaseline
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi

SITE="${SITE:-vega}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/dnn_rebaseline/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
SELECTIONS="${SELECTIONS:-${ONLY:-}}"
SEEDS="${SEEDS:-${SEED:-}}"
ARCHITECTURES="${ARCHITECTURES:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
MAX_EPOCH="${MAX_EPOCH:-}"
RESUME="${RESUME:-true}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
JOB_TOKEN="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out") 2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)

module purge
module load Python/3.10.8-GCCcore-12.2.0

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing Vega runtime virtualenv at ${VENV_DIR}." >&2
  exit 2
fi

source "${VENV_DIR}/bin/activate"
if [[ -f "${REPO_ROOT}/_vega/korali/env.sh" ]]; then
  source "${REPO_ROOT}/_vega/korali/env.sh"
fi
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

selection_args=()
for selection in ${SELECTIONS}; do
  selection_args+=(--only "${selection}")
done

seed_args=()
for seed in ${SEEDS}; do
  seed_args+=(--seed "${seed}")
done

resume_arg=(--resume)
case "${RESUME,,}" in
  false|0|no)
    resume_arg=(--no-resume)
    ;;
esac

command=(
  "${PYTHON_BIN}"
  scripts/platforms/vega/run_dnn_rebaseline_matrix.py
  --site "${SITE}"
  --output-root "${OUTPUT_ROOT}"
  --run-tag "${RUN_TAG}"
  --python-bin "${PYTHON_BIN}"
  "${resume_arg[@]}"
  "${selection_args[@]}"
  "${seed_args[@]}"
)

if [[ -n "${ARCHITECTURES}" ]]; then
  command+=(--architectures "${ARCHITECTURES}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  command+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${LR}" ]]; then
  command+=(--lr "${LR}")
fi
if [[ -n "${MAX_EPOCH}" ]]; then
  command+=(--max-epoch "${MAX_EPOCH}")
fi

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
