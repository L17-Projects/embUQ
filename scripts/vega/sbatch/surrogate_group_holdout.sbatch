#!/usr/bin/env bash
#SBATCH --job-name=mesouq-surrogate-holdout
#SBATCH --partition=dev
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/vega/surrogate_group_holdout/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FAMILIES="${MODEL_FAMILIES:-dnn bnn}"
SELECTIONS="${SELECTIONS:-}"

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  openmpi/4.1.2.1 \
  CUDA/12.2.2

source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

"$PYTHON_BIN" scripts/vega/run_surrogate_group_holdout.py \
  --site vega \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --python-bin "${PYTHON_BIN}" \
  $(for fam in ${MODEL_FAMILIES}; do printf -- "--surrogate-family %s " "${fam}"; done) \
  $(for sel in ${SELECTIONS}; do printf -- "--only %s " "${sel}"; done)
