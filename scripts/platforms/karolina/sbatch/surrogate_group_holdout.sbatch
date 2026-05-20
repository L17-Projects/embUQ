#!/usr/bin/env bash
#SBATCH --job-name=mesouq-surrogate-holdout
#SBATCH --partition=qgpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/karolina/surrogate_group_holdout/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FAMILIES="${MODEL_FAMILIES:-dnn bnn}"
SELECTIONS="${SELECTIONS:-}"

cd "${REPO_ROOT}"
source scripts/platforms/karolina/env_karolina.sh
module load Python/3.10.8-GCCcore-12.2.0
VENV_DIR="${VENV_DIR:-${MESOUQ_SITE_RUNTIME_ROOT}/venv}"
source "${VENV_DIR}/bin/activate"

"$PYTHON_BIN" scripts/platforms/hpc/run_surrogate_group_holdout.py \
  --site karolina \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --python-bin "${PYTHON_BIN}" \
  $(for fam in ${MODEL_FAMILIES}; do printf -- "--surrogate-family %s " "${fam}"; done) \
  $(for sel in ${SELECTIONS}; do printf -- "--only %s " "${sel}"; done)
