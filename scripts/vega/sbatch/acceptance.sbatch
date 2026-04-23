#!/usr/bin/env bash
#SBATCH --job-name=mesouq-acceptance
#SBATCH --partition=dev
#SBATCH --time=00:29:00
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

SELECTIONS="${SELECTIONS:-compression:reduced-model:validation indentation:reduced-model:validation}"
SITE="${SITE:-vega}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/acceptance/${RUN_TAG}}"
CPU_RANKS="${CPU_RANKS:-1}"
POPULATION_SIZE="${POPULATION_SIZE:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  openmpi/4.1.2.1 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0

source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

command=(
  "$PYTHON_BIN"
  scripts/run_vega_acceptance.py
  --output-root "$OUTPUT_ROOT"
  --site "$SITE"
  --run-tag "$RUN_TAG"
  --cpu-ranks "$CPU_RANKS"
  --python-bin "$PYTHON_BIN"
  --workflows
)

for workflow in $SELECTIONS; do
  command+=("$workflow")
done

if [[ -n "$POPULATION_SIZE" ]]; then
  command+=(--population-size "$POPULATION_SIZE")
fi

"${command[@]}"
