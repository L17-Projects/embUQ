#!/usr/bin/env bash
#SBATCH --job-name=mesouq-validation-matrix
#SBATCH --partition=dev
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi
EXPERIMENTS="${EXPERIMENTS:-compression indentation}"
MODEL_FAMILIES="${MODEL_FAMILIES:-full-model reduced-model}"
PROFILE="${PROFILE:-validation}"
SITE="${SITE:-vega}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/validation_matrix/${RUN_TAG}}"
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${PROFILE}" != "validation" ]]; then
  echo "validation_matrix.sbatch only supports PROFILE=validation." >&2
  exit 2
fi

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

"$PYTHON_BIN" scripts/platforms/vega/run_validation_matrix.py \
  --experiments $EXPERIMENTS \
  --model-families $MODEL_FAMILIES \
  --output-root "$OUTPUT_ROOT" \
  --site "$SITE" \
  --run-tag "$RUN_TAG" \
  --phase2-cpu-ranks "$PHASE2_CPU_RANKS" \
  --python-bin "$PYTHON_BIN"
