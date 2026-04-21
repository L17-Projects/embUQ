#!/usr/bin/env bash
#SBATCH --job-name=mesouq-validation-matrix
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

EXPERIMENTS="${EXPERIMENTS:-compression indentation}"
MODEL_FAMILIES="${MODEL_FAMILIES:-full-model reduced-model}"
SITE="${SITE:-karolina}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/validation_matrix/${RUN_TAG}}"
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
source scripts/karolina/env_karolina.sh
module load Python/3.10.8-GCCcore-12.2.0
module load GSL/2.7-GCC-12.3.0
module load Eigen/3.4.0-GCCcore-12.2.0
source _vega/venv/bin/activate
source _vega/korali/env.sh

"$PYTHON_BIN" scripts/karolina/run_validation_matrix.py \
  --experiments $EXPERIMENTS \
  --model-families $MODEL_FAMILIES \
  --output-root "$OUTPUT_ROOT" \
  --site "$SITE" \
  --run-tag "$RUN_TAG" \
  --phase2-cpu-ranks "$PHASE2_CPU_RANKS" \
  --python-bin "$PYTHON_BIN"
