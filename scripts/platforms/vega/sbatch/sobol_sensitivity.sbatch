#!/usr/bin/env bash
#SBATCH --job-name=mesouq-sobol
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/vega/sobol/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# MODEL_FAMILIES: space-separated families to run. Default: dnn bnn
MODEL_FAMILIES="${MODEL_FAMILIES:-dnn bnn}"
DEVICE="${DEVICE:-gpu}"
# SELECTIONS: space-separated <modality>_<diameter>um keys to restrict runs.
# Leave empty to run all cases.
SELECTIONS="${SELECTIONS:-}"

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0

source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"
export MESOUQ_SITE=vega

FAMILY_ARGS=""
for fam in ${MODEL_FAMILIES}; do
  FAMILY_ARGS="${FAMILY_ARGS} --surrogate-family ${fam}"
done
ONLY_ARGS=""
for sel in ${SELECTIONS}; do
  ONLY_ARGS="${ONLY_ARGS} --only ${sel}"
done

"$PYTHON_BIN" scripts/platforms/hpc/run_sobol_matrix.py \
  --site vega \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --python-bin "${PYTHON_BIN}" \
  --device "${DEVICE}" \
  --n-samples 1024 \
  --n-axis-points 20 \
  --predictive-mc-samples 64 \
  --predictive-mc-chunk-size 8 \
  ${FAMILY_ARGS} \
  ${ONLY_ARGS}
