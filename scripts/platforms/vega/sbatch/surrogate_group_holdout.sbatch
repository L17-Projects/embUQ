#!/usr/bin/env bash
#SBATCH --job-name=mesouq-surrogate-holdout
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

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/vega/surrogate_group_holdout/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FAMILIES="${MODEL_FAMILIES:-dnn bnn}"
SELECTIONS="${SELECTIONS:-}"
SEED="${SEED:-20260317}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
PRED_MC_SAMPLES="${PRED_MC_SAMPLES:-64}"
PRED_MC_CHUNK="${PRED_MC_CHUNK:-8}"
DEVICE="${DEVICE:-gpu}"
DISP_SOURCE="${DISP_SOURCE:-auto}"
RUPTURE_RATIO="${RUPTURE_RATIO:-2.0}"

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  openmpi/4.1.2.1 \
  CUDA/12.2.2

ENV_ROOT="${MESOUQ_ENV_ROOT:-${REPO_ROOT}/_vega/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment: ${ENV_SCRIPT}" >&2
  exit 2
fi
source "${ENV_SCRIPT}"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

"$PYTHON_BIN" scripts/platforms/hpc/run_surrogate_group_holdout.py \
  --site vega \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --seed "${SEED}" \
  --val-fraction "${VAL_FRACTION}" \
  --predictive-mc-samples "${PRED_MC_SAMPLES}" \
  --predictive-mc-chunk-size "${PRED_MC_CHUNK}" \
  --device "${DEVICE}" \
  --disp-source "${DISP_SOURCE}" \
  --rupture-ratio "${RUPTURE_RATIO}" \
  --python-bin "${PYTHON_BIN}" \
  $(for fam in ${MODEL_FAMILIES}; do printf -- "--surrogate-family %s " "${fam}"; done) \
  $(for sel in ${SELECTIONS}; do printf -- "--only %s " "${sel}"; done)
