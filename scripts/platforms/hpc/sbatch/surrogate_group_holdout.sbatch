#!/usr/bin/env bash
#SBATCH --job-name=mesouq-surrogate-holdout
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi

SITE="${SITE:-${HPC_SITE:-vega}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/surrogate_group_holdout/${RUN_TAG}}"
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

cd "${REPO_ROOT}"
export HPC_SITE="${SITE}"

"$PYTHON_BIN" scripts/platforms/hpc/run_surrogate_group_holdout.py \
  --site "${SITE}" \
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
