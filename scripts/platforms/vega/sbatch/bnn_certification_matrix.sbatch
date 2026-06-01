#!/bin/bash -l
#SBATCH --job-name=mesouq-bnn-certify
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gn10
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

legacy_name="HPC""_SITE"
if [[ ${!legacy_name+x} ]]; then
  echo "${legacy_name} is no longer supported. Use MESOUQ_SITE." >&2
  exit 2
fi
unset legacy_name

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi

if [[ -n "${SITE:-}" && "${SITE}" != "vega" ]]; then
  echo "This site template is fixed to vega; use scripts/platforms/hpc/sbatch for SITE=${SITE}." >&2
  exit 2
fi
if [[ -n "${MESOUQ_SITE:-}" && "${MESOUQ_SITE}" != "vega" ]]; then
  echo "This site template is fixed to vega; MESOUQ_SITE=${MESOUQ_SITE} conflicts." >&2
  exit 2
fi
SITE="vega"
export MESOUQ_SITE="${SITE}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DNN_ROOT="${DNN_ROOT:-}"
BNN_ROOT="${BNN_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/bnn_certification/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
ENV_ROOT="${MESOUQ_ENV_ROOT:-${MESOUQ_SITE_RUNTIME_ROOT:?Set MESOUQ_SITE_RUNTIME_ROOT before using this sbatch script.}/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_ROOT}/bin/python}"
SELECTIONS="${SELECTIONS:-${ONLY:-}}"
SEEDS="${SEEDS:-${SEED:-}}"
EXPECTED_SEED_COUNT="${EXPECTED_SEED_COUNT:-}"
VAL_FRACTION="${VAL_FRACTION:-}"
PREDICTIVE_MC_SAMPLES="${PREDICTIVE_MC_SAMPLES:-}"
PREDICTIVE_MC_CHUNK_SIZE="${PREDICTIVE_MC_CHUNK_SIZE:-}"
DEVICE="${DEVICE:-cuda}"
DISP_SOURCE="${DISP_SOURCE:-}"
RUPTURE_RATIO="${RUPTURE_RATIO:-}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-}"
ACCEPTANCE_UPPER_BOUND="${ACCEPTANCE_UPPER_BOUND:-}"
RESUME="${RESUME:-true}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
JOB_TOKEN="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out") 2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)

if [[ -z "${DNN_ROOT}" || -z "${BNN_ROOT}" ]]; then
  echo "Set both DNN_ROOT and BNN_ROOT before submitting certification." >&2
  exit 2
fi

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  CUDA/12.2.2

if [[ ! -x "${PYTHON_BIN}" || ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment at ${ENV_ROOT}." >&2
  exit 2
fi

source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${DEVICE}" == "cuda" ]]; then
  "${PYTHON_BIN}" -c "
import sys
import torch

if not torch.cuda.is_available():
    print('ERROR: CUDA not available on this node.', file=sys.stderr)
    sys.exit(1)
print('CUDA OK:', torch.cuda.get_device_name(0))
"
fi

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
  scripts/platforms/hpc/run_bnn_certification_matrix.py
  --site "${SITE}"
  --dnn-root "${DNN_ROOT}"
  --bnn-root "${BNN_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --run-tag "${RUN_TAG}"
  --python-bin "${PYTHON_BIN}"
  --device "${DEVICE}"
  "${resume_arg[@]}"
  "${selection_args[@]}"
  "${seed_args[@]}"
)

if [[ -n "${EXPECTED_SEED_COUNT}" ]]; then
  command+=(--expected-seed-count "${EXPECTED_SEED_COUNT}")
fi
if [[ -n "${VAL_FRACTION}" ]]; then
  command+=(--val-fraction "${VAL_FRACTION}")
fi
if [[ -n "${PREDICTIVE_MC_SAMPLES}" ]]; then
  command+=(--predictive-mc-samples "${PREDICTIVE_MC_SAMPLES}")
fi
if [[ -n "${PREDICTIVE_MC_CHUNK_SIZE}" ]]; then
  command+=(--predictive-mc-chunk-size "${PREDICTIVE_MC_CHUNK_SIZE}")
fi
if [[ -n "${DISP_SOURCE}" ]]; then
  command+=(--disp-source "${DISP_SOURCE}")
fi
if [[ -n "${RUPTURE_RATIO}" ]]; then
  command+=(--rupture-ratio "${RUPTURE_RATIO}")
fi
if [[ -n "${BOOTSTRAP_RESAMPLES}" ]]; then
  command+=(--bootstrap-resamples "${BOOTSTRAP_RESAMPLES}")
fi
if [[ -n "${BOOTSTRAP_SEED}" ]]; then
  command+=(--bootstrap-seed "${BOOTSTRAP_SEED}")
fi
if [[ -n "${CONFIDENCE_LEVEL}" ]]; then
  command+=(--confidence-level "${CONFIDENCE_LEVEL}")
fi
if [[ -n "${ACCEPTANCE_UPPER_BOUND}" ]]; then
  command+=(--acceptance-upper-bound "${ACCEPTANCE_UPPER_BOUND}")
fi

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
