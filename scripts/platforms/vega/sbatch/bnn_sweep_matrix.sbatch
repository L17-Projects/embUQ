#!/bin/bash -l
#SBATCH --job-name=mesouq-bnn-sweep
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

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
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/bnn_sweep/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/_vega/venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
DNN_ROOT="${DNN_ROOT:-}"
SELECTIONS="${SELECTIONS:-${ONLY:-}}"
SEEDS="${SEEDS:-${SEED:-}}"
ARCHITECTURES="${ARCHITECTURES:-}"
ARCHITECTURE_SOURCE="${ARCHITECTURE_SOURCE:-explicit}"
STAGE1_PRIOR_SCALES="${STAGE1_PRIOR_SCALES:-}"
STAGE1_OBS_NOISE_PRIOR_SCALES="${STAGE1_OBS_NOISE_PRIOR_SCALES:-}"
STAGE1_LRS="${STAGE1_LRS:-}"
STAGE1_PRIOR_SCALE="${STAGE1_PRIOR_SCALE:-}"
STAGE1_OBS_NOISE_PRIOR_SCALE="${STAGE1_OBS_NOISE_PRIOR_SCALE:-}"
STAGE1_LR="${STAGE1_LR:-}"
TOP_K="${TOP_K:-}"
PRIOR_SCALES="${PRIOR_SCALES:-}"
OBS_NOISE_PRIOR_SCALES="${OBS_NOISE_PRIOR_SCALES:-}"
LRS="${LRS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_STEPS="${MAX_STEPS:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
PREDICTIVE_MC_SAMPLES="${PREDICTIVE_MC_SAMPLES:-}"
MAX_WALLTIME_SECONDS="${MAX_WALLTIME_SECONDS:-}"
PARITY_TOL="${PARITY_TOL:-}"
DEVICE="${DEVICE:-cuda}"
REQUIRE_PARITY="${REQUIRE_PARITY:-false}"
RESUME="${RESUME:-true}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
JOB_TOKEN="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out") 2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  CUDA/12.2.2

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing Vega runtime virtualenv at ${VENV_DIR}." >&2
  exit 2
fi

source "${VENV_DIR}/bin/activate"
if [[ -f "${REPO_ROOT}/_vega/korali/env.sh" ]]; then
  source "${REPO_ROOT}/_vega/korali/env.sh"
fi
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

require_parity_arg=(--no-require-parity)
case "${REQUIRE_PARITY,,}" in
  true|1|yes)
    require_parity_arg=(--require-parity)
    ;;
esac

command=(
  "${PYTHON_BIN}"
  scripts/platforms/hpc/run_bnn_sweep_matrix.py
  --site "${SITE}"
  --output-root "${OUTPUT_ROOT}"
  --run-tag "${RUN_TAG}"
  --python-bin "${PYTHON_BIN}"
  --architecture-source "${ARCHITECTURE_SOURCE}"
  --device "${DEVICE}"
  "${resume_arg[@]}"
  "${require_parity_arg[@]}"
  "${selection_args[@]}"
  "${seed_args[@]}"
)

if [[ -n "${DNN_ROOT}" ]]; then
  command+=(--dnn-root "${DNN_ROOT}")
fi
if [[ -n "${ARCHITECTURES}" ]]; then
  command+=(--architectures "${ARCHITECTURES}")
fi
if [[ -n "${STAGE1_PRIOR_SCALES}" ]]; then
  command+=(--stage1-prior-scales "${STAGE1_PRIOR_SCALES}")
fi
if [[ -n "${STAGE1_OBS_NOISE_PRIOR_SCALES}" ]]; then
  command+=(--stage1-obs-noise-prior-scales "${STAGE1_OBS_NOISE_PRIOR_SCALES}")
fi
if [[ -n "${STAGE1_LRS}" ]]; then
  command+=(--stage1-lrs "${STAGE1_LRS}")
fi
if [[ -n "${STAGE1_PRIOR_SCALE}" ]]; then
  command+=(--stage1-prior-scale "${STAGE1_PRIOR_SCALE}")
fi
if [[ -n "${STAGE1_OBS_NOISE_PRIOR_SCALE}" ]]; then
  command+=(--stage1-obs-noise-prior-scale "${STAGE1_OBS_NOISE_PRIOR_SCALE}")
fi
if [[ -n "${STAGE1_LR}" ]]; then
  command+=(--stage1-lr "${STAGE1_LR}")
fi
if [[ -n "${TOP_K}" ]]; then
  command+=(--top-k "${TOP_K}")
fi
if [[ -n "${PRIOR_SCALES}" ]]; then
  command+=(--prior-scales "${PRIOR_SCALES}")
fi
if [[ -n "${OBS_NOISE_PRIOR_SCALES}" ]]; then
  command+=(--obs-noise-prior-scales "${OBS_NOISE_PRIOR_SCALES}")
fi
if [[ -n "${LRS}" ]]; then
  command+=(--lrs "${LRS}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  command+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  command+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${EVAL_EVERY}" ]]; then
  command+=(--eval-every "${EVAL_EVERY}")
fi
if [[ -n "${PREDICTIVE_MC_SAMPLES}" ]]; then
  command+=(--predictive-mc-samples "${PREDICTIVE_MC_SAMPLES}")
fi
if [[ -n "${MAX_WALLTIME_SECONDS}" ]]; then
  command+=(--max-walltime-seconds "${MAX_WALLTIME_SECONDS}")
fi
if [[ -n "${PARITY_TOL}" ]]; then
  command+=(--parity-tol "${PARITY_TOL}")
fi

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
