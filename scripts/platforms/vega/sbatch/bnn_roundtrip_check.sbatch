#!/bin/bash -l
#SBATCH --job-name=mesouq-bnn-roundtrip
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/bnn_roundtrip/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
ENV_ROOT="${MESOUQ_ENV_ROOT:-${MESOUQ_SITE_RUNTIME_ROOT:?Set MESOUQ_SITE_RUNTIME_ROOT before using this sbatch script.}/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_ROOT}/bin/python}"
SELECTION="${SELECTION:-indentation_3.2um}"
SEED="${SEED:-}"
DEVICE="${DEVICE:-cuda}"
RELOAD_TOL_REL="${RELOAD_TOL_REL:-}"
WIDTH="${WIDTH:-}"
DEPTH="${DEPTH:-}"
PRIOR_SCALE="${PRIOR_SCALE:-}"
OBS_NOISE_PRIOR_SCALE="${OBS_NOISE_PRIOR_SCALE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
MAX_STEPS="${MAX_STEPS:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
PREDICTIVE_MC_SAMPLES="${PREDICTIVE_MC_SAMPLES:-}"
PREDICTIVE_MC_CHUNK_SIZE="${PREDICTIVE_MC_CHUNK_SIZE:-}"
MAX_WALLTIME_SECONDS="${MAX_WALLTIME_SECONDS:-}"
PARITY_TOL="${PARITY_TOL:-}"
DISP_SOURCE="${DISP_SOURCE:-}"
RUPTURE_RATIO="${RUPTURE_RATIO:-}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
JOB_TOKEN="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out") 2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)

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

command=(
  "${PYTHON_BIN}"
  scripts/platforms/hpc/run_bnn_roundtrip_check.py
  --site "${SITE}"
  --output-root "${OUTPUT_ROOT}"
  --run-tag "${RUN_TAG}"
  --selection "${SELECTION}"
  --device "${DEVICE}"
)

if [[ -n "${SEED}" ]]; then
  command+=(--seed "${SEED}")
fi
if [[ -n "${RELOAD_TOL_REL}" ]]; then
  command+=(--reload-tol-rel "${RELOAD_TOL_REL}")
fi
if [[ -n "${WIDTH}" ]]; then
  command+=(--width "${WIDTH}")
fi
if [[ -n "${DEPTH}" ]]; then
  command+=(--depth "${DEPTH}")
fi
if [[ -n "${PRIOR_SCALE}" ]]; then
  command+=(--prior-scale "${PRIOR_SCALE}")
fi
if [[ -n "${OBS_NOISE_PRIOR_SCALE}" ]]; then
  command+=(--obs-noise-prior-scale "${OBS_NOISE_PRIOR_SCALE}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  command+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${LR}" ]]; then
  command+=(--lr "${LR}")
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
if [[ -n "${PREDICTIVE_MC_CHUNK_SIZE}" ]]; then
  command+=(--predictive-mc-chunk-size "${PREDICTIVE_MC_CHUNK_SIZE}")
fi
if [[ -n "${MAX_WALLTIME_SECONDS}" ]]; then
  command+=(--max-walltime-seconds "${MAX_WALLTIME_SECONDS}")
fi
if [[ -n "${PARITY_TOL}" ]]; then
  command+=(--parity-tol "${PARITY_TOL}")
fi
if [[ -n "${DISP_SOURCE}" ]]; then
  command+=(--disp-source "${DISP_SOURCE}")
fi
if [[ -n "${RUPTURE_RATIO}" ]]; then
  command+=(--rupture-ratio "${RUPTURE_RATIO}")
fi

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
