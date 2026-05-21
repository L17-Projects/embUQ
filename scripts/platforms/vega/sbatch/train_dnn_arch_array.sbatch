#!/usr/bin/env bash
#SBATCH --job-name=mesouq-dnn-arch
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

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
TRAIN_SCRIPT="${TRAIN_SCRIPT:?TRAIN_SCRIPT not set}"
EXPERIMENT="${EXPERIMENT:?EXPERIMENT not set}"
DIAMETER="${DIAMETER:?DIAMETER not set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR not set}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:?BATCH_SIZE not set}"
LR="${LR:?LR not set}"
MAX_EPOCH="${MAX_EPOCH:?MAX_EPOCH not set}"
ARCH_INDEX="${ARCH_INDEX:-${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}}"
RESULT_JSON="${OUTPUT_DIR}/result_${ARCH_INDEX}.json"
DATA_DIAMETER="${DATA_DIAMETER:-${DIAMETER}}"
DATA_FILE="${DATA_FILE:-}"
DISP_SOURCE="${DISP_SOURCE:-auto}"

module purge
module load Python/3.10.8-GCCcore-12.2.0
ENV_ROOT="${MESOUQ_ENV_ROOT:-${MESOUQ_SITE_RUNTIME_ROOT:?Set MESOUQ_SITE_RUNTIME_ROOT before using this sbatch script.}/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment: ${ENV_SCRIPT}" >&2
  exit 2
fi
source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

common_args=(
  --diameter "${DIAMETER}"
  --data-diameter "${DATA_DIAMETER}"
  --arch-index "${ARCH_INDEX}"
  --output-dir "${OUTPUT_DIR}"
  --result-json "${RESULT_JSON}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --max-epoch "${MAX_EPOCH}"
)
if [[ -n "${DATA_FILE}" ]]; then
  common_args+=(--data-file "${DATA_FILE}")
fi
if [[ "${EXPERIMENT}" == "indentation" ]]; then
  common_args+=(--disp-source "${DISP_SOURCE}" --num-workers 0)
fi

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${common_args[@]}"
