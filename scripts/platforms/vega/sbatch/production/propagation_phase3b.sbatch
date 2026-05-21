#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

set -eo pipefail

[[ -z "${REPO_ROOT:-}" ]] && { echo "ERROR: REPO_ROOT not set"; exit 1; }

module purge
module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.2.2 \
            GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0
OMPI_BIN_DIR="$(ls -d /opt/mpi/openmpi/4.1.*/bin 2>/dev/null | sort -V | tail -1)"
if [[ -n "${OMPI_BIN_DIR}" ]]; then
  OMPI_ROOT="$(dirname "${OMPI_BIN_DIR}")"
  export PATH="${OMPI_BIN_DIR}:${PATH}"
  export LD_LIBRARY_PATH="${OMPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi
ENV_ROOT="${MESOUQ_ENV_ROOT:-${REPO_ROOT}/_vega/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment: ${ENV_SCRIPT}" >&2
  exit 2
fi
source "${ENV_SCRIPT}"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-compression}"
MODEL_FAMILY="${MODEL_FAMILY:-full-model}"
PROFILE="${PROFILE:-production}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROPAGATION_DEVICE="${PROPAGATION_DEVICE:-gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-_runs/${EXPERIMENT}/${RUN_TAG}/production/setup}"

if [[ -z "${CONFIG_PATH:-}" ]]; then
  echo "ERROR: CONFIG_PATH not set" >&2
  exit 1
fi

python scripts/platforms/hpc/run_propagation.py \
  --experiment "${EXPERIMENT}" \
  --model-family "${MODEL_FAMILY}" \
  --profile "${PROFILE}" \
  --stage phase3b \
  --device "${PROPAGATION_DEVICE}" \
  --python-bin "${PYTHON_BIN}" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}"
