#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks=64
#SBATCH --mem=256000
#SBATCH --time=06:00:00

set -eo pipefail

[[ -z "${REPO_ROOT:-}" ]] && { echo "ERROR: REPO_ROOT not set"; exit 1; }

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 \
            GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0
ENV_ROOT="${MESOUQ_ENV_ROOT:-${MESOUQ_SITE_RUNTIME_ROOT:?Set MESOUQ_SITE_RUNTIME_ROOT before using this sbatch script.}/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment: ${ENV_SCRIPT}" >&2
  exit 2
fi
source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"

# Korali GPU builds may require libcuda.so.1 symbol resolution on CPU nodes.
export LD_LIBRARY_PATH="/cvmfs/sling.si/modules/el7/software/CUDA/12.6.0/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-compression}"
MODEL_FAMILY="${MODEL_FAMILY:-full-model}"
PROFILE="${PROFILE:-production}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS:-64}"
PHASE2_BACKEND="${PHASE2_BACKEND:-cpu-mpi}"
OUTPUT_DIR="${OUTPUT_DIR:-_runs/${EXPERIMENT}/${RUN_TAG}/production/setup}"

if [[ -z "${CONFIG_PATH:-}" ]]; then
  echo "ERROR: CONFIG_PATH not set" >&2
  exit 1
fi

python scripts/platforms/hpc/run_inference_stage.py \
  --experiment "${EXPERIMENT}" \
  --model-family "${MODEL_FAMILY}" \
  --profile "${PROFILE}" \
  --stage phase2 \
  --phase2-backend "${PHASE2_BACKEND}" \
  --cpu-ranks "${PHASE2_CPU_RANKS}" \
  --python-bin "${PYTHON_BIN}" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}"
