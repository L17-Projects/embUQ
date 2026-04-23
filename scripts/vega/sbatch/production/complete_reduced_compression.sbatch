#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=47:00:00
#SBATCH --output=/dev/null

set -eo pipefail

[[ -z "${REPO_ROOT:-}" ]] && { echo "ERROR: REPO_ROOT not set"; exit 1; }

route_gpu_partition() {
  local limit="$1"
  local hh mm ss
  IFS=":" read -r hh mm ss <<< "${limit}"
  [[ -z "${hh}" || -z "${mm}" || -z "${ss}" ]] && {
    echo "ERROR: invalid time limit '${limit}' (expected HH:MM:SS)" >&2
    exit 1
  }
  local total_seconds=$((10#${hh} * 3600 + 10#${mm} * 60 + 10#${ss}))
  if ((total_seconds < 1800)); then
    echo "dev"
  else
    echo "gpu"
  fi
}

EXPERIMENT="compression"
MODEL_FAMILY="reduced-model"
PROFILE="${PROFILE:-production}"
PHASE3B_MEM_ARG="${PHASE3B_MEM_ARG:-}"
DEVICE="${DEVICE:-gpu}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS:-64}"
PHASE2_BACKEND="${PHASE2_BACKEND:-native-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/reduced/configs/production/reduced_config_compression.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-_runs/${EXPERIMENT}/${RUN_TAG}/production/setup}"

PHASE1_TIME="${PHASE1_TIME:-12:00:00}"
PHASE2_TIME="${PHASE2_TIME:-06:00:00}"
PHASE3B_TIME="${PHASE3B_TIME:-12:00:00}"
PROP3B_TIME="${PROP3B_TIME:-03:00:00}"

PHASE1_GPU_PARTITION="$(route_gpu_partition "${PHASE1_TIME}")"
PHASE2_GPU_PARTITION="$(route_gpu_partition "${PHASE2_TIME}")"
PHASE3B_GPU_PARTITION="$(route_gpu_partition "${PHASE3B_TIME}")"
PROP3B_GPU_PARTITION="$(route_gpu_partition "${PROP3B_TIME}")"

module purge
module load Python/3.10.8-GCCcore-12.2.0 openmpi/4.1.2.1 CUDA/12.2.2 \
            GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0
source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"

LOGS_DIR="${REPO_ROOT}/_runs/${EXPERIMENT}/${RUN_TAG}/production/logs"
mkdir -p "${LOGS_DIR}"
exec >"${LOGS_DIR}/complete_${SLURM_JOB_ID}.out" 2>"${LOGS_DIR}/complete_${SLURM_JOB_ID}.err"

echo "[complete] experiment=${EXPERIMENT} model_family=${MODEL_FAMILY} profile=${PROFILE}"
echo "[complete] config=${CONFIG_PATH} output=${OUTPUT_DIR}"
echo "[complete] phase2_backend=${PHASE2_BACKEND}"
echo "[complete] partitions phase1=${PHASE1_GPU_PARTITION} phase2=${PHASE2_GPU_PARTITION} phase3b=${PHASE3B_GPU_PARTITION} prop3b=${PROP3B_GPU_PARTITION}"

EXPORT_ENV="ALL,REPO_ROOT=${REPO_ROOT},CONFIG_PATH=${CONFIG_PATH},EXPERIMENT=${EXPERIMENT},MODEL_FAMILY=${MODEL_FAMILY},PROFILE=${PROFILE},RUN_TAG=${RUN_TAG},OUTPUT_DIR=${OUTPUT_DIR},PATH=${PATH},PYTHONPATH=${PYTHONPATH:-},LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},DEVICE=${DEVICE},PHASE2_CPU_RANKS=${PHASE2_CPU_RANKS},PHASE2_BACKEND=${PHASE2_BACKEND},PYTHON_BIN=${PYTHON_BIN}"

PHASE1_JOB=$(REPO_ROOT="${REPO_ROOT}" CONFIG_PATH="${CONFIG_PATH}" \
  sbatch --wait \
    --partition="${PHASE1_GPU_PARTITION}" --gres=gpu:1 --ntasks=1 \
    --time="${PHASE1_TIME}" --exclude="" \
    --job-name=p1_${EXPERIMENT} \
    --output="${LOGS_DIR}/phase1_%j.out" --error="${LOGS_DIR}/phase1_%j.err" \
    --export="${EXPORT_ENV}" \
    "${REPO_ROOT}/scripts/vega/sbatch/production/phase1_gpu.sbatch" \
  | awk '{print $4}')
[[ -z "${PHASE1_JOB}" ]] && { echo "ERROR: Phase 1 failed"; exit 1; }

if [[ "${PHASE2_BACKEND}" == "native-cuda" ]]; then
  PHASE2_JOB=$(REPO_ROOT="${REPO_ROOT}" CONFIG_PATH="${CONFIG_PATH}" \
    sbatch --wait \
      --partition="${PHASE2_GPU_PARTITION}" --gres=gpu:1 --ntasks=1 \
      --time="${PHASE2_TIME}" --exclude="" \
      --job-name=p2_${EXPERIMENT} \
      --output="${LOGS_DIR}/phase2_%j.out" --error="${LOGS_DIR}/phase2_%j.err" \
      --export="${EXPORT_ENV}" \
      "${REPO_ROOT}/scripts/vega/sbatch/production/phase2_native_cuda.sbatch" \
    | awk '{print $4}')
else
  PHASE2_JOB=$(REPO_ROOT="${REPO_ROOT}" CONFIG_PATH="${CONFIG_PATH}" \
    sbatch --wait \
      --partition=cpu --ntasks="${PHASE2_CPU_RANKS}" --mem=256000 \
      --time="${PHASE2_TIME}" --exclude="" \
      --job-name=p2_${EXPERIMENT} \
      --output="${LOGS_DIR}/phase2_%j.out" --error="${LOGS_DIR}/phase2_%j.err" \
      --export="${EXPORT_ENV}" \
      "${REPO_ROOT}/scripts/vega/sbatch/production/phase2_cpu.sbatch" \
    | awk '{print $4}')
fi
[[ -z "${PHASE2_JOB}" ]] && { echo "ERROR: Phase 2 failed"; exit 1; }

PHASE3B_JOB=$(REPO_ROOT="${REPO_ROOT}" CONFIG_PATH="${CONFIG_PATH}" \
  sbatch --wait \
    --partition="${PHASE3B_GPU_PARTITION}" --gres=gpu:1 --ntasks=1 \
    --exclusive ${PHASE3B_MEM_ARG} --time="${PHASE3B_TIME}" --exclude="" \
    --job-name=p3b_${EXPERIMENT} \
    --output="${LOGS_DIR}/phase3b_%j.out" --error="${LOGS_DIR}/phase3b_%j.err" \
    --export="${EXPORT_ENV}" \
    "${REPO_ROOT}/scripts/vega/sbatch/production/phase3b_gpu.sbatch" \
  | awk '{print $4}')
[[ -z "${PHASE3B_JOB}" ]] && { echo "ERROR: Phase 3b failed"; exit 1; }

PROP_JOB=$(REPO_ROOT="${REPO_ROOT}" CONFIG_PATH="${CONFIG_PATH}" \
  sbatch --wait \
    --partition="${PROP3B_GPU_PARTITION}" --ntasks=1 --mem=4000 --gres=gpu:1 \
    --time="${PROP3B_TIME}" --exclude="" \
    --job-name=prop3b_${EXPERIMENT} \
    --output="${LOGS_DIR}/prop3b_%j.out" --error="${LOGS_DIR}/prop3b_%j.err" \
    --export="${EXPORT_ENV}" \
    "${REPO_ROOT}/scripts/vega/sbatch/production/propagation_phase3b.sbatch" \
  | awk '{print $4}')
[[ -z "${PROP_JOB}" ]] && { echo "ERROR: Propagation Phase 3b failed"; exit 1; }

echo "[complete] success phase1=${PHASE1_JOB} phase2=${PHASE2_JOB} phase3b=${PHASE3B_JOB} prop=${PROP_JOB}"
