#!/bin/bash
# Submit from the repo root or set REPO_ROOT explicitly before sbatch.
# Required selection variables:
#   EXPERIMENT=compression|indentation
#   MODEL_FAMILY=full-model|reduced-model
#   PROFILE=production|validation

#SBATCH --job-name=mesouq-map-mirheo
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi
EXPERIMENT="${EXPERIMENT:-compression}"
MODEL_FAMILY="${MODEL_FAMILY:-full-model}"
PROFILE="${PROFILE:-validation}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAP_MIRHEO_N_DISPLACEMENTS="${MAP_MIRHEO_N_DISPLACEMENTS:-15}"
# Leave production unset so the Mirheo payload uses its scientific defaults.
# The dedicated sanity runner exports the supported 200/200 micro-canary floor explicitly.
MAP_MIRHEO_NUMSTEPS="${MAP_MIRHEO_NUMSTEPS:-}"
MAP_MIRHEO_NUMSTEPS_EQ="${MAP_MIRHEO_NUMSTEPS_EQ:-}"
MAP_MIRHEO_TIMEOUT_SECONDS="${MAP_MIRHEO_TIMEOUT_SECONDS:-}"
MAP_MIRHEO_MAX_RETRIES="${MAP_MIRHEO_MAX_RETRIES:-1}"
MAP_MIRHEO_RETRY_DT_SCALE="${MAP_MIRHEO_RETRY_DT_SCALE:-0.5}"
MAP_MIRHEO_CLEANUP_BUFFER_SECONDS="${MAP_MIRHEO_CLEANUP_BUFFER_SECONDS:-120}"
MAP_MIRHEO_DATASET_NAME="${MAP_MIRHEO_DATASET_NAME:-}"
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-02:00:00}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: OUTPUT_DIR is required for workflow_map_mirheo.sbatch." >&2
  exit 2
fi

module purge
module load Python/3.10.8-GCCcore-12.2.0 OpenMPI/4.1.4-GCC-12.2.0 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0 HDF5/1.14.0-gompi-2022b MPFR/4.2.0-GCCcore-12.2.0 GMP/6.2.1-GCCcore-12.2.0

source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"
if [[ -f "${REPO_ROOT}/_vega/gv_venv/env.sh" ]]; then
  source "${REPO_ROOT}/_vega/gv_venv/env.sh"
elif [[ -f "${REPO_ROOT}/_vega/mirheo/env.sh" ]]; then
  source "${REPO_ROOT}/_vega/mirheo/env.sh"
else
  echo "ERROR: repo-local Mirheo bootstrap missing. Run scripts/platforms/hpc/bootstrap_mirheo.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}"

EXPECTED_PARTITION="$("${PYTHON_BIN}" scripts/platforms/hpc/resolve_gpu_partition.py --time-limit "${GPU_TIME_LIMIT}")"
if [[ -n "${SLURM_JOB_PARTITION:-}" && "${SLURM_JOB_PARTITION}" != "${EXPECTED_PARTITION}" ]]; then
  echo "ERROR: strict GPU partition policy violation for MAP Mirheo job: partition=${SLURM_JOB_PARTITION} expected=${EXPECTED_PARTITION} time_limit=${GPU_TIME_LIMIT}" >&2
  exit 1
fi
echo "[workflow_map_mirheo] partition=${SLURM_JOB_PARTITION:-unknown} expected=${EXPECTED_PARTITION} time_limit=${GPU_TIME_LIMIT}"

parse_slurm_seconds() {
  local limit="$1"
  "${PYTHON_BIN}" - "${limit}" <<'PY'
import sys
from meso_uq.scheduler_routing import parse_slurm_time_limit

try:
    print(parse_slurm_time_limit(sys.argv[1]))
except ValueError as exc:
    print(f"ERROR: invalid Slurm time limit {sys.argv[1]!r}: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY
}

if [[ ! "${MAP_MIRHEO_MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAP_MIRHEO_MAX_RETRIES must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "${MAP_MIRHEO_CLEANUP_BUFFER_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAP_MIRHEO_CLEANUP_BUFFER_SECONDS must be a non-negative integer." >&2
  exit 2
fi

walltime_seconds="$(parse_slurm_seconds "${GPU_TIME_LIMIT}")"
attempt_count=$((MAP_MIRHEO_MAX_RETRIES + 1))
available_budget=$((walltime_seconds - MAP_MIRHEO_CLEANUP_BUFFER_SECONDS))
if (( available_budget <= attempt_count )); then
  echo "ERROR: GPU_TIME_LIMIT=${GPU_TIME_LIMIT} leaves no runtime budget after cleanup buffer (${MAP_MIRHEO_CLEANUP_BUFFER_SECONDS}s)." >&2
  exit 2
fi

max_timeout_per_attempt=$((available_budget / attempt_count))
default_timeout_seconds=1800
if [[ -z "${MAP_MIRHEO_TIMEOUT_SECONDS}" ]]; then
  if (( default_timeout_seconds < max_timeout_per_attempt )); then
    MAP_MIRHEO_TIMEOUT_SECONDS="${default_timeout_seconds}"
  else
    MAP_MIRHEO_TIMEOUT_SECONDS="${max_timeout_per_attempt}"
  fi
fi
if [[ ! "${MAP_MIRHEO_TIMEOUT_SECONDS}" =~ ^[0-9]+$ || "${MAP_MIRHEO_TIMEOUT_SECONDS}" -lt 1 ]]; then
  echo "ERROR: MAP_MIRHEO_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if (( MAP_MIRHEO_TIMEOUT_SECONDS > max_timeout_per_attempt )); then
  echo "ERROR: MAP_MIRHEO_TIMEOUT_SECONDS=${MAP_MIRHEO_TIMEOUT_SECONDS}s exceeds safe per-attempt budget (${max_timeout_per_attempt}s) for GPU_TIME_LIMIT=${GPU_TIME_LIMIT} with MAP_MIRHEO_MAX_RETRIES=${MAP_MIRHEO_MAX_RETRIES}." >&2
  exit 2
fi

common_args=(--experiment "${EXPERIMENT}" --model-family "${MODEL_FAMILY}" --profile "${PROFILE}" --python-bin "${PYTHON_BIN}")
common_args+=(--output-dir "${OUTPUT_DIR}")
if [[ -n "${MAP_MIRHEO_NUMSTEPS}" ]]; then
  common_args+=(--numsteps "${MAP_MIRHEO_NUMSTEPS}")
fi
if [[ -n "${MAP_MIRHEO_NUMSTEPS_EQ}" ]]; then
  common_args+=(--numsteps-eq "${MAP_MIRHEO_NUMSTEPS_EQ}")
fi
common_args+=(--timeout-seconds "${MAP_MIRHEO_TIMEOUT_SECONDS}")
common_args+=(--max-retries "${MAP_MIRHEO_MAX_RETRIES}")
common_args+=(--retry-dt-scale-factor "${MAP_MIRHEO_RETRY_DT_SCALE}")
if [[ -n "${MAP_MIRHEO_DATASET_NAME}" ]]; then
  common_args+=(--dataset-name "${MAP_MIRHEO_DATASET_NAME}")
fi

python -u scripts/platforms/hpc/run_map_mirheo.py "${common_args[@]}" --n-displacements "${MAP_MIRHEO_N_DISPLACEMENTS}"
