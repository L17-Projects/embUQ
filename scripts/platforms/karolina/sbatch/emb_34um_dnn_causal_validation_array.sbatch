#!/usr/bin/env bash
#SBATCH --job-name=mesouq-emb-34um-dnn-causal-array
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --exclude=acn05,acn13,acn14
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi
export REPO_ROOT
cd "${REPO_ROOT}"

EXECUTION_MODE="${EXECUTION_MODE:-execute}"
MESOUQ_DPD_ARRAY_LAUNCH_STAGGER_SECONDS="${MESOUQ_DPD_ARRAY_LAUNCH_STAGGER_SECONDS:-4}"
if [[ "${EXECUTION_MODE}" == "execute" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ && "${MESOUQ_DPD_ARRAY_LAUNCH_STAGGER_SECONDS}" =~ ^[0-9]+$ ]]; then
    STAGGER_SLOT=$((10#${SLURM_ARRAY_TASK_ID} % 8))
    STAGGER_SLEEP=$((STAGGER_SLOT * 10#${MESOUQ_DPD_ARRAY_LAUNCH_STAGGER_SECONDS}))
    if [[ "${STAGGER_SLEEP}" -gt 0 ]]; then
      echo "Staggering DPD array launch by ${STAGGER_SLEEP}s before Karolina environment load."
      sleep "${STAGGER_SLEEP}"
    fi
  fi
fi

if [[ -z "${MESOUQ_SITE_RUNTIME_ROOT:-}" ]]; then
  echo "MESOUQ_SITE_RUNTIME_ROOT must be set before using this Karolina sbatch script." >&2
  exit 2
fi
source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"
source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env karolina "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/emb/indentation:${REPO_ROOT}/emb/indentation/evalkit${PYTHONPATH:+:${PYTHONPATH}}"

TIMESTAMP="${TIMESTAMP:-}"
MODE="${MODE:-}"
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
VAULT_ROOT="${VAULT_ROOT:-}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-emb-34um-dnn-causal-validation}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-}"
BATCH_DIR_OVERRIDE="${BATCH_DIR_OVERRIDE:-}"
BATCH_SUMMARY="${BATCH_SUMMARY:-}"
BATCH_NAME="${BATCH_NAME:-}"
REPLICA="${REPLICA:-}"
CYCLE="${CYCLE:-0}"
BRANCH="${BRANCH:-dnn_causal}"
CONCURRENT_JOBS="${CONCURRENT_JOBS:-30}"
RETRY_LIMIT="${RETRY_LIMIT:-0}"

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE="$(command -v python || true)"
else
  PYTHON_EXECUTABLE_RESOLVED="$(command -v "${PYTHON_EXECUTABLE}" || true)"
  if [[ -n "${PYTHON_EXECUTABLE_RESOLVED}" ]]; then
    PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE_RESOLVED}"
  fi
fi
if [[ -z "${PYTHON_EXECUTABLE}" || ! -x "${PYTHON_EXECUTABLE}" ]]; then
  echo "Unable to resolve an executable Python interpreter after loading the MesoUQ environment." >&2
  exit 2
fi
export PYTHON_EXECUTABLE
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/workflows/emb/active_learning/run_emb_34um_final_gate_candidate.py}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-${REPO_ROOT}/scripts/workflows/dpd/run_dpd_production_preflight.py}"
DPD_PREFLIGHT_HELPER="${DPD_PREFLIGHT_HELPER:-${REPO_ROOT}/scripts/workflows/dpd/dpd_production_preflight.sh}"
DPD_PREFLIGHT="${DPD_PREFLIGHT:-1}"
DPD_PREFLIGHT_MIN_FREE_BYTES="${DPD_PREFLIGHT_MIN_FREE_BYTES:-1000000000}"
DPD_PREFLIGHT_HDF5_DRIVER="${DPD_PREFLIGHT_HDF5_DRIVER:-serial}"
DPD_PREFLIGHT_HDF5_SMOKE_TEST="${DPD_PREFLIGHT_HDF5_SMOKE_TEST:-1}"

if [[ "${EXECUTION_MODE}" != "execute" && "${EXECUTION_MODE}" != "render-only" && "${EXECUTION_MODE}" != "dry-run" ]]; then
  echo "EXECUTION_MODE must be 'execute', 'render-only', or 'dry-run'." >&2
  exit 2
fi

if [[ -n "${BATCH_DIR_OVERRIDE}" ]]; then
  BATCH_DIR="${BATCH_DIR_OVERRIDE}"
elif [[ -n "${CAMPAIGN_ROOT}" && "${MODE}" == "unseen_test" ]]; then
  BATCH_DIR="${CAMPAIGN_ROOT}/unseen_test"
elif [[ -n "${CAMPAIGN_ROOT}" && -n "${REPLICA}" ]]; then
  if [[ "${MODE}" == "shared_initial" ]]; then
    BATCH_DIR="${CAMPAIGN_ROOT}/replica-$(printf '%03d' "${REPLICA}")/shared_initial"
  elif [[ "${MODE}" == lhs-step-* || "${MODE}" == al-step-* ]]; then
    BATCH_DIR="${CAMPAIGN_ROOT}/replica-$(printf '%03d' "${REPLICA}")/${MODE}"
  else
    echo "Unsupported MODE for DNN causal validation: ${MODE}." >&2
    exit 2
  fi
else
  echo "BATCH_DIR_OVERRIDE or CAMPAIGN_ROOT/REPLICA must be set." >&2
  exit 2
fi

if [[ -z "${TIMESTAMP}" ]]; then
  echo "TIMESTAMP is required." >&2
  exit 2
fi
if [[ -z "${SCRATCH_ROOT}" ]]; then
  echo "SCRATCH_ROOT is required." >&2
  exit 2
fi
if [[ -z "${VAULT_ROOT}" ]]; then
  echo "VAULT_ROOT is required." >&2
  exit 2
fi
if [[ ! -d "${BATCH_DIR}" ]]; then
  echo "Expected batch directory does not exist: ${BATCH_DIR}" >&2
  exit 2
fi

if [[ -z "${BATCH_SUMMARY}" ]]; then
  BATCH_SUMMARY="${BATCH_DIR}/emb_34um_batch_summary.json"
fi
if [[ ! -f "${BATCH_SUMMARY}" ]]; then
  echo "Expected batch summary missing: ${BATCH_SUMMARY}" >&2
  exit 2
fi
if [[ -z "${BATCH_NAME}" ]]; then
  BATCH_NAME="$(basename "${BATCH_DIR}")"
fi
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
  echo "Runner script missing: ${RUNNER_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${PREFLIGHT_SCRIPT}" ]]; then
  echo "Preflight script missing: ${PREFLIGHT_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${DPD_PREFLIGHT_HELPER}" ]]; then
  echo "DPD preflight helper missing: ${DPD_PREFLIGHT_HELPER}" >&2
  exit 2
fi
source "${DPD_PREFLIGHT_HELPER}"

echo "EMB 3.4um DNN causal validation array"
echo "Mode: ${MODE}"
echo "Execution mode: ${EXECUTION_MODE}"
echo "Branch: ${BRANCH}"
echo "Batch directory: ${BATCH_DIR}"
echo "Batch name: ${BATCH_NAME}"
echo "Batch summary: ${BATCH_SUMMARY}"
echo "Run-id prefix: ${RUN_ID_PREFIX}"
echo "Replica: ${REPLICA:-n/a}"
echo "Cycle: ${CYCLE}"
echo "Concurrent jobs: ${CONCURRENT_JOBS}"
echo "Retry limit: ${RETRY_LIMIT}"
echo "Python executable: ${PYTHON_EXECUTABLE}"
echo "Runner script: ${RUNNER_SCRIPT}"
echo "Preflight script: ${PREFLIGHT_SCRIPT}"
echo "Preflight helper: ${DPD_PREFLIGHT_HELPER}"
echo "Vault root: ${VAULT_ROOT}"

# These EMB runs use two MPI ranks on one Karolina node. Avoid OpenMPI's
# openib BTL, which can crash in libibverbs during MPI initialization on
# Karolina GPU nodes and is not needed for same-node Mirheo execution.
export OMPI_MCA_btl="${MESOUQ_KAROLINA_OMPI_MCA_BTL:-self,vader,tcp}"
export OMPI_MCA_btl_openib_warn_default_gid_prefix="${OMPI_MCA_btl_openib_warn_default_gid_prefix:-0}"
echo "OpenMPI BTL: ${OMPI_MCA_btl}"

if [[ "${EXECUTION_MODE}" == "render-only" ]]; then
  echo "Render-only validation completed. No candidate execution performed."
  exit 0
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
elif [[ "${MODE}" == "unseen_test" ]]; then
  TASK_INDEX=0
else
  echo "MODE=${MODE} requires submission with an sbatch --array range." >&2
  exit 2
fi

CANDIDATE_MANIFEST="$("${PYTHON_EXECUTABLE}" - "${BATCH_SUMMARY}" "${TASK_INDEX}" <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
sys.path.insert(0, str(repo_root))
from scripts.workflows.emb.active_learning.run_emb_34um_final_gate_candidate import select_candidate_manifest

print(select_candidate_manifest(sys.argv[1], int(sys.argv[2])))
PY
)"

echo "Array task index: ${TASK_INDEX}"
echo "Candidate manifest: ${CANDIDATE_MANIFEST}"

mesouq_sanitize_nested_srun_cpu_env
mesouq_run_dpd_production_preflight

TIMEOUT_MARKED=0
mark_failed_from_signal() {
  signal_name="${1:-TERM}"
  if [[ "${TIMEOUT_MARKED}" == "1" ]]; then
    exit 143
  fi
  TIMEOUT_MARKED=1
  echo "Received ${signal_name}; marking candidate failed before Slurm cleanup." >&2
  "${PYTHON_EXECUTABLE}" "${RUNNER_SCRIPT}" \
    --candidate-manifest "${CANDIDATE_MANIFEST}" \
    --retry-attempt "${attempt:-0}" \
    --mark-failed \
    --error-message "Candidate interrupted by Slurm signal ${signal_name} in job ${SLURM_JOB_ID:-unknown}." || true
  exit 143
}
trap 'mark_failed_from_signal TERM' TERM
trap 'mark_failed_from_signal INT' INT

"${PYTHON_EXECUTABLE}" "${RUNNER_SCRIPT}" \
  --candidate-manifest "${CANDIDATE_MANIFEST}" \
  --dry-run

DRY_RUN_FLAG=()
if [[ "${EXECUTION_MODE}" == "dry-run" ]]; then
  DRY_RUN_FLAG=(--dry-run)
fi

SRUN_ENV=(
  "PATH=${PATH}"
  "PYTHONPATH=${PYTHONPATH:-}"
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
  "REPO_ROOT=${REPO_ROOT}"
  "OMPI_MCA_btl=${OMPI_MCA_btl}"
  "OMPI_MCA_btl_openib_warn_default_gid_prefix=${OMPI_MCA_btl_openib_warn_default_gid_prefix}"
)

attempt=0
while [[ "${attempt}" -le "${RETRY_LIMIT}" ]]; do
  echo "Running candidate attempt ${attempt}/${RETRY_LIMIT}"
  if srun --ntasks=2 --kill-on-bad-exit=1 /usr/bin/env "${SRUN_ENV[@]}" "${PYTHON_EXECUTABLE}" "${RUNNER_SCRIPT}" \
    --candidate-manifest "${CANDIDATE_MANIFEST}" \
    --retry-attempt "${attempt}" \
    "${DRY_RUN_FLAG[@]}"; then
    echo "Candidate completed at attempt ${attempt}"
    exit 0
  fi
  attempt=$((attempt + 1))
done

echo "Candidate failed after $((RETRY_LIMIT + 1)) attempts: ${CANDIDATE_MANIFEST}" >&2
"${PYTHON_EXECUTABLE}" "${RUNNER_SCRIPT}" \
  --candidate-manifest "${CANDIDATE_MANIFEST}" \
  --retry-attempt "${RETRY_LIMIT}" \
  --mark-failed \
  --error-message "Candidate failed after $((RETRY_LIMIT + 1)) attempts in Slurm job ${SLURM_JOB_ID:-unknown}."
exit 1
