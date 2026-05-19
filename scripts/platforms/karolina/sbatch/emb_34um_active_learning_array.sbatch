#!/usr/bin/env bash
#SBATCH --job-name=mesouq-emb-34um-al-array
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --gpus=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi
export REPO_ROOT

source /scratch/project/eu-26-17/eubrieucb/mesouq/load_mesouq_karolina.sh

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/emb/indentation:${REPO_ROOT}/emb/indentation/evalkit${PYTHONPATH:+:${PYTHONPATH}}"

TIMESTAMP="${TIMESTAMP:-}"
MODE="${MODE:-}"
EXECUTION_MODE="${EXECUTION_MODE:-execute}"
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
VAULT_ROOT="${VAULT_ROOT:-}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-emb-34um-final-gate}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-}"
CONCURRENT_JOBS="${CONCURRENT_JOBS:-30}"
RETRY_LIMIT="${RETRY_LIMIT:-3}"
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
if [[ "${MODE}" != "full" && "${MODE}" != "canary" ]]; then
  echo "MODE must be 'full' or 'canary'." >&2
  exit 2
fi
if [[ "${EXECUTION_MODE}" != "execute" && "${EXECUTION_MODE}" != "render-only" && "${EXECUTION_MODE}" != "dry-run" ]]; then
  echo "EXECUTION_MODE must be 'execute', 'render-only', or 'dry-run'." >&2
  exit 2
fi

if [[ -z "${CAMPAIGN_ROOT}" ]]; then
  CAMPAIGN_ROOT="${SCRATCH_ROOT}/${TIMESTAMP}"
fi
if [[ ! -d "${CAMPAIGN_ROOT}" ]]; then
  echo "CAMPAIGN_ROOT does not exist: ${CAMPAIGN_ROOT}" >&2
  exit 2
fi

echo "EMB 3.4um final-gate campaign"
echo "Mode: ${MODE}"
echo "Execution mode: ${EXECUTION_MODE}"
echo "Campaign root: ${CAMPAIGN_ROOT}"
echo "Run-id prefix: ${RUN_ID_PREFIX}"
echo "Concurrent jobs: ${CONCURRENT_JOBS}"
echo "Retry limit: ${RETRY_LIMIT}"
echo "Python executable: ${PYTHON_EXECUTABLE}"
echo "Vault root: ${VAULT_ROOT}"

if [[ "${MODE}" == "full" ]]; then
  BATCH_DIR="${CAMPAIGN_ROOT}/full_gate"
else
  BATCH_DIR="${CAMPAIGN_ROOT}/canary"
fi

if [[ ! -d "${BATCH_DIR}" ]]; then
  echo "Expected batch directory does not exist: ${BATCH_DIR}" >&2
  exit 2
fi

if [[ ! -f "${BATCH_DIR}/emb_34um_batch_summary.json" ]]; then
  echo "Expected batch summary missing: ${BATCH_DIR}/emb_34um_batch_summary.json" >&2
  exit 2
fi

BATCH_SUMMARY="${BATCH_DIR}/emb_34um_batch_summary.json"

if [[ "${EXECUTION_MODE}" == "render-only" ]]; then
  echo "Render-only validation completed. No candidate execution performed."
  exit 0
fi

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
  echo "Runner script missing: ${RUNNER_SCRIPT}" >&2
  exit 2
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
elif [[ "${MODE}" == "canary" ]]; then
  TASK_INDEX=0
else
  echo "MODE=full requires submission with an sbatch --array range." >&2
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
)

attempt=0
while [[ "${attempt}" -le "${RETRY_LIMIT}" ]]; do
  echo "Running candidate attempt ${attempt}/${RETRY_LIMIT}"
  if srun --ntasks=2 /usr/bin/env "${SRUN_ENV[@]}" "${PYTHON_EXECUTABLE}" "${RUNNER_SCRIPT}" \
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
