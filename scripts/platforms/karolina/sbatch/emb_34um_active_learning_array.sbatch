#!/usr/bin/env bash
#SBATCH --job-name=mesouq-emb-34um-al-array
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --gpus=1
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
export REPO_ROOT

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
EXECUTION_MODE="${EXECUTION_MODE:-execute}"
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
VAULT_ROOT="${VAULT_ROOT:-}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-emb-34um-final-gate}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-}"
BATCH_DIR_OVERRIDE="${BATCH_DIR_OVERRIDE:-}"
SEED="${SEED:-}"
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
case "${MODE}" in
  full|canary|lhs|benchmark|validation|shared_initial|al-step-[0-9][0-9]|lhs-step-[0-9][0-9]) ;;
  *)
  echo "MODE must be 'full', 'canary', 'lhs', 'benchmark', 'validation', 'shared_initial', 'al-step-XX', or 'lhs-step-XX'." >&2
  exit 2
esac
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

if [[ -n "${BATCH_DIR_OVERRIDE}" ]]; then
  BATCH_DIR="${BATCH_DIR_OVERRIDE}"
elif [[ "${MODE}" == "full" ]]; then
  BATCH_DIR="${CAMPAIGN_ROOT}/full_gate"
elif [[ "${MODE}" == "lhs" ]]; then
  BATCH_DIR="${CAMPAIGN_ROOT}/lhs_gate"
elif [[ "${MODE}" == "benchmark" ]]; then
  BATCH_DIR="${CAMPAIGN_ROOT}/benchmark"
elif [[ "${MODE}" == "validation" && -z "${SEED}" ]]; then
  BATCH_DIR="${CAMPAIGN_ROOT}/validation"
else
  if [[ "${MODE}" == "canary" ]]; then
    BATCH_DIR="${CAMPAIGN_ROOT}/canary"
  else
    if [[ -z "${SEED}" ]]; then
      echo "SEED is required for causal stages or canary-like stage batches." >&2
      exit 2
    fi
    if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
      echo "SEED must be a non-negative integer." >&2
      exit 2
    fi
    SEED_PADDED="$(printf '%03d' "${SEED}")"
    if [[ "${MODE}" == "shared_initial" ]]; then
      BATCH_DIR="${CAMPAIGN_ROOT}/seed-${SEED_PADDED}/shared_initial"
    elif [[ "${MODE}" == "validation" ]]; then
      BATCH_DIR="${CAMPAIGN_ROOT}/seed-${SEED_PADDED}/validation"
    elif [[ "${MODE}" == al-step-* || "${MODE}" == lhs-step-* ]]; then
      BATCH_DIR="${CAMPAIGN_ROOT}/seed-${SEED_PADDED}/${MODE}"
    else
      echo "Unsupported causal mode: ${MODE}." >&2
      exit 2
    fi
  fi
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
elif [[ "${MODE}" == "canary" || "${MODE}" == "lhs" || "${MODE}" == "benchmark" || ( "${MODE}" == "validation" && -z "${SEED}" ) ]]; then
  TASK_INDEX=0
else
  if [[ "${MODE}" == "full" ]]; then
    echo "MODE=full requires submission with an sbatch --array range." >&2
  else
    echo "MODE=${MODE} requires submission with an sbatch --array range." >&2
  fi
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
