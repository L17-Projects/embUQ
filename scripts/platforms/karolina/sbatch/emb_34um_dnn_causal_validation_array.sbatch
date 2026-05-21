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
EXECUTION_MODE="${EXECUTION_MODE:-render-only}"
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
VAULT_ROOT="${VAULT_ROOT:-}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-emb-34um-dnn-causal-validation}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-}"
BATCH_DIR_OVERRIDE="${BATCH_DIR_OVERRIDE:-}"
REPLICA="${REPLICA:-}"
CYCLE="${CYCLE:-0}"
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

if [[ "${EXECUTION_MODE}" != "render-only" && "${EXECUTION_MODE}" != "dry-run" ]]; then
  echo "EXECUTION_MODE must be 'render-only' or 'dry-run'." >&2
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
if [[ ! -f "${BATCH_DIR}/emb_34um_batch_summary.json" ]]; then
  echo "Expected batch summary missing: ${BATCH_DIR}/emb_34um_batch_summary.json" >&2
  exit 2
fi

echo "EMB 3.4um DNN causal validation array"
echo "Mode: ${MODE}"
echo "Execution mode: ${EXECUTION_MODE}"
echo "Batch directory: ${BATCH_DIR}"
echo "Run-id prefix: ${RUN_ID_PREFIX}"
echo "Concurrent jobs: ${CONCURRENT_JOBS}"
echo "Retry limit: ${RETRY_LIMIT}"
echo "Python executable: ${PYTHON_EXECUTABLE}"
echo "Vault root: ${VAULT_ROOT}"

echo "Render-only validation completed. No DPD submission performed."
exit 0
