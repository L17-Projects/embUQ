#!/usr/bin/env bash
#SBATCH --job-name=mesouq-emb-34um-al-array
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"

TIMESTAMP="${TIMESTAMP:-}"
MODE="${MODE:-}"
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
VAULT_ROOT="${VAULT_ROOT:-}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-emb-34um-final-gate}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-}"
CONCURRENT_JOBS="${CONCURRENT_JOBS:-30}"
RETRY_LIMIT="${RETRY_LIMIT:-3}"

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

if [[ -z "${CAMPAIGN_ROOT}" ]]; then
  CAMPAIGN_ROOT="${SCRATCH_ROOT}/${TIMESTAMP}"
fi
if [[ ! -d "${CAMPAIGN_ROOT}" ]]; then
  echo "CAMPAIGN_ROOT does not exist: ${CAMPAIGN_ROOT}" >&2
  exit 2
fi

echo "Render-only EMB 3.4um final-gate campaign"
echo "Mode: ${MODE}"
echo "Campaign root: ${CAMPAIGN_ROOT}"
echo "Run-id prefix: ${RUN_ID_PREFIX}"
echo "Concurrent jobs: ${CONCURRENT_JOBS}"
echo "Retry limit: ${RETRY_LIMIT}"
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

echo "This script is render-only. No candidate submissions are performed."
