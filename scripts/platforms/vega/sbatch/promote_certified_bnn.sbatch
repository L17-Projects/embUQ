#!/bin/bash -l
#SBATCH --job-name=mesouq-bnn-promote
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

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

CERTIFICATION_ROOT="${CERTIFICATION_ROOT:-}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
DRY_RUN="${DRY_RUN:-false}"
# Promotion consumes one certification root at a time; keep the selection axis visible for
# operator parity with other Vega sbatch entrypoints.
SELECTIONS="${SELECTIONS:-}"
DEFAULT_LOG_BASE="${CERTIFICATION_ROOT:-${REPO_ROOT}/_runs/vega/bnn_promotion_preflight}"
LOG_ROOT="${LOG_ROOT:-${DEFAULT_LOG_BASE}/logs}"
ENV_ROOT="${MESOUQ_ENV_ROOT:-${MESOUQ_SITE_RUNTIME_ROOT:?Set MESOUQ_SITE_RUNTIME_ROOT before using this sbatch script.}/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_ROOT}/bin/python}"

cd "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}"
JOB_TOKEN="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out") 2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)

if [[ -z "${CERTIFICATION_ROOT}" ]]; then
  echo "Set CERTIFICATION_ROOT before submitting promotion." >&2
  exit 2
fi

module purge
module load Python/3.10.8-GCCcore-12.2.0

if [[ ! -x "${PYTHON_BIN}" || ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment at ${ENV_ROOT}." >&2
  exit 2
fi

source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

command=(
  "${PYTHON_BIN}"
  scripts/platforms/hpc/promote_certified_bnn.py
  --certification-root "${CERTIFICATION_ROOT}"
)

if [[ -n "${MANIFEST_PATH}" ]]; then
  command+=(--manifest-path "${MANIFEST_PATH}")
fi

case "${DRY_RUN,,}" in
  true|1|yes)
    command+=(--dry-run)
    ;;
esac

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
