#!/usr/bin/env bash
#SBATCH --job-name=mesouq-dpd-preflight
#SBATCH --partition=dev
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH --exclude=gn10
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi
if [[ -n "${SITE:-}" && "${SITE}" != "vega" ]]; then
  echo "This site template is fixed to vega; use scripts/platforms/hpc/sbatch for SITE=${SITE}." >&2
  exit 2
fi
if [[ -n "${MESOUQ_SITE:-}" && "${MESOUQ_SITE}" != "vega" ]]; then
  echo "This site template is fixed to vega; MESOUQ_SITE=${MESOUQ_SITE} conflicts." >&2
  exit 2
fi
SITE="vega"
export MESOUQ_SITE="${SITE}"

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  OpenMPI/4.1.4-GCC-12.2.0 \
  CUDA/12.2.2 \
  HDF5/1.14.0-gompi-2022b \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0 \
  MPFR/4.2.0-GCCcore-12.2.0 \
  GMP/6.2.1-GCCcore-12.2.0

source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${MESOUQ_SCRATCH_ROOT:-}}"
if [[ -z "${SCRATCH_ROOT}" ]]; then
  echo "SCRATCH_ROOT or MESOUQ_SCRATCH_ROOT is required." >&2
  exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/runs/dpd_preflight_canary/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-${MESOUQ_ENV_ROOT}/bin/python}"
DPD_PREFLIGHT_MIN_FREE_BYTES="${DPD_PREFLIGHT_MIN_FREE_BYTES:-1000000000}"
DPD_PREFLIGHT_HDF5_DRIVER="${DPD_PREFLIGHT_HDF5_DRIVER:-serial}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-${REPO_ROOT}/scripts/workflows/dpd/run_dpd_production_preflight.py}"
DPD_PREFLIGHT_HELPER="${DPD_PREFLIGHT_HELPER:-${REPO_ROOT}/scripts/workflows/dpd/dpd_production_preflight.sh}"

source "${DPD_PREFLIGHT_HELPER}"
mesouq_sanitize_nested_srun_cpu_env

"${PYTHON_BIN}" "${PREFLIGHT_SCRIPT}" \
  --repo-root "${REPO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --scratch-root "${SCRATCH_ROOT}" \
  --min-free-bytes "${DPD_PREFLIGHT_MIN_FREE_BYTES}" \
  --hdf5-smoke-test \
  --hdf5-driver "${DPD_PREFLIGHT_HDF5_DRIVER}"
