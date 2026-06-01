#!/usr/bin/env bash
#SBATCH --job-name=mesouq-prod-sanity
#SBATCH --partition=dev
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi

SELECTIONS="${SELECTIONS:-compression:full-model:production}"
ALL_LANES="${ALL_LANES:-0}"
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
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/production_sanity/${RUN_TAG}}"
PHASE2_BACKEND="${PHASE2_BACKEND:-}"
if [[ -z "${PHASE2_CPU_RANKS:-}" ]]; then
  if [[ "${PHASE2_BACKEND}" == "cpu-mpi" ]]; then
    PHASE2_CPU_RANKS="64"
  else
    PHASE2_CPU_RANKS="1"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  OpenMPI/4.1.4-GCC-12.2.0 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0

ENV_ROOT="${MESOUQ_ENV_ROOT:-${MESOUQ_SITE_RUNTIME_ROOT:?Set MESOUQ_SITE_RUNTIME_ROOT before using this sbatch script.}/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment: ${ENV_SCRIPT}" >&2
  exit 2
fi
source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"

cd "${REPO_ROOT}"

command=(
  "$PYTHON_BIN"
  scripts/platforms/hpc/run_production_sanity.py
  --output-root "$OUTPUT_ROOT"
  --site "$SITE"
  --run-tag "$RUN_TAG"
  --phase2-cpu-ranks "$PHASE2_CPU_RANKS"
  --python-bin "$PYTHON_BIN"
)

if [[ -n "$PHASE2_BACKEND" ]]; then
  command+=(--phase2-backend "$PHASE2_BACKEND")
fi

if [[ "$ALL_LANES" == "1" ]]; then
  command+=(--all-lanes)
fi

for selection in $SELECTIONS; do
  command+=(--selection "$selection")
done

"${command[@]}"
