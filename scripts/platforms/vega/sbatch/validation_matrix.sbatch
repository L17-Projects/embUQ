#!/usr/bin/env bash
#SBATCH --job-name=mesouq-validation-matrix
#SBATCH --partition=dev
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi
EXPERIMENTS="${EXPERIMENTS:-compression indentation}"
MODEL_FAMILIES="${MODEL_FAMILIES:-full-model reduced-model}"
PROFILE="${PROFILE:-validation}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/validation_matrix/${RUN_TAG}}"
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS:-4}"
RUN_MAP_MIRHEO="${RUN_MAP_MIRHEO:-false}"
MAP_MIRHEO_N_DISPLACEMENTS="${MAP_MIRHEO_N_DISPLACEMENTS:-1}"
SKIP_RELEASE_MANIFEST="${SKIP_RELEASE_MANIFEST:-true}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${PROFILE}" != "validation" ]]; then
  echo "validation_matrix.sbatch only supports PROFILE=validation." >&2
  exit 2
fi

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  openmpi/4.1.2.1 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0

ENV_ROOT="${MESOUQ_ENV_ROOT:-${REPO_ROOT}/_vega/env}"
ENV_SCRIPT="${MESOUQ_ENV_SCRIPT:-${ENV_ROOT}/env.sh}"
if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing canonical MesoUQ environment: ${ENV_SCRIPT}" >&2
  exit 2
fi
source "${ENV_SCRIPT}"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

extra_args=()
case "${RUN_MAP_MIRHEO,,}" in
  true|1|yes|on)
    extra_args+=(--run-map-mirheo --map-mirheo-n-displacements "$MAP_MIRHEO_N_DISPLACEMENTS")
    ;;
  false|0|no|off) ;;
  *)
    echo "RUN_MAP_MIRHEO must be true or false, got '${RUN_MAP_MIRHEO}'." >&2
    exit 2
    ;;
esac

case "${SKIP_RELEASE_MANIFEST,,}" in
  true|1|yes|on)
    extra_args+=(--skip-release-manifest)
    ;;
  false|0|no|off) ;;
  *)
    echo "SKIP_RELEASE_MANIFEST must be true or false, got '${SKIP_RELEASE_MANIFEST}'." >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" scripts/platforms/hpc/run_validation_matrix.py \
  --experiments $EXPERIMENTS \
  --model-families $MODEL_FAMILIES \
  --output-root "$OUTPUT_ROOT" \
  --site "$SITE" \
  --run-tag "$RUN_TAG" \
  --phase2-cpu-ranks "$PHASE2_CPU_RANKS" \
  --python-bin "$PYTHON_BIN" \
  "${extra_args[@]}"
