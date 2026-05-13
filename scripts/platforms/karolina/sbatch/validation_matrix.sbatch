#!/usr/bin/env bash
#SBATCH --job-name=mesouq-validation-matrix
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
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi

EXPERIMENTS="${EXPERIMENTS:-compression indentation}"
MODEL_FAMILIES="${MODEL_FAMILIES:-full-model reduced-model}"
PROFILE="${PROFILE:-validation}"
SITE="${SITE:-karolina}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS:-4}"
RUN_MAP_MIRHEO="${RUN_MAP_MIRHEO:-false}"
MAP_MIRHEO_N_DISPLACEMENTS="${MAP_MIRHEO_N_DISPLACEMENTS:-1}"
SKIP_RELEASE_MANIFEST="${SKIP_RELEASE_MANIFEST:-true}"

if [[ "${PROFILE}" != "validation" ]]; then
  echo "validation_matrix.sbatch only supports PROFILE=validation." >&2
  exit 2
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-${MESOUQ_RUNS_ROOT}/validation_matrix/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

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

"$PYTHON_BIN" scripts/platforms/karolina/run_validation_matrix.py \
  --experiments $EXPERIMENTS \
  --model-families $MODEL_FAMILIES \
  --output-root "$OUTPUT_ROOT" \
  --site "$SITE" \
  --run-tag "$RUN_TAG" \
  --phase2-cpu-ranks "$PHASE2_CPU_RANKS" \
  --python-bin "$PYTHON_BIN" \
  "${extra_args[@]}"
