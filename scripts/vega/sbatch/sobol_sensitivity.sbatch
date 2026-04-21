#!/usr/bin/env bash
#SBATCH --job-name=mesouq-sobol
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/vega/sobol/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# MODEL_FAMILIES: space-separated families to run. Default: dnn bnn
MODEL_FAMILIES="${MODEL_FAMILIES:-dnn bnn}"
# SELECTIONS: space-separated <modality>_<diameter>um keys to restrict runs.
# Leave empty to run all cases.
SELECTIONS="${SELECTIONS:-}"

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0

source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

fail=0

_want() {
  local key="$1"
  [[ -z "${SELECTIONS}" ]] && return 0
  for s in ${SELECTIONS}; do [[ "$s" == "$key" ]] && return 0; done
  return 1
}

# --- Compression (dnn + bnn) ---
for fam in ${MODEL_FAMILIES}; do
  for d in 2.1 2.9 3.0; do
    _want "compression_${d}um" || continue
    out="${OUTPUT_ROOT}/compression/${fam}/sobol_vs_disp_${d}um.csv"
    mkdir -p "$(dirname "$out")"
    if [[ "$fam" == "dnn" ]]; then
      model="compression/surrogate/diameters/${d}um/trained/microbubble_force_BEST.pkl"
    else
      model="compression/surrogate/diameters/${d}um/trained/microbubble_force_BNN.pt"
    fi
    echo "[sobol] compression/${fam}/${d}um"
    "$PYTHON_BIN" compression/surrogate/sensitivity/scripts/run_sobol_vs_disp.py \
      --model "$model" --surrogate-family "$fam" --output "$out" \
      --n-samples 1024 --n-displacements 20 \
      --predictive-mc-samples 64 --predictive-mc-chunk-size 8 \
      --device cpu || { echo "FAILED: compression/${fam}/${d}um" >&2; fail=1; }
  done
done

# --- Indentation (dnn + bnn) ---
for fam in ${MODEL_FAMILIES}; do
  for d in 3.2 3.4 5.8; do
    _want "indentation_${d}um" || continue
    out="${OUTPUT_ROOT}/indentation/${fam}/sobol_vs_force_${d}um.csv"
    mkdir -p "$(dirname "$out")"
    if [[ "$fam" == "dnn" ]]; then
      model="indentation/surrogate/diameters/${d}um/trained/microbubble_displacement_BEST.pkl"
    else
      model="indentation/surrogate/diameters/${d}um/trained/microbubble_displacement_BNN.pt"
    fi
    echo "[sobol] indentation/${fam}/${d}um"
    "$PYTHON_BIN" indentation/surrogate/sensitivity/scripts/run_sobol_vs_force.py \
      --model "$model" --surrogate-family "$fam" --output "$out" \
      --n-samples 1024 --n-forces 20 \
      --predictive-mc-samples 64 --predictive-mc-chunk-size 8 \
      --device cpu || { echo "FAILED: indentation/${fam}/${d}um" >&2; fail=1; }
  done
done

exit $fail
