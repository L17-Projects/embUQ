#!/usr/bin/env bash
#SBATCH --job-name=mesouq-bnn-train
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit from the repo root." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/vega/bnn_training/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-20260317}"
# SELECTIONS: space-separated spec names to run, e.g. "indentation_3.2um indentation_3.4um".
# Leave empty to run all specs.
SELECTIONS="${SELECTIONS:-}"
ONLY_ARGS=""
for sel in ${SELECTIONS}; do ONLY_ARGS="${ONLY_ARGS} --only ${sel}"; done

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  openmpi/4.1.2.1 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0

source "${REPO_ROOT}/_vega/venv/bin/activate"
source "${REPO_ROOT}/_vega/korali/env.sh"

cd "${REPO_ROOT}"

# Hard gate: fail immediately if no GPU is visible to PyTorch.
"$PYTHON_BIN" -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available on this node.', file=sys.stderr)
    sys.exit(1)
print('CUDA OK:', torch.cuda.get_device_name(0))
"

"$PYTHON_BIN" scripts/vega/train_bnn_surrogates.py \
  --site vega \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --python-bin "${PYTHON_BIN}" \
  --device cuda \
  --seed "${SEED}" \
  ${ONLY_ARGS}
