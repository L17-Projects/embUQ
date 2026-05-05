#!/bin/bash
# Submit from the repo root or set REPO_ROOT explicitly before sbatch.
# Required variables:
#   CAMPAIGN_ID=<unique replay campaign id>
# Optional variables:
#   LANES="stretching buckling torsion eigenmodes"

#SBATCH --job-name=mesouq-gv-paper-replay
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi

CAMPAIGN_ID="${CAMPAIGN_ID:-}"
if [[ -z "${CAMPAIGN_ID}" ]]; then
  echo "CAMPAIGN_ID is required." >&2
  exit 2
fi

LANES="${LANES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/gv/figure_replay/${CAMPAIGN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  OpenMPI/4.1.4-GCC-12.2.0 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0 \
  HDF5/1.14.0-gompi-2022b \
  MPFR/4.2.0-GCCcore-12.2.0 \
  GMP/6.2.1-GCCcore-12.2.0

if [[ ! -f "${REPO_ROOT}/_vega/gv_venv/env.sh" ]]; then
  echo "Missing required GV runtime environment: ${REPO_ROOT}/_vega/gv_venv/env.sh" >&2
  exit 1
fi

source "${REPO_ROOT}/_vega/gv_venv/env.sh"
export MESOUQ_GV_MPI_RANKS="${MESOUQ_GV_MPI_RANKS:-2}"
export MESOUQ_GV_EIGENMODES_MPI_RANKS="${MESOUQ_GV_EIGENMODES_MPI_RANKS:-1}"

cd "${REPO_ROOT}"

command=(
  "${PYTHON_BIN}"
  scripts/workflows/gv/run_paper_figure_replay.py
  --campaign-id "${CAMPAIGN_ID}"
  --output-root "${OUTPUT_ROOT}"
)

for lane in ${LANES}; do
  command+=(--lane "${lane}")
done

"${command[@]}"
