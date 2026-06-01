#!/bin/bash
# Submit from the repo root or set REPO_ROOT explicitly before sbatch.
# Required variables:
#   CAMPAIGN_ID=<unique replay campaign id>
# Optional variables:
#   LANES="stretching buckling torsion eigenmodes"
#   PAPER_EXACT=1
#   STRETCHING_POINT_START=<inclusive index>
#   STRETCHING_POINT_STOP=<exclusive index>
#   BUCKLING_BUCK_MAX=<float>
#   BUCKLING_POINT_COUNT=<int>
#   BUCKLING_TIMEOUT_SECONDS=<int>
#
# For lane-aware walltime defaults, prefer:
#   scripts/platforms/vega/submit_gv_paper_figure_replay.sh

#SBATCH --job-name=mesouq-gv-paper-replay
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=gn10
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

CAMPAIGN_ID="${CAMPAIGN_ID:-}"
if [[ -z "${CAMPAIGN_ID}" ]]; then
  echo "CAMPAIGN_ID is required." >&2
  exit 2
fi

LANES="${LANES:-}"
PAPER_EXACT="${PAPER_EXACT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/gv/figure_replay/${CAMPAIGN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STRETCHING_POINT_START="${STRETCHING_POINT_START:-}"
STRETCHING_POINT_STOP="${STRETCHING_POINT_STOP:-}"
BUCKLING_BUCK_MAX="${BUCKLING_BUCK_MAX:-}"
BUCKLING_POINT_COUNT="${BUCKLING_POINT_COUNT:-}"
BUCKLING_TIMEOUT_SECONDS="${BUCKLING_TIMEOUT_SECONDS:-}"

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

source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"
export MESOUQ_GV_MPI_RANKS="${MESOUQ_GV_MPI_RANKS:-2}"
export MESOUQ_GV_EIGENMODES_MPI_RANKS="${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}"
export MESOUQ_GV_EIGENMODES_DOMAIN_RANKS="${MESOUQ_GV_EIGENMODES_DOMAIN_RANKS:-1,1,1}"

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

if [[ "${PAPER_EXACT}" == "1" || "${PAPER_EXACT}" == "true" || "${PAPER_EXACT}" == "TRUE" ]]; then
  command+=(--paper-exact)
fi

if [[ -n "${STRETCHING_POINT_START}" ]]; then
  command+=(--stretching-point-start "${STRETCHING_POINT_START}")
fi

if [[ -n "${STRETCHING_POINT_STOP}" ]]; then
  command+=(--stretching-point-stop "${STRETCHING_POINT_STOP}")
fi

if [[ -n "${BUCKLING_BUCK_MAX}" ]]; then
  command+=(--buckling-buck-max "${BUCKLING_BUCK_MAX}")
fi

if [[ -n "${BUCKLING_POINT_COUNT}" ]]; then
  command+=(--buckling-point-count "${BUCKLING_POINT_COUNT}")
fi

if [[ -n "${BUCKLING_TIMEOUT_SECONDS}" ]]; then
  command+=(--buckling-timeout-seconds "${BUCKLING_TIMEOUT_SECONDS}")
fi

"${command[@]}"
