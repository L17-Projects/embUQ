#!/usr/bin/env bash
# Submit from the repo root or set REPO_ROOT explicitly before sbatch.
# Required variables:
#   CAMPAIGN_ID=<unique replay campaign id>
# Optional variables:
#   LANES="stretching buckling torsion eigenmodes"
#   PAPER_EXACT=1
#   OUTPUT_ROOT=_runs/gv/figure_replay/<campaign>
#   GV_PAPER_REPLAY_TIME_LIMIT=HH:MM:SS   # explicit override
#   STRETCHING_POINT_START=<int>
#   STRETCHING_POINT_STOP=<int>
#   BUCKLING_BUCK_MAX=<float>
#   BUCKLING_POINT_COUNT=<int>
#   BUCKLING_TIMEOUT_SECONDS=<int>

#SBATCH --job-name=mesouq-gv-paper-replay
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
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
PAPER_EXACT="${PAPER_EXACT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STRETCHING_POINT_START="${STRETCHING_POINT_START:-}"
STRETCHING_POINT_STOP="${STRETCHING_POINT_STOP:-}"
BUCKLING_BUCK_MAX="${BUCKLING_BUCK_MAX:-}"
BUCKLING_POINT_COUNT="${BUCKLING_POINT_COUNT:-}"
BUCKLING_TIMEOUT_SECONDS="${BUCKLING_TIMEOUT_SECONDS:-}"

cd "${REPO_ROOT}"
source scripts/platforms/karolina/env_karolina.sh
OUTPUT_ROOT="${OUTPUT_ROOT:-${MESOUQ_RUNS_ROOT}/gv/figure_replay/${CAMPAIGN_ID}}"

GV_ENV_SCRIPT="${MESOUQ_GV_ENV_SCRIPT:-${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh}"
if [[ ! -f "${GV_ENV_SCRIPT}" ]]; then
  echo "Missing required GV runtime environment: ${GV_ENV_SCRIPT}" >&2
  exit 1
fi

source "${GV_ENV_SCRIPT}"
export MESOUQ_GV_MPI_RANKS="${MESOUQ_GV_MPI_RANKS:-2}"
export MESOUQ_GV_EIGENMODES_MPI_RANKS="${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}"

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
