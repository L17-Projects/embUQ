#!/usr/bin/env bash
#SBATCH --job-name=mesouq-karolina-bootstrap
#SBATCH --partition=qgpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi

cd "${REPO_ROOT}"
source scripts/platforms/karolina/env_karolina.sh
module load Python/3.10.8-GCCcore-12.2.0
module load GSL/2.7-GCC-12.3.0
module load Eigen/3.4.0-GCCcore-12.2.0

python -m venv _vega/venv
source _vega/venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test,mpi]"
pip install pybind11 meson ninja
python scripts/platforms/karolina/doctor_karolina.py
bash scripts/platforms/karolina/bootstrap_korali.sh --jobs 8
source _vega/korali/env.sh
python scripts/platforms/karolina/doctor_karolina.py --strict
