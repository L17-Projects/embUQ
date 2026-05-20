#!/usr/bin/env bash
#SBATCH --job-name=mesouq-karolina-bootstrap
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=02:00:00
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

cd "${REPO_ROOT}"
source scripts/platforms/karolina/env_karolina.sh
module load Python/3.10.8-GCCcore-12.2.0
module load GSL/2.7-GCC-12.3.0
module load Eigen/3.4.0-GCCcore-12.2.0

VENV_DIR="${VENV_DIR:-${MESOUQ_SITE_RUNTIME_ROOT}/venv}"
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install -e ".[test,mpi]"
pip install pybind11 meson ninja
python scripts/platforms/hpc/doctor_hpc.py --site karolina
bash scripts/platforms/hpc/bootstrap_korali.sh --site karolina --jobs 8
source "${MESOUQ_SITE_RUNTIME_ROOT}/korali/env.sh"
python scripts/platforms/hpc/doctor_hpc.py --site karolina --strict
