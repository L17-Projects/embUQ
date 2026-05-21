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

cd "${REPO_ROOT}"
if [[ -z "${MESOUQ_SITE_RUNTIME_ROOT:-}" ]]; then
  echo "MESOUQ_SITE_RUNTIME_ROOT must be set before using this Karolina sbatch script." >&2
  exit 2
fi
source scripts/platforms/karolina/env_karolina.sh
bash scripts/platforms/hpc/bootstrap_env.sh --site karolina --python-bin "$(command -v python)" --with-korali --jobs 8
source scripts/platforms/hpc/site_env.sh
mesouq_activate_site_env karolina "${REPO_ROOT}"
python scripts/platforms/hpc/doctor_hpc.py --site karolina --strict
