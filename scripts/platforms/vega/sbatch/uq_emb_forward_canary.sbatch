#!/usr/bin/env bash
#SBATCH --job-name=uq-emb-forward
#SBATCH --partition=dev
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=gn10
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must point at the MesoUQ checkout}"
: "${MESOUQ_SITE_RUNTIME_ROOT:?MESOUQ_SITE_RUNTIME_ROOT must identify the canonical runtime}"
: "${CONFIG_PATH:?CONFIG_PATH must be a materialized UQ_EMB config}"
: "${OUTPUT_PATH:?OUTPUT_PATH must be the canary JSON receipt path}"

module purge
module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.2.2 GSL/2.7-GCC-12.2.0 Eigen/3.4.0-GCCcore-12.2.0
source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"
mesouq_activate_site_env vega "${REPO_ROOT}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/workflows/emb/uq_emb/run_forward_canary.py \
  --site vega \
  --device cuda \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_PATH}"
