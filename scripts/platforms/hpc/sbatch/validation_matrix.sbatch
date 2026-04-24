#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SITE="${HPC_SITE:-vega}"

case "$SITE" in
  vega|karolina) ;;
  *)
    echo "Unsupported HPC_SITE=${SITE}. Expected one of: vega, karolina." >&2
    exit 2
    ;;
esac

exec "${REPO_ROOT}/scripts/${SITE}/sbatch/validation_matrix.sbatch"
