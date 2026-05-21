#!/usr/bin/env bash
#SBATCH --job-name=mes208-phase2-replicate
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=1

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must point at the MesoUQ checkout}"
: "${LANE_ROOT:?LANE_ROOT must be the per-replicate output root}"
: "${PHASE1_SOURCE:?PHASE1_SOURCE must point at results_phase_1}"
: "${PHASE2_BACKEND:?PHASE2_BACKEND must be cpu-mpi or native-cuda}"
: "${REPLICATE_ID:?REPLICATE_ID must identify this replicate}"

MESOUQ_SITE_RUNTIME_ROOT="${MESOUQ_SITE_RUNTIME_ROOT:-/scratch/project/eu-26-17/eubrieucb/mesouq/runtime}"
PYTHON_BIN="${PYTHON_BIN:-${MESOUQ_SITE_RUNTIME_ROOT}/env/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/inference/configs/production/inference_config_indentation.yaml}"
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS:-1}"
LOG_ROOT="${LOG_ROOT:-${LANE_ROOT}/logs}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${LANE_ROOT}/manifests}"
ENV_SCRIPT="${ENV_SCRIPT:-${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh}"
KORALI_ENV_SCRIPT="${KORALI_ENV_SCRIPT:-}"

case "${PHASE2_BACKEND}" in
  cpu-mpi)
    if [[ "${PHASE2_CPU_RANKS}" -lt 1 ]]; then
      echo "ERROR: PHASE2_CPU_RANKS must be >= 1 for cpu-mpi" >&2
      exit 1
    fi
    ;;
  native-cuda)
    if [[ "${PHASE2_CPU_RANKS}" != "1" ]]; then
      echo "ERROR: native-cuda Phase 2 must run with PHASE2_CPU_RANKS=1" >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: unsupported PHASE2_BACKEND=${PHASE2_BACKEND}" >&2
    exit 1
    ;;
esac

cd "${REPO_ROOT}"
source "${ENV_SCRIPT}"
if [[ -n "${KORALI_ENV_SCRIPT}" ]]; then
  source "${KORALI_ENV_SCRIPT}"
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LANE_ROOT}" "${LOG_ROOT}" "${MANIFEST_ROOT}"
if [[ -e "${LANE_ROOT}/results_phase_1" && ! -L "${LANE_ROOT}/results_phase_1" ]]; then
  echo "ERROR: ${LANE_ROOT}/results_phase_1 exists and is not a symlink" >&2
  exit 1
fi
ln -sfn "${PHASE1_SOURCE}" "${LANE_ROOT}/results_phase_1"

RUN_CMD=("${PYTHON_BIN}" "scripts/platforms/hpc/run_inference_stage.py")
RUN_CMD+=(
  --site karolina
  --experiment indentation
  --model-family full-model
  --profile production
  --stage phase2
  --phase2-backend "${PHASE2_BACKEND}"
  --cpu-ranks "${PHASE2_CPU_RANKS}"
  --config "${CONFIG_PATH}"
  --output-dir "${LANE_ROOT}"
  --python-bin "${PYTHON_BIN}"
)

{
  printf '[mes208] started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '[mes208] replicate_id=%s\n' "${REPLICATE_ID}"
  printf '[mes208] phase2_backend=%s\n' "${PHASE2_BACKEND}"
  printf '[mes208] phase2_cpu_ranks=%s\n' "${PHASE2_CPU_RANKS}"
  printf '[mes208] lane_root=%s\n' "${LANE_ROOT}"
  printf '[mes208] phase1_source=%s\n' "${PHASE1_SOURCE}"
  printf '[mes208] env_script=%s\n' "${ENV_SCRIPT}"
  printf '[mes208] korali_env_script=%s\n' "${KORALI_ENV_SCRIPT:-<default>}"
  "${PYTHON_BIN}" - <<'PYINFO'
import korali
print(f"[mes208] korali_file={korali.__file__}")
PYINFO
  printf '[mes208] slurm_job_id=%s\n' "${SLURM_JOB_ID:-manual}"
  printf '[mes208] node=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf '[mes208] command='
  printf '%q ' "${RUN_CMD[@]}"
  printf '\n'
} | tee "${LOG_ROOT}/environment.txt"

/usr/bin/time -v "${RUN_CMD[@]}" \
  > "${LOG_ROOT}/phase2.stdout.log" \
  2> "${LOG_ROOT}/phase2.stderr.log"

test -s "${LANE_ROOT}/results_phase_2/latest"

REPO_ROOT="${REPO_ROOT}" \
LANE_ROOT="${LANE_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
MANIFEST_ROOT="${MANIFEST_ROOT}" \
CONFIG_PATH="${CONFIG_PATH}" \
PHASE1_SOURCE="${PHASE1_SOURCE}" \
PHASE2_BACKEND="${PHASE2_BACKEND}" \
PHASE2_CPU_RANKS="${PHASE2_CPU_RANKS}" \
REPLICATE_ID="${REPLICATE_ID}" \
PYTHON_BIN="${PYTHON_BIN}" \
ENV_SCRIPT="${ENV_SCRIPT}" \
KORALI_ENV_SCRIPT="${KORALI_ENV_SCRIPT}" \
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from meso_uq.inference.posterior_equivalence import (
    load_phase2_posterior_samples,
    summarize_posterior_samples,
)

repo_root = Path(os.environ["REPO_ROOT"])
lane_root = Path(os.environ["LANE_ROOT"])
log_root = Path(os.environ["LOG_ROOT"])
manifest_root = Path(os.environ["MANIFEST_ROOT"])
phase2 = lane_root / "results_phase_2"
latest = phase2 / "latest"
backend = os.environ["PHASE2_BACKEND"]
replicate_id = os.environ["REPLICATE_ID"]
summary_backend = f"{backend}-{replicate_id}"

samples = load_phase2_posterior_samples(phase2)
summary = summarize_posterior_samples(samples, backend=summary_backend, source=latest)
summary_path = phase2 / "posterior_summary.json"
summary_path.write_text(json.dumps(summary.to_manifest(), indent=2), encoding="utf-8")

state = json.loads(latest.read_text(encoding="utf-8"))
try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    commit = None

manifest = {
    "schema_version": "mes208.phase2_replicate.v1",
    "linear_issue": "MES-208",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "repo_root": str(repo_root),
    "commit": commit,
    "platform": "karolina",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    "replicate_id": replicate_id,
    "phase2_backend": backend,
    "phase2_cpu_ranks": int(os.environ["PHASE2_CPU_RANKS"]),
    "config": os.environ["CONFIG_PATH"],
    "phase1_source": os.environ["PHASE1_SOURCE"],
    "env_script": os.environ["ENV_SCRIPT"],
    "korali_env_script": os.environ.get("KORALI_ENV_SCRIPT") or None,
    "lane_root": str(lane_root),
    "phase2_latest": str(latest),
    "posterior_summary": str(summary_path),
    "korali_run_id": state.get("Run ID"),
    "korali_random_seed": state.get("Random Seed"),
    "korali_current_generation": state.get("Current Generation"),
    "korali_is_finished": state.get("Is Finished"),
    "sample_count": summary.sample_count,
    "finite_logposterior_ratio": summary.finite_logposterior_ratio,
    "max_logposterior": summary.logposterior_max,
    "logs": {
        "environment": str(log_root / "environment.txt"),
        "phase2_stdout": str(log_root / "phase2.stdout.log"),
        "phase2_stderr": str(log_root / "phase2.stderr.log"),
    },
    "status": "completed",
}
manifest_root.mkdir(parents=True, exist_ok=True)
manifest_path = manifest_root / "replicate_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY
