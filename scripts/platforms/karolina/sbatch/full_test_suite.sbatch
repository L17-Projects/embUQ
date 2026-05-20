#!/usr/bin/env bash
# Submit from the repo root or set REPO_ROOT explicitly before sbatch.
# Optional positional arguments are forwarded to pytest; no arguments means the
# full suite from pyproject.toml.
#SBATCH --job-name=mesouq-full-test-suite
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu_exp
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --gpus=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2
  exit 2
fi
export REPO_ROOT

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MESOUQ_PROJECT_ID="${MESOUQ_PROJECT_ID:-eu-26-17}"
DEFAULT_SCRATCH_ROOT="${MESOUQ_SCRATCH_ROOT:-/scratch/project/${MESOUQ_PROJECT_ID}/eubrieucb/mesouq}"
DEFAULT_RUNS_ROOT="${MESOUQ_RUNS_ROOT:-${DEFAULT_SCRATCH_ROOT}/runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_RUNS_ROOT}/full_test_suite/${RUN_TAG}}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
JOB_TOKEN="${SLURM_JOB_ID:-manual}"
export OUTPUT_ROOT

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out")
exec 2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"

# Keep module/runtime library setup from env_karolina.sh, but do not let
# site override variables leak into tests that assert repo-local defaults.
unset \
  MESOUQ_SITE \
  MESOUQ_RUNS_ROOT \
  MESOUQ_SCRATCH_ROOT \
  MESOUQ_SITE_RUNTIME_ROOT \
  MESOUQ_PROVENANCE_ROOT \
  MESOUQ_GV_ENV_SCRIPT \
  GV_SCALE_SPACE_BINARY \
  GV_CGAL_TOOLS_ROOT

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PYTEST_LOG="${OUTPUT_ROOT}/pytest.log"
SUMMARY_JSON="${OUTPUT_ROOT}/full_test_suite_summary.json"
PYTEST_ARGS=("$@")
command=("${PYTHON_BIN}" -m pytest "${PYTEST_ARGS[@]}")
pytest_env=(
  env
  -u SLURM_JOB_ID
  -u SLURM_ARRAY_JOB_ID
)

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[mesouq-full-test-suite] repo_root=${REPO_ROOT}"
echo "[mesouq-full-test-suite] output_root=${OUTPUT_ROOT}"
echo "[mesouq-full-test-suite] python_bin=${PYTHON_BIN}"
echo "[mesouq-full-test-suite] command=${command[*]}"
echo "[mesouq-full-test-suite] host=$(hostname -f 2>/dev/null || hostname)"
echo "[mesouq-full-test-suite] slurm_job_id=${SLURM_JOB_ID:-manual}"
echo "[mesouq-full-test-suite] slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[mesouq-full-test-suite] slurm_gpus=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-unknown}}"
"${PYTHON_BIN}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

set +e
"${pytest_env[@]}" "${command[@]}" 2>&1 | tee "${PYTEST_LOG}"
pytest_status=${PIPESTATUS[0]}
set -e

result_line="$(
  "${PYTHON_BIN}" - "${PYTEST_LOG}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
patterns = (
    re.compile(r"=+ .* (passed|failed|skipped|error|errors|xfailed|xpassed).* in .* =+"),
    re.compile(r"^(?=.*\b(passed|failed|skipped|error|errors|xfailed|xpassed)\b).*\bin\s+\d+(?:\.\d+)?s$"),
    re.compile(r"=+ (no tests ran|.* interrupted: .*) =+"),
)
lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
for line in reversed(lines):
    if any(pattern.search(line) for pattern in patterns):
        print(line.strip())
        break
else:
    print("pytest summary line not found")
PY
)"

"${PYTHON_BIN}" - "${SUMMARY_JSON}" "${pytest_status}" "${result_line}" "${PYTHON_BIN}" "${PYTEST_ARGS[@]}" <<'PY'
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
pytest_status = int(sys.argv[2])
result_line = sys.argv[3]
python_bin = sys.argv[4]
pytest_args = sys.argv[5:]


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip() or None


summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "site": "karolina",
    "repo_root": os.environ.get("REPO_ROOT"),
    "git": {
        "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "status_short": run_text(["git", "status", "--short"]) or "",
    },
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("SLURM_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS"),
    },
    "runtime": {
        "hostname": socket.getfqdn(),
        "python_bin": python_bin,
        "python_version": run_text([python_bin, "--version"]),
        "mesouq_site_runtime_root": os.environ.get("MESOUQ_SITE_RUNTIME_ROOT"),
        "mesouq_runs_root": os.environ.get("MESOUQ_RUNS_ROOT"),
    },
    "test_environment": {
        "site_overrides_unset": {
            key: key not in os.environ
            for key in (
                "MESOUQ_SITE",
                "MESOUQ_RUNS_ROOT",
                "MESOUQ_SITE_RUNTIME_ROOT",
                "MESOUQ_GV_ENV_SCRIPT",
                "GV_SCALE_SPACE_BINARY",
            )
        },
        "policy": (
            "Karolina modules and runtime libraries are loaded, but MesoUQ "
            "site override variables are unset before pytest so unit tests "
            "exercise repo-local defaults unless a test sets overrides itself. "
            "Slurm job identity variables are hidden from pytest while the "
            "wrapper still records them in this summary."
        ),
        "slurm_job_identity_hidden_from_pytest": [
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
        ],
    },
    "command": [python_bin, "-m", "pytest", *pytest_args],
    "result": {
        "exit_code": pytest_status,
        "summary": result_line,
        "passed": pytest_status == 0,
    },
    "artifacts": {
        "pytest_log": str(summary_path.parent / "pytest.log"),
        "summary_json": str(summary_path),
        "slurm_stdout": str(summary_path.parent / "logs" / f"slurm-{os.environ.get('SLURM_JOB_ID', 'manual')}.out"),
        "slurm_stderr": str(summary_path.parent / "logs" / f"slurm-{os.environ.get('SLURM_JOB_ID', 'manual')}.err"),
    },
    "mpi_gpu_applicability": {
        "suite_scope": "repository pytest suite",
        "mpi": "single Slurm task; this command does not launch mpirun/srun MPI ranks",
        "gpu": "one A100 allocation is requested for Karolina compute-node/runtime parity and CUDA visibility; the pytest suite is not a dedicated GPU workflow proof",
    },
}

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[mesouq-full-test-suite] summary_json={summary_path}")
PY

exit "${pytest_status}"
