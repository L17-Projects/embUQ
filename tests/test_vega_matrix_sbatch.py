from __future__ import annotations

from pathlib import Path

import pytest


SBATCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "platforms" / "vega" / "sbatch"

TEMPLATES = (
    "dnn_rebaseline_matrix.sbatch",
    "bnn_sweep_matrix.sbatch",
    "bnn_roundtrip_check.sbatch",
)


def _read(name: str) -> str:
    return (SBATCH_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("template", TEMPLATES)
def test_matrix_template_exists(template: str) -> None:
    assert (SBATCH_DIR / template).exists(), f"Missing template: {template}"


@pytest.mark.parametrize("template", TEMPLATES)
def test_matrix_templates_use_login_shell_repo_venv_and_repo_pythonpath(template: str) -> None:
    text = _read(template)
    assert text.startswith("#!/bin/bash -l\n")
    assert 'VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"' in text
    assert 'source "${VENV_DIR}/bin/activate"' in text
    assert 'export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in text
    assert 'source "${REPO_ROOT}/_vega/korali/env.sh"' in text


@pytest.mark.parametrize("template", TEMPLATES)
def test_matrix_templates_redirect_logs_into_output_root(template: str) -> None:
    text = _read(template)
    assert "#SBATCH --output=/dev/null" in text
    assert "#SBATCH --error=/dev/null" in text
    assert 'LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"' in text
    assert 'mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"' in text
    assert 'exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out")' in text
    assert '2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)' in text


def test_dnn_rebaseline_template_dispatches_cpu_runner_with_only_seed_and_resume_knobs() -> None:
    text = _read("dnn_rebaseline_matrix.sbatch")
    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "CUDA/12.2.2" not in text
    assert "openmpi" not in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/dnn_rebaseline/${RUN_TAG}}"' in text
    assert 'SELECTIONS="${SELECTIONS:-${ONLY:-}}"' in text
    assert 'SEEDS="${SEEDS:-${SEED:-}}"' in text
    assert 'selection_args+=(--only "${selection}")' in text
    assert 'seed_args+=(--seed "${seed}")' in text
    assert "--resume" in text
    assert "--no-resume" in text
    assert "run_dnn_rebaseline_matrix.py" in text


def test_bnn_sweep_template_dispatches_gpu_runner_without_openmpi() -> None:
    text = _read("bnn_sweep_matrix.sbatch")
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "CUDA/12.2.2" in text
    assert "openmpi" not in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/bnn_sweep/${RUN_TAG}}"' in text
    assert 'SELECTIONS="${SELECTIONS:-${ONLY:-}}"' in text
    assert 'SEEDS="${SEEDS:-${SEED:-}}"' in text
    assert 'DEVICE="${DEVICE:-cuda}"' in text
    assert 'REQUIRE_PARITY="${REQUIRE_PARITY:-false}"' in text
    assert 'selection_args+=(--only "${selection}")' in text
    assert 'seed_args+=(--seed "${seed}")' in text
    assert "--require-parity" in text
    assert "--no-require-parity" in text
    assert "--max-walltime-seconds" in text
    assert "torch.cuda.is_available()" in text
    assert "run_bnn_sweep_matrix.py" in text


def test_bnn_roundtrip_template_dispatches_gpu_runner_with_selection_and_reload_knobs() -> None:
    text = _read("bnn_roundtrip_check.sbatch")
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "CUDA/12.2.2" in text
    assert "openmpi" not in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/bnn_roundtrip/${RUN_TAG}}"' in text
    assert 'SELECTION="${SELECTION:-indentation_3.2um}"' in text
    assert 'command+=(--reload-tol-rel "${RELOAD_TOL_REL}")' in text
    assert 'command+=(--predictive-mc-chunk-size "${PREDICTIVE_MC_CHUNK_SIZE}")' in text
    assert "torch.cuda.is_available()" in text
    assert "run_bnn_roundtrip_check.py" in text
