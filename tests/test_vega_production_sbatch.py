"""Validate the structure and key knobs of all production SBATCH templates.

Tests are pure text-file assertions — no shell execution required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SBATCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "platforms"
    / "vega"
    / "sbatch"
    / "production"
)

# ---------------------------------------------------------------------------
# Individual template files
# ---------------------------------------------------------------------------

PHASE_TEMPLATES = [
    "phase1_gpu.sbatch",
    "phase2_cpu.sbatch",
    "phase2_native_cuda.sbatch",
    "phase3b_gpu.sbatch",
    "propagation_phase3b.sbatch",
]

COMPLETE_ORCHESTRATORS = [
    "complete_inference_compression.sbatch",
    "complete_inference_indentation.sbatch",
    "complete_reduced_compression.sbatch",
    "complete_reduced_indentation.sbatch",
]

VALIDATION_TEMPLATES = [
    "validation_phase1_to_3b.sbatch",
    "validation_propagation.sbatch",
]

ALL_TEMPLATES = PHASE_TEMPLATES + COMPLETE_ORCHESTRATORS + VALIDATION_TEMPLATES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(name: str) -> str:
    return (SBATCH_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_sbatch_template_exists(template: str) -> None:
    assert (SBATCH_DIR / template).exists(), f"Missing template: {template}"


# ---------------------------------------------------------------------------
# REPO_ROOT guard: every template must abort if REPO_ROOT is unset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_repo_root_guard_present(template: str) -> None:
    text = _read(template)
    assert "REPO_ROOT" in text, f"{template}: no REPO_ROOT reference"
    # Guard pattern must abort on missing REPO_ROOT (exit 1 or exit 2)
    assert "exit 1" in text or "exit 2" in text, f"{template}: no exit guard for REPO_ROOT"


# ---------------------------------------------------------------------------
# CONFIG_PATH handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", PHASE_TEMPLATES)
def test_phase_templates_check_config_path(template: str) -> None:
    text = _read(template)
    # Individual phase scripts must error out if CONFIG_PATH is empty
    assert "CONFIG_PATH" in text, f"{template}: no CONFIG_PATH reference"
    assert "exit 1" in text, f"{template}: no exit guard for CONFIG_PATH"


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_set_config_path(template: str) -> None:
    text = _read(template)
    # Orchestrators must define CONFIG_PATH (with a default based on REPO_ROOT)
    assert "CONFIG_PATH=" in text, f"{template}: does not set CONFIG_PATH"
    assert "REPO_ROOT" in text, f"{template}: CONFIG_PATH default not derived from REPO_ROOT"


# ---------------------------------------------------------------------------
# --exclude="" guard on child sbatch calls (prevents accidental node exclusion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_use_exclude_empty(template: str) -> None:
    text = _read(template)
    assert (
        '--exclude=""' in text
    ), f'{template}: child sbatch calls must include --exclude="" to prevent stale exclusions'


# ---------------------------------------------------------------------------
# Phase 3b must run as an array job with one GPU per diameter, without node exclusivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_phase3b_does_not_request_exclusive_nodes(template: str) -> None:
    text = _read(template)
    assert "--exclusive" not in text, f"{template}: phase3b should not request exclusive nodes"


# ---------------------------------------------------------------------------
# Propagation memory budget in orchestrators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_propagation_mem_4000(template: str) -> None:
    text = _read(template)
    assert "--mem=4000" in text, f"{template}: propagation step missing --mem=4000"


# ---------------------------------------------------------------------------
# Child script paths in orchestrators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_reference_phase_sbatch_files(template: str) -> None:
    text = _read(template)
    for child in (
        "phase1_gpu.sbatch",
        "phase3b_gpu.sbatch",
        "propagation_phase3b.sbatch",
    ):
        assert child in text, f"{template}: no reference to child script {child}"
    assert "phase2_native_cuda.sbatch" in text
    assert "phase2_cpu.sbatch" in text


# ---------------------------------------------------------------------------
# GPU partition for GPU phases in orchestrators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_gpu_partition_for_phase1_and_phase3b(template: str) -> None:
    text = _read(template)
    assert "route_gpu_partition" in text, f"{template}: missing GPU partition router"
    assert "PHASE1_GPU_PARTITION" in text
    assert "PHASE3B_GPU_PARTITION" in text
    assert "PROP3B_GPU_PARTITION" in text


# ---------------------------------------------------------------------------
# phase1_gpu.sbatch specifics
# ---------------------------------------------------------------------------


def test_phase1_gpu_partition_is_gpu() -> None:
    text = _read("phase1_gpu.sbatch")
    assert "#SBATCH --partition=gpu" in text


def test_phase1_gpu_requests_gpu_resource() -> None:
    text = _read("phase1_gpu.sbatch")
    assert "--gres=gpu:1" in text


def test_phase1_gpu_calls_run_inference_stage_with_phase1() -> None:
    text = _read("phase1_gpu.sbatch")
    assert "run_inference_stage.py" in text
    assert "--stage phase1" in text


def test_phase1_gpu_passes_device_flag() -> None:
    text = _read("phase1_gpu.sbatch")
    assert '--device "${DEVICE}"' in text


# ---------------------------------------------------------------------------
# phase2_cpu.sbatch specifics
# ---------------------------------------------------------------------------


def test_phase2_cpu_partition_is_cpu() -> None:
    text = _read("phase2_cpu.sbatch")
    assert "#SBATCH --partition=cpu" in text


def test_phase2_cpu_requests_64_tasks() -> None:
    text = _read("phase2_cpu.sbatch")
    assert "#SBATCH --ntasks=64" in text
    assert "#SBATCH --mem=256000" in text


def test_phase2_cpu_calls_run_inference_stage_with_phase2() -> None:
    text = _read("phase2_cpu.sbatch")
    assert "run_inference_stage.py" in text
    assert "--stage phase2" in text


def test_phase2_cpu_passes_cpu_ranks() -> None:
    text = _read("phase2_cpu.sbatch")
    assert "--cpu-ranks" in text
    assert "PHASE2_CPU_RANKS" in text
    assert "--phase2-backend" in text


def test_phase2_native_cuda_partition_is_gpu() -> None:
    text = _read("phase2_native_cuda.sbatch")
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --mem=64000" in text
    assert "--gres=gpu:1" in text


def test_phase2_native_cuda_enforces_backend_and_calls_stage2() -> None:
    text = _read("phase2_native_cuda.sbatch")
    assert "PHASE2_BACKEND" in text
    assert "native-cuda" in text
    assert "run_inference_stage.py" in text
    assert "--stage phase2" in text
    assert "--phase2-backend" in text


def test_complete_workflows_default_native_cuda_to_single_phase2_rank() -> None:
    for name in (
        "complete_inference_compression.sbatch",
        "complete_inference_indentation.sbatch",
        "complete_reduced_compression.sbatch",
        "complete_reduced_indentation.sbatch",
    ):
        text = _read(name)
        assert 'PHASE2_BACKEND="${PHASE2_BACKEND:-native-cuda}"' in text
        assert 'PHASE2_CPU_RANKS="1"' in text
        assert 'PHASE2_CPU_RANKS="64"' in text


# ---------------------------------------------------------------------------
# phase3b_gpu.sbatch specifics
# ---------------------------------------------------------------------------


def test_phase3b_gpu_partition_is_gpu() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert "#SBATCH --partition=gpu" in text


def test_phase3b_gpu_requests_exclusive() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert "#SBATCH --exclusive" not in text
    assert "#SBATCH --mem=70000" in text


def test_phase3b_gpu_calls_run_inference_stage_with_phase3b() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert "run_inference_stage.py" in text
    assert "--stage phase3b" in text


def test_phase3b_gpu_passes_device_flag() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert '--device "${DEVICE}"' in text


def test_phase3b_gpu_resolves_dataset_from_array_task() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "list_experiment_datasets.py" in text
    assert "--dataset-name" in text


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_submit_phase3b_as_array_job(template: str) -> None:
    text = _read(template)
    assert "--array=" in text
    assert "PHASE3B_ARRAY_MAX" in text
    assert "list_experiment_datasets.py" in text
    assert "phase3b_%A_%a.out" in text


# ---------------------------------------------------------------------------
# propagation_phase3b.sbatch specifics
# ---------------------------------------------------------------------------


def test_propagation_partition_is_gpu() -> None:
    text = _read("propagation_phase3b.sbatch")
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text


def test_propagation_mem_matches_orchestrator_budget() -> None:
    text = _read("propagation_phase3b.sbatch")
    assert "#SBATCH --mem=4000" in text


def test_propagation_calls_run_propagation_with_phase3b() -> None:
    text = _read("propagation_phase3b.sbatch")
    assert "run_propagation.py" in text
    assert "--stage phase3b" in text


def test_propagation_passes_device_flag() -> None:
    text = _read("propagation_phase3b.sbatch")
    assert "PROPAGATION_DEVICE" in text
    assert "--device" in text


# ---------------------------------------------------------------------------
# set -eo pipefail present in all templates (safety)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_set_eo_pipefail_present(template: str) -> None:
    text = _read(template)
    # Accept both "set -eo pipefail" and "set -euo pipefail"
    assert (
        "set -eo pipefail" in text or "set -euo pipefail" in text
    ), f"{template}: missing 'set -eo pipefail' or 'set -euo pipefail'"
