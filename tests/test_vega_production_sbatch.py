"""Validate the structure and key knobs of all production SBATCH templates.

Tests are pure text-file assertions — no shell execution required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SBATCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vega" / "sbatch" / "production"

# ---------------------------------------------------------------------------
# Individual template files
# ---------------------------------------------------------------------------

PHASE_TEMPLATES = [
    "phase1_gpu.sbatch",
    "phase2_cpu.sbatch",
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
    # Guard pattern: [[ -z "${REPO_ROOT...}" ]] && { ... exit 1; }
    assert "exit 1" in text, f"{template}: no exit-1 guard"


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
# Phase 3b must run with --exclusive in orchestrators (GPU memory isolation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_phase3b_exclusive(template: str) -> None:
    text = _read(template)
    assert "--exclusive" in text, f"{template}: phase3b sbatch call missing --exclusive"


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
        "phase2_cpu.sbatch",
        "phase3b_gpu.sbatch",
        "propagation_phase3b.sbatch",
    ):
        assert child in text, f"{template}: no reference to child script {child}"


# ---------------------------------------------------------------------------
# GPU partition for GPU phases in orchestrators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", COMPLETE_ORCHESTRATORS)
def test_complete_orchestrators_gpu_partition_for_phase1_and_phase3b(template: str) -> None:
    text = _read(template)
    # GPU_PARTITION variable should be used for GPU job submissions
    assert "GPU_PARTITION" in text, f"{template}: GPU_PARTITION not set or referenced"


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


def test_phase2_cpu_calls_run_inference_stage_with_phase2() -> None:
    text = _read("phase2_cpu.sbatch")
    assert "run_inference_stage.py" in text
    assert "--stage phase2" in text


def test_phase2_cpu_passes_cpu_ranks() -> None:
    text = _read("phase2_cpu.sbatch")
    assert "--cpu-ranks" in text
    assert "PHASE2_CPU_RANKS" in text


# ---------------------------------------------------------------------------
# phase3b_gpu.sbatch specifics
# ---------------------------------------------------------------------------


def test_phase3b_gpu_partition_is_gpu() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert "#SBATCH --partition=gpu" in text


def test_phase3b_gpu_requests_exclusive() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert "#SBATCH --exclusive" in text


def test_phase3b_gpu_calls_run_inference_stage_with_phase3b() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert "run_inference_stage.py" in text
    assert "--stage phase3b" in text


def test_phase3b_gpu_passes_device_flag() -> None:
    text = _read("phase3b_gpu.sbatch")
    assert '--device "${DEVICE}"' in text


# ---------------------------------------------------------------------------
# propagation_phase3b.sbatch specifics
# ---------------------------------------------------------------------------


def test_propagation_partition_is_cpu() -> None:
    text = _read("propagation_phase3b.sbatch")
    assert "#SBATCH --partition=cpu" in text


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
    assert "set -eo pipefail" in text, f"{template}: missing 'set -eo pipefail'"
