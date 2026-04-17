"""Tests for device flag propagation through stage command builders.

Coverage targets:
- build_inference_command: phase3b gpu → no mpi, has --device gpu
- build_inference_command: phase2 with device kwarg ignored
- build_propagation_command: phase3b cpu/gpu device flag
- build_propagation_command: phase1 omits device flag
"""

from pathlib import Path

from meso_uq.vega_workflows import (
    VegaWorkflowSelection,
    build_inference_command,
    build_propagation_command,
    resolve_workflow_config_path,
    resolve_workflow_output_root,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_build_inference_command_phase3b_gpu_skips_mpirun() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase3b",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        device="gpu",
    )

    assert command[0] == "python"
    assert "mpirun" not in command
    assert "--device" in command
    assert command[command.index("--device") + 1] == "gpu"


def test_build_inference_command_phase3b_gpu_reduced_model() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("indentation", "reduced-model", "production")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase3b",
        python_bin="python3",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        device="gpu",
    )

    assert command[0] == "python3"
    assert "mpirun" not in command
    assert "--device" in command
    assert command[command.index("--device") + 1] == "gpu"


def test_build_inference_command_phase2_omits_device_flag() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "validation")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase2",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        cpu_ranks=2,
    )

    assert "--device" not in command


def test_build_propagation_command_phase3b_cpu_device_flag() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    command = build_propagation_command(
        repo_root,
        "phase3b",
        "python",
        resolve_workflow_config_path(repo_root, selection),
        resolve_workflow_output_root(repo_root, selection),
        device="cpu",
    )

    assert "--device" in command
    assert command[command.index("--device") + 1] == "cpu"


def test_build_propagation_command_phase3b_default_device() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "reduced-model", "validation")
    command = build_propagation_command(
        repo_root,
        "phase3b",
        "python",
        resolve_workflow_config_path(repo_root, selection),
        resolve_workflow_output_root(repo_root, selection),
    )

    # Device flag should be present (default device)
    assert "--device" in command


def test_build_propagation_command_phase1_no_device_flag() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    command = build_propagation_command(
        repo_root,
        "phase1",
        "python",
        resolve_workflow_config_path(repo_root, selection),
        resolve_workflow_output_root(repo_root, selection),
    )

    assert "--device" not in command


def test_build_inference_command_phase1_cpu_multi_rank_device_flag() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("indentation", "full-model", "validation")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase1",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        device="cpu",
        cpu_ranks=None,
    )

    assert "mpirun" in command
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cpu"
