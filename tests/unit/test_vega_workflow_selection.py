from pathlib import Path

import pytest

import meso_uq.vega_workflows as vega_workflows
from meso_uq.vega_workflows import (
    VegaWorkflowSelection,
    build_inference_command,
    build_propagation_command,
    expand_selection_matrix,
    load_workflow_datasets,
    parse_selection,
    resolve_inference_stage_driver,
    resolve_propagation_driver,
    resolve_workflow_config_path,
    resolve_workflow_output_root,
    selection_key,
    selection_slug,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_workflow_config_paths_resolve_across_model_family_and_profile() -> None:
    repo_root = _repo_root()

    assert (
        resolve_workflow_config_path(
            repo_root,
            VegaWorkflowSelection("compression", "full-model", "production"),
        )
        == repo_root / "inference" / "configs" / "production" / "inference_config_compression.yaml"
    )
    assert (
        resolve_workflow_config_path(
            repo_root,
            VegaWorkflowSelection("indentation", "full-model", "validation"),
        )
        == repo_root / "inference" / "configs" / "validation" / "validation_config_indentation.yaml"
    )
    assert (
        resolve_workflow_config_path(
            repo_root,
            VegaWorkflowSelection("compression", "reduced-model", "production"),
        )
        == repo_root / "reduced" / "configs" / "production" / "reduced_config_compression.yaml"
    )
    assert (
        resolve_workflow_config_path(
            repo_root,
            VegaWorkflowSelection("indentation", "reduced-model", "validation"),
        )
        == repo_root / "reduced" / "configs" / "validation" / "validation_config_indentation.yaml"
    )


def test_workflow_output_root_separates_model_family_from_profile() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "reduced-model", "validation")
    assert resolve_workflow_output_root(
        repo_root,
        selection,
        site="karolina",
        run_tag="20260421_120000",
    ) == (
        repo_root
        / "_runs"
        / "karolina"
        / "runs"
        / "20260421_120000"
        / "compression"
        / "reduced-model"
        / "validation"
    )


def test_selection_helpers_preserve_explicit_axes() -> None:
    parsed = parse_selection("compression:full-model:validation")

    assert parsed == VegaWorkflowSelection("compression", "full-model", "validation")
    assert selection_key(parsed) == "compression:full-model:validation"
    assert selection_slug(parsed) == "compression__full-model__validation"


def test_selection_helpers_reject_legacy_aliases() -> None:
    with pytest.raises(ValueError):
        parse_selection("indentation_reduced")


def test_selection_rejects_unsupported_model_family_and_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported model family"):
        VegaWorkflowSelection("compression", "legacy-model", "production")
    with pytest.raises(ValueError, match="Unsupported profile"):
        VegaWorkflowSelection("compression", "full-model", "legacy")


def test_expand_selection_matrix_builds_cartesian_product() -> None:
    selections = expand_selection_matrix(
        experiments=["compression", "indentation"],
        model_families=["full-model"],
        profiles=["validation", "production"],
    )

    assert selections == [
        VegaWorkflowSelection("compression", "full-model", "validation"),
        VegaWorkflowSelection("compression", "full-model", "production"),
        VegaWorkflowSelection("indentation", "full-model", "validation"),
        VegaWorkflowSelection("indentation", "full-model", "production"),
    ]


def test_stage_driver_resolution_uses_reduced_wrappers_for_all_reduced_stages() -> None:
    repo_root = _repo_root()
    assert resolve_inference_stage_driver(repo_root, "phase1", "reduced-model") == (
        repo_root / "reduced" / "scripts" / "run_phase_1.py"
    )
    assert resolve_inference_stage_driver(repo_root, "phase2", "reduced-model") == (
        repo_root / "reduced" / "scripts" / "run_phase_2.py"
    )
    assert resolve_inference_stage_driver(repo_root, "phase3b", "reduced-model") == (
        repo_root / "reduced" / "scripts" / "run_phase_3b.py"
    )
    assert resolve_inference_stage_driver(repo_root, "phase1", "full-model") == (
        repo_root / "inference" / "scripts" / "run_phase_1.py"
    )
    assert resolve_inference_stage_driver(repo_root, "phase2", "full-model") == (
        repo_root / "inference" / "scripts" / "run_phase_2.py"
    )
    assert resolve_inference_stage_driver(repo_root, "phase3b", "full-model") == (
        repo_root / "inference" / "scripts" / "run_phase_3b.py"
    )


def test_build_inference_command_handles_phase2_mpirun_only() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "validation")
    config_path = resolve_workflow_config_path(repo_root, selection)
    output_root = resolve_workflow_output_root(repo_root, selection)

    command = build_inference_command(
        repo_root,
        selection,
        stage="phase2",
        python_bin="python",
        config_path=config_path,
        output_root=output_root,
        cpu_ranks=4,
    )
    assert command[:6] == [
        "mpirun",
        "--bind-to",
        "none",
        "--oversubscribe",
        "-np",
        "4",
    ]
    assert str(repo_root / "inference" / "scripts" / "run_phase_2.py") in command


def test_build_propagation_command_resolves_public_script() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("indentation", "reduced-model", "production")
    config_path = resolve_workflow_config_path(repo_root, selection)
    output_root = resolve_workflow_output_root(repo_root, selection)
    command = build_propagation_command(repo_root, "phase3b", "python", config_path, output_root)

    assert (
        resolve_propagation_driver(repo_root, "phase3b")
        == repo_root / "propagation" / "scripts" / "run_phase3b_propagation.py"
    )
    assert command[0] == "python"
    assert command[1] == str(repo_root / "propagation" / "scripts" / "run_phase3b_propagation.py")


def test_load_workflow_datasets_reads_selected_experiment_only() -> None:
    repo_root = _repo_root()
    config_path = (
        repo_root / "inference" / "configs" / "validation" / "validation_config_compression.yaml"
    )
    datasets = load_workflow_datasets(repo_root, config_path, "compression")

    assert datasets
    assert all(name.startswith("compression_") for _, name in datasets)


def test_load_workflow_datasets_raises_when_experiment_has_no_enabled_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiments: []\n", encoding="utf-8")

    class _Spec:
        def __init__(self, *, enabled: bool, name: str):
            self.enabled = enabled
            self.name = name
            self.diameters = [2.5]

        def dataset_name(self, diameter: float) -> str:
            return f"{self.name}_{diameter:.1f}um"

    monkeypatch.setattr(
        vega_workflows,
        "load_experiments",
        lambda config, repo_root: [
            _Spec(enabled=False, name="compression"),
            _Spec(enabled=True, name="indentation"),
        ],
    )

    with pytest.raises(ValueError, match="No enabled datasets found for experiment 'compression'"):
        load_workflow_datasets(_repo_root(), config_path, "compression")


def test_build_inference_command_phase2_single_rank_uses_direct_python() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "validation")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase2",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        cpu_ranks=1,
    )

    assert command[0] == "python"
    assert "--device" not in command
    assert command[-2:] == ["--phase2-backend", "cpu-mpi"]


def test_build_inference_command_phase2_production_defaults_to_native_cuda() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase2",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        cpu_ranks=1,
    )

    assert command[0] == "python"
    assert "mpirun" not in command
    assert command[-2:] == ["--phase2-backend", "native-cuda"]


def test_build_inference_command_phase2_production_cpu_mpi_override_uses_mpirun() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase2",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        cpu_ranks=4,
        phase2_backend="cpu-mpi",
    )

    assert command[:6] == ["mpirun", "--bind-to", "none", "--oversubscribe", "-np", "4"]
    assert command[-2:] == ["--phase2-backend", "cpu-mpi"]


def test_build_inference_command_rejects_invalid_phase2_backend() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    with pytest.raises(ValueError, match="Unsupported phase2_backend"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase2",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            phase2_backend="bad-backend",
        )


def test_build_inference_command_rejects_native_cuda_with_multiple_ranks() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    with pytest.raises(ValueError, match="requires cpu_ranks=1"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase2",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            cpu_ranks=2,
            phase2_backend="native-cuda",
        )


def test_build_inference_command_rejects_phase2_backend_outside_phase2() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    with pytest.raises(ValueError, match="phase2_backend is only supported for phase2"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase1",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            phase2_backend="native-cuda",
        )


def test_build_inference_command_phase2_native_cuda_rejects_multi_rank() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    with pytest.raises(ValueError, match="native-cuda backend requires cpu_ranks=1"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase2",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            cpu_ranks=2,
            phase2_backend="native-cuda",
        )


def test_build_inference_command_phase1_cpu_uses_mpi_and_device_flag() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "reduced-model", "production")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase1",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        cpu_ranks=1,
        device="cpu",
        restart=True,
        dry_run=True,
    )

    assert command[:5] == ["mpirun", "--bind-to", "none", "-np", "1"]
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cpu"
    assert "--restart" in command
    assert "--dry_run" in command


def test_build_inference_command_phase1_gpu_skips_mpi() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("indentation", "full-model", "validation")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase1",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        device="gpu",
    )

    assert command[0] == "python"
    assert "mpirun" not in command
    assert command[-2:] == ["--device", "gpu"]


def test_build_inference_command_phase3b_cpu_uses_mpi_and_device_flag() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    command = build_inference_command(
        repo_root,
        selection,
        stage="phase3b",
        python_bin="python",
        config_path=resolve_workflow_config_path(repo_root, selection),
        output_root=resolve_workflow_output_root(repo_root, selection),
        device="cpu",
    )

    assert command[:5] == ["mpirun", "--bind-to", "none", "-np", "1"]
    assert command[-2:] == ["--device", "cpu"]


def test_build_inference_command_phase3b_dataset_filter_is_forwarded() -> None:
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
        dataset_name="compression_2.1um",
    )

    assert "--dataset-name" in command
    assert command[command.index("--dataset-name") + 1] == "compression_2.1um"


def test_build_inference_command_phase3b_repeat_seed_mode_is_forwarded() -> None:
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
        korali_random_seed=3104,
        phase3b_seed_mode="repeat",
    )

    assert command[command.index("--phase3b-seed-mode") + 1] == "repeat"


def test_build_inference_command_rejects_phase3b_seed_mode_outside_phase3b() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    with pytest.raises(ValueError, match="phase3b_seed_mode is only supported for phase3b"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase1",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            phase3b_seed_mode="repeat",
        )


def test_build_inference_command_rejects_invalid_phase3b_seed_mode() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    with pytest.raises(ValueError, match="phase3b_seed_mode must be either"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase3b",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            phase3b_seed_mode="bad-mode",
        )


def test_build_inference_command_rejects_phase3b_conflicting_filters() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "production")
    with pytest.raises(ValueError, match="Use either dataset_name or diameter, not both."):
        build_inference_command(
            repo_root,
            selection,
            stage="phase3b",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            device="gpu",
            dataset_name="compression_2.1um",
            diameter=2.1,
        )


def test_build_propagation_command_phase3b_includes_device_flag() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("indentation", "full-model", "production")
    command = build_propagation_command(
        repo_root,
        "phase3b",
        "python",
        resolve_workflow_config_path(repo_root, selection),
        resolve_workflow_output_root(repo_root, selection),
        device="gpu",
    )

    assert command[-2:] == ["--device", "gpu"]


def test_build_propagation_command_phase1_includes_device_flag() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "validation")
    command = build_propagation_command(
        repo_root,
        "phase1",
        "python",
        resolve_workflow_config_path(repo_root, selection),
        resolve_workflow_output_root(repo_root, selection),
    )

    assert command[-2:] == ["--device", "cpu"]


def test_build_inference_command_rejects_non_phase2_cpu_ranks() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "validation")
    with pytest.raises(ValueError, match="cpu_ranks is only supported for phase2"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase1",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            cpu_ranks=2,
        )


def test_build_inference_command_rejects_restart_outside_phase1() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "validation")
    with pytest.raises(ValueError, match="restart is only supported for phase1"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase2",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            restart=True,
        )


def test_build_inference_command_rejects_dry_run_outside_phase1() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "validation")
    with pytest.raises(ValueError, match="dry_run is only supported for phase1"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase3b",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            dry_run=True,
        )


def test_build_inference_command_rejects_phase2_backend_outside_phase2() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "full-model", "validation")
    with pytest.raises(ValueError, match="phase2_backend is only supported for phase2"):
        build_inference_command(
            repo_root,
            selection,
            stage="phase1",
            python_bin="python",
            config_path=resolve_workflow_config_path(repo_root, selection),
            output_root=resolve_workflow_output_root(repo_root, selection),
            phase2_backend="cpu-mpi",
        )
