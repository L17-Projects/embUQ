from pathlib import Path

import pytest

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

    assert resolve_workflow_config_path(
        repo_root,
        VegaWorkflowSelection("compression", "full-model", "production"),
    ) == repo_root / "inference" / "configs" / "production" / "inference_config_compression.yaml"
    assert resolve_workflow_config_path(
        repo_root,
        VegaWorkflowSelection("indentation", "full-model", "validation"),
    ) == repo_root / "inference" / "configs" / "validation" / "validation_config_indentation.yaml"
    assert resolve_workflow_config_path(
        repo_root,
        VegaWorkflowSelection("compression", "reduced-model", "production"),
    ) == repo_root / "reduced" / "configs" / "production" / "reduced_config_compression.yaml"
    assert resolve_workflow_config_path(
        repo_root,
        VegaWorkflowSelection("indentation", "reduced-model", "validation"),
    ) == repo_root / "reduced" / "configs" / "validation" / "validation_config_indentation.yaml"


def test_workflow_output_root_separates_model_family_from_profile() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("compression", "reduced-model", "validation")
    assert resolve_workflow_output_root(repo_root, selection) == (
        repo_root / "_vega" / "runs" / "compression" / "reduced-model" / "validation"
    )


def test_selection_helpers_preserve_explicit_axes() -> None:
    parsed = parse_selection("compression:full-model:validation")

    assert parsed == VegaWorkflowSelection("compression", "full-model", "validation")
    assert selection_key(parsed) == "compression:full-model:validation"
    assert selection_slug(parsed) == "compression__full-model__validation"


def test_selection_helpers_reject_legacy_aliases() -> None:
    with pytest.raises(ValueError):
        parse_selection("indentation_reduced")


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
    assert command[:6] == ["mpirun", "--bind-to", "none", "--oversubscribe", "-np", "4"]
    assert str(repo_root / "inference" / "scripts" / "run_phase_2.py") in command


def test_build_propagation_command_resolves_public_script() -> None:
    repo_root = _repo_root()
    selection = VegaWorkflowSelection("indentation", "reduced-model", "production")
    config_path = resolve_workflow_config_path(repo_root, selection)
    output_root = resolve_workflow_output_root(repo_root, selection)
    command = build_propagation_command(repo_root, "phase3b", "python", config_path, output_root)

    assert resolve_propagation_driver(repo_root, "phase3b") == repo_root / "propagation" / "scripts" / "run_phase3b_propagation.py"
    assert command[0] == "python"
    assert command[1] == str(repo_root / "propagation" / "scripts" / "run_phase3b_propagation.py")


def test_load_workflow_datasets_reads_selected_experiment_only() -> None:
    repo_root = _repo_root()
    config_path = repo_root / "inference" / "configs" / "validation" / "validation_config_compression.yaml"
    datasets = load_workflow_datasets(repo_root, config_path, "compression")

    assert datasets
    assert all(name.startswith("compression_") for _, name in datasets)
