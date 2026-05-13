from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


EMB_WORKFLOW_ENTRYPOINTS = (
    # Legacy runtime preparation surfaces.
    "src/meso_uq/agents/emb/workflows.py",
    "src/meso_uq/simulation/emb_generation.py",
    "compression/src/generate.py",
    "compression/src/parameters.py",
    "indentation/src/generate.py",
    "indentation/src/parameters.py",
    # DNN/BNN surrogate evaluation.
    "compression/surrogate/evaluate.py",
    "compression/surrogate/evaluate_bnn.py",
    "indentation/surrogate/evaluate.py",
    "indentation/surrogate/evaluate_bnn.py",
    # DNN/BNN training + grouped holdout + multi-arch sweeps.
    "compression/surrogate/scripts/emb_train.py",
    "compression/surrogate/scripts/emb_train_bnn.py",
    "compression/surrogate/scripts/run_group_holdout.py",
    "compression/surrogate/scripts/train_multi_arch.py",
    "indentation/surrogate/scripts/emb_train.py",
    "indentation/surrogate/scripts/emb_train_bnn.py",
    "indentation/surrogate/scripts/run_group_holdout.py",
    "indentation/surrogate/scripts/train_multi_arch.py",
    # Validation / matrix wrappers that reference these surfaces.
    "scripts/platforms/vega/run_validation_matrix.py",
    "scripts/platforms/vega/run_workflow_matrix.py",
    "scripts/platforms/vega/run_dnn_rebaseline_matrix.py",
    "scripts/platforms/vega/run_bnn_sweep_matrix.py",
    "scripts/platforms/vega/run_bnn_roundtrip_check.py",
    "scripts/platforms/vega/run_bnn_certification_matrix.py",
    "scripts/platforms/vega/promote_certified_bnn.py",
    # EMB campaign wrappers that still participate in compatibility planning.
    "scripts/workflows/emb/huq_emb/run_vega_50k_campaign.py",
    "scripts/workflows/emb/huq_emb/run_paper_data_campaign.py",
    "scripts/workflows/emb/huq_emb/run_exact_uqdpd_asset_port.py",
)


WORKFLOW_DOC_EXPECTED_REFERENCES = (
    "compression/surrogate/evaluate.py",
    "compression/surrogate/scripts/emb_train.py",
    "compression/surrogate/scripts/train_multi_arch.py",
    "indentation/surrogate/evaluate.py",
    "indentation/surrogate/scripts/emb_train.py",
    "indentation/surrogate/scripts/train_multi_arch.py",
    "scripts/platforms/vega/run_dnn_surrogate_training.py",
    "scripts/workflows/emb/huq_emb/run_vega_50k_campaign.py",
    "scripts/workflows/emb/huq_emb/run_exact_uqdpd_asset_port.py",
    "scripts/platforms/vega/run_validation_matrix.py",
    "run_workflow_matrix.py",
)


VEGA_MATRIX_DOC_EXPECTED_REFERENCES = (
    "scripts/platforms/vega/run_validation_matrix.py",
    "scripts/platforms/vega/run_workflow_matrix.py",
)


EMB_WORKFLOW_TEST_GUARDS = (
    "tests/test_compression_generate.py",
    "tests/test_compression_parameters_runtime.py",
    "tests/test_indentation_generate_runtime.py",
    "tests/test_indentation_parameters_runtime.py",
    "tests/unit/test_compression_emb_train_script.py",
    "tests/unit/test_compression_emb_train_bnn_script.py",
    "tests/unit/test_indentation_emb_train_script.py",
    "tests/unit/test_indentation_emb_train_bnn_script.py",
    "tests/test_surrogate_evaluators.py",
    "tests/test_surrogate_group_holdout_orchestration.py",
    "tests/test_validation_matrix.py",
    "tests/test_vega_matrix_sbatch.py",
)


def _missing_paths(relative_paths: tuple[str, ...]) -> list[str]:
    return [path for path in relative_paths if not (REPO_ROOT / path).exists()]


def test_emb_workflow_entrypoints_exist() -> None:
    missing = _missing_paths(EMB_WORKFLOW_ENTRYPOINTS)
    assert not missing, f"Missing EMB workflow entrypoints: {missing}"


def test_emb_workflow_docs_reference_public_surfaces() -> None:
    workflows_doc = (REPO_ROOT / "docs/WORKFLOWS.md").read_text(encoding="utf-8")
    for reference in WORKFLOW_DOC_EXPECTED_REFERENCES:
        assert reference in workflows_doc, f"docs/WORKFLOWS.md no longer references {reference}"

    vega_doc = (REPO_ROOT / "docs/VEGA_VALIDATION_MATRIX.md").read_text(encoding="utf-8")
    for reference in VEGA_MATRIX_DOC_EXPECTED_REFERENCES:
        assert reference in vega_doc, (
            f"docs/VEGA_VALIDATION_MATRIX.md no longer references {reference}"
        )


def test_emb_workflow_regression_test_guards_exist() -> None:
    missing = _missing_paths(EMB_WORKFLOW_TEST_GUARDS)
    assert not missing, f"Missing EMB workflow regression tests: {missing}"
