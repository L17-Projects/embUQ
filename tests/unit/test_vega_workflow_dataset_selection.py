from __future__ import annotations

from pathlib import Path

import pytest

from meso_uq.vega_workflows import (
    VegaWorkflowSelection,
    expand_selection_matrix,
    load_workflow_datasets,
    parse_selection,
    resolve_workflow_config_path,
    resolve_workflow_output_root,
    select_workflow_datasets,
    selection_key,
    selection_slug,
)


def _datasets():
    return [(2.1, "compression_2.1um"), (2.8, "compression_2.8um")]


def test_select_workflow_datasets_returns_all_when_no_filter() -> None:
    selected = select_workflow_datasets(_datasets())
    assert selected == _datasets()


def test_select_workflow_datasets_filters_by_name() -> None:
    selected = select_workflow_datasets(_datasets(), dataset_name="compression_2.8um")
    assert selected == [(2.8, "compression_2.8um")]


def test_select_workflow_datasets_filters_by_diameter() -> None:
    selected = select_workflow_datasets(_datasets(), diameter=2.1)
    assert selected == [(2.1, "compression_2.1um")]


def test_select_workflow_datasets_rejects_conflicting_filters() -> None:
    with pytest.raises(ValueError, match="Use either dataset_name or diameter, not both."):
        select_workflow_datasets(_datasets(), dataset_name="compression_2.1um", diameter=2.1)


def test_legacy_emb_selection_remains_backward_compatible() -> None:
    selection = VegaWorkflowSelection("compression", "full-model", "validation")

    assert selection.structure == "emb"
    assert selection_key(selection) == "compression:full-model:validation"
    assert selection_slug(selection) == "compression__full-model__validation"


def test_structure_aware_emb_selection_includes_structure_identity() -> None:
    selection = VegaWorkflowSelection("compression", "full-model", "validation", structure="emb")

    assert selection_key(selection) == "emb:compression:full-model:validation"
    assert selection_slug(selection) == "emb__compression__full-model__validation"
    assert selection != VegaWorkflowSelection("compression", "full-model", "validation")


def test_parse_selection_accepts_structure_aware_gv_selection() -> None:
    selection = parse_selection("gv:stretching:full-model:validation")

    assert selection.structure == "gv"
    assert selection.experiment == "stretching"


def test_parse_selection_rejects_gv_experiment_without_explicit_structure() -> None:
    with pytest.raises(ValueError, match="requires an explicit structure"):
        parse_selection("stretching:full-model:validation")


def test_selection_rejects_experiment_for_wrong_structure() -> None:
    with pytest.raises(ValueError, match="does not belong to structure"):
        VegaWorkflowSelection("compression", "full-model", "validation", structure="gv")


def test_resolve_workflow_output_root_preserves_legacy_emb_layout(tmp_path: Path) -> None:
    selection = VegaWorkflowSelection("compression", "full-model", "production")

    output = resolve_workflow_output_root(tmp_path, selection, run_tag="run-1", site="vega")

    assert output.parts[-3:] == ("compression", "full-model", "production")


def test_resolve_workflow_output_root_includes_structure_for_new_api(tmp_path: Path) -> None:
    selection = VegaWorkflowSelection("compression", "full-model", "production", structure="emb")

    output = resolve_workflow_output_root(tmp_path, selection, run_tag="run-1", site="vega")

    assert output.parts[-4:] == ("emb", "compression", "full-model", "production")


def test_resolve_workflow_config_path_rejects_gv_runtime_before_dispatch(tmp_path: Path) -> None:
    selection = VegaWorkflowSelection("stretching", "full-model", "validation", structure="gv")

    with pytest.raises(ValueError, match="GV workflow runtime/config resolution is not implemented yet"):
        resolve_workflow_config_path(tmp_path, selection)


def test_load_workflow_datasets_filters_to_emb_structure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiments:",
                "  - name: compression",
                "    structure: gv",
                "    enabled: true",
                "    geometries: [gv_rad2_height14_28]",
                "  - name: compression",
                "    structure: emb",
                "    enabled: true",
                "    geometries: [diameter_2.1um]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected = load_workflow_datasets(tmp_path, config_path, "compression", structure="emb")

    assert selected == [(2.1, "compression_2.1um")]


def test_load_workflow_datasets_rejects_gv_structure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiments: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="GV workflow dataset loading is not implemented yet"):
        load_workflow_datasets(tmp_path, config_path, "stretching", structure="gv")


def test_expand_selection_matrix_supports_structure_and_experiment_axes() -> None:
    selections = expand_selection_matrix(
        experiments=["stretching"],
        model_families=["full-model"],
        profiles=["validation"],
        structures=["gv"],
    )

    assert len(selections) == 1
    assert selections[0] == VegaWorkflowSelection(
        "stretching", "full-model", "validation", structure="gv"
    )
