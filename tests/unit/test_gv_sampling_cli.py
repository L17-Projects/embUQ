from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.structures.gv.sampling import artifacts

SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_sampling.py"
_GV_STRETCHING_MATERIALS = [
    "ka=1.1",
    "kb=1.2",
    "mu=0.9",
    "b1=0.1",
    "b2=0.2",
    "a3=0.3",
    "a4=0.4",
    "mu_l=0.5",
    "c=0.6",
]


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _material_arguments() -> list[str]:
    return list(_GV_STRETCHING_MATERIALS)


def test_resolve_selected_experiment_supports_default_selection_path() -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")
    args = argparse.Namespace(structure="gv", selection=None, experiment="stretching")

    assert module._resolve_selected_experiment(args) == ("gv", "stretching")


def test_resolve_selected_experiment_reports_invalid_and_conflicting_values() -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")

    with pytest.raises(ValueError, match="must use the form gv:<experiment>"):
        module._resolve_selected_experiment(
            argparse.Namespace(structure="gv", selection="gv/stretching", experiment=None)
        )

    with pytest.raises(ValueError, match="must target structure 'gv'"):
        module._resolve_selected_experiment(
            argparse.Namespace(structure="gv", selection="emb:stretching", experiment=None)
        )

    with pytest.raises(ValueError, match="Conflicting selection values"):
        module._resolve_selected_experiment(
            argparse.Namespace(structure="gv", selection="gv:stretching", experiment="buckling")
        )

    with pytest.raises(ValueError, match="Conflicting structure values"):
        module._resolve_selected_experiment(
            argparse.Namespace(structure="emb", selection="gv:stretching", experiment=None)
        )


def test_parse_material_overrides_handles_invalid_inputs() -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")

    assert module._parse_material_overrides([]) is None

    with pytest.raises(ValueError, match="Invalid material override"):
        module._parse_material_overrides(["bad-entry"])

    with pytest.raises(ValueError, match="Duplicate material override"):
        module._parse_material_overrides(["ka=1.1", "ka=1.2"])


def test_parse_control_value_requires_non_empty_csv_sweep() -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")

    assert module._parse_control_value("1") == 1.0
    with pytest.raises(ValueError, match="at least one value"):
        module._parse_control_value(",,")


def test_parse_controls_merges_duplicate_values_and_validates_input() -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")
    from meso_uq.structures.registry import get_structure

    structure = get_structure("gv")
    controls = module._parse_controls(
        ["tot_force=1", "tot_force=2", "bpress=-90"],
        structure=structure,
        experiment_name="stretching",
    )

    assert controls == {"tot_force": (1.0, 2.0), "bpress": -90.0}

    with pytest.raises(ValueError, match="Invalid control override"):
        module._parse_controls(["bad-control"], structure=structure, experiment_name="stretching")

    with pytest.raises(ValueError, match="Unknown control"):
        module._parse_controls(["oops=10"], structure=structure, experiment_name="stretching")

    with pytest.raises(ValueError, match="At least one control"):
        module._parse_controls([], structure=structure, experiment_name="stretching")


def test_resolve_geometry_and_output_root_handle_all_inputs(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")
    from meso_uq.structures.registry import get_structure

    structure = get_structure("gv")

    geometry = module._resolve_geometry(
        argparse.Namespace(geometry_id="gv_rad2_height14_28", radius=None, height=None),
        structure,
    )
    assert geometry.id == "gv_rad2_height14_28"

    default_geometry = module._resolve_geometry(
        argparse.Namespace(geometry_id=None, radius=None, height=None),
        structure,
    )
    assert default_geometry.id == module.DEFAULT_GV_GEOMETRY.id

    with pytest.raises(ValueError, match="Custom geometry requires both"):
        module._resolve_geometry(
            argparse.Namespace(geometry_id=None, radius=2.5, height=None),
            structure=structure,
        )

    resolved = module._resolve_output_root(argparse.Namespace(output_root=str(tmp_path / "runs")))
    assert resolved == (tmp_path / "runs").resolve()


def test_load_fixture_rejects_non_mapping_payload(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")
    list_path = tmp_path / "fixture.json"
    list_path.write_text("[1, 2, 3]", encoding="utf-8")

    assert module._load_fixture(None) is None
    with pytest.raises(ValueError, match="must contain a JSON object"):
        module._load_fixture(str(list_path))


def test_main_reports_experiment_and_material_validation_errors() -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")

    with pytest.raises(ValueError, match="Either --experiment or --selection must be provided"):
        module.main(["--campaign-id", "campaign-cli", "--control", "tot_force=600", "--control", "bpress=-91"])

    with pytest.raises(ValueError, match="All GV material parameters must be provided"):
        module.main(
            [
                "--experiment",
                "stretching",
                "--campaign-id",
                "campaign-cli",
                "--control",
                "tot_force=600",
                "--control",
                "bpress=-91",
            ]
        )


def test_sampling_cli_invokes_sample_gv_with_expected_args(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")
    fixture = {
        "controls": {"tot_force": 600.0, "bpress": -91.0},
        "channels": {
            "tot_force": [500.0, 600.0],
            "force": [1.0, 2.0],
            "displacement": [0.0, 0.1],
        },
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture, sort_keys=True), encoding="utf-8")

    captured = {}

    def _fake_sample_gv(**kwargs):
        captured.update(kwargs)
        return artifacts.GVSamplingArtifactsResult(
            manifest={"campaign_id": kwargs["campaign_id"], "experiment": kwargs["experiment"], "dataset_id": "fake"},
            channels=kwargs.get("controls", {}).copy() if kwargs.get("channels") is None else dict(kwargs["channels"]),
            manifest_path=tmp_path / "manifest.json",
            hdf5_path=tmp_path / "numerical_dataset.h5",
            campaign_id=kwargs["campaign_id"],
            experiment=kwargs["experiment"],
    )

    module.sample_gv = _fake_sample_gv  # type: ignore[assignment]
    output_root = tmp_path / "sampling-run"
    argv = [
        "--selection",
        "gv:stretching",
        "--campaign-id",
        "campaign-cli",
        "--control",
        "tot_force=600",
        "--control",
        "bpress=-91",
        "--radius",
        "2.5",
        "--height",
        "15.0",
        "--fixture-path",
        str(fixture_path),
        "--output-root",
        str(output_root),
        "--dry-run",
    ]
    for item in _material_arguments():
        argv.extend(["--material", item])

    rc = module.main(argv)

    assert rc == 0
    assert captured["campaign_id"] == "campaign-cli"
    assert captured["experiment"] == "stretching"
    assert captured["geometry"] == {"radGV": 2.5, "height": 15.0}
    assert captured["controls"] == {"tot_force": (600.0,), "bpress": -91.0}
    assert captured["runtime_options"].output_root == output_root.resolve()
    assert captured["dry_run"] is True
    assert captured["write_artifacts"] is False
    assert captured["material_parameters"] == {
        "ka": 1.1,
        "kb": 1.2,
        "mu": 0.9,
        "b1": 0.1,
        "b2": 0.2,
        "a3": 0.3,
        "a4": 0.4,
        "mu_l": 0.5,
        "c": 0.6,
    }


def test_main_prints_fixture_reference_with_valid_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")
    fixture = {
        "controls": {"tot_force": 600.0, "bpress": -91.0},
        "channels": {
            "tot_force": [500.0, 600.0],
            "force": [1.0, 2.0],
            "displacement": [0.0, 0.1],
        },
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture, sort_keys=True), encoding="utf-8")

    def _fake_sample_gv(**kwargs):
        return artifacts.GVSamplingArtifactsResult(
            manifest={
                "campaign_id": kwargs["campaign_id"],
                "experiment": kwargs["experiment"],
                "dataset_id": "gv:stretching:gv_rad2_height14_28:tot_force_600__bpress_-91",
            },
            channels=kwargs.get("controls", {}).copy() if kwargs.get("channels") is None else dict(kwargs["channels"]),
            manifest_path=tmp_path / "manifest.json",
            hdf5_path=tmp_path / "numerical_dataset.h5",
            campaign_id=kwargs["campaign_id"],
            experiment=kwargs["experiment"],
        )

    module.sample_gv = _fake_sample_gv  # type: ignore[assignment]
    output_root = tmp_path / "sampling-run"

    argv = [
        "--selection",
        "gv:stretching",
        "--campaign-id",
        "campaign-cli",
        "--control",
        "tot_force=600",
        "--control",
        "bpress=-91",
        "--fixture-path",
        str(fixture_path),
        "--dry-run",
    ]
    for item in _GV_STRETCHING_MATERIALS:
        argv.extend(["--material", item])

    rc = module.main(argv)
    output = capsys.readouterr().out

    assert rc == 0
    assert f"provenance_fixture={fixture_path}" in output
