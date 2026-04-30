from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from meso_uq.experiments import ExperimentSpec
import meso_uq.inference.gv_hbi as gv_hbi


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "inference" / "scripts" / "run_phase_1.py"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _gv_phase1_config(manifest_path: Path, *, backend: str = "dnn") -> dict[str, object]:
    return {
        "pop_size": 32,
        "max_gen": 1,
        "target_cov": 0.8,
        "covariance_scaling": 0.04,
        "structure": "gv",
        "structures": ["gv"],
        "use_surrogate": True,
        "surrogate": {"backend": backend},
        "experimental_gv_hbi": True,
        "prior_ka": [0.1, 1.1],
        "prior_kb": [0.2, 1.2],
        "prior_mu": [0.3, 1.3],
        "prior_b1": [0.4, 1.4],
        "prior_b2": [0.5, 1.5],
        "prior_a3": [0.6, 1.6],
        "prior_a4": [0.7, 1.7],
        "prior_mu_l": [0.8, 1.8],
        "prior_c": [0.9, 1.9],
        "prior_sigma": [0.01, 0.10],
        "hyperprior_mu_ka": [0.1, 1.1],
        "hyperprior_sigma_ka": [0.01, 0.2],
        "hyperprior_mu_kb": [0.2, 1.2],
        "hyperprior_sigma_kb": [0.01, 0.2],
        "hyperprior_mu_mu": [0.3, 1.3],
        "hyperprior_sigma_mu": [0.01, 0.2],
        "hyperprior_mu_b1": [0.4, 1.4],
        "hyperprior_sigma_b1": [0.01, 0.2],
        "hyperprior_mu_b2": [0.5, 1.5],
        "hyperprior_sigma_b2": [0.01, 0.2],
        "hyperprior_mu_a3": [0.6, 1.6],
        "hyperprior_sigma_a3": [0.01, 0.2],
        "hyperprior_mu_a4": [0.7, 1.7],
        "hyperprior_sigma_a4": [0.01, 0.2],
        "hyperprior_mu_mu_l": [0.8, 1.8],
        "hyperprior_sigma_mu_l": [0.01, 0.2],
        "hyperprior_mu_c": [0.9, 1.9],
        "hyperprior_sigma_c": [0.01, 0.2],
        "experiments": [
            {
                "structure": "gv",
                "name": "torsion",
                "geometries": ["gv_rad2_height14_28"],
                "controls": ["theta_0.03"],
                "surrogate_manifest": str(manifest_path),
            }
        ],
    }


def _gv_experiment(
    *,
    structure: str = "gv",
    name: str = "torsion",
    enabled: bool = True,
    geometries: list[str] | None = None,
    controls: list[str] | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        structure=structure,
        name=name,
        geometries=geometries or ["gv_rad2_height14_28"],
        data_dir=Path("."),
        data_prefix="",
        surrogate_dir=Path("."),
        enabled=enabled,
        prior_d0=[0.0, 0.5],
        prior_sigma=[0.0, 1.0],
        controls=controls or ["theta_0.03"],
    )


def _write_surrogate_manifest(
    tmp_path: Path,
    *,
    payload_updates: dict[str, object] | None = None,
    artifact_updates: dict[str, object] | None = None,
    relative_artifacts: bool = False,
) -> Path:
    model_path = tmp_path / "gv_surrogate_dnn_smoke.pkl"
    model_path.write_text("artifact", encoding="utf-8")
    reference_manifest_path = tmp_path / "gv_reference_manifest.json"
    reference_manifest_path.write_text("{}", encoding="utf-8")
    artifacts: dict[str, object] = {
        "model_path": model_path.name if relative_artifacts else str(model_path),
        "reference_manifest": reference_manifest_path.name if relative_artifacts else str(reference_manifest_path),
    }
    if artifact_updates is not None:
        artifacts.update(artifact_updates)
    payload: dict[str, object] = {
        "structure": "gv",
        "experiment": "torsion",
        "geometry": "gv_rad2_height14_28",
        "controls": {"theta": 0.03},
        "reference_kind": "synthetic",
        "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
        "backend": "dnn",
        "artifacts": artifacts,
    }
    if payload_updates is not None:
        payload.update(payload_updates)
    manifest_path = tmp_path / "gv_dnn_surrogate_smoke_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _build_setup_manifest(config: dict[str, object], tmp_path: Path) -> dict[str, object]:
    return gv_hbi.build_gv_phase1_setup_manifest(
        config,
        [_gv_experiment()],
        repo_root=tmp_path,
        output_root=tmp_path / "out",
    )


def test_gv_phase1_dry_run_writes_setup_manifest_without_loading_korali(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("gv_phase1_setup_test")
    reference_manifest_path = tmp_path / "gv_reference_manifest.json"
    reference_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "geometry_spec": {
                    "id": "gv_rad2_height14_28",
                    "parameters": {"radius": 2.0, "height": 14.28},
                    "label": "GV radius 2.0 height 14.28",
                    "source": "test",
                },
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                "surrogate_backend": "dnn",
                "calibrated_parameter_names": ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"],
                "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
                "points": [0.0, 0.5, 1.0],
                "values": [0.0, 0.1, 0.2],
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "gv_surrogate_dnn_smoke.pkl"
    model_path.write_text("artifact", encoding="utf-8")
    training_report_path = tmp_path / "gv_surrogate_dnn_training_report.json"
    training_report_path.write_text("{}", encoding="utf-8")
    surrogate_manifest_path = tmp_path / "gv_dnn_surrogate_smoke_manifest.json"
    surrogate_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                "backend": "dnn",
                "artifacts": {
                    "reference_manifest": str(reference_manifest_path),
                    "model_path": str(model_path),
                    "training_report": str(training_report_path),
                },
                "runtime_stage": {"known_issues": []},
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "gv_phase1.yaml"
    config_path.write_text(yaml.safe_dump(_gv_phase1_config(surrogate_manifest_path)), encoding="utf-8")

    monkeypatch.setattr(
        module,
        "_load_korali_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("Korali runtime should not be loaded for GV dry-run setup validation")),
    )

    output_dir = tmp_path / "phase1_out"
    module.run_inference(
        dry_run=True,
        config_path=str(config_path),
        output_dir=str(output_dir),
        device="cpu",
    )

    manifest = json.loads(
        (output_dir / "results_phase_1" / "gv_phase1_setup_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "setup_validated"
    assert manifest["workflow"] == "gv_phase1_setup"
    assert manifest["structure"] == "gv"
    assert manifest["experimental"] is True
    assert manifest["enabled_by"] == "config:experimental_gv_hbi"
    assert manifest["chain"] == ["Mirheo", "DNN surrogate", "hierarchical inference"]
    assert manifest["parameter_contract"]["calibrated"] == ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"]
    assert manifest["phase1"]["variable_names"] == ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c", "sigma"]
    assert manifest["phase1"]["noise_model"] == {"kind": "multiplicative", "parameter": "sigma"}
    assert manifest["phase2"]["pooled_variable_names"] == ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"]
    assert manifest["controls"]["names"] == ["theta"]
    assert manifest["controls"]["configured"] == ["theta_0.03"]
    assert manifest["controls"]["policy"] == "GV controls are design inputs and are excluded from calibrated vectors."
    entry = manifest["datasets"][0]
    assert entry["dataset_id"] == "gv:torsion:gv_rad2_height14_28:theta_0.03"
    assert entry["control"] == "theta_0.03"
    assert entry["surrogate"]["reference_manifest"] == str(reference_manifest_path)
    assert entry["surrogate"]["artifact"] == str(model_path.resolve())


def test_gv_phase1_dry_run_rejects_non_dnn_backend(tmp_path: Path) -> None:
    module = _load_module("gv_phase1_setup_backend_error_test")
    surrogate_manifest_path = tmp_path / "gv_dnn_surrogate_smoke_manifest.json"
    surrogate_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "gv_phase1_bnn.yaml"
    config_path.write_text(
        yaml.safe_dump(_gv_phase1_config(surrogate_manifest_path, backend="bnn")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only the DNN surrogate backend"):
        module.run_inference(
            dry_run=True,
            config_path=str(config_path),
            output_dir=str(tmp_path / "phase1_out"),
            device="cpu",
        )


def test_gv_phase1_dry_run_requires_experimental_flag(tmp_path: Path) -> None:
    module = _load_module("gv_phase1_setup_experimental_flag_test")
    surrogate_manifest_path = tmp_path / "gv_dnn_surrogate_smoke_manifest.json"
    surrogate_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                "backend": "dnn",
                "artifacts": {"artifact_path": str(tmp_path / "missing.pkl")},
            }
        ),
        encoding="utf-8",
    )
    config = _gv_phase1_config(surrogate_manifest_path)
    config.pop("experimental_gv_hbi")
    config_path = tmp_path / "gv_phase1_no_flag.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(PermissionError, match="experimental"):
        module.run_inference(
            dry_run=True,
            config_path=str(config_path),
            output_dir=str(tmp_path / "phase1_out"),
            device="cpu",
        )


def test_gv_phase1_dry_run_requires_existing_dnn_artifact(tmp_path: Path) -> None:
    module = _load_module("gv_phase1_setup_missing_artifact_test")
    surrogate_manifest_path = tmp_path / "gv_dnn_surrogate_smoke_manifest.json"
    surrogate_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                "backend": "dnn",
                "artifacts": {"artifact_path": str(tmp_path / "missing.pkl")},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "gv_phase1_missing_artifact.yaml"
    config_path.write_text(yaml.safe_dump(_gv_phase1_config(surrogate_manifest_path)), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing GV surrogate artifact"):
        module.run_inference(
            dry_run=True,
            config_path=str(config_path),
            output_dir=str(tmp_path / "phase1_out"),
            device="cpu",
        )


def test_gv_phase1_dry_run_requires_reference_manifest(tmp_path: Path) -> None:
    module = _load_module("gv_phase1_setup_missing_reference_test")
    model_path = tmp_path / "gv_surrogate_dnn_smoke.pkl"
    model_path.write_text("artifact", encoding="utf-8")
    surrogate_manifest_path = tmp_path / "gv_dnn_surrogate_smoke_manifest.json"
    surrogate_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                "backend": "dnn",
                "artifacts": {"model_path": str(model_path)},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "gv_phase1_missing_reference.yaml"
    config_path.write_text(yaml.safe_dump(_gv_phase1_config(surrogate_manifest_path)), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing GV reference manifest path"):
        module.run_inference(
            dry_run=True,
            config_path=str(config_path),
            output_dir=str(tmp_path / "phase1_out"),
            device="cpu",
        )


def test_gv_setup_manifest_accepts_env_flag_and_top_level_relative_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surrogate_manifest_path = _write_surrogate_manifest(tmp_path, relative_artifacts=True)
    config = _gv_phase1_config(surrogate_manifest_path)
    config.pop("experimental_gv_hbi")
    config["gv_surrogate_manifest"] = str(surrogate_manifest_path)
    for raw_experiment in config["experiments"]:
        raw_experiment.pop("surrogate_manifest")
    monkeypatch.setenv(gv_hbi.GV_HBI_EXPERIMENTAL_FLAG, "yes")

    manifest = _build_setup_manifest(config, tmp_path)

    assert manifest["enabled_by"] == f"env:{gv_hbi.GV_HBI_EXPERIMENTAL_FLAG}"
    assert manifest["datasets"][0]["surrogate"]["artifact"] == str((tmp_path / "gv_surrogate_dnn_smoke.pkl").resolve())
    assert manifest["datasets"][0]["surrogate"]["reference_manifest"] == str(
        (tmp_path / "gv_reference_manifest.json").resolve()
    )


def test_gv_setup_manifest_accepts_config_experimental_mapping_and_dataset_manifest_map(tmp_path: Path) -> None:
    surrogate_manifest_path = _write_surrogate_manifest(tmp_path)
    experiment = _gv_experiment()
    dataset_id = experiment.dataset_name("gv_rad2_height14_28", control="theta_0.03")
    config = _gv_phase1_config(surrogate_manifest_path)
    config.pop("experimental_gv_hbi")
    config["experimental"] = {"gv_hbi": "true"}
    config["gv_surrogate_manifests"] = {dataset_id: str(surrogate_manifest_path)}
    for raw_experiment in config["experiments"]:
        raw_experiment.pop("surrogate_manifest")

    manifest = gv_hbi.build_gv_phase1_setup_manifest(
        config,
        [experiment],
        repo_root=tmp_path,
        output_root=tmp_path / "out",
    )

    assert manifest["enabled_by"] == "config:experimental.gv_hbi"
    assert manifest["datasets"][0]["surrogate"]["manifest"] == str(surrogate_manifest_path.resolve())


def test_gv_inline_surrogate_manifest_map_ignores_unrelated_experiment_entries(tmp_path: Path) -> None:
    experiment = _gv_experiment()
    config = {
        "structure": "gv",
        "experiments": [
            123,
            {"structure": "emb", "name": "torsion", "surrogate_manifest": "wrong.json"},
            {"structure": "gv", "name": "torsion"},
            {"structure": "gv", "name": "torsion", "gv_setup_manifest": tmp_path / "right.json"},
        ],
    }

    inline = gv_hbi._inline_surrogate_manifest_map(config, [experiment])

    assert inline == {"gv:torsion:gv_rad2_height14_28:theta_0.03": str(tmp_path / "right.json")}


def test_gv_setup_manifest_rejects_missing_surrogate_manifest_file(tmp_path: Path) -> None:
    config = _gv_phase1_config(tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError, match="Missing GV surrogate manifest"):
        _build_setup_manifest(config, tmp_path)


def test_gv_setup_manifest_rejects_non_object_surrogate_manifest(tmp_path: Path) -> None:
    surrogate_manifest_path = tmp_path / "gv_dnn_surrogate_smoke_manifest.json"
    surrogate_manifest_path.write_text("[]", encoding="utf-8")
    config = _gv_phase1_config(surrogate_manifest_path)

    with pytest.raises(ValueError, match="must be a JSON object"):
        _build_setup_manifest(config, tmp_path)


def test_gv_setup_manifest_rejects_non_mapping_manifest_map(tmp_path: Path) -> None:
    config = _gv_phase1_config(tmp_path / "unused.json")
    config["gv_surrogate_manifests"] = ["not", "a", "mapping"]
    for raw_experiment in config["experiments"]:
        raw_experiment.pop("surrogate_manifest")

    with pytest.raises(ValueError, match="must be a mapping"):
        _build_setup_manifest(config, tmp_path)


def test_gv_setup_manifest_rejects_missing_manifest_selection_for_multiple_datasets(tmp_path: Path) -> None:
    config = _gv_phase1_config(tmp_path / "unused.json")
    config["experiments"][0]["geometries"] = ["gv_rad2_height14_28", "gv_rad3_height16"]
    for raw_experiment in config["experiments"]:
        raw_experiment.pop("surrogate_manifest")

    with pytest.raises(FileNotFoundError, match="Missing GV surrogate manifest selection"):
        gv_hbi.build_gv_phase1_setup_manifest(
            config,
            [_gv_experiment(geometries=["gv_rad2_height14_28", "gv_rad3_height16"])],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


@pytest.mark.parametrize(
    ("payload_updates", "artifact_updates", "match"),
    (
        ({"structure": "emb"}, None, "structure='gv'"),
        ({"experiment": "stretching"}, None, "experiment mismatch"),
        ({"geometry": "gv_rad3_height16"}, None, "geometry mismatch"),
        ({"dataset_id": "gv:torsion:other:theta_0.03"}, None, "dataset mismatch"),
        ({"backend": "bnn"}, None, "requires a DNN surrogate manifest"),
        ({"artifacts": None}, None, "missing artifacts"),
        (None, {"model_path": None, "reference_manifest": None}, "Missing GV surrogate artifact path"),
        (None, {"reference_manifest": "missing_reference.json"}, "Missing GV reference manifest"),
    ),
)
def test_gv_setup_manifest_rejects_invalid_surrogate_manifest_contract(
    tmp_path: Path,
    payload_updates: dict[str, object] | None,
    artifact_updates: dict[str, object] | None,
    match: str,
) -> None:
    surrogate_manifest_path = _write_surrogate_manifest(
        tmp_path,
        payload_updates=payload_updates,
        artifact_updates=artifact_updates,
    )
    config = _gv_phase1_config(surrogate_manifest_path)

    with pytest.raises((FileNotFoundError, ValueError), match=match):
        _build_setup_manifest(config, tmp_path)


def test_gv_setup_manifest_rejects_mixed_structures(tmp_path: Path) -> None:
    config = _gv_phase1_config(_write_surrogate_manifest(tmp_path))

    with pytest.raises(ValueError, match="requires only GV experiments"):
        gv_hbi.build_gv_phase1_setup_manifest(
            config,
            [_gv_experiment(), _gv_experiment(structure="emb", name="compression")],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_gv_setup_manifest_rejects_empty_enabled_experiment_selection(tmp_path: Path) -> None:
    config = _gv_phase1_config(_write_surrogate_manifest(tmp_path))

    with pytest.raises(ValueError, match="No enabled GV experiments"):
        gv_hbi.build_gv_phase1_setup_manifest(
            config,
            [_gv_experiment(enabled=False)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_gv_setup_manifest_rejects_control_names_in_inference_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _gv_phase1_config(_write_surrogate_manifest(tmp_path))
    monkeypatch.setattr(gv_hbi, "phase1_prior_specs", lambda _config: [("theta", [0.0, 1.0])])
    monkeypatch.setattr(gv_hbi, "phase2_hyperprior_specs", lambda _config: [])
    monkeypatch.setattr(gv_hbi, "active_hierarchical_variable_names", lambda _config: [])

    with pytest.raises(ValueError, match="controls must not appear"):
        _build_setup_manifest(config, tmp_path)


def test_gv_setup_bounds_payload_rejects_unknown_spec_shape() -> None:
    assert gv_hbi._as_bool(1) is True

    with pytest.raises(ValueError, match="Unsupported bounds spec shape"):
        gv_hbi._bounds_payload([("bad",)])
