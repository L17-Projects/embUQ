from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


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
