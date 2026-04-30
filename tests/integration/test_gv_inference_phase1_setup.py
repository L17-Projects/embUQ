from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "inference" / "scripts" / "run_phase_1.py"
SMOKE_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dnn_smoke.py"
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.inference import gv_hbi


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_module(name: str):
    return _load_script(SCRIPT_PATH, name)


class _AutoDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = type(self)()
        return dict.__getitem__(self, key)


class _FakeExperiment(_AutoDict):
    pass


class _FakeEngine(_AutoDict):
    def __init__(self) -> None:
        super().__init__()
        self.mpi_comm = None
        self.run_argument = None
        self.sample_data = None

    def setMPIComm(self, comm) -> None:
        self.mpi_comm = comm

    def run(self, experiments) -> None:
        self.run_argument = experiments
        experiment = experiments[0]
        sample = {"Parameters": [0.5] * len(experiment["Variables"])}
        experiment["Problem"]["Computational Model"](sample)
        self.sample_data = sample
        state_path = Path(experiment["File Output"]["Path"])
        state_path.mkdir(parents=True, exist_ok=True)
        (state_path / "latest").write_text("{}", encoding="utf-8")


class _FakeKorali(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.created_experiments: list[_FakeExperiment] = []
        self.created_engines: list[_FakeEngine] = []

    def Experiment(self) -> _FakeExperiment:
        experiment = _FakeExperiment()
        self.created_experiments.append(experiment)
        return experiment

    def Engine(self) -> _FakeEngine:
        engine = _FakeEngine()
        self.created_engines.append(engine)
        return engine


class _FakeComm:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1


def _install_fake_gv_korali(
    module,
    monkeypatch: pytest.MonkeyPatch,
    *,
    patch_model: bool = True,
) -> _FakeKorali:
    fake_korali = _FakeKorali()
    fake_mpi = types.SimpleNamespace(COMM_WORLD=_FakeComm())
    monkeypatch.setattr(module, "_load_korali_runtime", lambda: (fake_korali, fake_mpi))
    monkeypatch.setattr(module, "configure_device_conduit", lambda *args, **kwargs: None)
    if patch_model:
        monkeypatch.setattr(gv_hbi, "_load_gv_dnn_model_state", lambda _path: object())
        monkeypatch.setattr(gv_hbi, "_predict_gv_dnn", lambda _state, x_raw: np.ones(x_raw.shape[0]))
    return fake_korali


def _gv_phase1_config(
    manifest_path: Path,
    *,
    backend: str = "dnn",
    control: str = "theta_0.03",
) -> dict[str, object]:
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
                "controls": [control],
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
    assert "theta" not in manifest["phase1"]["variable_names"]
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


def test_gv_phase1_execution_requires_experimental_flag(tmp_path: Path) -> None:
    module = _load_module("gv_phase1_execution_experimental_flag_test")
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
    config_path = tmp_path / "gv_phase1_no_flag_execution.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(PermissionError, match="experimental"):
        module.run_inference(
            dry_run=False,
            config_path=str(config_path),
            output_dir=str(tmp_path / "phase1_out"),
            device="cpu",
        )


def test_gv_phase1_restart_remains_blocked(tmp_path: Path) -> None:
    module = _load_module("gv_phase1_restart_blocked_test")
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
    config_path = tmp_path / "gv_phase1_restart.yaml"
    config_path.write_text(yaml.safe_dump(_gv_phase1_config(surrogate_manifest_path)), encoding="utf-8")

    with pytest.raises(NotImplementedError, match="restart is not implemented"):
        module.run_inference(
            restart=True,
            dry_run=True,
            config_path=str(config_path),
            output_dir=str(tmp_path / "phase1_out"),
            device="cpu",
        )


def test_gv_phase1_mixed_emb_gv_configuration_remains_blocked(tmp_path: Path) -> None:
    module = _load_module("gv_phase1_mixed_blocked_test")
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
    config["experiments"].append(
        {
            "structure": "emb",
            "name": "compression",
            "diameters": [2.1],
        }
    )
    config_path = tmp_path / "gv_phase1_mixed.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Mixed EMB/GV"):
        module.run_inference(
            dry_run=True,
            config_path=str(config_path),
            output_dir=str(tmp_path / "phase1_out"),
            device="cpu",
        )


def test_gv_phase1_paths_do_not_import_emb_runtime_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("gv_phase1_lazy_emb_import_test")
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in {"compression", "indentation"}:
            raise AssertionError(f"Unexpected EMB import on GV path: {name}")
        return original_import(name, globals, locals, fromlist, level)

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
                "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
            }
        ),
        encoding="utf-8",
    )
    dataset_csv_path = tmp_path / "gv_surrogate_smoke_dataset.csv"
    dataset_csv_path.write_text(
        "ka,kb,mu,b1,b2,a3,a4,mu_l,c,radius,height,theta,torsion_coord,torsion_response,source_curve_id\n"
        "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2.0,14.28,0.03,0.0,0.0,curve_0\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "gv_surrogate_dnn_smoke.pkl"
    model_path.write_text("artifact", encoding="utf-8")
    training_report_path = tmp_path / "gv_surrogate_dnn_training_report.json"
    training_report_path.write_text(json.dumps({"status": "passed", "backend": "dnn"}), encoding="utf-8")
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
                    "dataset_csv": str(dataset_csv_path),
                    "training_report": str(training_report_path),
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "gv_phase1_lazy_import.yaml"
    config_path.write_text(yaml.safe_dump(_gv_phase1_config(surrogate_manifest_path)), encoding="utf-8")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _install_fake_gv_korali(module, monkeypatch)
    monkeypatch.setattr(
        gv_hbi,
        "_build_execution_artifact_probe",
        lambda artifact_path: {
            "status": "loaded",
            "artifact_path": str(Path(artifact_path).resolve()),
            "model_class": "MLP",
            "input_dim": 13,
            "output_dim": 1,
        },
    )

    module.run_inference(
        dry_run=True,
        config_path=str(config_path),
        output_dir=str(tmp_path / "phase1_setup_out"),
        device="cpu",
    )
    module.run_inference(
        dry_run=False,
        config_path=str(config_path),
        output_dir=str(tmp_path / "phase1_execution_out"),
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


def test_gv_phase1_execution_runs_single_lane_dnn_korali_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("gv_phase1_execution_test")
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
                "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
            }
        ),
        encoding="utf-8",
    )
    dataset_csv_path = tmp_path / "gv_surrogate_smoke_dataset.csv"
    dataset_csv_path.write_text(
        "ka,kb,mu,b1,b2,a3,a4,mu_l,c,radius,height,theta,torsion_coord,torsion_response,source_curve_id\n"
        "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2.0,14.28,0.03,0.0,0.0,curve_0\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "gv_surrogate_dnn_smoke.pkl"
    model_path.write_text("artifact", encoding="utf-8")
    training_report_path = tmp_path / "gv_surrogate_dnn_training_report.json"
    training_report_path.write_text(json.dumps({"status": "passed", "backend": "dnn"}), encoding="utf-8")
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
                    "dataset_csv": str(dataset_csv_path),
                    "training_report": str(training_report_path),
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "gv_phase1_execution.yaml"
    config_path.write_text(yaml.safe_dump(_gv_phase1_config(surrogate_manifest_path)), encoding="utf-8")

    fake_korali = _install_fake_gv_korali(module, monkeypatch)
    monkeypatch.setattr(
        gv_hbi,
        "_build_execution_artifact_probe",
        lambda artifact_path: {
            "status": "loaded",
            "artifact_path": str(Path(artifact_path).resolve()),
            "model_class": "MLP",
            "input_dim": 13,
            "output_dim": 1,
        },
    )

    output_dir = tmp_path / "phase1_execution_out"
    module.run_inference(
        dry_run=False,
        config_path=str(config_path),
        output_dir=str(output_dir),
        device="cpu",
    )

    setup_manifest = json.loads(
        (output_dir / "results_phase_1" / gv_hbi.GV_PHASE1_SETUP_MANIFEST).read_text(encoding="utf-8")
    )
    execution_manifest = json.loads(
        (output_dir / "results_phase_1" / gv_hbi.GV_PHASE1_EXECUTION_MANIFEST).read_text(encoding="utf-8")
    )
    assert setup_manifest["status"] == "setup_validated"
    assert execution_manifest["workflow"] == "gv_phase1_execution"
    assert execution_manifest["status"] == "phase1_korali_completed"
    assert execution_manifest["execution_model"] == "single_lane_dnn_korali"
    assert execution_manifest["dataset"]["dataset_id"] == "gv:torsion:gv_rad2_height14_28:theta_0.03"
    assert execution_manifest["dataset"]["control_values"] == {"theta": 0.03}
    assert execution_manifest["controls"]["fixed_outside_inferred_variables"] is True
    assert execution_manifest["phase1"]["variable_names"] == ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c", "sigma"]
    assert execution_manifest["phase1"]["noise_model"] == {"kind": "multiplicative", "parameter": "sigma"}
    assert execution_manifest["provenance"]["noise_model"] == {"kind": "multiplicative", "parameter": "sigma"}
    assert execution_manifest["provenance"]["dataset_csv"] == str(dataset_csv_path.resolve())
    assert execution_manifest["provenance"]["training_report"] == str(training_report_path.resolve())
    assert execution_manifest["artifact_probe"]["status"] == "loaded"
    assert execution_manifest["runtime"]["korali_invoked"] is True
    assert execution_manifest["runtime"]["phase1_completed"] is True
    assert execution_manifest["runtime"]["full_hbi_completed"] is False
    assert execution_manifest["runtime"]["latest_state_exists"] is True
    assert execution_manifest["reference"] == {
        "axis_column": "torsion_coord",
        "target_column": "torsion_response",
        "num_points": 1,
    }
    experiment = fake_korali.created_experiments[0]
    assert experiment["Problem"]["Type"] == "Bayesian/Reference"
    assert experiment["Problem"]["Reference Data"] == [0.0]
    assert fake_korali.created_engines[-1].sample_data["Reference Evaluations"] == [1.0]


def test_gv_phase1_execution_consumes_smoke_dnn_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    smoke_module = _load_script(SMOKE_SCRIPT_PATH, "gv_phase1_real_smoke_dnn_source")
    phase1_module = _load_module("gv_phase1_real_smoke_dnn_execution")
    smoke_root = tmp_path / "smoke"
    assert (
        smoke_module.main(
            [
                "--output-root",
                str(smoke_root),
                "--experiment",
                "torsion",
                "--control",
                "theta=0.03",
                "--seed",
                "5",
                "--num-curves",
                "3",
                "--points-per-curve",
                "3",
                "--max-epoch",
                "2",
                "--width",
                "5",
                "--depth",
                "2",
                "--batch-size",
                "4",
            ]
        )
        == 0
    )
    surrogate_manifest = smoke_root / "gv_dnn_surrogate_smoke_manifest.json"
    config_path = tmp_path / "gv_phase1_real_smoke.yaml"
    config_path.write_text(yaml.safe_dump(_gv_phase1_config(surrogate_manifest, control="theta_0_03")), encoding="utf-8")
    fake_korali = _install_fake_gv_korali(phase1_module, monkeypatch, patch_model=False)

    output_dir = tmp_path / "phase1_real_smoke_out"
    phase1_module.run_inference(
        dry_run=False,
        config_path=str(config_path),
        output_dir=str(output_dir),
        device="cpu",
    )

    manifest = json.loads(
        (output_dir / "results_phase_1" / gv_hbi.GV_PHASE1_EXECUTION_MANIFEST).read_text(encoding="utf-8")
    )
    experiment = fake_korali.created_experiments[0]
    reference_data = experiment["Problem"]["Reference Data"]
    reference_evaluations = fake_korali.created_engines[-1].sample_data["Reference Evaluations"]
    assert manifest["status"] == "phase1_korali_completed"
    assert manifest["runtime"]["korali_invoked"] is True
    assert manifest["provenance"]["surrogate_artifact"].endswith("gv_surrogate_dnn_smoke.pkl")
    assert len(reference_evaluations) == len(reference_data) == 3
