from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_operational_canary.py"
PHASE1_SCRIPT_PATH = REPO_ROOT / "inference" / "scripts" / "run_phase_1.py"
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.inference import gv_hbi


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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

    def Barrier(self) -> None:
        return None


def _install_fake_phase1_runtime(module, monkeypatch: pytest.MonkeyPatch) -> _FakeKorali:
    fake_korali = _FakeKorali()
    fake_mpi = types.SimpleNamespace(COMM_WORLD=_FakeComm())
    monkeypatch.setattr(module, "_load_korali_runtime", lambda: (fake_korali, fake_mpi))
    monkeypatch.setattr(module, "configure_device_conduit", lambda *args, **kwargs: None)
    monkeypatch.setattr(gv_hbi, "_load_gv_dnn_model_state", lambda _path: object())
    monkeypatch.setattr(gv_hbi, "_predict_gv_dnn", lambda _state, x_raw: np.ones(x_raw.shape[0]))
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
    return fake_korali


def test_gv_operational_canary_requires_runtime_flag(tmp_path: Path) -> None:
    module = _load_module(SCRIPT_PATH, "gv_operational_canary_flag_test")

    with pytest.raises(SystemExit, match="2"):
        module.main(["--output-root", str(tmp_path / "gv_operational_canary")])


def test_gv_operational_canary_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(SCRIPT_PATH, "gv_operational_canary_helper_test")

    non_object_json = tmp_path / "list.json"
    non_object_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        module._read_json(non_object_json)

    assert module._as_bool(True) is True
    assert module._as_bool(None) is False
    assert module._as_bool(1) is True
    assert module._control_cli_items(["theta=0.03", "flow=1.0"]) == [
        "--control",
        "theta=0.03",
        "--control",
        "flow=1.0",
    ]
    assert module._resolve_selection("gv:torsion") == ("gv", "torsion")
    with pytest.raises(ValueError, match="form gv:<experiment>"):
        module._resolve_selection("torsion")
    with pytest.raises(ValueError, match="only supports structure 'gv'"):
        module._resolve_selection("emb:compression")

    with pytest.raises(ValueError, match="missing artifacts"):
        module._require_phase1_compatible_surrogate_artifact(surrogate_manifest={})
    with pytest.raises(FileNotFoundError, match="missing a surrogate artifact path"):
        module._require_phase1_compatible_surrogate_artifact(surrogate_manifest={"artifacts": {}})

    assert module._build_verdict(
        surrogate_report={"status": "failed"},
        phase1_execution_manifest={"status": "phase1_korali_completed"},
    ) == "fail"
    assert module._build_verdict(
        surrogate_report={"status": "passed"},
        phase1_execution_manifest={"status": "setup_validated"},
    ) == "fail"

    monkeypatch.setenv(module.GV_HBI_EXPERIMENTAL_FLAG, "yes")
    assert module._require_runtime_flag() == f"env:{module.GV_HBI_EXPERIMENTAL_FLAG}"


def test_gv_operational_canary_stitches_runtime_surrogate_and_phase1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(SCRIPT_PATH, "gv_operational_canary_test")
    phase1_module = _load_module(PHASE1_SCRIPT_PATH, "gv_operational_canary_phase1_runtime")
    fake_korali = _install_fake_phase1_runtime(phase1_module, monkeypatch)
    original_loader = module._load_module

    def patched_loader(module_name: str, path: Path):
        if path == module.PHASE1_SCRIPT:
            return phase1_module
        return original_loader(module_name, path)

    monkeypatch.setattr(module, "_load_module", patched_loader)
    monkeypatch.setenv(gv_hbi.GV_HBI_EXPERIMENTAL_FLAG, "1")
    output_root = tmp_path / "gv_operational_canary"

    rc = module.main(
        [
            "--output-root",
            str(output_root),
            "--selection",
            "gv:torsion",
            "--geometry-id",
            "gv_rad2_height14_28",
            "--control",
            "theta=0.03",
            "--include-experimental",
            "--max-epoch",
            "4",
            "--width",
            "6",
            "--depth",
            "2",
            "--batch-size",
            "4",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_root / "gv_operational_canary_manifest.json").read_text(encoding="utf-8"))
    runtime_manifest = json.loads((output_root / "runtime" / "gv_runtime_dry_run_manifest.json").read_text(encoding="utf-8"))
    surrogate_manifest = json.loads((output_root / "surrogate" / "gv_dnn_surrogate_smoke_manifest.json").read_text(encoding="utf-8"))
    phase1_setup_manifest = json.loads(
        (output_root / "phase1" / "results_phase_1" / gv_hbi.GV_PHASE1_SETUP_MANIFEST).read_text(encoding="utf-8")
    )
    phase1_execution_manifest = json.loads(
        (output_root / "phase1" / "results_phase_1" / gv_hbi.GV_PHASE1_EXECUTION_MANIFEST).read_text(encoding="utf-8")
    )

    assert manifest["workflow"] == "gv_operational_canary"
    assert manifest["enabled_by"] == f"env:{gv_hbi.GV_HBI_EXPERIMENTAL_FLAG}"
    assert manifest["structure"] == "gv"
    assert manifest["selection"] == "gv:torsion"
    assert manifest["experiment"] == "torsion"
    assert manifest["geometry"] == runtime_manifest["geometry"]
    assert manifest["controls"] == {"theta": 0.03}
    assert manifest["control_id"] == runtime_manifest["control_id"]
    assert manifest["dataset_id"] == runtime_manifest["dataset_id"]
    assert manifest["reference_kind"] == "synthetic"
    assert manifest["surrogate_backend"] == "dnn"
    assert manifest["noise_model"] == {"kind": "multiplicative", "parameter": "sigma"}
    assert manifest["calibrated_parameters"] == ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"]
    assert manifest["nuisance_parameters"] == []
    assert manifest["noise_parameters"] == ["sigma"]
    assert "theta" not in set(manifest["calibrated_parameters"]) | set(manifest["nuisance_parameters"])
    assert manifest["checks"]["runtime_flag_enabled"] is True
    assert manifest["checks"]["controls_excluded_from_calibrated_and_nuisance"] is True
    assert manifest["checks"]["phase1_status"] == "phase1_korali_completed"
    assert manifest["checks"]["phase1_execution_model"] == "single_lane_dnn_korali"
    assert manifest["verdict"] == "pass"

    assert surrogate_manifest["dataset_id"] == manifest["dataset_id"]
    assert phase1_setup_manifest["enabled_by"] == f"env:{gv_hbi.GV_HBI_EXPERIMENTAL_FLAG}"
    assert phase1_setup_manifest["datasets"][0]["dataset_id"] == manifest["dataset_id"]
    assert phase1_execution_manifest["dataset"]["dataset_id"] == manifest["dataset_id"]
    assert phase1_execution_manifest["controls"]["fixed_outside_inferred_variables"] is True
    assert phase1_execution_manifest["runtime"]["korali_invoked"] is True
    assert fake_korali.created_engines[-1].sample_data["Reference Evaluations"] == [1.0] * 4
    assert Path(manifest["artifacts"]["phase1_config"]).is_file()
    assert Path(manifest["artifacts"]["reference_manifest"]).is_file()
