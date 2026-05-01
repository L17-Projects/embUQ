from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml


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
    parsed_experiment_only = module.build_parser().parse_args(
        ["--output-root", str(tmp_path / "experiment_only"), "--experiment", "stretching"]
    )
    assert parsed_experiment_only.selection is None
    assert module._resolve_runtime_selection(
        structure=parsed_experiment_only.structure,
        experiment=parsed_experiment_only.experiment,
        selection=parsed_experiment_only.selection,
    ) == ("gv", "stretching")
    assert module._resolve_selection("gv:torsion") == ("gv", "torsion")
    with pytest.raises(ValueError, match="form gv:<experiment>"):
        module._resolve_selection("torsion")
    with pytest.raises(ValueError, match="only supports structure 'gv'"):
        module._resolve_selection("emb:compression")
    with pytest.raises(ValueError, match="Conflicting runtime selections"):
        module._resolve_runtime_selection(structure="gv", experiment="stretching", selection="gv:torsion")
    with pytest.raises(ValueError, match="Conflicting runtime structure"):
        module._resolve_runtime_selection(structure="emb", experiment="torsion", selection="gv:torsion")
    with pytest.raises(ValueError, match="Either --selection or --experiment"):
        module._resolve_runtime_selection(structure="gv", experiment=None, selection=None)
    with pytest.raises(ValueError, match="only supports structure 'gv'"):
        module._resolve_runtime_selection(structure="emb", experiment="compression", selection=None)
    assert module._resolve_runtime_selection(structure=None, experiment="torsion", selection=None) == ("gv", "torsion")

    with pytest.raises(ValueError, match="missing artifacts"):
        module._require_phase1_compatible_surrogate_artifact(
            surrogate_manifest={},
            surrogate_root=tmp_path,
        )
    with pytest.raises(FileNotFoundError, match="missing a surrogate artifact path"):
        module._require_phase1_compatible_surrogate_artifact(
            surrogate_manifest={"artifacts": {}},
            surrogate_root=tmp_path,
            surrogate_status="passed",
        )
    existing_artifact_manifest = {"artifacts": {"artifact_path": str(tmp_path / "model.pt")}}
    assert (
        module._require_phase1_compatible_surrogate_artifact(
            surrogate_manifest=existing_artifact_manifest,
            surrogate_root=tmp_path,
            surrogate_status="passed",
        )
        is existing_artifact_manifest
    )
    dry_run_manifest = {"artifacts": {}}
    returned_manifest = module._require_phase1_compatible_surrogate_artifact(
        surrogate_manifest=dry_run_manifest,
        surrogate_root=tmp_path,
        surrogate_status="dry-run",
    )
    placeholder_path = Path(returned_manifest["artifacts"]["artifact_path"])
    assert placeholder_path.is_file()
    assert returned_manifest["artifacts"]["model_path"] == str(placeholder_path)

    assert module._build_verdict(
        surrogate_report={"status": "failed"},
        phase1_execution_manifest={"status": "phase1_korali_completed"},
    ) == "fail"
    assert module._build_verdict(
        surrogate_report={"status": "passed"},
        phase1_execution_manifest={"status": "setup_validated"},
    ) == "fail"
    assert module._summarize_runtime_known_issues({"known_issues": "not-a-list"}) == []
    assert module._summarize_runtime_known_issues(
        {
            "known_issues": [
                {
                    "id": "a",
                    "summary": "Blocking runtime issue",
                    "evidence": "runtime",
                    "severity": "critical",
                },
                {
                    "id": "b",
                    "summary": "Notice",
                    "evidence": "runtime",
                    "severity": "notice",
                },
                "not-a-dict",
            ]
        }
    ) == [
        {
            "id": "a",
            "summary": "Blocking runtime issue",
            "evidence": "runtime",
            "severity": "critical",
            "classification": "experimental_blocked",
        },
        {
            "id": "b",
            "summary": "Notice",
            "evidence": "runtime",
            "severity": "notice",
            "classification": "observed",
        },
    ]
    assert module._validate_output_root(tmp_path / "safe-canary-output") == (tmp_path / "safe-canary-output").resolve()
    with pytest.raises(ValueError, match="must not be inside repository source trees"):
        module._validate_output_root(module.GV_SIMULATION_ROOT / "runtime")
    with pytest.raises(ValueError, match="must not be inside repository source trees"):
        module._validate_output_root(module.REPO_ROOT / "gv" / "scratch")
    with pytest.raises(ValueError, match="must not be inside repository source trees"):
        module._validate_output_root(module.REPO_ROOT / "src" / "meso_uq")
    with pytest.raises(ValueError, match="must not be inside repository source trees"):
        module._validate_output_root(module.REPO_ROOT / "scripts" / "workflows")
    with pytest.raises(ValueError, match="must not be inside repository source trees"):
        module._validate_output_root(module.REPO_ROOT / "tests" / "unit")

    monkeypatch.setenv(module.GV_HBI_EXPERIMENTAL_FLAG, "yes")
    assert module._require_runtime_flag() == f"env:{module.GV_HBI_EXPERIMENTAL_FLAG}"


def test_gv_operational_canary_cli_defaults_do_not_shadow_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(SCRIPT_PATH, "gv_operational_canary_cli_default_test")
    runtime_calls: list[list[str]] = []

    def _fake_runtime_main(argv: list[str]) -> int:
        runtime_calls.append(list(argv))
        runtime_dir = Path(argv[argv.index("--output-root") + 1]).resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        if "--selection" in argv:
            experiment = str(argv[argv.index("--selection") + 1]).split(":", maxsplit=1)[1]
        else:
            experiment = str(argv[argv.index("--experiment") + 1])
        control_id = "theta_0_03" if experiment == "torsion" else "elongation_0_1"
        controls = {"theta": 0.03} if experiment == "torsion" else {"elongation": 0.1}
        runtime_manifest = {
            "structure": "gv",
            "experiment": experiment,
            "geometry": "gv_rad2_height14_28",
            "geometry_spec": {
                "id": "gv_rad2_height14_28",
                "parameters": {"radius": 2.0, "height": 14.28},
            },
            "controls": controls,
            "control_id": control_id,
            "dataset_id": f"gv:{experiment}:gv_rad2_height14_28:{control_id}",
            "runtime_package": "mirheoOBMD",
            "source_root": str(tmp_path / "sources" / experiment),
            "experimental": True,
            "known_issues": [],
        }
        runtime_dir.joinpath("gv_runtime_dry_run_manifest.json").write_text(json.dumps(runtime_manifest), encoding="utf-8")
        return 0

    def _fake_surrogate_main(argv: list[str]) -> int:
        surrogate_dir = Path(argv[argv.index("--output-root") + 1]).resolve()
        runtime_manifest = json.loads(Path(argv[argv.index("--runtime-manifest") + 1]).read_text(encoding="utf-8"))
        surrogate_dir.mkdir(parents=True, exist_ok=True)
        surrogate_dir.joinpath("gv_dnn_surrogate_smoke_manifest.json").write_text(
            json.dumps(
                {
                    "workflow": "gv_dnn_surrogate_smoke",
                    "reference_kind": "synthetic",
                    "backend": "dnn",
                    "dataset_id": runtime_manifest["dataset_id"],
                    "artifacts": {
                        "artifact_path": str(surrogate_dir / "model.pt"),
                        "model_path": str(surrogate_dir / "model.pt"),
                    },
                }
            ),
            encoding="utf-8",
        )
        surrogate_dir.joinpath("gv_dnn_surrogate_smoke_report.json").write_text(
            json.dumps({"status": "passed"}),
            encoding="utf-8",
        )
        return 0

    def _fake_run_inference(*, dry_run: bool, config_path: str, output_dir: str, device: str) -> None:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        dataset_id = (
            Path(config["experiments"][0]["surrogate_manifest"]).read_text(encoding="utf-8")
        )
        dataset_id = json.loads(dataset_id)["dataset_id"]
        phase1_manifest_root = Path(output_dir) / "results_phase_1"
        phase1_manifest_root.mkdir(parents=True, exist_ok=True)
        phase1_manifest_root.joinpath(gv_hbi.GV_PHASE1_SETUP_MANIFEST).write_text(
            json.dumps(
                {
                    "enabled_by": f"env:{gv_hbi.GV_HBI_EXPERIMENTAL_FLAG}",
                    "datasets": [{"dataset_id": dataset_id}],
                    "parameter_contract": {"calibrated": ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"]},
                }
            ),
            encoding="utf-8",
        )
        phase1_manifest_root.joinpath(gv_hbi.GV_PHASE1_EXECUTION_MANIFEST).write_text(
            json.dumps(
                {
                    "status": "phase1_korali_completed",
                    "execution_model": "single_lane_dnn_korali",
                    "dataset": {"dataset_id": dataset_id},
                    "phase1": {
                        "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
                        "variable_names": ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c", "sigma"],
                    },
                }
            ),
            encoding="utf-8",
        )

    def patched_loader(module_name: str, path: Path):
        if path == module.RUNTIME_SCRIPT:
            return types.SimpleNamespace(main=_fake_runtime_main)
        if path == module.SURROGATE_SCRIPT:
            return types.SimpleNamespace(main=_fake_surrogate_main)
        if path == module.PHASE1_SCRIPT:
            return types.SimpleNamespace(run_inference=_fake_run_inference)
        raise AssertionError(f"Unexpected module load: {module_name} from {path}")

    monkeypatch.setattr(module, "_load_module", patched_loader)
    monkeypatch.setenv(gv_hbi.GV_HBI_EXPERIMENTAL_FLAG, "1")

    default_output = tmp_path / "default_selection"
    assert module.main(["--output-root", str(default_output), "--include-experimental"]) == 0
    default_manifest = json.loads((default_output / module.FINAL_MANIFEST).read_text(encoding="utf-8"))
    assert default_manifest["selection"] == module.DEFAULT_SELECTION
    assert "--selection" in runtime_calls[-1]
    assert runtime_calls[-1][runtime_calls[-1].index("--selection") + 1] == module.DEFAULT_SELECTION

    experiment_output = tmp_path / "experiment_only"
    assert module.main(["--output-root", str(experiment_output), "--experiment", "stretching", "--include-experimental"]) == 0
    experiment_manifest = json.loads((experiment_output / module.FINAL_MANIFEST).read_text(encoding="utf-8"))
    assert experiment_manifest["selection"] == "gv:stretching"
    assert "--selection" not in runtime_calls[-1]
    assert runtime_calls[-1][runtime_calls[-1].index("--experiment") + 1] == "stretching"


def test_gv_operational_canary_surfaces_stage_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(SCRIPT_PATH, "gv_operational_canary_failure_branch_test")

    def _write_runtime_manifest(runtime_dir: Path) -> None:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_manifest = {
            "structure": "gv",
            "experiment": "torsion",
            "geometry": "gv_rad2_height14_28",
            "geometry_spec": {
                "id": "gv_rad2_height14_28",
                "parameters": {"radius": 2.0, "height": 14.28},
            },
            "controls": {"theta": 0.03},
            "control_id": "theta_0_03",
            "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0_03",
            "runtime_package": "mirheoOBMD",
            "source_root": str(tmp_path / "sources" / "torsion"),
            "experimental": True,
            "known_issues": [],
        }
        runtime_dir.joinpath("gv_runtime_dry_run_manifest.json").write_text(json.dumps(runtime_manifest), encoding="utf-8")

    def _runtime_failure(argv: list[str]) -> int:
        return 9

    def _runtime_success(argv: list[str]) -> int:
        _write_runtime_manifest(Path(argv[argv.index("--output-root") + 1]).resolve())
        return 0

    def _surrogate_failure(argv: list[str]) -> int:
        return 7

    def patched_loader_factory(runtime_main, surrogate_main):
        def patched_loader(module_name: str, path: Path):
            if path == module.RUNTIME_SCRIPT:
                return types.SimpleNamespace(main=runtime_main)
            if path == module.SURROGATE_SCRIPT:
                return types.SimpleNamespace(main=surrogate_main)
            if path == module.PHASE1_SCRIPT:
                return types.SimpleNamespace(run_inference=lambda **kwargs: None)
            raise AssertionError(f"Unexpected module load: {module_name} from {path}")

        return patched_loader

    monkeypatch.setenv(gv_hbi.GV_HBI_EXPERIMENTAL_FLAG, "1")
    monkeypatch.setattr(module, "_load_module", patched_loader_factory(_runtime_failure, _surrogate_failure))
    with pytest.raises(RuntimeError, match="GV runtime dry-run failed with code 9"):
        module.main(["--output-root", str(tmp_path / "runtime_failure"), "--selection", "gv:torsion", "--include-experimental"])

    monkeypatch.setattr(module, "_load_module", patched_loader_factory(_runtime_success, _surrogate_failure))
    with pytest.raises(RuntimeError, match="GV DNN smoke workflow failed with code 7"):
        module.main(["--output-root", str(tmp_path / "surrogate_failure"), "--selection", "gv:torsion", "--include-experimental"])


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
    runtime_manifest_path = output_root / "runtime" / "gv_runtime_dry_run_manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
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
    assert manifest["checks"]["runtime_dry_run"] is True
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
    runtime_render_manifest = json.loads((output_root / "runtime" / "gv_runtime_render_manifest.json").read_text(encoding="utf-8"))
    assert runtime_render_manifest["runtime_manifest"] == str(runtime_manifest_path)
    assert all(command["status"] == "skipped-dry-run" for command in runtime_render_manifest["commands"])
    assert Path(manifest["artifacts"]["phase1_config"]).is_file()
    assert Path(manifest["artifacts"]["reference_manifest"]).is_file()


def test_gv_operational_canary_records_runtime_blocked_issue_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(SCRIPT_PATH, "gv_operational_canary_runtime_issue_test")

    def _fake_runtime_main(argv: list[str]) -> int:
        runtime_root_index = argv.index("--output-root")
        runtime_dir = Path(argv[runtime_root_index + 1]).resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        work_dir = runtime_dir / "shear_flow" / "gv_rad2_height14_28" / "ptan_0_4__afsi_0__bpress_-91" / "work"
        runtime_manifest = {
            "structure": "gv",
            "experiment": "shear_flow",
            "geometry": "gv_rad2_height14_28",
            "geometry_spec": {
                "id": "gv_rad2_height14_28",
                "parameters": {"radius": 2.0, "height": 14.28},
            },
            "controls": {"ptan": 0.4, "afsi": 0.0, "bpress": -91.0},
            "control_id": "ptan_0_4__afsi_0__bpress_-91",
            "dataset_id": "gv:shear_flow:gv_rad2_height14_28:ptan_0_4__afsi_0__bpress_-91",
            "output_root": str(runtime_dir),
            "work_dir": str(work_dir),
            "source_root": str(tmp_path / "sources" / "shear_flow"),
            "runtime_package": "mirheoOBMD",
            "experimental": True,
            "known_issues": [
                {
                    "id": "shear-flow-bouncer-candidates",
                    "summary": "Bouncer overflow indicates blocked experimental behavior.",
                    "evidence": "source/fixtures/bouncer.txt",
                    "severity": "blocked",
                },
            ],
            "commands": [],
            "analysis_commands": [],
            "generated_subdirs": [],
        }
        (runtime_dir / "gv_runtime_dry_run_manifest.json").write_text(json.dumps(runtime_manifest), encoding="utf-8")
        return 0

    def _fake_surrogate_main(argv: list[str]) -> int:
        surrogate_dir = Path(argv[argv.index("--output-root") + 1]).resolve()
        runtime_manifest = json.loads(Path(argv[argv.index("--runtime-manifest") + 1]).read_text(encoding="utf-8"))
        surrogate_dir.mkdir(parents=True, exist_ok=True)
        surrogate_manifest = {
            "workflow": "gv_dnn_surrogate_smoke",
            "reference_kind": "synthetic",
            "backend": "dnn",
            "dataset_id": runtime_manifest["dataset_id"],
            "artifacts": {"artifact_path": str(surrogate_dir / "model.pt"), "model_path": str(surrogate_dir / "model.pt")},
        }
        surrogate_dir.joinpath("gv_dnn_surrogate_smoke_manifest.json").write_text(
            json.dumps(surrogate_manifest),
            encoding="utf-8",
        )
        surrogate_dir.joinpath("gv_dnn_surrogate_smoke_report.json").write_text(
            json.dumps({"status": "passed"}),
            encoding="utf-8",
        )
        return 0

    def _fake_run_inference(*, dry_run: bool, config_path: str, output_dir: str, device: str) -> None:
        phase1_manifest_root = Path(output_dir) / "results_phase_1"
        phase1_manifest_root.mkdir(parents=True, exist_ok=True)
        phase1_manifest_root.joinpath(gv_hbi.GV_PHASE1_SETUP_MANIFEST).write_text(
            json.dumps(
                {
                    "enabled_by": f"env:{gv_hbi.GV_HBI_EXPERIMENTAL_FLAG}",
                    "datasets": [{"dataset_id": "gv:shear_flow:gv_rad2_height14_28:ptan_0_4__afsi_0__bpress_-91"}],
                    "parameter_contract": {"calibrated": ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"]},
                }
            ),
            encoding="utf-8",
        )
        phase1_manifest_root.joinpath(gv_hbi.GV_PHASE1_EXECUTION_MANIFEST).write_text(
            json.dumps(
                {
                    "status": "phase1_korali_completed",
                    "execution_model": "single_lane_dnn_korali",
                    "dataset": {"dataset_id": "gv:shear_flow:gv_rad2_height14_28:ptan_0_4__afsi_0__bpress_-91"},
                    "phase1": {
                        "noise_model": {"parameter": "sigma"},
                        "variable_names": ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c", "sigma"],
                    },
                }
            ),
            encoding="utf-8",
        )

    phase1_module = types.SimpleNamespace(run_inference=_fake_run_inference)

    original_loader = module._load_module

    def patched_loader(module_name: str, path: Path):
        if path == module.RUNTIME_SCRIPT:
            return types.SimpleNamespace(main=_fake_runtime_main)
        if path == module.SURROGATE_SCRIPT:
            return types.SimpleNamespace(main=_fake_surrogate_main)
        if path == module.PHASE1_SCRIPT:
            return phase1_module
        return original_loader(module_name, path)

    monkeypatch.setattr(module, "_load_module", patched_loader)
    monkeypatch.setenv(gv_hbi.GV_HBI_EXPERIMENTAL_FLAG, "1")

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "gv_operational_canary"),
            "--selection",
            "gv:shear_flow",
            "--include-experimental",
            "--geometry-id",
            "gv_rad2_height14_28",
        ]
    )

    assert rc == 0
    manifest_path = tmp_path / "gv_operational_canary" / module.FINAL_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checks"]["runtime_experimental"] is True
    assert manifest["checks"]["runtime_blocked_issue_count"] == 1
    assert manifest["runtime_stage"]["experimental"] is True
    assert manifest["runtime_stage"]["runtime_package"] == "mirheoOBMD"
    assert manifest["runtime_stage"]["source_root"] == str(tmp_path / "sources" / "shear_flow")
    assert manifest["runtime_stage"]["known_issues"] == [
        {
            "id": "shear-flow-bouncer-candidates",
            "summary": "Bouncer overflow indicates blocked experimental behavior.",
            "evidence": "source/fixtures/bouncer.txt",
            "severity": "blocked",
            "classification": "experimental_blocked",
        }
    ]
