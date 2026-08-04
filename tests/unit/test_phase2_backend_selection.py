from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml


class _AutoDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = type(self)()
        return dict.__getitem__(self, key)


class _FakeExperiment(_AutoDict):
    def __init__(self, owner=None) -> None:
        super().__init__()
        self.owner = owner
        self.loaded_state_path: str | None = None
        self.load_state_result = True

    def loadState(self, path: str) -> bool:
        self.loaded_state_path = path
        if self.owner is not None and self.owner.next_load_state_results:
            return self.owner.next_load_state_results.pop(0)
        return self.load_state_result


class _FakeEngine(_AutoDict):
    def __init__(self) -> None:
        super().__init__()
        self.mpi_comm = None
        self.run_argument = None

    def setMPIComm(self, comm) -> None:
        self.mpi_comm = comm

    def run(self, experiment) -> None:
        self.run_argument = experiment


class _FakeComm:
    def __init__(self, *, rank: int = 0, size: int = 1) -> None:
        self.rank = rank
        self.size = size
        self.barrier_calls = 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def Barrier(self) -> None:
        self.barrier_calls += 1


class _FakeStudy:
    def __init__(self, diameters: list[float], *, enabled: bool = True) -> None:
        self.diameters = diameters
        self.enabled = enabled

    def dataset_name(self, diameter_um: float) -> str:
        return f"compression_{diameter_um}um"


class _FakeKoraliModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("korali")
        self.created_experiments: list[_FakeExperiment] = []
        self.created_engines: list[_FakeEngine] = []
        self.next_load_state_results: list[bool] = []

    def Experiment(self) -> _FakeExperiment:
        exp = _FakeExperiment(owner=self)
        self.created_experiments.append(exp)
        return exp

    def Engine(self) -> _FakeEngine:
        engine = _FakeEngine()
        self.created_engines.append(engine)
        return engine


class _FakeMPI4PY(types.ModuleType):
    def __init__(self, comm: _FakeComm) -> None:
        super().__init__("mpi4py")
        self.MPI = types.SimpleNamespace(COMM_WORLD=comm)


@pytest.fixture
def phase2_module(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "inference" / "scripts" / "run_phase_2.py"
    key = "mesouq_test_inference_phase2_backend"
    sys.modules.pop(key, None)
    monkeypatch.syspath_prepend(str(repo_root / "src"))
    monkeypatch.syspath_prepend(str(repo_root))

    fake_korali = _FakeKoraliModule()
    fake_comm = _FakeComm()
    fake_mpi4py = _FakeMPI4PY(fake_comm)
    monkeypatch.setitem(sys.modules, "korali", fake_korali)
    monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi4py)

    spec = importlib.util.spec_from_file_location(key, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, fake_korali, fake_comm


@pytest.fixture
def phase2_runtime(monkeypatch: pytest.MonkeyPatch, phase2_module):
    module, fake_korali, fake_comm = phase2_module
    recorded: dict[str, object] = {
        "dated_prints": [],
        "conduit": None,
        "single_rank_calls": [],
        "to_korali_path": None,
    }

    def fake_dated_print(message: str) -> None:
        recorded["dated_prints"].append(message)

    def fake_phase2_hyperprior_specs(_config):
        return [
            ("Yt", (1.0, 2.0), (0.1, 0.2)),
            ("kb", (3.0, 4.0), (0.3, 0.4)),
            ("d0", (0.0, 0.5), (0.0, 0.3)),
        ]

    def fake_load_experiments(_config, _root):
        return [_FakeStudy([2.1], enabled=True), _FakeStudy([2.9], enabled=False)]

    def fake_configure_korali_conduit(engine, **kwargs):
        recorded["conduit"] = {"engine": engine, **kwargs}

    def fake_require_single_rank(comm, label):
        recorded["single_rank_calls"].append((comm, label))

    def fake_to_korali_path(path: str, *, base_dir: str) -> str:
        recorded["to_korali_path"] = {"path": path, "base_dir": base_dir}
        return f"KORALI::{path}"

    monkeypatch.setattr(module, "datedPrint", fake_dated_print)
    monkeypatch.setattr(module, "phase2_hyperprior_specs", fake_phase2_hyperprior_specs)
    monkeypatch.setattr(module, "load_experiments", fake_load_experiments)
    monkeypatch.setattr(module, "configure_korali_conduit", fake_configure_korali_conduit)
    monkeypatch.setattr(module, "require_single_rank", fake_require_single_rank)
    monkeypatch.setattr(module, "to_korali_path", fake_to_korali_path)

    return module, fake_korali, fake_comm, recorded



def _write_phase2_config(path: Path, *, include_backend: str | None = None) -> None:
    payload: dict[str, object] = {
        "prior_Yt": [1.0, 2.0],
        "prior_kb": [3.0, 4.0],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.0, 1.0],
        "fixed_params": {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0},
        "hbi_pop_size": 8,
        "hbi_burn_in": 2,
        "hbi_target_cov": 0.4,
        "hbi_covariance_scaling": 0.7,
        "hbi_max_gen": 3,
    }
    if include_backend is not None:
        payload["phase2_backend"] = include_backend
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_direct_phase2_config(path: Path) -> None:
    payload: dict[str, object] = {
        "structure": "emb",
        "phase1_contract_mode": "emb_direct_ka_kb",
        "prior_ka": [1000.0, 100000.0],
        "prior_kb": [100.0, 100000.0],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.001, 0.5],
        "hyperprior_mu_ka": [1000.0, 100000.0],
        "hyperprior_sigma_ka": [0.0, 50000.0],
        "hyperprior_mu_kb": [100.0, 100000.0],
        "hyperprior_sigma_kb": [0.0, 50000.0],
        "hbi_pop_size": 8,
        "hbi_burn_in": 2,
        "hbi_target_cov": 0.4,
        "hbi_covariance_scaling": 0.7,
        "hbi_max_gen": 3,
        "phase2_backend": "cpu-mpi",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")



def test_resolve_phase2_backend_profile_defaults(phase2_module) -> None:
    mod, _, _ = phase2_module
    assert mod._resolve_phase2_backend({}, None, profile_hint="production") == "native-cuda"
    assert mod._resolve_phase2_backend({}, None, profile_hint="validation") == "cpu-mpi"



def test_resolve_phase2_backend_explicit_override_wins(phase2_module) -> None:
    mod, _, _ = phase2_module
    cfg = {"phase2_backend": "cpu-mpi"}
    assert mod._resolve_phase2_backend(cfg, "native-cuda", profile_hint="validation") == "native-cuda"



def test_resolve_phase2_backend_config_value_used_when_explicit_missing(phase2_module) -> None:
    mod, _, _ = phase2_module
    cfg = {"phase2_backend": "native-cuda"}
    assert mod._resolve_phase2_backend(cfg, None, profile_hint="validation") == "native-cuda"



def test_resolve_phase2_backend_rejects_unknown_backend(phase2_module) -> None:
    mod, _, _ = phase2_module
    with pytest.raises(ValueError, match="Unsupported phase2_backend"):
        mod._resolve_phase2_backend({}, "gpu-batch", profile_hint="production")



def test_detect_native_cuda_batch_support_reports_disabled(tmp_path: Path, phase2_module, monkeypatch) -> None:
    mod, _, _ = phase2_module
    (tmp_path / "extern" / "korali").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(runtime_root))
    build_opts = runtime_root / "korali" / "build" / "meson-info" / "intro-buildoptions.json"
    build_opts.parent.mkdir(parents=True, exist_ok=True)
    build_opts.write_text(
        json.dumps([{"name": "native_cuda_batch", "value": False}]),
        encoding="utf-8",
    )

    supported, source = mod._detect_native_cuda_batch_support(tmp_path)
    assert supported is False
    assert "native_cuda_batch" in source



def test_detect_native_cuda_batch_support_reports_missing_metadata(tmp_path: Path, phase2_module, monkeypatch) -> None:
    mod, _, _ = phase2_module
    (tmp_path / "extern" / "korali").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(tmp_path / "runtime"))
    supported, source = mod._detect_native_cuda_batch_support(tmp_path)
    assert supported is None
    assert "build options not found" in source



def test_detect_native_cuda_batch_support_reports_invalid_json(tmp_path: Path, phase2_module, monkeypatch) -> None:
    mod, _, _ = phase2_module
    (tmp_path / "extern" / "korali").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(runtime_root))
    build_opts = runtime_root / "korali" / "build" / "meson-info" / "intro-buildoptions.json"
    build_opts.parent.mkdir(parents=True, exist_ok=True)
    build_opts.write_text("not-json", encoding="utf-8")

    supported, source = mod._detect_native_cuda_batch_support(tmp_path)
    assert supported is None
    assert "failed reading" in source



def test_run_hierarchical_inference_defaults_to_native_cuda_for_production_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase2_runtime
) -> None:
    mod, fake_korali, fake_comm, recorded = phase2_runtime
    config_path = tmp_path / "configs" / "production" / "inference.yaml"
    _write_phase2_config(config_path)
    output_dir = tmp_path / "phase2_native_cuda"
    monkeypatch.setattr(mod, "_detect_native_cuda_batch_support", lambda *_: (None, "meson-missing"))

    mod.run_hierarchical_inference(config_path=str(config_path), output_dir=str(output_dir))

    assert recorded["single_rank_calls"] == [(fake_comm, "Phase 2 native-cuda backend")]
    engine = fake_korali.created_engines[-1]
    experiment = fake_korali.created_experiments[0]
    assert engine["Conduit"]["Type"] == "Sequential"
    assert experiment["Problem"]["Use Batch Evaluation"] is True
    assert experiment["Problem"]["Batch Evaluation Backend"] == "NativeCuda"
    assert recorded["conduit"] is None
    assert any("could not be verified" in msg for msg in recorded["dated_prints"])
    assert any("phase2_backend=native-cuda" in msg for msg in recorded["dated_prints"])
    assert experiment["Problem"]["Sub Experiments"][0].loaded_state_path == str(
        output_dir / "results_phase_1" / "compression_2.1um" / "latest"
    )
    assert engine.run_argument is experiment



def test_run_hierarchical_inference_uses_default_config_resolution_when_config_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase2_runtime
) -> None:
    mod, fake_korali, _fake_comm, recorded = phase2_runtime
    resolved_config = tmp_path / "configs" / "compression_default.yaml"
    _write_phase2_config(resolved_config, include_backend="cpu-mpi")
    output_dir = tmp_path / "phase2_default_cfg"
    monkeypatch.setattr(mod, "resolve_inference_config_path", lambda *_args, **_kwargs: resolved_config)

    mod.run_hierarchical_inference(config_path=None, output_dir=str(output_dir))

    engine = fake_korali.created_engines[-1]
    assert recorded["conduit"] is not None
    assert recorded["conduit"]["mpi_ranks"] == 1
    assert recorded["conduit"]["ranks_per_worker"] == 1
    assert recorded["conduit"]["concurrent_jobs"] == 1
    assert engine.mpi_comm is mod.MPI.COMM_WORLD
    assert any("phase2_backend=cpu-mpi" in msg for msg in recorded["dated_prints"])



def test_run_hierarchical_inference_cpu_mpi_configures_mpi_conduit(
    tmp_path: Path, phase2_runtime
) -> None:
    mod, fake_korali, fake_comm, recorded = phase2_runtime
    fake_comm.size = 4
    config_path = tmp_path / "configs" / "validation" / "inference.yaml"
    _write_phase2_config(config_path)
    output_dir = tmp_path / "phase2_cpu_mpi"

    mod.run_hierarchical_inference(config_path=str(config_path), output_dir=str(output_dir))

    engine = fake_korali.created_engines[-1]
    assert engine.mpi_comm is mod.MPI.COMM_WORLD
    assert recorded["conduit"] is not None
    assert recorded["conduit"]["mpi_ranks"] == 4
    assert recorded["conduit"]["ranks_per_worker"] == 1
    assert recorded["conduit"]["concurrent_jobs"] == 1
    assert recorded["single_rank_calls"] == []
    assert any("phase2_backend=cpu-mpi" in msg for msg in recorded["dated_prints"])


def test_run_hierarchical_inference_direct_contract_adds_nuisance_conditionals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase2_runtime
) -> None:
    mod, fake_korali, _fake_comm, _recorded = phase2_runtime
    config_path = tmp_path / "configs" / "validation" / "direct.yaml"
    _write_direct_phase2_config(config_path)
    monkeypatch.setattr(
        mod,
        "load_experiments",
        lambda _config, _root: [_FakeStudy([4.0, 3.2], enabled=True)],
    )
    monkeypatch.setattr(
        mod,
        "phase2_hyperprior_specs",
        lambda _config: [
            ("ka", (1000.0, 100000.0), (0.0, 50000.0)),
            ("kb", (100.0, 100000.0), (0.0, 50000.0)),
        ],
    )

    mod.run_hierarchical_inference(config_path=str(config_path), output_dir=str(tmp_path / "out"))

    experiment = fake_korali.created_experiments[0]
    assert experiment["Problem"]["Conditional Priors"] == [
        "Conditional ka",
        "Conditional kb",
        "Conditional d0",
        "Conditional sigma",
    ]
    variables = experiment["Variables"]
    assert [variables[i]["Name"] for i in range(4)] == ["mu_ka", "sigma_ka", "mu_kb", "sigma_kb"]
    distributions = experiment["Distributions"]
    assert distributions[0]["Name"] == "Conditional ka"
    assert distributions[3]["Name"] == "Conditional kb"
    assert distributions[6]["Name"] == "Conditional d0"
    assert distributions[6]["Minimum"] == 0.0
    assert distributions[6]["Maximum"] == 0.5
    assert distributions[7]["Name"] == "Conditional sigma"
    assert distributions[7]["Minimum"] == 0.001
    assert distributions[7]["Maximum"] == 0.5


def test_run_hierarchical_inference_raises_when_phase1_state_missing(
    tmp_path: Path, phase2_runtime
) -> None:
    mod, fake_korali, _fake_comm, _recorded = phase2_runtime
    config_path = tmp_path / "configs" / "validation" / "inference.yaml"
    _write_phase2_config(config_path, include_backend="cpu-mpi")
    fake_korali.next_load_state_results.append(False)

    with pytest.raises(FileNotFoundError, match="could not load Phase 1 state"):
        mod.run_hierarchical_inference(config_path=str(config_path), output_dir=str(tmp_path / "out"))



def test_run_hierarchical_inference_native_cuda_disabled_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase2_runtime
) -> None:
    mod, fake_korali, fake_comm, recorded = phase2_runtime
    config_path = tmp_path / "configs" / "production" / "inference.yaml"
    _write_phase2_config(config_path)
    monkeypatch.setattr(mod, "_detect_native_cuda_batch_support", lambda *_: (False, "meson-flag"))

    with pytest.raises(RuntimeError, match="native_cuda_batch is disabled"):
        mod.run_hierarchical_inference(config_path=str(config_path), output_dir=str(tmp_path / "out"))

    assert recorded["single_rank_calls"] == [(fake_comm, "Phase 2 native-cuda backend")]
    engine = fake_korali.created_engines[-1]
    assert engine.run_argument is None



def test_main_forwards_default_arguments(monkeypatch: pytest.MonkeyPatch, phase2_module) -> None:
    mod, _, _ = phase2_module
    captured: dict[str, object] = {}

    def fake_run_hierarchical_inference(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mod, "run_hierarchical_inference", fake_run_hierarchical_inference)

    mod.main([])

    assert captured == {
        "profiling": False,
        "config_path": None,
        "output_dir": "_setup",
        "phase2_backend": None,
        "korali_random_seed": None,
    }



def test_main_forwards_explicit_arguments(monkeypatch: pytest.MonkeyPatch, phase2_module) -> None:
    mod, _, _ = phase2_module
    captured: dict[str, object] = {}

    def fake_run_hierarchical_inference(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mod, "run_hierarchical_inference", fake_run_hierarchical_inference)

    mod.main([
        "--profiling",
        "--config",
        "config.yaml",
        "--output-dir",
        "phase2",
        "--phase2-backend",
        "native-cuda",
    ])

    assert captured == {
        "profiling": True,
        "config_path": "config.yaml",
        "output_dir": "phase2",
        "phase2_backend": "native-cuda",
        "korali_random_seed": None,
    }
