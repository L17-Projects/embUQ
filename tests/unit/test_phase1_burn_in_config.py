from __future__ import annotations

import importlib.util
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
    pass


class _FakeEngine(_AutoDict):
    def __init__(self) -> None:
        super().__init__()
        self.mpi_comm = None
        self.run_argument = None

    def setMPIComm(self, comm) -> None:
        self.mpi_comm = comm

    def run(self, experiments) -> None:
        self.run_argument = experiments


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


class _FakeKoraliModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("korali")
        self.created_experiments: list[_FakeExperiment] = []
        self.created_engines: list[_FakeEngine] = []

    def Experiment(self) -> _FakeExperiment:
        exp = _FakeExperiment()
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
def phase1_runtime(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "inference" / "scripts" / "run_phase_1.py"
    key = "mesouq_test_inference_phase1_burnin"
    sys.modules.pop(key, None)
    monkeypatch.syspath_prepend(str(repo_root / "src"))
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.syspath_prepend(str(repo_root / "compression"))
    monkeypatch.syspath_prepend(str(repo_root / "compression" / "evalkit"))
    monkeypatch.syspath_prepend(str(repo_root / "indentation"))
    monkeypatch.syspath_prepend(str(repo_root / "indentation" / "evalkit"))

    fake_korali = _FakeKoraliModule()
    fake_comm = _FakeComm()
    fake_mpi4py = _FakeMPI4PY(fake_comm)
    monkeypatch.setitem(sys.modules, "korali", fake_korali)
    monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi4py)

    comp_mod = types.ModuleType("compression.evalkit.posterior_compression")
    comp_mod.compute_compression = lambda *a, **kw: None
    comp_mod.compute_compression_surrogate = lambda *a, **kw: None
    comp_mod.compute_compression_surrogate_batch = lambda *a, **kw: None
    comp_mod.preload_compression_surrogate = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "compression.evalkit.posterior_compression", comp_mod)

    comp_tools = types.ModuleType("compression.evalkit.tools")
    comp_tools.datedPrint = lambda *a, **kw: None
    comp_tools.prepareCompression = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "compression.evalkit.tools", comp_tools)

    ind_mod = types.ModuleType("indentation.evalkit.posterior_indentation")
    ind_mod.compute_indentation_surrogate = lambda *a, **kw: None
    ind_mod.compute_indentation_surrogate_batch = lambda *a, **kw: None
    ind_mod.preload_indentation_surrogate = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "indentation.evalkit.posterior_indentation", ind_mod)

    ind_prep = types.ModuleType("indentation.evalkit.prepare_env")
    ind_prep.prepareIndentation = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "indentation.evalkit.prepare_env", ind_prep)

    spec = importlib.util.spec_from_file_location(key, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, fake_korali, fake_comm


def test_phase1_uses_phase1_burn_in_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase1_runtime
) -> None:
    mod, fake_korali, _fake_comm = phase1_runtime
    config_path = tmp_path / "reduced_indentation_phase1.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pop_size": 50000,
                "max_gen": -1,
                "target_cov": 0.8,
                "covariance_scaling": 0.04,
                "phase1_burn_in": 1,
                "hbi_burn_in": 0,
                "use_surrogate": True,
                "surrogate": {"backend": "dnn"},
                "prior_Yt": [1.0, 2.0],
                "prior_kb": [3.0, 4.0],
                "prior_d0": [5.0, 6.0],
                "prior_sigma": [7.0, 8.0],
            }
        ),
        encoding="utf-8",
    )

    class _FakeStudy:
        name = "indentation"
        enabled = True
        diameters = [3.2]
        prior_d0 = None
        prior_sigma = None
        data_dir = tmp_path
        data_prefix = "indentation_data_"

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"indentation_{diameter_um}um"

        @staticmethod
        def get_reference_points(_diameter_um: float) -> list[float]:
            return [0.0, 1.0]

        @staticmethod
        def get_reference_data(_diameter_um: float) -> list[float]:
            return [0.0, 1.0]

        @staticmethod
        def data_file(_diameter_um: float) -> Path:
            return tmp_path / "dummy.csv"

    monkeypatch.setattr(mod, "load_experiments", lambda config, root: [_FakeStudy()])
    monkeypatch.setattr(
        mod,
        "phase1_prior_specs",
        lambda config, prior_d0, prior_sigma: [
            ("Yt", config["prior_Yt"]),
            ("kb", config["prior_kb"]),
            ("d0", prior_d0),
            ("sigma", prior_sigma),
        ],
    )
    monkeypatch.setattr(mod, "configure_device_conduit", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "to_korali_path", lambda path, *, base_dir: path)

    output_dir = tmp_path / "phase1_output"
    mod.run_inference(config_path=str(config_path), output_dir=str(output_dir), device="gpu")

    experiment = fake_korali.created_experiments[0]
    assert experiment["Solver"]["Burn In"] == 1
    assert fake_korali.created_engines[-1].run_argument == fake_korali.created_experiments


def test_phase1_falls_back_to_hbi_burn_in_when_phase1_knob_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase1_runtime
) -> None:
    mod, fake_korali, _fake_comm = phase1_runtime
    config_path = tmp_path / "reduced_indentation_phase1_legacy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pop_size": 50000,
                "max_gen": -1,
                "target_cov": 0.8,
                "covariance_scaling": 0.04,
                "hbi_burn_in": 2,
                "use_surrogate": True,
                "surrogate": {"backend": "dnn"},
                "prior_Yt": [1.0, 2.0],
                "prior_kb": [3.0, 4.0],
                "prior_d0": [5.0, 6.0],
                "prior_sigma": [7.0, 8.0],
            }
        ),
        encoding="utf-8",
    )

    class _FakeStudy:
        name = "indentation"
        enabled = True
        diameters = [3.2]
        prior_d0 = None
        prior_sigma = None
        data_dir = tmp_path
        data_prefix = "indentation_data_"

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"indentation_{diameter_um}um"

        @staticmethod
        def get_reference_points(_diameter_um: float) -> list[float]:
            return [0.0, 1.0]

        @staticmethod
        def get_reference_data(_diameter_um: float) -> list[float]:
            return [0.0, 1.0]

        @staticmethod
        def data_file(_diameter_um: float) -> Path:
            return tmp_path / "dummy.csv"

    monkeypatch.setattr(mod, "load_experiments", lambda config, root: [_FakeStudy()])
    monkeypatch.setattr(
        mod,
        "phase1_prior_specs",
        lambda config, prior_d0, prior_sigma: [
            ("Yt", config["prior_Yt"]),
            ("kb", config["prior_kb"]),
            ("d0", prior_d0),
            ("sigma", prior_sigma),
        ],
    )
    monkeypatch.setattr(mod, "configure_device_conduit", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "to_korali_path", lambda path, *, base_dir: path)

    output_dir = tmp_path / "phase1_output_legacy"
    mod.run_inference(config_path=str(config_path), output_dir=str(output_dir), device="gpu")

    experiment = fake_korali.created_experiments[0]
    assert experiment["Solver"]["Burn In"] == 2


def test_phase1_falls_back_to_hbi_burn_in_when_phase1_knob_is_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase1_runtime
) -> None:
    mod, fake_korali, _fake_comm = phase1_runtime
    config_path = tmp_path / "reduced_indentation_phase1_null.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pop_size": 50000,
                "max_gen": -1,
                "target_cov": 0.8,
                "covariance_scaling": 0.04,
                "phase1_burn_in": None,
                "hbi_burn_in": 2,
                "use_surrogate": True,
                "surrogate": {"backend": "dnn"},
                "prior_Yt": [1.0, 2.0],
                "prior_kb": [3.0, 4.0],
                "prior_d0": [5.0, 6.0],
                "prior_sigma": [7.0, 8.0],
            }
        ),
        encoding="utf-8",
    )

    class _FakeStudy:
        name = "indentation"
        enabled = True
        diameters = [3.2]
        prior_d0 = None
        prior_sigma = None
        data_dir = tmp_path
        data_prefix = "indentation_data_"

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"indentation_{diameter_um}um"

        @staticmethod
        def get_reference_points(_diameter_um: float) -> list[float]:
            return [0.0, 1.0]

        @staticmethod
        def get_reference_data(_diameter_um: float) -> list[float]:
            return [0.0, 1.0]

        @staticmethod
        def data_file(_diameter_um: float) -> Path:
            return tmp_path / "dummy.csv"

    monkeypatch.setattr(mod, "load_experiments", lambda config, root: [_FakeStudy()])
    monkeypatch.setattr(
        mod,
        "phase1_prior_specs",
        lambda config, prior_d0, prior_sigma: [
            ("Yt", config["prior_Yt"]),
            ("kb", config["prior_kb"]),
            ("d0", prior_d0),
            ("sigma", prior_sigma),
        ],
    )
    monkeypatch.setattr(mod, "configure_device_conduit", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "to_korali_path", lambda path, *, base_dir: path)

    output_dir = tmp_path / "phase1_output_null"
    mod.run_inference(config_path=str(config_path), output_dir=str(output_dir), device="gpu")

    experiment = fake_korali.created_experiments[0]
    assert experiment["Solver"]["Burn In"] == 2


def test_phase1_cpu_surrogate_path_covers_dry_run_profiling_and_max_gen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase1_runtime,
) -> None:
    mod, fake_korali, fake_comm = phase1_runtime
    config_path = tmp_path / "compression_phase1_cpu.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pop_size": 4,
                "max_gen": 2,
                "target_cov": 0.7,
                "covariance_scaling": 0.03,
                "phase1_burn_in": 1,
                "use_surrogate": True,
                "surrogate": {"backend": "dnn"},
                "prior_Yt": [1.0, 2.0],
                "prior_kb": [3.0, 4.0],
                "prior_d0": [0.0, 0.5],
                "prior_sigma": [0.01, 0.2],
            }
        ),
        encoding="utf-8",
    )

    class _CompressionStudy:
        name = "compression"
        enabled = True
        diameters = [2.1]
        prior_d0 = None
        prior_sigma = None

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"compression_{diameter_um}um"

        @staticmethod
        def get_reference_points(_diameter_um: float) -> list[float]:
            return [0.0, 1.0]

        @staticmethod
        def get_reference_data(_diameter_um: float) -> list[float]:
            return [0.0, 1.0]

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_experiments", lambda config, root: [_CompressionStudy()])
    monkeypatch.setattr(mod, "configure_device_conduit", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "to_korali_path", lambda path, *, base_dir: path)
    monkeypatch.setattr(
        mod,
        "phase1_prior_specs",
        lambda config, prior_d0, prior_sigma: [
            ("Yt", config["prior_Yt"]),
            ("kb", config["prior_kb"]),
            ("d0", prior_d0),
            ("sigma", prior_sigma),
        ],
    )

    mod.run_inference(
        profiling=True,
        dry_run=True,
        config_path=str(config_path),
        output_dir=str(tmp_path / "phase1_out"),
        device="cpu",
    )

    engine = fake_korali.created_engines[-1]
    experiment = fake_korali.created_experiments[0]
    assert engine.mpi_comm is fake_comm
    assert engine["Profiling"]["Detail"] == "Full"
    assert experiment["Solver"]["Termination Criteria"]["Max Generations"] == 2
    assert experiment["Problem"]["Type"] == "Bayesian/Reference"
    assert fake_comm.barrier_calls >= 2


def test_phase1_restart_gpu_surrogate_restores_reference_data_and_batch_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase1_runtime,
) -> None:
    mod, fake_korali, _fake_comm = phase1_runtime
    config_path = tmp_path / "compression_phase1_restart.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pop_size": 4,
                "max_gen": -1,
                "target_cov": 0.7,
                "covariance_scaling": 0.03,
                "phase1_burn_in": 1,
                "use_surrogate": True,
                "surrogate": {"backend": "dnn"},
                "prior_Yt": [1.0, 2.0],
                "prior_kb": [3.0, 4.0],
                "prior_d0": [0.0, 0.5],
                "prior_sigma": [0.01, 0.2],
            }
        ),
        encoding="utf-8",
    )

    class _RestartExperiment(_FakeExperiment):
        def loadState(self, _path: str) -> bool:
            self["Problem"]["Reference Data"] = [10.0]
            return True

    def _experiment_factory() -> _RestartExperiment:
        experiment = _RestartExperiment()
        fake_korali.created_experiments.append(experiment)
        return experiment

    class _CompressionStudy:
        name = "compression"
        enabled = True
        diameters = [2.1]
        prior_d0 = None
        prior_sigma = None

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"compression_{diameter_um}um"

        @staticmethod
        def get_reference_points(_diameter_um: float) -> list[float]:
            return [0.0, 1.0, 2.0]

    monkeypatch.setattr(fake_korali, "Experiment", _experiment_factory)
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_experiments", lambda config, root: [_CompressionStudy()])
    monkeypatch.setattr(mod, "configure_device_conduit", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "to_korali_path", lambda path, *, base_dir: path)

    mod.run_inference(
        restart=True,
        config_path=str(config_path),
        output_dir=str(tmp_path / "phase1_out"),
        device="gpu",
    )

    experiment = fake_korali.created_experiments[0]
    assert experiment["Problem"]["Reference Data"] == [10.0]
    assert experiment["Problem"]["Use Batch Evaluation"] is True
    assert "Batch Computational Model" in experiment["Problem"]
    assert fake_korali.created_engines[-1].run_argument == [experiment]


def test_phase1_helper_functions_cover_paths_backend_alignment_and_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase1_runtime,
) -> None:
    mod, _fake_korali, _fake_comm = phase1_runtime

    assert mod._resolve_surrogate_backend({"surrogate": None}) == "dnn"
    with pytest.raises(ValueError, match="Expected 'surrogate' config section"):
        mod._resolve_surrogate_backend({"surrogate": []})
    with pytest.raises(ValueError, match="Unsupported surrogate backend"):
        mod._resolve_surrogate_backend({"surrogate": {"backend": "foo"}})

    ref_points, ref_data = mod._align_reference_data([0.0, 1.0, 2.0], [3.0], "exp", rank=0)
    assert ref_points == [0.0]
    assert ref_data == [3.0]

    default_config = tmp_path / "default.yaml"
    default_config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(mod, "resolve_inference_config_path", lambda *args, **kwargs: default_config)
    assert mod._resolve_config_path(None) == default_config

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    relative_config = tmp_path / "relative.yaml"
    relative_config.write_text("{}\n", encoding="utf-8")
    assert mod._resolve_config_path("relative.yaml") == relative_config.resolve()
    assert mod._resolve_output_dir("phase1_out") == (tmp_path / "phase1_out").resolve()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert mod._resolve_output_dir("~/phase1_home_out") == (tmp_path / "phase1_home_out").resolve()

    cwd_before = Path.cwd()
    with mod._working_directory(tmp_path):
        assert Path.cwd() == tmp_path
    assert Path.cwd() == cwd_before


def test_phase1_prepare_environment_and_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase1_runtime,
) -> None:
    mod, _fake_korali, _fake_comm = phase1_runtime

    prepare_calls = []
    monkeypatch.setattr(mod, "prepareCompression", lambda diameter_um: prepare_calls.append(("compression", diameter_um)))
    monkeypatch.setattr(
        mod,
        "prepareIndentation",
        lambda diameter_um, data_dir, data_prefix, data_file: prepare_calls.append(
            ("indentation", diameter_um, data_dir, data_prefix, data_file)
        ),
    )

    compression_exp = types.SimpleNamespace(name="compression", diameters=[2.1], data_dir=tmp_path)
    indentation_exp = types.SimpleNamespace(
        name="indentation",
        diameters=[3.2],
        data_dir=tmp_path,
        data_prefix="indentation_data_",
        data_file=lambda diameter_um: tmp_path / f"{diameter_um}.csv",
    )
    unknown_exp = types.SimpleNamespace(name="mystery", diameters=[1.0], data_dir=tmp_path)

    mod._prepare_experiment_environment([compression_exp, indentation_exp], rank=1)
    assert prepare_calls == []

    mod._prepare_experiment_environment([compression_exp, indentation_exp], rank=0)
    assert prepare_calls == [
        ("compression", 2.1),
        ("indentation", 3.2, str(tmp_path), "indentation_data_", str(tmp_path / "3.2.csv")),
    ]

    with pytest.raises(ValueError, match="Unsupported experiment type 'mystery'"):
        mod._prepare_experiment_environment([unknown_exp], rank=0)

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    warnings = []
    monkeypatch.setattr(mod, "datedPrint", lambda message: warnings.append(message))
    parameter_dir = tmp_path / "_init_compression_2.1um" / "parameter"
    parameter_dir.mkdir(parents=True)
    base_payload = {"numsteps": 1, "numsteps_eq": 2, "keep": 3}
    for filename in ["parameters-default00001.yaml", "parameters-default00001eq.yaml"]:
        (parameter_dir / filename).write_text(yaml.safe_dump(base_payload), encoding="utf-8")

    mod._apply_compression_dry_run(
        [types.SimpleNamespace(name="compression", diameters=[2.1, 2.9])],
        rank=1,
    )
    mod._apply_compression_dry_run(
        [types.SimpleNamespace(name="compression", diameters=[2.1, 2.9])],
        rank=0,
    )

    for filename in ["parameters-default00001.yaml", "parameters-default00001eq.yaml"]:
        payload = yaml.safe_load((parameter_dir / filename).read_text(encoding="utf-8"))
        assert payload["numsteps"] == 100
        assert payload["numsteps_eq"] == 100
        assert payload["keep"] == 3
    assert any("parameter template not found" in message for message in warnings)


def test_phase1_main_forwards_cli_arguments(monkeypatch: pytest.MonkeyPatch, phase1_runtime) -> None:
    mod, _fake_korali, _fake_comm = phase1_runtime
    captured = {}

    def _fake_run_inference(
        restart=False,
        profiling=False,
        dry_run=False,
        config_path=None,
        output_dir="_setup",
        device="cpu",
        setup_only=False,
    ) -> None:
        captured.update(
            {
                "restart": restart,
                "profiling": profiling,
                "dry_run": dry_run,
                "config_path": config_path,
                "output_dir": output_dir,
                "device": device,
                "setup_only": setup_only,
            }
        )

    monkeypatch.setattr(mod, "run_inference", _fake_run_inference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_1.py",
            "--restart",
            "--profiling",
            "--dry_run",
            "--config",
            "phase1.yaml",
            "--output-dir",
            "results",
            "--device",
            "gpu",
        ],
    )

    mod.main([])

    assert captured == {
        "restart": True,
        "profiling": True,
        "dry_run": True,
        "config_path": "phase1.yaml",
        "output_dir": "results",
        "device": "gpu",
        "setup_only": False,
    }
