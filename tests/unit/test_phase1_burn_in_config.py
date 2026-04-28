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
