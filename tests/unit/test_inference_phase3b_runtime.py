from __future__ import annotations

import importlib.util
import os
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
    def __init__(self, *, load_state_result: bool = True, ref_data=None) -> None:
        super().__init__()
        self.load_state_result = load_state_result
        self.ref_data = ref_data
        self.loaded_paths = []

    def loadState(self, path: str) -> bool:
        self.loaded_paths.append(path)
        if self.ref_data is not None:
            self["Problem"]["Reference Data"] = list(self.ref_data)
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
    def __init__(self, *, rank: int = 0, size: int = 2) -> None:
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
        self.experiment_queue: list[_FakeExperiment] = []
        self.created_experiments: list[_FakeExperiment] = []
        self.created_engines: list[_FakeEngine] = []

    def Experiment(self) -> _FakeExperiment:
        if self.experiment_queue:
            experiment = self.experiment_queue.pop(0)
        else:
            experiment = _FakeExperiment()
        self.created_experiments.append(experiment)
        return experiment

    def Engine(self) -> _FakeEngine:
        engine = _FakeEngine()
        self.created_engines.append(engine)
        return engine


class _FakeMPI4PY(types.ModuleType):
    def __init__(self, comm: _FakeComm) -> None:
        super().__init__("mpi4py")
        self.MPI = types.SimpleNamespace(COMM_WORLD=comm)


@pytest.fixture
def phase3b_runtime(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "inference" / "scripts" / "run_phase_3b.py"
    key = "mesouq_test_phase3b_runtime"
    sys.modules.pop(key, None)
    monkeypatch.syspath_prepend(str(repo_root / "src"))
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.syspath_prepend(str(repo_root / "emb" / "compression"))
    monkeypatch.syspath_prepend(str(repo_root / "emb" / "compression" / "evalkit"))
    monkeypatch.syspath_prepend(str(repo_root / "emb" / "indentation"))
    monkeypatch.syspath_prepend(str(repo_root / "emb" / "indentation" / "evalkit"))

    fake_korali = _FakeKoraliModule()
    fake_comm = _FakeComm()
    monkeypatch.setitem(sys.modules, "korali", fake_korali)
    monkeypatch.setitem(sys.modules, "mpi4py", _FakeMPI4PY(fake_comm))

    comp_mod = types.ModuleType("emb.compression.evalkit.posterior_compression")
    comp_mod.compute_compression_surrogate = lambda *args, **kwargs: None
    comp_mod.compute_compression_surrogate_batch = lambda *args, **kwargs: None
    comp_mod.preload_compression_surrogate = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "emb.compression.evalkit.posterior_compression", comp_mod)

    comp_tools = types.ModuleType("emb.compression.evalkit.tools")
    comp_tools.datedPrint = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "emb.compression.evalkit.tools", comp_tools)

    ind_mod = types.ModuleType("emb.indentation.evalkit.posterior_indentation")
    ind_mod.compute_indentation_surrogate = lambda *args, **kwargs: None
    ind_mod.compute_indentation_surrogate_batch = lambda *args, **kwargs: None
    ind_mod.preload_indentation_surrogate = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "emb.indentation.evalkit.posterior_indentation", ind_mod)

    spec = importlib.util.spec_from_file_location(key, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, fake_korali, fake_comm


def test_run_phase3b_dataset_cpu_configures_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase3b_runtime,
) -> None:
    mod, fake_korali, fake_comm = phase3b_runtime
    output_root = tmp_path / "output"
    (output_root / "results_phase_2" / "latest").mkdir(parents=True)
    (output_root / "results_phase_1" / "compression_2.1um" / "latest").mkdir(parents=True)
    fake_korali.experiment_queue = [
        _FakeExperiment(load_state_result=True),
        _FakeExperiment(load_state_result=True, ref_data=[5.0]),
        _FakeExperiment(),
    ]

    conduit_calls = []
    monkeypatch.setattr(
        mod,
        "configure_device_conduit",
        lambda engine, device, mpi_ranks: conduit_calls.append((engine, device, mpi_ranks)),
    )
    monkeypatch.setattr(mod, "to_korali_path", lambda path, *, base_dir: path)

    compute_calls = []

    def _fake_model(sample_data, ref_points, diameter_um, device="cpu") -> None:
        compute_calls.append((sample_data, ref_points, diameter_um, device))

    mod.run_phase_3b_dataset(
        experiment_name="compression",
        diameter_um=2.1,
        reference_points=[0.0, 1.0],
        compute_model=_fake_model,
        pop_size=17,
        max_gen=4,
        target_cov=0.55,
        output_root=output_root,
        profiling=True,
        device="cpu",
    )

    sub_experiment = fake_korali.created_experiments[1]
    final_experiment = fake_korali.created_experiments[2]
    final_engine = fake_korali.created_engines[-1]

    sub_experiment["Problem"]["Computational Model"]("sample")

    assert compute_calls == [("sample", [0.0], 2.1, "cpu")]
    assert final_engine.mpi_comm is fake_comm
    assert conduit_calls == [(final_engine, "cpu", 2)]
    assert final_engine.run_argument is final_experiment
    assert final_experiment["Problem"]["Type"] == "Hierarchical/Theta"
    assert final_experiment["Problem"]["Psi Experiment"] is fake_korali.created_experiments[0]
    assert final_experiment["Problem"]["Sub Experiment"] is sub_experiment
    assert final_experiment["Solver"]["Population Size"] == 17
    assert final_experiment["Solver"]["Burn In"] == 1
    assert final_experiment["Solver"]["Target Coefficient Of Variation"] == 0.55
    assert final_experiment["Solver"]["Covariance Scaling"] == 0.04
    assert final_experiment["Solver"]["Termination Criteria"]["Max Generations"] == 4
    assert final_engine["Profiling"]["Detail"] == "Full"
    assert final_engine["Profiling"]["Frequency"] == 0.5
    assert (output_root / "results_phase_3b" / "compression_2.1um").is_dir()
    assert fake_comm.barrier_calls >= 2


def test_run_phase3b_dataset_uses_registry_dataset_name_for_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase3b_runtime,
) -> None:
    mod, fake_korali, _fake_comm = phase3b_runtime
    output_root = tmp_path / "output"
    dataset_name = "compression_soft_4.10um"
    (output_root / "results_phase_2" / "latest").mkdir(parents=True)
    (output_root / "results_phase_1" / dataset_name / "latest").mkdir(parents=True)
    fake_korali.experiment_queue = [
        _FakeExperiment(load_state_result=True),
        _FakeExperiment(load_state_result=True, ref_data=[5.0]),
        _FakeExperiment(),
    ]

    monkeypatch.setattr(mod, "configure_device_conduit", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "to_korali_path", lambda path, *, base_dir: path)

    mod.run_phase_3b_dataset(
        experiment_name="compression",
        diameter_um=4.1,
        reference_points=[0.0],
        compute_model=lambda *args, **kwargs: None,
        pop_size=5,
        max_gen=1,
        target_cov=0.5,
        output_root=output_root,
        dataset_name=dataset_name,
    )

    assert fake_korali.created_experiments[1].loaded_paths == [
        str(output_root / "results_phase_1" / dataset_name / "latest")
    ]
    assert (output_root / "results_phase_3b" / dataset_name).is_dir()


def test_run_phase3b_dataset_gpu_uses_batch_conduit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase3b_runtime,
) -> None:
    mod, fake_korali, _fake_comm = phase3b_runtime
    output_root = tmp_path / "output"
    (output_root / "results_phase_2" / "latest").mkdir(parents=True)
    (output_root / "results_phase_1" / "indentation_3.2um" / "latest").mkdir(parents=True)
    fake_korali.experiment_queue = [
        _FakeExperiment(load_state_result=True),
        _FakeExperiment(load_state_result=True, ref_data=[1.0, 2.0]),
        _FakeExperiment(),
    ]

    monkeypatch.setattr(mod, "configure_device_conduit", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "to_korali_path", lambda path, *, base_dir: path)

    batch_calls = []
    single_calls = []
    gpu_conduit_calls = []

    monkeypatch.setattr(
        mod,
        "compute_indentation_surrogate_batch",
        lambda sample_data, ref_points, diameter_um, device="gpu": batch_calls.append(
            (sample_data, ref_points, diameter_um, device)
        ),
    )

    def _fake_model(sample_data, ref_points, diameter_um, device="gpu") -> None:
        single_calls.append((sample_data, ref_points, diameter_um, device))

    def _fake_gpu_conduit(sub, batch_model_fn, single_model_fn) -> None:
        gpu_conduit_calls.append((sub, batch_model_fn, single_model_fn))

    monkeypatch.setattr(mod, "configure_gpu_batch_sub_experiment", _fake_gpu_conduit)

    mod.run_phase_3b_dataset(
        experiment_name="indentation",
        diameter_um=3.2,
        reference_points=[0.0, 1.0],
        compute_model=_fake_model,
        pop_size=9,
        max_gen=0,
        target_cov=0.6,
        output_root=output_root,
        profiling=False,
        device="gpu",
    )

    sub_experiment, batch_model_fn, single_model_fn = gpu_conduit_calls[0]
    batch_model_fn("batch")
    single_model_fn("single")

    assert batch_calls == [("batch", [0.0, 1.0], 3.2, "gpu")]
    assert single_calls == [("single", [0.0, 1.0], 3.2, "gpu")]
    assert sub_experiment is fake_korali.created_experiments[1]
    assert fake_korali.created_engines[-1].mpi_comm is None


def test_run_phase3b_dataset_raises_for_missing_prerequisites_or_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase3b_runtime,
) -> None:
    mod, fake_korali, _fake_comm = phase3b_runtime
    monkeypatch.setattr(mod, "configure_device_conduit", lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match="Phase 2 results not found"):
        mod.run_phase_3b_dataset(
            experiment_name="compression",
            diameter_um=2.1,
            reference_points=[0.0],
            compute_model=lambda *args, **kwargs: None,
            pop_size=1,
            max_gen=1,
            target_cov=0.5,
            output_root=tmp_path,
        )

    output_root = tmp_path / "second"
    (output_root / "results_phase_2" / "latest").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Phase 1 results not found"):
        mod.run_phase_3b_dataset(
            experiment_name="compression",
            diameter_um=2.1,
            reference_points=[0.0],
            compute_model=lambda *args, **kwargs: None,
            pop_size=1,
            max_gen=1,
            target_cov=0.5,
            output_root=output_root,
        )

    failing_root = tmp_path / "third"
    (failing_root / "results_phase_2" / "latest").mkdir(parents=True)
    (failing_root / "results_phase_1" / "compression_2.1um" / "latest").mkdir(parents=True)
    fake_korali.experiment_queue = [
        _FakeExperiment(load_state_result=False),
        _FakeExperiment(load_state_result=True),
    ]
    with pytest.raises(RuntimeError, match="Failed to load Phase 2 state"):
        mod.run_phase_3b_dataset(
            experiment_name="compression",
            diameter_um=2.1,
            reference_points=[0.0],
            compute_model=lambda *args, **kwargs: None,
            pop_size=1,
            max_gen=1,
            target_cov=0.5,
            output_root=failing_root,
        )

    fake_korali.experiment_queue = [
        _FakeExperiment(load_state_result=True),
        _FakeExperiment(load_state_result=False),
    ]
    with pytest.raises(RuntimeError, match="Failed to load Phase 1 state"):
        mod.run_phase_3b_dataset(
            experiment_name="compression",
            diameter_um=2.1,
            reference_points=[0.0],
            compute_model=lambda *args, **kwargs: None,
            pop_size=1,
            max_gen=1,
            target_cov=0.5,
            output_root=failing_root,
        )


def test_run_phase3b_dispatches_selected_target_and_preloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase3b_runtime,
) -> None:
    mod, _fake_korali, _fake_comm = phase3b_runtime
    config_path = tmp_path / "phase3b.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "surrogate": {"backend": "bnn"},
                "phase3b_pop_size": 77,
                "phase3b_max_gen": 5,
                "phase3b_target_cov": 0.42,
                "phase3b_covariance_scaling": 0.015,
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    (output_root / "results_phase_2" / "latest").mkdir(parents=True)
    (output_root / "results_phase_1" / "compression_2.9um" / "latest").mkdir(parents=True)

    experiment = types.SimpleNamespace(
        name="compression",
        enabled=True,
        diameters=[2.1, 2.9],
        dataset_name=lambda diameter_um: f"compression_{diameter_um}um",
        get_reference_points=lambda diameter_um: [diameter_um, diameter_um + 1.0],
    )
    monkeypatch.setattr(mod, "load_experiments", lambda config, root: [experiment])
    preload_calls = []
    monkeypatch.setattr(
        mod,
        "preload_compression_surrogate",
        lambda diameter_um, device="cpu", backend="dnn": preload_calls.append(
            (diameter_um, device, backend)
        ),
    )
    run_calls = []
    monkeypatch.setattr(mod, "run_phase_3b_dataset", lambda **kwargs: run_calls.append(kwargs))

    mod.run_phase_3b(
        config_path=str(config_path),
        output_dir=str(output_root),
        device="gpu",
        diameter=2.9,
        profiling=True,
    )

    assert os.environ["HUQ_INFERENCE_CONFIG"] == str(config_path.resolve())
    assert preload_calls == [(2.9, "gpu", "bnn")]
    assert run_calls == [
        {
            "experiment_name": "compression",
            "diameter_um": 2.9,
            "reference_points": [2.9, 3.9],
            "compute_model": mod.compute_compression_surrogate,
            "compute_batch_model": mod.compute_compression_surrogate_batch,
            "pop_size": 77,
            "max_gen": 5,
            "target_cov": 0.42,
            "covariance_scaling": 0.015,
            "output_root": output_root.resolve(),
            "profiling": True,
            "device": "gpu",
            "dataset_name": "compression_2.9um",
            "korali_random_seed": None,
            "restart": False,
        }
    ]


def test_run_phase3b_offsets_seed_for_each_selected_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase3b_runtime,
) -> None:
    mod, _fake_korali, _fake_comm = phase3b_runtime
    config_path = tmp_path / "phase3b.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "surrogate": {"backend": "bnn"},
                "phase3b_target_experiments": ["compression"],
                "phase3b_target_diameters": [2.1, 2.9],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    (output_root / "results_phase_2" / "latest").mkdir(parents=True)
    for diameter in (2.1, 2.9):
        (output_root / "results_phase_1" / f"compression_{diameter}um" / "latest").mkdir(
            parents=True
        )

    experiment = types.SimpleNamespace(
        name="compression",
        enabled=True,
        diameters=[2.1, 2.9],
        dataset_name=lambda diameter_um: f"compression_{diameter_um}um",
        get_reference_points=lambda diameter_um: [diameter_um],
    )
    monkeypatch.setattr(mod, "load_experiments", lambda config, root: [experiment])
    monkeypatch.setattr(mod, "preload_compression_surrogate", lambda *args, **kwargs: None)
    run_calls = []
    monkeypatch.setattr(mod, "run_phase_3b_dataset", lambda **kwargs: run_calls.append(kwargs))

    mod.run_phase_3b(
        config_path=str(config_path),
        output_dir=str(output_root),
        korali_random_seed=3104,
    )

    assert [call["korali_random_seed"] for call in run_calls] == [3104, 3105]


def test_run_phase3b_rejects_unknown_preload_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase3b_runtime,
) -> None:
    mod, _fake_korali, _fake_comm = phase3b_runtime
    config_path = tmp_path / "phase3b.yaml"
    config_path.write_text(yaml.safe_dump({"surrogate": {"backend": "dnn"}}), encoding="utf-8")
    experiment = types.SimpleNamespace(
        name="mystery",
        enabled=True,
        diameters=[1.0],
        dataset_name=lambda diameter_um: "mystery_1.0um",
        get_reference_points=lambda diameter_um: [],
    )
    monkeypatch.setattr(mod, "load_experiments", lambda config, root: [experiment])

    with pytest.raises(ValueError, match="No surrogate preload function registered"):
        mod.run_phase_3b(config_path=str(config_path), output_dir=str(tmp_path / "out"))
