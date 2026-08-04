import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


class _FakeComm:
    def Get_rank(self):
        return 0


def _install_phase3b_backend_stubs():
    sys.modules.setdefault("korali", types.ModuleType("korali"))

    mpi4py_module = types.ModuleType("mpi4py")
    mpi4py_module.MPI = types.SimpleNamespace(COMM_WORLD=_FakeComm())
    sys.modules.setdefault("mpi4py", mpi4py_module)

    compression_module = types.ModuleType("emb.compression.evalkit.posterior_compression")
    compression_module.compute_compression_surrogate = lambda *args, **kwargs: None
    compression_module.compute_compression_surrogate_batch = lambda *args, **kwargs: None
    compression_module.preload_compression_surrogate = lambda *args, **kwargs: None
    sys.modules.setdefault("emb.compression.evalkit.posterior_compression", compression_module)

    indentation_module = types.ModuleType("emb.indentation.evalkit.posterior_indentation")
    indentation_module.compute_indentation_surrogate = lambda *args, **kwargs: None
    indentation_module.compute_indentation_surrogate_batch = lambda *args, **kwargs: None
    indentation_module.preload_indentation_surrogate = lambda *args, **kwargs: None
    sys.modules.setdefault("emb.indentation.evalkit.posterior_indentation", indentation_module)


def _load_phase3b_module():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "emb" / "compression"))
    sys.path.insert(0, str(repo_root / "emb" / "compression" / "evalkit"))
    sys.path.insert(0, str(repo_root / "emb" / "indentation"))
    sys.path.insert(0, str(repo_root / "emb" / "indentation" / "evalkit"))
    backend_names = (
        "emb.compression.evalkit.posterior_compression",
        "emb.indentation.evalkit.posterior_indentation",
    )
    previous_backends = {name: sys.modules.get(name) for name in backend_names}
    for name in backend_names:
        sys.modules.pop(name, None)
    _install_phase3b_backend_stubs()
    module_path = repo_root / "inference" / "scripts" / "run_phase_3b.py"
    spec = importlib.util.spec_from_file_location("mesouq_test_run_phase3b", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_backends.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def test_run_phase3b_exports_active_config(monkeypatch, tmp_path):
    module = _load_phase3b_module()
    config_path = tmp_path / "validation.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    output_root = tmp_path / "output"
    phase2_dir = output_root / "results_phase_2"
    phase2_dir.mkdir(parents=True)
    (phase2_dir / "latest").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "load_experiments", lambda config, root: [])
    monkeypatch.delenv("HUQ_INFERENCE_CONFIG", raising=False)

    module.run_phase_3b(config_path=str(config_path), output_dir=str(output_root))

    assert os.environ["HUQ_INFERENCE_CONFIG"] == str(config_path.resolve())


def test_run_phase3b_requires_phase1_results(monkeypatch, tmp_path):
    module = _load_phase3b_module()
    config_path = tmp_path / "validation.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    output_root = tmp_path / "output"
    phase2_dir = output_root / "results_phase_2"
    phase2_dir.mkdir(parents=True)
    (phase2_dir / "latest").write_text("{}", encoding="utf-8")

    experiment = types.SimpleNamespace(
        name="compression",
        enabled=True,
        diameters=[2.1],
        dataset_name=lambda diameter_um: f"compression_{diameter_um}um",
        get_reference_points=lambda diameter_um: [],
    )
    monkeypatch.setattr(module, "load_experiments", lambda config, root: [experiment])

    with pytest.raises(SystemExit) as excinfo:
        module.run_phase_3b(config_path=str(config_path), output_dir=str(output_root))

    assert excinfo.value.code == 1


def test_phase3b_main_forwards_device_and_paths(monkeypatch):
    module = _load_phase3b_module()
    captured = {}

    def _fake_run_phase_3b(
        profiling=False,
        config_path=None,
        output_dir="_setup",
        device="cpu",
        dataset_name=None,
        diameter=None,
        korali_random_seed=None,
        restart=False,
    ):
        captured["profiling"] = profiling
        captured["config_path"] = config_path
        captured["output_dir"] = output_dir
        captured["device"] = device
        captured["dataset_name"] = dataset_name
        captured["diameter"] = diameter
        captured["korali_random_seed"] = korali_random_seed
        captured["restart"] = restart

    monkeypatch.setattr(module, "run_phase_3b", _fake_run_phase_3b)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_3b.py",
            "--profiling",
            "--config",
            "phase3b.yaml",
            "--output-dir",
            "results",
            "--device",
            "gpu",
            "--dataset-name",
            "compression_2.1um",
        ],
    )
    module.main([])

    assert captured == {
        "profiling": True,
        "config_path": "phase3b.yaml",
        "output_dir": "results",
        "device": "gpu",
        "dataset_name": "compression_2.1um",
        "diameter": None,
        "korali_random_seed": None,
        "restart": False,
    }
