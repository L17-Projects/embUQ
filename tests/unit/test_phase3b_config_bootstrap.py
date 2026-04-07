import importlib.util
import os
import sys
from pathlib import Path


def _load_phase3b_module():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "compression"))
    sys.path.insert(0, str(repo_root / "compression" / "evalkit"))
    sys.path.insert(0, str(repo_root / "indentation"))
    sys.path.insert(0, str(repo_root / "indentation" / "evalkit"))
    module_path = repo_root / "inference" / "scripts" / "run_phase_3b.py"
    spec = importlib.util.spec_from_file_location("mesouq_test_run_phase3b", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
