from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_karolina_validation_matrix_wrapper_calls_workflow_matrix_with_validation_profile(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_validation_matrix.py",
        "karolina_validation_matrix_wrapper_test",
    )
    captured = {}

    def fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--experiments",
            "compression",
            "--model-families",
            "full-model",
            "--output-root",
            str(tmp_path / "validation"),
            "--phase2-cpu-ranks",
            "4",
            "--python-bin",
            "python",
            "--run-map-mirheo",
            "--map-mirheo-n-displacements",
            "1",
            "--skip-release-manifest",
            "--site",
            "karolina",
        ]
    )

    assert rc == 0
    assert captured["cwd"] == str(repo_root)
    assert captured["command"][0] == "python"
    assert captured["command"][1] == str(
        repo_root / "scripts" / "platforms" / "vega" / "run_workflow_matrix.py"
    )
    assert "--profiles" in captured["command"]
    assert _arg_value(captured["command"], "--profiles") == "validation"
    assert _arg_value(captured["command"], "--site") == "karolina"
    assert "--run-map-mirheo" in captured["command"]
    assert _arg_value(captured["command"], "--map-mirheo-n-displacements") == "1"
    assert "--skip-release-manifest" in captured["command"]
    assert str(tmp_path / "validation") in captured["command"]


def test_karolina_validation_matrix_wrapper_uses_default_runs_root_for_relative_output(
    monkeypatch,
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_validation_matrix.py",
        "karolina_validation_matrix_wrapper_relative_output_test",
    )
    captured = {}

    def fake_default_runs_root(root, workflow_name, *, site=None, run_tag=None, env=None):  # noqa: ANN001
        captured["default_runs_root"] = {
            "root": root,
            "workflow_name": workflow_name,
            "site": site,
            "run_tag": run_tag,
        }
        return repo_root / "_runs" / "karolina" / workflow_name / "tag1"

    def fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module, "default_runs_root", fake_default_runs_root)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--selection", "compression:full-model:validation", "--python-bin", "python"])

    assert rc == 0
    assert captured["default_runs_root"] == {
        "root": repo_root,
        "workflow_name": "validation_matrix",
        "site": "karolina",
        "run_tag": None,
    }
    assert captured["cwd"] == str(repo_root)
    assert _arg_value(captured["command"], "--output-root").endswith("/_runs/karolina/validation_matrix/tag1")
