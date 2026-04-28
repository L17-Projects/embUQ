from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _parse_arg(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def _assert_runner_import_without_torch(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root}{':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''}"
    script = f"""
import builtins
import importlib.util

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.', 1)[0] == 'torch':
        raise ModuleNotFoundError('blocked torch import')
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
spec = importlib.util.spec_from_file_location('runner_import_test', {str(path)!r})
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _make_spec(tmp_path: Path, name: str) -> dict[str, str]:
    data = tmp_path / f"{name}.dat"
    script = tmp_path / f"{name}_train.py"
    dnn = tmp_path / f"{name}.pkl"
    bnn = tmp_path / f"{name}.pt"
    for path in (data, script):
        path.write_text("placeholder", encoding="utf-8")
    return {
        "name": name,
        "modality": "compression",
        "diameter_um": "2.1",
        "data": str(data),
        "dnn_artifact": str(dnn),
        "bnn_artifact": str(bnn),
        "dnn_train_script": str(script),
        "dnn_multi_arch_script": str(script),
        "bnn_train_script": str(script),
        "group_holdout_script": str(script),
    }


def test_dnn_rebaseline_runner_writes_selection_and_state(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_dnn_rebaseline_matrix.py",
        "run_dnn_rebaseline_matrix_main_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    called: list[list[str]] = []

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        called.append(command)
        out_path = Path(_parse_arg(command, "--out"))
        report_path = Path(_parse_arg(command, "--report-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("artifact", encoding="utf-8")
        val_loss = 0.25 if "w32_d2" in out_path.stem else 0.5
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"train_loss": 0.1, "val_loss": val_loss, "out": str(out_path)}),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "101",
            "--architectures",
            "w32_d2,w64_d2",
        ]
    )
    assert rc == 0
    assert len(called) == 2

    selection_path = tmp_path / "out" / "spec_a" / "seed_101" / "selection.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    assert payload["best_architecture"] == "w32_d2"
    assert payload["best_val_loss"] == pytest.approx(0.25)

    matrix_path = tmp_path / "out" / "dnn_rebaseline_matrix_report__seeds-101.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["status"] == "passed"
    assert matrix["completed_at"] is not None
    assert len(matrix["runs"]) == 2
    assert matrix["config"]["requested_specs"] == []
    assert matrix["config"]["requested_seeds"] == [101]
    assert matrix["config"]["resolved_specs"] == ["spec_a"]


def test_dnn_rebaseline_runner_resume_skips_completed(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_dnn_rebaseline_matrix.py",
        "run_dnn_rebaseline_matrix_resume_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    seed_root = tmp_path / "out" / "spec_a" / "seed_20260317"
    artifact_path = seed_root / "candidates" / "w32_d2.pkl"
    report_path = seed_root / "reports" / "w32_d2.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("artifact", encoding="utf-8")
    report_path.write_text(json.dumps({"train_loss": 0.1, "val_loss": 0.2}), encoding="utf-8")

    called: list[list[str]] = []

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        called.append(command)
        out_path = Path(_parse_arg(command, "--out"))
        candidate_report = Path(_parse_arg(command, "--report-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("artifact", encoding="utf-8")
        candidate_report.parent.mkdir(parents=True, exist_ok=True)
        candidate_report.write_text(
            json.dumps({"train_loss": 0.2, "val_loss": 0.3, "out": str(out_path)}),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "20260317",
            "--architectures",
            "w32_d2,w64_d2",
        ]
    )
    assert rc == 0
    assert len(called) == 1
    assert "w64_d2" in " ".join(called[0])


def test_dnn_rebaseline_runner_writes_invocation_scoped_report_for_filtered_runs(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_dnn_rebaseline_matrix.py",
        "run_dnn_rebaseline_matrix_filtered_report_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        out_path = Path(_parse_arg(command, "--out"))
        report_path = Path(_parse_arg(command, "--report-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("artifact", encoding="utf-8")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"train_loss": 0.1, "val_loss": 0.2, "out": str(out_path)}),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--only",
            "spec_a",
            "--seed",
            "20260317",
            "--architectures",
            "w32_d2",
        ]
    )
    assert rc == 0

    default_matrix_path = tmp_path / "out" / "dnn_rebaseline_matrix_report.json"
    filtered_matrix_path = (
        tmp_path / "out" / "dnn_rebaseline_matrix_report__specs-spec_a__seeds-20260317.json"
    )
    assert default_matrix_path.exists() is False
    matrix = json.loads(filtered_matrix_path.read_text(encoding="utf-8"))
    assert matrix["status"] == "passed"
    assert matrix["completed_at"] is not None
    assert matrix["config"]["requested_specs"] == ["spec_a"]
    assert matrix["config"]["requested_seeds"] == [20260317]
    assert matrix["config"]["resolved_specs"] == ["spec_a"]


def test_dnn_rebaseline_runner_import_does_not_require_torch() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _assert_runner_import_without_torch(
        repo_root / "scripts" / "platforms" / "karolina" / "run_dnn_rebaseline_matrix.py"
    )


def test_dnn_rebaseline_runner_rejects_empty_or_unknown_architecture_names(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_dnn_rebaseline_matrix.py",
        "run_dnn_rebaseline_matrix_parse_test",
    )

    with pytest.raises(ValueError, match="non-empty"):
        module._parse_architecture_names(" , ")
    with pytest.raises(ValueError, match="Unknown DNN architecture"):
        module._parse_architecture_names("unknown_arch")
    assert module._resolve_seeds([]) == list(module.DEFAULT_SEEDS)

    seed_root = tmp_path / "seed"
    artifact_path, report_path = module._candidate_paths(seed_root, "w32_d2")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("artifact", encoding="utf-8")
    report_path.write_text("{broken json", encoding="utf-8")
    assert module._candidate_completed(report_path, artifact_path) is False
    assert (
        module._matrix_report_path(tmp_path, requested_specs=[], requested_seeds=[])
        == tmp_path / "dnn_rebaseline_matrix_report.json"
    )
    assert (
        module._matrix_report_path(tmp_path, requested_specs=["spec_a"], requested_seeds=[20260317])
        == tmp_path / "dnn_rebaseline_matrix_report__specs-spec_a__seeds-20260317.json"
    )


def test_dnn_rebaseline_runner_rejects_unknown_only_filter(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_dnn_rebaseline_matrix.py",
        "run_dnn_rebaseline_matrix_only_filter_test",
    )
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [_make_spec(tmp_path, "spec_a")])

    with pytest.raises(ValueError, match="No EMB dataset specs matched"):
        module.main(["--output-root", str(tmp_path / "out"), "--only", "missing"])


def test_hpc_dnn_rebaseline_wrapper_dispatches_to_selected_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_dnn_rebaseline_matrix.py"),
        "hpc_dnn_rebaseline_dispatch_test",
    )
    monkeypatch.setenv("HPC_SITE", "karolina")
    captured: list[list[str]] = []

    def _fake_call(cmd):  # noqa: ANN001
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    rc = module.main(["--seed", "123"])
    assert rc == 0
    assert captured
    assert sys.executable in captured[0][0]
    assert "scripts/platforms/karolina/run_dnn_rebaseline_matrix.py" in " ".join(captured[0])


def test_hpc_dnn_rebaseline_wrapper_rejects_unknown_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_dnn_rebaseline_matrix.py"),
        "hpc_dnn_rebaseline_invalid_site_test",
    )
    monkeypatch.setenv("HPC_SITE", "unknown")
    with pytest.raises(SystemExit, match="Unsupported HPC_SITE"):
        module.main([])
