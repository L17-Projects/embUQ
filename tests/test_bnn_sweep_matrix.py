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
    dnn = tmp_path / f"{name}.pkl"
    script = tmp_path / f"{name}_train.py"
    for path in (data, dnn, script):
        path.write_text("placeholder", encoding="utf-8")
    return {
        "name": name,
        "modality": "compression",
        "diameter_um": "2.1",
        "data": str(data),
        "dnn_artifact": str(dnn),
        "bnn_artifact": str(tmp_path / f"{name}.pt"),
        "dnn_train_script": str(script),
        "dnn_multi_arch_script": str(script),
        "bnn_train_script": str(script),
        "group_holdout_script": str(script),
    }


def test_bnn_sweep_runner_selects_best_stage2_candidate(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_main_test",
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
        prior_scale = float(_parse_arg(command, "--prior-scale"))
        metric = 0.5
        if "stage1__w32_d2" in out_path.stem:
            metric = 0.20
        elif "stage1__w64_d2" in out_path.stem:
            metric = 0.30
        elif prior_scale == 0.5:
            metric = 0.11
        elif prior_scale == 1.0:
            metric = 0.15
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"training": {"best_val_rmse": metric, "final_val_rmse": metric + 0.01}}),
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
            "--top-k",
            "1",
            "--prior-scales",
            "0.5,1.0",
            "--obs-noise-prior-scales",
            "0.1",
            "--lrs",
            "0.001",
        ]
    )
    assert rc == 0
    selection_path = tmp_path / "out" / "spec_a" / "seed_101" / "selection.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    assert payload["top_architectures"] == ["w32_d2"]
    assert payload["best_candidate_metric"] == pytest.approx(0.11)
    assert "prior0.5" in payload["best_candidate_report_path"]
    assert called


def test_bnn_sweep_runner_resume_skips_completed(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_resume_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    seed_root = tmp_path / "out" / "spec_a" / "seed_20260317"
    artifact_path, report_path = module._candidate_paths(seed_root, "stage1__w32_d2__prior1__obs1__lr0.001")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("artifact", encoding="utf-8")
    report_path.write_text(
        json.dumps({"training": {"best_val_rmse": 0.2, "final_val_rmse": 0.21}}),
        encoding="utf-8",
    )

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
            json.dumps({"training": {"best_val_rmse": 0.25, "final_val_rmse": 0.26}}),
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
            "--top-k",
            "1",
            "--prior-scales",
            "1.0",
            "--obs-noise-prior-scales",
            "1.0",
            "--lrs",
            "0.001",
        ]
    )
    assert rc == 0
    assert called
    assert all("w32_d2" not in " ".join(command) or "stage2" in " ".join(command) for command in called)


def test_bnn_sweep_runner_import_does_not_require_torch() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _assert_runner_import_without_torch(repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py")


def test_hpc_bnn_sweep_wrapper_dispatches_to_selected_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_bnn_sweep_matrix.py"),
        "hpc_bnn_sweep_dispatch_test",
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
    assert "scripts/platforms/karolina/run_bnn_sweep_matrix.py" in " ".join(captured[0])


def test_hpc_bnn_sweep_wrapper_rejects_unknown_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_bnn_sweep_matrix.py"),
        "hpc_bnn_sweep_invalid_site_test",
    )
    monkeypatch.setenv("HPC_SITE", "unknown")
    with pytest.raises(SystemExit, match="Unsupported HPC_SITE"):
        module.main([])
