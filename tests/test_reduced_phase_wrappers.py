from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(path: Path, key: str):
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_reduced_phase1_wrapper_forwards_flags(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "reduced" / "scripts" / "run_phase_1.py",
        "mesouq_reduced_phase1",
    )

    captured = {}

    def _fake_call(cmd, cwd=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_1.py",
            "--config",
            "cfg.yaml",
            "--output-dir",
            "out",
            "--profiling",
            "--restart",
            "--dry_run",
            "--device",
            "gpu",
        ],
    )

    rc = module.main()
    assert rc == 0
    assert captured["cwd"] == str(module.PROJECT_ROOT)
    assert captured["cmd"][1] == str(module.MAIN_DRIVER)
    assert captured["cmd"][-2:] == ["--device", "gpu"]
    assert "--profiling" in captured["cmd"]
    assert "--restart" in captured["cmd"]
    assert "--dry_run" in captured["cmd"]


def test_reduced_phase2_wrapper_uses_default_config(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "reduced" / "scripts" / "run_phase_2.py",
        "mesouq_reduced_phase2",
    )

    captured = {}

    def _fake_call(cmd, cwd=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    monkeypatch.setattr(sys, "argv", ["run_phase_2.py"])

    rc = module.main()
    assert rc == 0
    assert captured["cwd"] == str(module.PROJECT_ROOT)
    assert captured["cmd"][1] == str(module.MAIN_DRIVER)
    assert module.DEFAULT_CONFIG.as_posix() in " ".join(captured["cmd"])


def test_reduced_phase2_wrapper_forwards_backend(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "reduced" / "scripts" / "run_phase_2.py",
        "mesouq_reduced_phase2_backend_forwarding",
    )
    captured = {}

    def _fake_call(cmd, cwd=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_phase_2.py", "--phase2-backend", "native-cuda", "--profiling"],
    )

    rc = module.main()
    assert rc == 0
    assert captured["cwd"] == str(module.PROJECT_ROOT)
    assert "--phase2-backend" in captured["cmd"]
    assert captured["cmd"][-1] == "native-cuda"
    assert "--profiling" in captured["cmd"]


def test_reduced_phase3b_wrapper_forwards_device(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "reduced" / "scripts" / "run_phase_3b.py",
        "mesouq_reduced_phase3b",
    )

    captured = {}

    def _fake_call(cmd, cwd=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_phase_3b.py", "--output-dir", "out", "--profiling", "--device", "gpu"],
    )

    rc = module.main()
    assert rc == 0
    assert captured["cwd"] == str(module.PROJECT_ROOT)
    assert captured["cmd"][1] == str(module.MAIN_DRIVER)
    assert "--profiling" in captured["cmd"]
    assert captured["cmd"][-2:] == ["--device", "gpu"]


def test_reduced_phase3b_wrapper_forwards_repeat_seed_mode(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "reduced" / "scripts" / "run_phase_3b.py",
        "mesouq_reduced_phase3b_repeat_seed",
    )
    captured = {}

    monkeypatch.setattr(
        module.subprocess,
        "call",
        lambda cmd, cwd=None: captured.update(cmd=cmd, cwd=cwd) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_phase_3b.py", "--phase3b-seed-mode", "repeat"],
    )

    assert module.main() == 0
    assert captured["cmd"][-2:] == ["--phase3b-seed-mode", "repeat"]
