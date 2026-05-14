from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_run_surrogate_group_holdout_orchestrator_builds_and_records_runs(
    tmp_path,
    monkeypatch,
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_surrogate_group_holdout.py",
        "run_surrogate_group_holdout_test",
    )

    data_file = tmp_path / "data.dat"
    data_file.write_text("1 0 2 0.1 0.2 0.3 0.4 0 0 0\n", encoding="utf-8")
    model_file = tmp_path / "model.pkl"
    model_file.write_text("dummy", encoding="utf-8")

    captured_commands: list[list[str]] = []

    def _fake_run(command, cwd=None):
        captured_commands.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        module,
        "_SPECS",
        [
            {
                "name": "compression_2.1um",
                "modality": "compression",
                "diameter": "2.1",
                "data": data_file,
                "dnn_model": model_file,
                "bnn_model": model_file,
                "script": repo_root / "emb" / "compression" / "surrogate" / "scripts" / "run_group_holdout.py",
            }
        ],
    )

    output_root = tmp_path / "runs"
    rc = module.main(
        [
            "--output-root",
            str(output_root),
            "--surrogate-family",
            "dnn",
            "--python-bin",
            sys.executable,
        ]
    )
    assert rc == 0
    assert captured_commands
    joined = " ".join(captured_commands[0])
    assert "--surrogate-family dnn" in joined
    assert "--diameter 2.1" in joined

    report_csv = output_root / "surrogate_group_holdout_report.csv"
    report_df = pd.read_csv(report_csv)
    assert len(report_df) == 1
    assert report_df.loc[0, "status"] == "passed"


def test_run_surrogate_group_holdout_orchestrator_handles_missing_models_and_failures(
    tmp_path,
    monkeypatch,
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_surrogate_group_holdout.py",
        "run_surrogate_group_holdout_failure_test",
    )

    data_file = tmp_path / "data.dat"
    data_file.write_text("1 0 2 0.1 0.2 0.3 0.4 0 0 0\n", encoding="utf-8")
    model_file = tmp_path / "model.pkl"
    model_file.write_text("dummy", encoding="utf-8")
    missing_model = tmp_path / "missing.pt"

    calls: list[list[str]] = []

    def _fake_run(command, cwd=None):
        calls.append(list(command))
        # Force a command failure to cover failure handling/return path.
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        module,
        "_SPECS",
        [
            {
                "name": "indentation_3.2um",
                "modality": "indentation",
                "diameter": "3.2",
                "data": data_file,
                "dnn_model": model_file,
                "bnn_model": missing_model,
                "script": repo_root / "emb" / "indentation" / "surrogate" / "scripts" / "run_group_holdout.py",
            }
        ],
    )

    # `--only` mismatch path.
    with pytest.raises(ValueError, match="No specs matched"):
        module.main(["--only", "not-a-spec"])

    output_root = tmp_path / "runs_failure"
    rc = module.main(
        [
            "--output-root",
            str(output_root),
            "--surrogate-family",
            "dnn",
            "--python-bin",
            sys.executable,
        ]
    )
    assert rc == 1
    assert calls, "subprocess.run should have been invoked for existing model"
    assert "--disp-source" in " ".join(calls[0])

    report_df = pd.read_csv(output_root / "surrogate_group_holdout_report.csv")
    assert report_df.loc[0, "status"] == "failed"

    # Missing model is reported as failure when skip-missing-model is disabled.
    strict_root = tmp_path / "runs_missing_strict"
    rc_strict = module.main(
        [
            "--output-root",
            str(strict_root),
            "--surrogate-family",
            "bnn",
            "--no-skip-missing-model",
            "--python-bin",
            sys.executable,
        ]
    )
    assert rc_strict == 1
    strict_df = pd.read_csv(strict_root / "surrogate_group_holdout_report.csv")
    assert strict_df.loc[0, "status"] == "skipped_missing_model"
