from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _run_script_with_args(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *extra_args: str,
):
    data = tmp_path / "F_Delta.dat"
    out = tmp_path / "out.pkl"
    report = tmp_path / "report.json"
    data.write_text("placeholder", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_read(path: str, *, curve_axis_name: str, value_name: str):
        captured["read_path"] = path
        captured["curve_axis_name"] = curve_axis_name
        captured["value_name"] = value_name
        return {"dummy": True}

    def fake_train(df, **kwargs):
        captured["df"] = df
        captured["train_kwargs"] = kwargs
        return {
            "out": kwargs["out_path"],
            "train_loss": 1.0,
            "val_loss": 2.0,
        }

    monkeypatch.setattr(module, "read_compression_training_table", fake_read)
    monkeypatch.setattr(module, "train_tabular_surrogate", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "emb_train.py",
            str(data),
            "--out",
            str(out),
            "--report-path",
            str(report),
            *extra_args,
        ],
    )

    module.main()
    return captured


def test_compression_emb_train_accepts_seed_and_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "emb_train.py",
        "compression_emb_train_script_test",
    )

    captured = _run_script_with_args(module, monkeypatch, tmp_path, "--seed", "123")

    assert captured["curve_axis_name"] == "disp"
    assert captured["value_name"] == "F"
    kwargs = captured["train_kwargs"]
    assert kwargs["input_cols"] == ["Yt", "kb", "b1", "b2", "a3", "a4", "disp"]
    assert kwargs["target_col"] == "F"
    assert kwargs["seed"] == 123
    assert kwargs["report_path"].endswith("report.json")
