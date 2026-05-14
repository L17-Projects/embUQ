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
    data = tmp_path / "samples_all.dat"
    out = tmp_path / "out.pkl"
    report = tmp_path / "report.json"
    data.write_text("placeholder", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_read_indentation(path: str, *, disp_source: str, rupture_ratio_threshold):
        captured["read_path"] = path
        captured["disp_source"] = disp_source
        captured["rupture_ratio_threshold"] = rupture_ratio_threshold
        return {"dummy": True}

    def fake_train(df, **kwargs):
        captured["df"] = df
        captured["train_kwargs"] = kwargs
        return {
            "out": kwargs["out_path"],
            "train_loss": 1.0,
            "val_loss": 2.0,
        }

    monkeypatch.setattr(module, "read_indentation_table", fake_read_indentation)
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


def test_indentation_emb_train_uses_shared_cleaned_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "emb" / "indentation" / "surrogate" / "scripts" / "emb_train.py",
        "indentation_emb_train_reader_test",
    )

    assert not hasattr(module, "read_wide_curve_table"), (
        "emb_train.py must not import read_wide_curve_table; "
        "indentation training must use the shared cleaned loader."
    )

    captured = _run_script_with_args(
        module,
        monkeypatch,
        tmp_path,
        "--seed",
        "321",
        "--disp-source",
        "displacement",
        "--rupture-ratio",
        "0",
    )

    assert captured["disp_source"] == "displacement"
    assert captured["rupture_ratio_threshold"] is None
    kwargs = captured["train_kwargs"]
    assert kwargs["input_cols"] == ["Yt", "kb", "b1", "b2", "a3", "a4", "F"]
    assert kwargs["target_col"] == "disp"
    assert kwargs["seed"] == 321
    assert kwargs["report_path"].endswith("report.json")
