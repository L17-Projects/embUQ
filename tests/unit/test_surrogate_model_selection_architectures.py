from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from meso_uq.surrogate import model_selection


def test_grid_search_tabular_surrogate_honors_explicit_architecture_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_train(df, *, input_cols, target_col, out_path, width, depth, batch_size, lr, max_epoch):
        _ = df, input_cols, target_col, out_path, batch_size, lr, max_epoch
        calls.append((width, depth))
        return {"train_loss": float(width + depth), "val_loss": float(width + depth)}

    monkeypatch.setattr(model_selection, "train_tabular_surrogate", fake_train)

    df = pd.DataFrame({"x": [0.0], "y": [0.0]})
    result = model_selection.grid_search_tabular_surrogate(
        df,
        input_cols=["x"],
        target_col="y",
        output_dir=tmp_path,
        widths=[8],
        depths=[1],
        architectures=[(16, 2), (32, 3)],
        batch_size=1,
        lr=1e-3,
        max_epoch=1,
    )

    assert calls == [(16, 2), (32, 3)]
    leaderboard = pd.read_csv(result["leaderboard_path"])
    assert list(zip(leaderboard["width"], leaderboard["depth"])) == [(16, 2), (32, 3)]


def _load_compression_model_select_module():
    repo_root = Path(__file__).resolve().parents[2]
    key = "mesouq_test_compression_emb_model_select"
    sys.modules.pop(key, None)
    module_path = repo_root / "compression" / "surrogate" / "scripts" / "emb_model_select.py"
    spec = importlib.util.spec_from_file_location(key, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_compression_model_select_defaults_to_12_architectures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_compression_model_select_module()

    captured: dict[str, object] = {}

    def fake_read(path, *, curve_axis_name, value_name):
        captured["read"] = (path, curve_axis_name, value_name)
        return {"dummy": True}

    def fake_grid(df, **kwargs):
        _ = df
        captured["kwargs"] = kwargs
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "leaderboard.csv").write_text("width,depth,val_loss,train_loss,model_path\n", encoding="utf-8")
        (out_dir / "best_model.json").write_text(
            json.dumps({"width": 64, "depth": 3, "val_loss": 1.0, "train_loss": 1.0}),
            encoding="utf-8",
        )
        return {
            "best": {"width": 64, "depth": 3, "val_loss": 1.0, "train_loss": 1.0},
            "leaderboard_path": str(out_dir / "leaderboard.csv"),
            "best_path": str(out_dir / "best_model.json"),
        }

    monkeypatch.setattr(mod, "read_wide_curve_table", fake_read)
    monkeypatch.setattr(mod, "grid_search_tabular_surrogate", fake_grid)

    data_path = tmp_path / "train.dat"
    data_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["emb_model_select.py", str(data_path), "--output-dir", str(tmp_path / "out")],
    )
    mod.main()

    kwargs = captured["kwargs"]
    architectures = kwargs["architectures"]
    assert len(architectures) == 12
    assert (64, 5) in architectures
    assert (256, 3) in architectures
    assert (256, 4) not in architectures


def test_compression_model_select_rejects_bad_architecture_token() -> None:
    mod = _load_compression_model_select_module()
    with pytest.raises(ValueError, match="Invalid architecture token"):
        mod._parse_architecture_list("64x3,badtoken")
