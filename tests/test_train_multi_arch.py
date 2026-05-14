from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg", force=True)
os.environ.setdefault("MPLBACKEND", "Agg")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_train_multi_arch_public_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "emb" / "indentation" / "surrogate" / "scripts" / "train_multi_arch.py",
        "train_multi_arch_test",
    )

    architectures = module.get_architectures()
    assert (32, 2, "w32_d2") in architectures
    assert (256, 3, "w256_d3") in architectures

    radp = np.array([5.0, 5.0])
    diameter_like = np.array([[9.8, 9.7], [9.6, 9.5]])
    displacement_like = np.array([[0.2, 0.3], [0.4, 0.5]])
    np.testing.assert_allclose(module._infer_displacement(diameter_like, radp, "diameter"), [[0.2, 0.3], [0.4, 0.5]])
    np.testing.assert_allclose(module._infer_displacement(displacement_like, radp, "displacement"), displacement_like)
    np.testing.assert_allclose(module._infer_displacement(diameter_like, radp, "auto"), [[0.2, 0.3], [0.4, 0.5]])

    data_path = tmp_path / "samples.dat"
    row = [10.0, 0.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
    np.savetxt(data_path, np.array([row]), fmt="%.6f")
    long_df = module.read_new_dat_to_long(str(data_path), disp_source="displacement")
    assert list(long_df.columns) == ["Yt", "kb", "b1", "b2", "a3", "a4", "F", "disp"]
    assert float(long_df.iloc[0]["F"]) == 0.0
    assert float(long_df.iloc[0]["disp"]) == 0.0

    with pytest.raises(ValueError):
        bad_path = tmp_path / "bad.dat"
        np.savetxt(bad_path, np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), fmt="%.6f")
        module.read_new_dat_to_long(str(bad_path))

    tensors_df = pd.DataFrame(
        {
            "Yt": [1.0, 1.0],
            "kb": [2.0, 2.0],
            "b1": [3.0, 4.0],
            "b2": [4.0, 5.0],
            "a3": [5.0, 6.0],
            "a4": [6.0, 7.0],
            "F": [1.0, 2.0],
            "disp": [0.1, 0.2],
        }
    )
    X, y, _x_mu, x_sd, _y_mu, y_sd = module.make_tensors(
        tensors_df, ["Yt", "kb", "b1", "b2", "a3", "a4", "F"], "disp"
    )
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32
    assert all(scale != 0 for scale in x_sd)
    assert y_sd[0] != 0

    monkeypatch.setattr(module, "read_new_dat_to_long", lambda *_args, **_kwargs: pd.DataFrame(
        {
            "Yt": np.linspace(1.0, 10.0, 20),
            "kb": np.linspace(2.0, 11.0, 20),
            "b1": np.linspace(3.0, 12.0, 20),
            "b2": np.linspace(4.0, 13.0, 20),
            "a3": np.linspace(5.0, 14.0, 20),
            "a4": np.linspace(6.0, 15.0, 20),
            "F": np.linspace(0.1, 2.0, 20),
            "disp": np.linspace(0.01, 0.2, 20),
        }
    ))

    def fake_train_model(model, loader, Xv, yv, lr, max_epoch, info_every):
        _ = loader, Xv, yv, lr, max_epoch, info_every
        return model, [1.0, 0.5], [1.2, 0.4]

    monkeypatch.setattr(module, "train_model", fake_train_model)
    monkeypatch.setattr(module, "check_negative_predictions", lambda *_args, **_kwargs: (False, 0.0, 0, 5))

    def fake_save_model_states(model, xshift, xscale, yshift, yscale, path):
        _ = model, xshift, xscale, yshift, yscale
        Path(path).write_text("model", encoding="utf-8")

    monkeypatch.setattr(module, "save_model_states", fake_save_model_states)

    result = module.train_single_architecture(
        (32, 2, "w32_d2"),
        data_path="unused.dat",
        output_dir=str(tmp_path / "trained"),
        disp_source="auto",
        max_epoch=2,
        num_workers=0,
    )
    assert result["name"] == "w32_d2"
    assert result["val_loss"] == 0.4
    assert Path(result["model_path"]).exists()

    summary_path = tmp_path / "plots" / "summary.png"
    module.plot_results_summary(
        [
            {
                "name": "w32_d2",
                "train_loss": 1.0,
                "val_loss": 0.5,
                "num_negatives": 0,
                "total_samples": 10,
                "min_prediction": 0.0,
            },
            {
                "name": "w64_d2",
                "train_loss": 1.2,
                "val_loss": 0.6,
                "num_negatives": 2,
                "total_samples": 10,
                "min_prediction": -0.1,
            },
        ],
        str(summary_path),
        "3.4",
    )
    assert summary_path.exists()

    surrogate_root = tmp_path / "indentation" / "surrogate"
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)
    data_dir = surrogate_root / "diameters" / "3.4um" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = data_dir / "samples_all.dat"
    expected.write_text("sample", encoding="utf-8")
    assert module._resolve_data_path("3.4", None) == expected


def test_train_multi_arch_main_single_arch_writes_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "emb" / "indentation" / "surrogate" / "scripts" / "train_multi_arch.py",
        "train_multi_arch_main_test",
    )

    surrogate_root = tmp_path / "indentation" / "surrogate"
    data_dir = surrogate_root / "diameters" / "3.4um" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "samples_all.dat").write_text("sample", encoding="utf-8")
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)
    monkeypatch.setattr(
        module,
        "train_single_architecture",
        lambda *args, **kwargs: {
            "name": "w32_d2",
            "width": 32,
            "depth": 2,
            "train_loss": 1.0,
            "val_loss": 0.5,
            "has_negatives": False,
            "min_prediction": 0.0,
            "num_negatives": 0,
            "total_samples": 10,
            "model_path": str(tmp_path / "trained" / "w32_d2.pkl"),
        },
    )

    result_json = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "3.4",
            "--arch-name",
            "w32_d2",
            "--result-json",
            str(result_json),
            "--output-dir",
            str(tmp_path / "trained"),
        ],
    )

    module.main()
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["name"] == "w32_d2"
