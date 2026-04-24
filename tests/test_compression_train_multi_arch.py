from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
os.environ.setdefault("MPLBACKEND", "Agg")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_compression_train_multi_arch_helpers(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_test",
    )

    architectures = module.get_architectures()
    assert (32, 2, "w32_d2") in architectures
    assert (256, 3, "w256_d3") in architectures

    data_path = tmp_path / "F_Delta.dat"
    row = [10.0, 0.0, 2.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
    np.savetxt(data_path, np.array([row]), fmt="%.6f")
    long_df = module.read_new_dat_to_long(str(data_path))
    assert list(long_df.columns) == ["Yt", "kb", "b1", "b2", "a3", "a4", "disp", "F"]

    monkeypatch.setattr(
        module,
        "read_new_dat_to_long",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "Yt": np.linspace(1.0, 10.0, 20),
                "kb": np.linspace(2.0, 11.0, 20),
                "b1": np.linspace(3.0, 12.0, 20),
                "b2": np.linspace(4.0, 13.0, 20),
                "a3": np.linspace(5.0, 14.0, 20),
                "a4": np.linspace(6.0, 15.0, 20),
                "disp": np.linspace(0.1, 2.0, 20),
                "F": np.linspace(0.01, 0.2, 20),
            }
        ),
    )

    monkeypatch.setattr(
        module,
        "train_tabular_surrogate",
        lambda *_args, **_kwargs: {"train_loss": 1.0, "val_loss": 0.4, "out": str(tmp_path / "m.pkl")},
    )

    result = module.train_single_architecture((32, 2, "w32_d2"), str(data_path), str(tmp_path / "trained"), max_epoch=2)
    assert result["name"] == "w32_d2"
    assert result["val_loss"] == 0.4

    summary_path = tmp_path / "plots" / "summary.png"
    module.plot_results_summary(
        [
            {"name": "w32_d2", "train_loss": 1.0, "val_loss": 0.5},
            {"name": "w64_d2", "train_loss": 1.2, "val_loss": 0.6},
        ],
        str(summary_path),
        "2.1",
    )
    assert summary_path.exists()


def test_compression_train_multi_arch_main_writes_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_main_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    data_dir = surrogate_root / "diameters" / "2.1um" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "F_Delta.dat").write_text("sample", encoding="utf-8")
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)
    model_path = tmp_path / "trained" / "w32_d2.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "train_single_architecture",
        lambda *args, **kwargs: {
            "name": "w32_d2",
            "width": 32,
            "depth": 2,
            "train_loss": 1.0,
            "val_loss": 0.5,
            "model_path": str(model_path),
        },
    )

    report_json = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
            "--output-dir",
            str(tmp_path / "trained"),
            "--report-json",
            str(report_json),
        ],
    )

    module.main()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["best"]["name"] == "w32_d2"
    assert payload["best_dest"].endswith("microbubble_force_BEST.pkl")
