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

matplotlib.use("Agg", force=True)
os.environ.setdefault("MPLBACKEND", "Agg")


def _load_module(path: Path, name: str):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))
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


def test_compression_resolve_data_path_uses_fallback_candidates(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_paths_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    data_dir = surrogate_root / "diameters" / "2.1um" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)

    assert module._resolve_data_path("2.1", None).name == "F_Delta.dat"
    (data_dir / "samples_all.dat").write_text("sample", encoding="utf-8")
    assert module._resolve_data_path("2.1", None).name == "samples_all.dat"
    assert module._resolve_data_path("2.1", "custom.dat").name == "custom.dat"


def test_compression_train_multi_arch_collect_only_and_error_paths(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_collect_only_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    data_dir = surrogate_root / "diameters" / "2.1um" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "F_Delta.dat").write_text("sample", encoding="utf-8")
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)

    output_dir = tmp_path / "trained"
    report_json = tmp_path / "collect_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
            "--output-dir",
            str(output_dir),
            "--collect-only",
            "--report-json",
            str(report_json),
        ],
    )

    with pytest.raises(FileNotFoundError, match="No per-architecture result JSONs found"):
        module.main()

    model_path = output_dir / "w32_d2.pkl"
    model_path.write_text("model", encoding="utf-8")
    result_payload = {
        "name": "w32_d2",
        "width": 32,
        "depth": 2,
        "train_loss": 1.0,
        "val_loss": 0.5,
        "model_path": str(model_path),
    }
    (output_dir / "result_w32_d2.json").write_text(json.dumps(result_payload), encoding="utf-8")

    module.main()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["best"]["name"] == "w32_d2"
    assert payload["data_path"].endswith("F_Delta.dat")


def test_compression_train_multi_arch_rejects_invalid_arch_selection(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_invalid_arch_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    data_dir = surrogate_root / "diameters" / "2.1um" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "F_Delta.dat").write_text("sample", encoding="utf-8")
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
            "--arch-index",
            "0",
            "--arch-name",
            "w32_d2",
        ],
    )
    with pytest.raises(ValueError, match="Use only one of --arch-index or --arch-name"):
        module.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
            "--arch-name",
            "bad-arch",
        ],
    )
    with pytest.raises(ValueError, match="Unknown arch-name"):
        module.main()


def test_compression_train_multi_arch_resolve_data_path_fallbacks(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_paths_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)
    data_dir = surrogate_root / "diameters" / "2.1um" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    explicit = data_dir / "custom.dat"
    explicit.write_text("explicit", encoding="utf-8")
    assert module._resolve_data_path("2.1", "custom.dat") == explicit

    default_path = data_dir / "F_Delta.dat"
    default_path.write_text("default", encoding="utf-8")
    assert module._resolve_data_path("2.1", None) == default_path

    default_path.unlink()
    samples_path = data_dir / "samples_all.dat"
    samples_path.write_text("samples", encoding="utf-8")
    assert module._resolve_data_path("2.1", None) == samples_path

    samples_path.unlink()
    assert module._resolve_data_path("2.1", None) == data_dir / "F_Delta.dat"


def test_compression_train_multi_arch_finalize_results_writes_artifacts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_finalize_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)

    output_dir = tmp_path / "trained"
    plots_dir = tmp_path / "plots"
    report_json = tmp_path / "reports" / "summary.json"
    model_a = output_dir / "w32_d2.pkl"
    model_b = output_dir / "w64_d2.pkl"
    model_a.parent.mkdir(parents=True, exist_ok=True)
    model_a.write_text("model-a", encoding="utf-8")
    model_b.write_text("model-b", encoding="utf-8")

    payload = module.finalize_results(
        diameter="2.1",
        output_dir=output_dir,
        results=[
            {"name": "w64_d2", "width": 64, "depth": 2, "train_loss": 0.9, "val_loss": 0.4, "model_path": str(model_b)},
            {"name": "w32_d2", "width": 32, "depth": 2, "train_loss": 1.0, "val_loss": 0.2, "model_path": str(model_a)},
        ],
        plots_dir=plots_dir,
        report_json=report_json,
        timestamp="20260424_120000",
    )

    best_dest = surrogate_root / "diameters" / "2.1um" / "trained" / "microbubble_force_BEST.pkl"
    assert payload["best"]["name"] == "w32_d2"
    assert Path(payload["training_results_csv"]).exists()
    assert Path(payload["summary_plot"]).exists()
    assert best_dest.read_text(encoding="utf-8") == "model-a"
    assert json.loads(report_json.read_text(encoding="utf-8"))["best"]["name"] == "w32_d2"


def test_compression_train_multi_arch_finalize_results_rejects_empty_results(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_finalize_empty_test",
    )

    with pytest.raises(ValueError, match="No architecture results"):
        module.finalize_results(
            diameter="2.1",
            output_dir=tmp_path / "trained",
            results=[],
            plots_dir=tmp_path / "plots",
        )


def test_compression_train_multi_arch_main_collect_only_aggregates_results(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_collect_only_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    data_dir = surrogate_root / "diameters" / "2.1um" / "data"
    output_dir = tmp_path / "trained"
    report_json = tmp_path / "collect_only_report.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "F_Delta.dat").write_text("sample", encoding="utf-8")
    model_a = output_dir / "w32_d2.pkl"
    model_b = output_dir / "w64_d2.pkl"
    model_a.write_text("model-a", encoding="utf-8")
    model_b.write_text("model-b", encoding="utf-8")
    (output_dir / "result_1.json").write_text(
        json.dumps({"name": "w64_d2", "width": 64, "depth": 2, "train_loss": 0.8, "val_loss": 0.5, "model_path": str(model_b)}),
        encoding="utf-8",
    )
    (output_dir / "result_0.json").write_text(
        json.dumps({"name": "w32_d2", "width": 32, "depth": 2, "train_loss": 0.9, "val_loss": 0.3, "model_path": str(model_a)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
            "--output-dir",
            str(output_dir),
            "--report-json",
            str(report_json),
            "--collect-only",
        ],
    )

    module.main()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["best"]["name"] == "w32_d2"
    assert payload["data_path"] == str(data_dir / "F_Delta.dat")
    assert Path(payload["best_dest"]).read_text(encoding="utf-8") == "model-a"


def test_compression_train_multi_arch_main_collect_only_requires_result_jsons(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_collect_only_missing_results_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    data_dir = surrogate_root / "diameters" / "2.1um" / "data"
    output_dir = tmp_path / "trained"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "F_Delta.dat").write_text("sample", encoding="utf-8")
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
            "--output-dir",
            str(output_dir),
            "--collect-only",
        ],
    )

    with pytest.raises(FileNotFoundError, match="No per-architecture result JSONs"):
        module.main()


def test_compression_train_multi_arch_main_rejects_invalid_single_arch_selection(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_single_arch_errors_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    data_dir = surrogate_root / "diameters" / "2.1um" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "F_Delta.dat").write_text("sample", encoding="utf-8")
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
            "--arch-index",
            "0",
            "--arch-name",
            "w32_d2",
        ],
    )
    with pytest.raises(ValueError, match="Use only one of --arch-index or --arch-name"):
        module.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
            "--arch-name",
            "missing-arch",
        ],
    )
    with pytest.raises(ValueError, match="Unknown arch-name 'missing-arch'"):
        module.main()


def test_compression_train_multi_arch_main_requires_existing_training_data(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "compression_train_multi_arch_missing_data_test",
    )

    surrogate_root = tmp_path / "compression" / "surrogate"
    monkeypatch.setattr(module, "SURROGATE_ROOT", surrogate_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_arch.py",
            "--diameter",
            "2.1",
        ],
    )

    with pytest.raises(FileNotFoundError, match="Training data not found"):
        module.main()
