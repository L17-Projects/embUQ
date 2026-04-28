from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_karolina_roundtrip_main_writes_pass_result(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_pass_test",
    )

    def _fake_check(args, run_dir):  # noqa: ANN001
        del args
        return {
            "status": "passed",
            "selection": "indentation_3.2um",
            "run_dir": str(run_dir),
            "updated_at": "2026-04-22T00:00:00+00:00",
        }

    monkeypatch.setattr(module, "run_roundtrip_check", _fake_check)

    rc = module.main(["--output-root", str(tmp_path), "--selection", "indentation_3.2um", "--site", "vega"])
    assert rc == 0

    result_path = tmp_path / "indentation_3.2um" / "roundtrip_strict_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"


def test_karolina_roundtrip_main_writes_failure_result(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_fail_test",
    )

    def _boom(args, run_dir):  # noqa: ANN001
        del args, run_dir
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(module, "run_roundtrip_check", _boom)

    rc = module.main(["--output-root", str(tmp_path), "--selection", "indentation_3.2um", "--site", "vega"])
    assert rc == 1

    result_path = tmp_path / "indentation_3.2um" / "roundtrip_strict_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "synthetic failure" in payload["error"]


def test_karolina_roundtrip_helpers_cover_selection_and_degradation() -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_helper_test",
    )
    spec = module._resolve_selection_spec("compression_2.1um")
    assert spec["modality"] == "compression"
    assert spec["diameter_um"] == "2.1"
    assert spec["data"].endswith("compression/surrogate/diameters/2.1um/data/F_Delta.dat")

    abs_deg, rel_deg = module._compute_degradation(0.08, 0.10)
    assert abs_deg == pytest.approx(0.02)
    assert rel_deg == pytest.approx(0.25)


def test_hpc_roundtrip_wrapper_dispatches_to_selected_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_bnn_roundtrip_check.py"),
        "hpc_roundtrip_dispatch_test",
    )
    monkeypatch.setenv("HPC_SITE", "karolina")
    captured: list[list[str]] = []

    def _fake_call(cmd):  # noqa: ANN001
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    rc = module.main(["--selection", "indentation_3.2um"])
    assert rc == 0
    assert captured
    joined = " ".join(captured[0])
    assert sys.executable in captured[0][0]
    assert "scripts/platforms/karolina/run_bnn_roundtrip_check.py" in joined
    assert "--selection indentation_3.2um" in joined


def test_hpc_roundtrip_wrapper_rejects_unknown_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_bnn_roundtrip_check.py"),
        "hpc_roundtrip_invalid_site_test",
    )
    monkeypatch.setenv("HPC_SITE", "unknown")
    with pytest.raises(SystemExit, match="Unsupported HPC_SITE"):
        module.main([])


def test_karolina_roundtrip_helper_branches(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_branch_test",
    )

    with pytest.raises(ValueError, match="Unknown --selection"):
        module._resolve_selection_spec("not-a-selection")

    assert module._compute_rmse(np.asarray([1.0, 2.0]), np.asarray([1.0, 4.0])) == pytest.approx(
        np.sqrt(2.0)
    )
    abs_deg, rel_deg = module._compute_degradation(0.0, 0.3)
    assert abs_deg == pytest.approx(0.3)
    assert rel_deg == float("inf")

    captured: dict[str, object] = {}

    def _fake_read_compression(path: str, *, curve_axis_name: str, value_name: str):
        captured["compression"] = (path, curve_axis_name, value_name)
        return "compression-df"

    def _fake_read_indentation(path: str, *, disp_source: str, rupture_ratio_threshold: float):
        captured["indentation"] = (path, disp_source, rupture_ratio_threshold)
        return "indentation-df"

    monkeypatch.setattr(module, "read_compression_training_table", _fake_read_compression)
    monkeypatch.setattr(module, "read_indentation_table", _fake_read_indentation)

    assert (
        module._load_training_dataframe({"modality": "compression", "data": "/tmp/c.dat"}, disp_source="auto", rupture_ratio=2.0)
        == "compression-df"
    )
    assert captured["compression"] == ("/tmp/c.dat", "disp", "F")
    assert (
        module._load_training_dataframe({"modality": "indentation", "data": "/tmp/i.dat"}, disp_source="diameter", rupture_ratio=3.0)
        == "indentation-df"
    )
    assert captured["indentation"] == ("/tmp/i.dat", "diameter", 3.0)


def test_karolina_roundtrip_split_and_env_helpers(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_split_env_test",
    )

    df = pd.DataFrame(
        {
            "Yt": [1.0, 2.0],
            "kb": [3.0, 4.0],
            "b1": [5.0, 6.0],
            "disp": [7.0, 8.0],
            "F": [9.0, 10.0],
        }
    )

    monkeypatch.setattr(
        module,
        "make_tensors",
        lambda _df, _inputs, _target: (
            np.asarray([[0.1], [0.2]], dtype=np.float32),
            np.asarray([[0.3], [0.4]], dtype=np.float32),
            None,
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        module,
        "_split_like_dnn",
        lambda *_args, **_kwargs: {
            "X_val_phys": np.asarray([[11.0, 12.0]], dtype=np.float32),
            "y_val_phys": np.asarray([13.0], dtype=np.float32),
        },
    )
    x_val, y_val = module._split_validation_phys(
        df,
        input_cols=["Yt", "kb"],
        target_col="F",
        seed=123,
    )
    np.testing.assert_allclose(x_val, np.asarray([[11.0, 12.0]], dtype=np.float32))
    np.testing.assert_allclose(y_val, np.asarray([13.0], dtype=np.float64))

    class _FakeTorch:
        __version__ = "2.test"

        class version:
            cuda = "12.3"

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def get_device_name(index: int) -> str:
                assert index == 0
                return "Fake GPU"

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    payload = module._collect_env()
    assert payload["cuda_available"] is True
    assert payload["cuda_device_name_0"] == "Fake GPU"


def test_karolina_roundtrip_run_roundtrip_check_covers_pass_and_fail(monkeypatch, tmp_path) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_runtime_test",
    )

    df = pd.DataFrame(
        {
            "Yt": [1.0, 2.0],
            "kb": [3.0, 4.0],
            "b1": [5.0, 6.0],
            "b2": [7.0, 8.0],
            "a3": [9.0, 10.0],
            "a4": [11.0, 12.0],
            "disp": [0.1, 0.2],
            "F": [0.3, 0.4],
        }
    )
    monkeypatch.setattr(
        module,
        "_resolve_selection_spec",
        lambda selection: {
            "name": selection,
            "modality": "compression",
            "diameter_um": "2.1",
            "data": "/tmp/data.dat",
            "dnn": "/tmp/model.pkl",
            "out_stem": "microbubble_force",
        },
    )
    monkeypatch.setattr(module, "_load_training_dataframe", lambda *args, **kwargs: df)
    monkeypatch.setattr(
        module,
        "train_tabular_bnn_surrogate",
        lambda *_args, **_kwargs: {
            "training": {"final_val_rmse": 0.1},
            "validation": {"val_predictive_std_mean": 0.25},
        },
    )
    monkeypatch.setattr(
        module,
        "_split_validation_phys",
        lambda *_args, **_kwargs: (
            np.asarray([[1.0] * 7], dtype=np.float32),
            np.asarray([0.1], dtype=np.float64),
        ),
    )

    class _Predictor:
        def __init__(self, artifact_path: str, *, device: str) -> None:
            assert artifact_path.endswith("_BNN_roundtrip_strict.pt")
            assert device == "cpu"

        def predict_mean_std(self, X_val_phys, *, predictive_mc_samples: int, predictive_mc_chunk_size: int):
            assert X_val_phys.shape == (1, 7)
            assert predictive_mc_samples == 5
            assert predictive_mc_chunk_size == 2
            return np.asarray([0.1], dtype=np.float64), np.asarray([0.05], dtype=np.float64)

    monkeypatch.setattr(module, "VariationalBNNPredictor", _Predictor)
    monkeypatch.setattr(module, "_collect_env", lambda: {"env": "ok"})
    monkeypatch.setattr(module, "_now_iso", lambda: "2026-04-23T00:00:00+00:00")

    args = SimpleNamespace(
        selection="compression_2.1um",
        disp_source="auto",
        rupture_ratio=2.0,
        width=32,
        depth=2,
        prior_scale=1.0,
        obs_noise_prior_scale=1.0,
        batch_size=16,
        lr=1e-3,
        max_steps=10,
        eval_every=2,
        predictive_mc_samples=5,
        predictive_mc_chunk_size=2,
        max_walltime_seconds=60,
        seed=7,
        parity_tol=1.1,
        device="cpu",
        reload_tol_rel=0.25,
    )

    result = module.run_roundtrip_check(args, tmp_path)
    assert result["status"] == "passed"
    assert result["config"]["device"] == "cpu"
    assert result["env"] == {"env": "ok"}

    class _FailingPredictor(_Predictor):
        def predict_mean_std(self, X_val_phys, *, predictive_mc_samples: int, predictive_mc_chunk_size: int):
            return np.asarray([0.3], dtype=np.float64), np.asarray([0.05], dtype=np.float64)

    monkeypatch.setattr(module, "VariationalBNNPredictor", _FailingPredictor)
    failed = module.run_roundtrip_check(args, tmp_path)
    assert failed["status"] == "failed"
    assert failed["degradation_rel"] > args.reload_tol_rel


def test_karolina_roundtrip_main_uses_default_runs_root(monkeypatch, tmp_path) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_default_root_test",
    )
    monkeypatch.setattr(module, "detect_hpc_site", lambda: "karolina")
    monkeypatch.setattr(module, "default_runs_root", lambda *_args, **_kwargs: tmp_path / "default-root")
    monkeypatch.setattr(
        module,
        "run_roundtrip_check",
        lambda args, run_dir: {"status": "passed", "selection": args.selection, "run_dir": str(run_dir)},
    )

    rc = module.main(["--selection", "compression_2.1um"])
    assert rc == 0
    result_path = tmp_path / "default-root" / "compression_2.1um" / "roundtrip_strict_result.json"
    assert result_path.exists()
