from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_map_curve_uses_map_mirheo_fallback_with_csv_map_layout(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "papers" / "huq_emb" / "uqdpd_generate_reduced_story_assets.py",
        "uqdpd_map_overlay_paths_test",
    )

    monkeypatch.setattr(
        module,
        "load_reference_curve",
        lambda modality, diameter: (np.asarray([0.1, 0.2]), np.asarray([1.0, 2.0])),
    )
    monkeypatch.setattr(
        module,
        "load_map_parameters",
        lambda modality, model_kind, diameter: {"parameters": [1.0, 2.0, 3.0]},
    )
    monkeypatch.setattr(
        module,
        "propagation_latest",
        lambda modality, model_kind, diameter: Path("/tmp/fake_latest.json"),
    )
    monkeypatch.setattr(
        module,
        "map_mirheo_path",
        lambda modality, model_kind, diameter: Path("/tmp/fake_result.json"),
    )

    def fake_load_latest(path: Path):
        if path.name == "fake_latest.json":
            return {"Samples": [{"Parameters": [9.0, 9.0, 9.0, 9.0], "Reference Evaluations": [7.0, 8.0]}]}
        if path.name == "fake_result.json":
            return {
                "displacement_points": [0.4, 0.5],
                "forces": [4.0, 5.0],
                "grid_config": {"d0_offset": 0.05},
            }
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(module, "load_latest", fake_load_latest)

    x_dpd, y_dpd = module.extract_map_curve("compression", "full", "2.9")
    assert x_dpd.tolist() == [0.45, 0.55]
    assert y_dpd.tolist() == [4.0, 5.0]


def test_extract_map_surrogate_curve_uses_map_parameters_and_surrogate(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "papers" / "huq_emb" / "uqdpd_generate_reduced_story_assets.py",
        "uqdpd_map_overlay_surrogate_test",
    )

    monkeypatch.setattr(
        module,
        "load_map_parameters",
        lambda modality, model_kind, diameter: {"parameters": [1.0, 2.0, 3.0], "sigma": 0.4},
    )
    monkeypatch.setattr(
        module,
        "load_reference_curve",
        lambda modality, diameter: (np.asarray([0.1, 0.2]), np.asarray([9.0, 8.0])),
    )
    monkeypatch.setattr(module, "workflow_config_path", lambda modality, model_kind: Path("/tmp/fake.yaml"))

    entered = []

    class _Ctx:
        def __enter__(self):
            entered.append(True)
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(module, "inference_config_environment", lambda path: _Ctx())

    def fake_compute(sample, xvals, diameter_um):
        assert sample["Parameters"] == [1.0, 2.0, 3.0, 0.4]
        assert xvals == [0.1, 0.2]
        assert diameter_um == 2.9
        sample["Reference Evaluations"] = [4.2, 5.3]

    import types
    import sys

    monkeypatch.setitem(
        sys.modules,
        "emb.compression.evalkit.posterior_compression",
        types.SimpleNamespace(
            compute_compression_surrogate=fake_compute,
            compute_compression_surrogate_batch=lambda *args, **kwargs: None,
            preload_compression_surrogate=lambda *args, **kwargs: None,
        ),
    )

    x_dpd, y_dpd = module.extract_map_surrogate_curve("compression", "reduced", "2.9")
    assert entered
    assert x_dpd.tolist() == [0.1, 0.2]
    assert y_dpd.tolist() == [4.2, 5.3]
