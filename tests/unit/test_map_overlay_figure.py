"""Tests for generate_map_overlay_figure.py — codex-connector P2 fixes."""
from __future__ import annotations

import json
import py_compile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_SCRIPT = REPO / "scripts" / "postprocess" / "generate_map_overlay_figure.py"


def test_script_parseable():
    py_compile.compile(str(_SCRIPT), doraise=True)


# ---------------------------------------------------------------------------
# P2 fix 1: model_family included in output stem
# ---------------------------------------------------------------------------

def test_make_overlay_figure_stem_includes_model_family(tmp_path, monkeypatch):
    """Output filename must include model_family to avoid overwriting."""
    import sys
    sys.path.insert(0, str(REPO / "src"))

    # Write a minimal manifest with model_family
    manifest = {
        "model_family": "reduced-model",
        "diameters": [],
    }
    manifest_path = tmp_path / "map_mirheo_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # Patch heavy dependencies so we can call make_overlay_figure
    import importlib, types
    import matplotlib
    matplotlib.use("Agg")

    from scripts.postprocess import generate_map_overlay_figure as mod

    saved_files = []

    def fake_savefig(path, **kw):
        saved_files.append(str(path))

    # Monkeypatch load_scaling and emb_yaml_path to avoid needing EMB YAML
    monkeypatch.setattr(mod, "load_scaling", lambda p: (250.0, 2.5e-3))
    monkeypatch.setattr(mod, "emb_yaml_path", lambda exp, root: tmp_path / "fake.yaml")

    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt.Figure, "savefig", lambda self, path, **kw: saved_files.append(str(path)))

    mod.make_overlay_figure(manifest_path, "indentation", tmp_path, repo_root=REPO)

    assert any("reduced-model" in p for p in saved_files), (
        f"Expected 'reduced-model' in output filenames, got: {saved_files}"
    )
    assert all("map_overlay_indentation_reduced-model" in p for p in saved_files), (
        f"Expected 'map_overlay_indentation_reduced-model' stem, got: {saved_files}"
    )


# ---------------------------------------------------------------------------
# P2 fix 2: relative result_json paths resolved against manifest directory
# ---------------------------------------------------------------------------

def test_make_overlay_figure_resolves_relative_result_paths(tmp_path, monkeypatch):
    """Relative result_json paths must be resolved against the manifest's directory."""
    import sys
    sys.path.insert(0, str(REPO / "src"))

    import matplotlib
    matplotlib.use("Agg")
    from scripts.postprocess import generate_map_overlay_figure as mod

    # Write a fake result JSON at a known location
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result_json = results_dir / "indentation_3.2um_result.json"
    result_json.write_text(json.dumps({
        "displacement_points": [0.1, 0.2],
        "forces": [0.5, 1.0],
        "grid_config": {"d0_offset": 0.05},
    }))

    # Manifest lives in tmp_path; relative path "results/..." should resolve correctly
    manifest = {
        "model_family": "reduced-model",
        "diameters": [
            {
                "status": "passed",
                "dataset_name": "indentation_3.2um",
                "result_json": "results/indentation_3.2um_result.json",
            }
        ],
    }
    manifest_path = tmp_path / "map_mirheo_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded_curves = []

    def fake_load_map_curve(rj, modality, lf, ff):
        loaded_curves.append(rj)
        import numpy as np
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    monkeypatch.setattr(mod, "_load_map_curve", fake_load_map_curve)
    monkeypatch.setattr(mod, "_load_reference", lambda *a, **kw: (__import__("numpy").array([]), __import__("numpy").array([])))
    monkeypatch.setattr(mod, "load_scaling", lambda p: (250.0, 2.5e-3))
    monkeypatch.setattr(mod, "emb_yaml_path", lambda exp, root: tmp_path / "fake.yaml")

    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt.Figure, "savefig", lambda self, path, **kw: None)

    mod.make_overlay_figure(manifest_path, "indentation", tmp_path, repo_root=REPO)

    assert len(loaded_curves) == 1, "Expected MAP curve to be loaded for 3.2um"
    assert loaded_curves[0] == result_json.resolve()
