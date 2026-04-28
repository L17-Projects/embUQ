"""Tests for scripts/platforms/vega/convert_map_manifest.py (PR 4/6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "platforms" / "vega"))

from convert_map_manifest import (
    _FULL_OUTPUT_NAMES,
    _REDUCED_OUTPUT_NAMES,
    convert_dataset,
    convert_manifest,
    detect_param_layout,
)


# ---------------------------------------------------------------------------
# detect_param_layout
# ---------------------------------------------------------------------------

def _full_dataset():
    return {
        "Yt": 1e7, "kb": 1e4, "b1": 0.1, "b2": 0.2, "a3": 0.3, "a4": 0.4,
        "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/run", "output_csv": "/tmp/out.csv",
    }


def _reduced_dataset():
    return {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/run", "output_csv": "/tmp/out.csv",
    }


def test_detect_param_layout_full():
    source_keys, output_names = detect_param_layout(_full_dataset())
    assert source_keys == ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]
    assert output_names == _FULL_OUTPUT_NAMES


def test_detect_param_layout_reduced():
    source_keys, output_names = detect_param_layout(_reduced_dataset())
    assert source_keys == ["Yt", "kb", "d0", "sigma"]
    assert output_names == _REDUCED_OUTPUT_NAMES


def test_detect_param_layout_unknown_raises():
    with pytest.raises(ValueError, match="Cannot determine"):
        detect_param_layout({"diameter_um": 3.2, "logPosterior": 1.0})


# ---------------------------------------------------------------------------
# convert_dataset
# ---------------------------------------------------------------------------

def test_convert_reduced_dataset():
    result = convert_dataset("indentation_3.2um", _reduced_dataset())
    assert result["diameter_um"] == pytest.approx(3.2)
    assert result["sample_id"] == 0
    assert result["generation"] == -1
    assert result["logPosterior"] == pytest.approx(5.0)
    assert result["parameter_names"] == ["Yt", "kb", "d0", "[Sigma]"]
    assert result["parameters"] == pytest.approx([1e7, 1e4, 0.1, 0.03])


def test_convert_full_dataset():
    result = convert_dataset("indentation_3.2um", _full_dataset())
    assert result["parameter_names"] == _FULL_OUTPUT_NAMES
    assert len(result["parameters"]) == 8
    assert result["parameters"][0] == pytest.approx(1e7)   # Yt
    assert result["parameters"][-1] == pytest.approx(0.03)  # sigma → [Sigma]


def test_convert_preserves_log_posterior():
    ds = _reduced_dataset()
    ds["logPosterior"] = 42.5
    result = convert_dataset("foo_3.2um", ds)
    assert result["logPosterior"] == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# convert_manifest (end-to-end with temp files)
# ---------------------------------------------------------------------------

def test_convert_manifest_reduced(tmp_path):
    manifest = {
        "experiment": "indentation",
        "datasets": {
            "indentation_3.2um": _reduced_dataset(),
        },
    }
    manifest_path = tmp_path / "phase3b_map_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    written = convert_manifest(manifest_path, tmp_path / "out")
    assert len(written) == 1
    out_file = written[0]
    assert out_file.name == "indentation_3.2um_map.json"
    result = json.loads(out_file.read_text())
    assert result["parameter_names"] == ["Yt", "kb", "d0", "[Sigma]"]
    assert result["diameter_um"] == pytest.approx(3.2)


def test_convert_manifest_full(tmp_path):
    manifest = {
        "experiment": "indentation",
        "datasets": {
            "indentation_3.4um": _full_dataset(),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    written = convert_manifest(manifest_path, tmp_path / "out")
    result = json.loads(written[0].read_text())
    assert result["parameter_names"] == _FULL_OUTPUT_NAMES
    assert len(result["parameters"]) == 8


def test_convert_manifest_output_filenames(tmp_path):
    manifest = {
        "datasets": {
            "compression_2.1um": _reduced_dataset(),
            "compression_2.9um": _reduced_dataset(),
        }
    }
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    written = convert_manifest(manifest_path, tmp_path / "out")
    names = {p.name for p in written}
    assert "compression_2.1um_map.json" in names
    assert "compression_2.9um_map.json" in names


def test_convert_manifest_empty_raises(tmp_path):
    manifest_path = tmp_path / "empty.json"
    manifest_path.write_text(json.dumps({"datasets": {}}))
    with pytest.raises(ValueError, match="no 'datasets'"):
        convert_manifest(manifest_path, tmp_path / "out")


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

def _write_manifest(path: Path, datasets: dict) -> None:
    path.write_text(json.dumps({"datasets": datasets}))


def test_main_returns_zero_on_success(tmp_path):
    from convert_map_manifest import main

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, {"indentation_3.2um": _reduced_dataset()})
    rc = main(["--manifest", str(manifest_path), "--output-dir", str(tmp_path / "out")])
    assert rc == 0


def test_main_returns_one_when_manifest_missing(tmp_path):
    from convert_map_manifest import main

    rc = main(["--manifest", str(tmp_path / "missing.json"),
               "--output-dir", str(tmp_path / "out")])
    assert rc == 1
