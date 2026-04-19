"""Tests for PR 2: indentation template files and radp lookup."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from meso_uq.mirheo.radp import RADP_LOOKUP, get_radp, infer_radp_for_diameter

_REPO = Path(__file__).resolve().parents[2]
_IND_SRC = _REPO / "indentation" / "src"


# ---------------------------------------------------------------------------
# EMB baseline validation (indentation/src/parameters-default.emb.yaml)
# ---------------------------------------------------------------------------

def _emb() -> dict:
    return yaml.safe_load((_IND_SRC / "parameters-default.emb.yaml").read_text())


def test_emb_baseline_fscale():
    assert _emb()["fscale"] == pytest.approx(0.0074)


def test_emb_baseline_shell_th():
    assert _emb()["shell_th"] == pytest.approx(5.0e-9)


def test_emb_baseline_numsteps():
    emb = _emb()
    assert emb["numsteps"] == 5000
    assert emb["numsteps_eq"] == 10000


# ---------------------------------------------------------------------------
# Parameter template files
# ---------------------------------------------------------------------------

def test_parameter_template_dir_exists():
    param_dir = _IND_SRC / "parameter"
    assert param_dir.is_dir()
    assert (param_dir / "parameters-default00001.yaml").exists()


def test_all_six_parameter_templates_present():
    param_dir = _IND_SRC / "parameter"
    expected = [
        "parameters-default00001.yaml",
        "parameters-default00001eq.yaml",
        "parameters.prms00001.yaml",
        "parameters.prms00001eq.yaml",
        "parameters00001.yaml",
        "parameters00001eq.yaml",
    ]
    for name in expected:
        assert (param_dir / name).exists(), f"Missing: {name}"


def test_posq_file_exists():
    assert (_IND_SRC / "posq.txt").exists()


def test_posq_file_has_seven_columns():
    """posq.txt stores position + quaternion: x y z qw qx qy qz."""
    lines = [ln for ln in (_IND_SRC / "posq.txt").read_text().splitlines() if ln.strip()]
    assert len(lines) >= 1
    assert len(lines[0].split()) == 7


# ---------------------------------------------------------------------------
# radp lookup table
# ---------------------------------------------------------------------------

def test_radp_lookup_indentation_keys():
    assert set(RADP_LOOKUP["indentation"]) == {3.2, 3.4, 5.8}


def test_radp_lookup_values():
    tbl = RADP_LOOKUP["indentation"]
    assert tbl[3.2] == pytest.approx(6.38)
    assert tbl[3.4] == pytest.approx(6.80)
    assert tbl[5.8] == pytest.approx(11.52)


def test_get_radp_returns_correct_values():
    assert get_radp("indentation", 3.2) == pytest.approx(6.38)
    assert get_radp("indentation", 3.4) == pytest.approx(6.80)
    assert get_radp("indentation", 5.8) == pytest.approx(11.52)


def test_get_radp_unknown_experiment_raises():
    with pytest.raises(ValueError, match="Unknown experiment"):
        get_radp("compression", 3.2)


def test_get_radp_unknown_diameter_raises():
    with pytest.raises(ValueError, match="No radp entry"):
        get_radp("indentation", 2.1)


def test_infer_radp_returns_lookup_when_known():
    assert infer_radp_for_diameter("indentation", 3.2) == pytest.approx(6.38)


def test_infer_radp_fallback_for_unknown_diameter():
    result = infer_radp_for_diameter("indentation", 4.0)
    assert result == pytest.approx(4.0 / 0.5)


def test_infer_radp_fallback_for_unknown_experiment():
    result = infer_radp_for_diameter("compression", 2.1)
    assert result == pytest.approx(2.1 / 0.5)
