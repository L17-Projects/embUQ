"""Tests for src/meso_uq/postprocess/paper_figures.py (PR 6)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from meso_uq.postprocess.paper_figures import (
    COLORS,
    LINESTYLES,
    convert_to_physical,
    emb_yaml_path,
    load_scaling,
)

REPO = Path(__file__).resolve().parents[2]
_IND_EMB = REPO / "emb" / "indentation" / "src" / "parameters-default.emb.yaml"
_CMP_EMB = REPO / "emb" / "compression" / "src" / "parameters-default.emb.yaml"


# ---------------------------------------------------------------------------
# emb_yaml_path
# ---------------------------------------------------------------------------

def test_emb_yaml_path_indentation():
    p = emb_yaml_path("indentation", REPO)
    assert p == REPO / "emb" / "indentation" / "src" / "parameters-default.emb.yaml"


def test_emb_yaml_path_compression():
    p = emb_yaml_path("compression", REPO)
    assert p == REPO / "emb" / "compression" / "src" / "parameters-default.emb.yaml"


def test_emb_yaml_path_unknown_raises():
    with pytest.raises(ValueError, match="Unknown modality"):
        emb_yaml_path("unknown", REPO)


# ---------------------------------------------------------------------------
# load_scaling — indentation EMB (ul=0.25e-6, fscale=0.0074)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _IND_EMB.exists(), reason="indentation EMB YAML not present")
def test_load_scaling_indentation_length_factor():
    length_factor, _ = load_scaling(_IND_EMB)
    # ul = 0.25e-6 m → 250 nm
    assert length_factor == pytest.approx(250.0, rel=1e-6)


@pytest.mark.skipif(not _IND_EMB.exists(), reason="indentation EMB YAML not present")
def test_load_scaling_indentation_force_factor_positive():
    _, force_factor = load_scaling(_IND_EMB)
    assert force_factor > 0


@pytest.mark.skipif(not _IND_EMB.exists(), reason="indentation EMB YAML not present")
def test_load_scaling_indentation_force_factor_value():
    """Regression: force_factor for indentation baseline params ≈ 2.24e-3 nN/DPD."""
    _, force_factor = load_scaling(_IND_EMB)
    # Computed from: ue=kbol*t0, um=rho_water*ul^3/rhow, ut=sqrt(um*ul^2/ue)
    # force_factor = (um*ul/ut^2) / fscale * 1e9 = (ue/ul) / fscale * 1e9
    ul = 0.25e-6
    kbol = 1.3805e-23
    t0 = 300.0
    energy_factor = 1.0
    fscale = 0.0074
    ue = energy_factor * kbol * t0
    expected = (ue / ul) / fscale * 1e9
    assert force_factor == pytest.approx(expected, rel=1e-6)


@pytest.mark.skipif(not _CMP_EMB.exists(), reason="compression EMB YAML not present")
def test_load_scaling_compression_length_factor_positive():
    length_factor, force_factor = load_scaling(_CMP_EMB)
    assert length_factor > 0
    assert force_factor > 0


# ---------------------------------------------------------------------------
# convert_to_physical
# ---------------------------------------------------------------------------

def test_convert_compression_axes():
    """Compression: x_dpd→nm (length), y_dpd→nN (force)."""
    lf, ff = 250.0, 2.5e-3
    x_phys, y_phys = convert_to_physical("compression",
                                          np.array([1.0, 2.0]),
                                          np.array([3.0, 4.0]),
                                          lf, ff)
    np.testing.assert_allclose(x_phys, [250.0, 500.0])
    np.testing.assert_allclose(y_phys, [7.5e-3, 10e-3])


def test_convert_indentation_axes():
    """Indentation: x_dpd→nN (force), y_dpd→nm (length) — axes are flipped."""
    lf, ff = 250.0, 2.5e-3
    x_phys, y_phys = convert_to_physical("indentation",
                                          np.array([1.0, 2.0]),
                                          np.array([3.0, 4.0]),
                                          lf, ff)
    # x: force → nN
    np.testing.assert_allclose(x_phys, [2.5e-3, 5e-3])
    # y: displacement → nm
    np.testing.assert_allclose(y_phys, [750.0, 1000.0])


def test_convert_unknown_modality_raises():
    with pytest.raises(ValueError, match="Unknown modality"):
        convert_to_physical("unknown", np.array([1.0]), np.array([1.0]), 1.0, 1.0)


def test_convert_scalar_values():
    """Handles scalar (0-d) arrays."""
    lf, ff = 100.0, 1.0
    x, y = convert_to_physical("compression",
                                np.float64(2.0), np.float64(3.0), lf, ff)
    assert float(x) == pytest.approx(200.0)
    assert float(y) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# COLORS and LINESTYLES palette
# ---------------------------------------------------------------------------

def test_colors_has_both_modalities():
    assert "indentation" in COLORS
    assert "compression" in COLORS


def test_colors_indentation_has_expected_diameters():
    assert set(COLORS["indentation"]) == {"3.2", "3.4", "5.8"}


def test_colors_compression_has_expected_diameters():
    assert set(COLORS["compression"]) == {"2.1", "2.9", "3.0"}


def test_linestyles_has_at_least_three():
    assert len(LINESTYLES) >= 3
