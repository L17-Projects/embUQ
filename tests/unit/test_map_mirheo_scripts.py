"""Tests for PR 4/6: MAP Mirheo evaluation scripts and validate_training_baseline."""
from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from meso_uq.mirheo.baseline import (
    EXPECTED_TRAINING_FSCALE,
    EXPECTED_TRAINING_NUMSTEPS,
    EXPECTED_TRAINING_NUMSTEPS_EQ,
    EXPECTED_TRAINING_SHELL_TH,
    validate_training_baseline,
)

REPO = Path(__file__).resolve().parents[2]
_IND_SCRIPT = REPO / "propagation" / "scripts" / "evaluate_map_mirheo_optimized_indentation.py"
_CMP_SCRIPT = REPO / "propagation" / "scripts" / "evaluate_map_mirheo_optimized.py"


# ---------------------------------------------------------------------------
# Syntax checks (no Mirheo needed)
# ---------------------------------------------------------------------------

def test_evaluate_indentation_script_parseable():
    py_compile.compile(str(_IND_SCRIPT), doraise=True)


def test_evaluate_compression_script_parseable():
    py_compile.compile(str(_CMP_SCRIPT), doraise=True)


def test_convert_map_manifest_script_parseable():
    script = REPO / "scripts" / "vega" / "convert_map_manifest.py"
    py_compile.compile(str(script), doraise=True)


# ---------------------------------------------------------------------------
# Baseline constants sanity check
# ---------------------------------------------------------------------------

def test_baseline_constants_values():
    assert EXPECTED_TRAINING_FSCALE == pytest.approx(0.0074)
    assert EXPECTED_TRAINING_SHELL_TH == pytest.approx(5.0e-9)
    assert EXPECTED_TRAINING_NUMSTEPS == 5000
    assert EXPECTED_TRAINING_NUMSTEPS_EQ == 10000


# ---------------------------------------------------------------------------
# validate_training_baseline — pure Python, no Mirheo/korali needed
# ---------------------------------------------------------------------------

def _valid_params():
    return {
        "fscale": 0.0074,
        "shell_th": 5.0e-9,
        "numsteps": 5000,
        "numsteps_eq": 10000,
    }


def test_validate_training_baseline_passes():
    validate_training_baseline(_valid_params(), "test")  # must not raise


def test_validate_training_baseline_wrong_fscale():
    params = {**_valid_params(), "fscale": 0.009}
    with pytest.raises(ValueError, match="fscale"):
        validate_training_baseline(params, "test")


def test_validate_training_baseline_wrong_shell_th():
    params = {**_valid_params(), "shell_th": 1.0e-9}
    with pytest.raises(ValueError, match="shell_th"):
        validate_training_baseline(params, "test")


def test_validate_training_baseline_wrong_numsteps():
    params = {**_valid_params(), "numsteps": 999}
    with pytest.raises(ValueError, match="numsteps"):
        validate_training_baseline(params, "test")


def test_validate_training_baseline_wrong_numsteps_eq():
    params = {**_valid_params(), "numsteps_eq": 1}
    with pytest.raises(ValueError, match="numsteps_eq"):
        validate_training_baseline(params, "test")
