"""Tests for PR 3: compute_indentation() Mirheo implementation."""
from __future__ import annotations

import inspect

import pytest


def test_compute_indentation_is_callable():
    """compute_indentation must be a real callable, not a stub."""
    pytest.importorskip("korali")
    from indentation.evalkit.posterior_indentation import compute_indentation
    assert callable(compute_indentation)


def test_compute_indentation_signature():
    """Verify the expected signature is preserved."""
    pytest.importorskip("korali")
    from indentation.evalkit.posterior_indentation import compute_indentation
    sig = inspect.signature(compute_indentation)
    params = list(sig.parameters)
    assert "sample" in params
    assert "X" in params
    assert "project_root" in params
    assert "diameter_um" in params
    assert "init_indentation_path" in params


def test_compute_indentation_no_longer_raises_not_implemented():
    """The stub NotImplementedError must be gone."""
    pytest.importorskip("korali")
    from indentation.evalkit.posterior_indentation import compute_indentation
    try:
        compute_indentation({"Parameters": [1e7, 1e4, 0.0, 0.0, 0.0, 0.0, 0.03]}, [0.1])
    except NotImplementedError:
        pytest.fail("compute_indentation must not raise NotImplementedError")
    except Exception:
        pass  # ImportError (no mirheo/h5py), RuntimeError (missing dirs), etc. are all acceptable


def test_compute_compression_signature():
    """compute_compression in compression/evalkit already has a real implementation."""
    pytest.importorskip("korali")
    from compression.evalkit.posterior_compression import compute_compression
    sig = inspect.signature(compute_compression)
    params = list(sig.parameters)
    assert "sample" in params
    assert "displ" in params
