"""
Unit tests for the Mirheo equil.py driver modules.

All tests guard against missing Mirheo/mpi4py with pytest.importorskip so
that CI (which runs without Mirheo installed) still passes.
"""
from __future__ import annotations

import inspect
from pathlib import Path


def test_compression_equil_module_exists():
    """The file must be present regardless of Mirheo availability."""
    path = Path(__file__).resolve().parents[2] / "compression" / "src" / "equil.py"
    assert path.exists(), f"compression/src/equil.py not found at {path}"


def test_indentation_equil_module_exists():
    """The file must be present regardless of Mirheo availability."""
    path = Path(__file__).resolve().parents[2] / "indentation" / "src" / "equil.py"
    assert path.exists(), f"indentation/src/equil.py not found at {path}"


def test_compression_equil_importable():
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from compression.src.equil import run_equil  # noqa: F401


def test_indentation_equil_importable():
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from indentation.src.equil import run_equil  # noqa: F401


def test_compression_run_equil_signature():
    """run_equil must accept the expected parameters."""
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from compression.src.equil import run_equil

    sig = inspect.signature(run_equil)
    params = list(sig.parameters)
    for expected in ("source_path", "simu_path", "simnum", "equil", "restart", "restart_path", "comm"):
        assert expected in params, f"Missing parameter '{expected}' in compression run_equil"


def test_indentation_run_equil_signature():
    """run_equil must accept the expected parameters (including vacuum)."""
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from indentation.src.equil import run_equil

    sig = inspect.signature(run_equil)
    params = list(sig.parameters)
    for expected in ("source_path", "simu_path", "simnum", "equil", "restart", "restart_path", "vacuum", "comm"):
        assert expected in params, f"Missing parameter '{expected}' in indentation run_equil"


def test_compression_equil_has_findpids():
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from compression.src import equil
    assert callable(equil.findpids)


def test_indentation_equil_has_findpids():
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from indentation.src import equil
    assert callable(equil.findpids)
