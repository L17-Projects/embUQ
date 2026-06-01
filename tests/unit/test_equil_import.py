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
    path = Path(__file__).resolve().parents[2] / "emb" / "compression" / "src" / "equil.py"
    assert path.exists(), f"emb/compression/src/equil.py not found at {path}"


def test_indentation_equil_module_exists():
    """The file must be present regardless of Mirheo availability."""
    path = Path(__file__).resolve().parents[2] / "emb" / "indentation" / "src" / "equil.py"
    assert path.exists(), f"emb/indentation/src/equil.py not found at {path}"


def test_indentation_internal_lj_minimum_is_below_cutoff():
    """The internal LJ minimum must stay strictly below Mirheo's cutoff."""
    path = Path(__file__).resolve().parents[2] / "emb" / "indentation" / "src" / "equil.py"
    text = path.read_text(encoding="utf-8")
    assert "lj_int_cutoff = lj_fac * rc" in text
    assert "lj_int_sigma = 0.99 * lj_int_cutoff / (2 ** (1 / 6))" in text


def test_compression_equil_config_lookup_reaches_repo_root_from_moved_source_dir():
    """The moved emb/compression/src driver needs one more parent than compression/src."""
    path = Path(__file__).resolve().parents[2] / "emb" / "compression" / "src" / "equil.py"
    text = path.read_text(encoding="utf-8")
    assert "../../../inference/configs/production/inference_config_compression.yaml" in text
    assert "file_based_root" in text


def test_compression_equil_importable():
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from emb.compression.src.equil import run_equil  # noqa: F401


def test_indentation_equil_importable():
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from emb.indentation.src.equil import run_equil  # noqa: F401


def test_compression_run_equil_signature():
    """run_equil must accept the expected parameters."""
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from emb.compression.src.equil import run_equil

    sig = inspect.signature(run_equil)
    params = list(sig.parameters)
    for expected in ("source_path", "simu_path", "simnum", "equil", "restart", "restart_path", "comm"):
        assert expected in params, f"Missing parameter '{expected}' in compression run_equil"


def test_indentation_run_equil_signature():
    """run_equil must accept the expected parameters (including vacuum)."""
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from emb.indentation.src.equil import run_equil

    sig = inspect.signature(run_equil)
    params = list(sig.parameters)
    for expected in ("source_path", "simu_path", "simnum", "equil", "restart", "restart_path", "vacuum", "comm"):
        assert expected in params, f"Missing parameter '{expected}' in indentation run_equil"


def test_compression_equil_has_findpids():
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from emb.compression.src import equil
    assert callable(equil.findpids)


def test_indentation_equil_has_findpids():
    pytest = __import__("pytest")
    pytest.importorskip("mirheo")
    from emb.indentation.src import equil
    assert callable(equil.findpids)
