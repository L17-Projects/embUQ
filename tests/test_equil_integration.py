"""
Integration-level guard test for the Mirheo equil.py drivers.

Verifies that when Mirheo is not installed the module import fails with
ImportError (not some unexpected crash) and that the test infrastructure
handles this gracefully.
"""
from __future__ import annotations

import importlib
import sys


def test_equil_skipped_without_mirheo():
    """When mirheo is not importable, importing the equil module raises ImportError."""
    if "mirheo" in sys.modules:
        import pytest
        pytest.skip("mirheo is installed — skip absence test")

    # Attempt to import; must raise ImportError (or ModuleNotFoundError, a subclass)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_compression_equil_test",
            str(__import__("pathlib").Path(__file__).resolve().parents[1] / "emb" / "compression" / "src" / "equil.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # If we reach here, mirheo must actually be importable — skip
        import pytest
        pytest.skip("mirheo appears to be importable — skipping absence test")
    except ImportError:
        pass  # Expected: mirheo (or mpi4py/trimesh) not installed
