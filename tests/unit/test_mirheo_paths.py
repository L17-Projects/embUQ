"""Tests for meso_uq.mirheo.paths — no Mirheo installation required."""
from __future__ import annotations

from pathlib import Path

import pytest

from meso_uq.mirheo.paths import find_equil_script, verify_equil_drivers


def test_find_equil_script_compression_returns_existing_path():
    path = find_equil_script("compression")
    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "equil.py"
    assert "compression" in str(path)


def test_find_equil_script_indentation_returns_existing_path():
    path = find_equil_script("indentation")
    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "equil.py"
    assert "indentation" in str(path)


def test_find_equil_script_unknown_experiment_raises_value_error():
    with pytest.raises(ValueError, match="Unknown experiment"):
        find_equil_script("stretching")


def test_verify_equil_drivers_returns_both_paths():
    result = verify_equil_drivers()
    assert set(result) == {"compression", "indentation"}
    for path_str in result.values():
        assert Path(path_str).exists()


def test_verify_equil_drivers_files_are_syntactically_valid():
    """py_compile must succeed — drivers must be valid Python."""
    result = verify_equil_drivers()
    assert len(result) == 2  # both checked without raising PyCompileError
