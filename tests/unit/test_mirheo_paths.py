"""Tests for meso_uq.mirheo.paths — no Mirheo installation required."""
from __future__ import annotations

from pathlib import Path

import pytest

import meso_uq.mirheo.paths as _paths_module
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


def test_find_equil_script_missing_file_raises_file_not_found(monkeypatch, tmp_path):
    """FileNotFoundError when the equil.py is absent from disk."""
    monkeypatch.setitem(_paths_module._EQUIL_PATHS, "compression", tmp_path / "nonexistent.py")
    with pytest.raises(FileNotFoundError, match="equil.py not found"):
        find_equil_script("compression")


def test_verify_equil_drivers_missing_file_raises_file_not_found(monkeypatch, tmp_path):
    """verify_equil_drivers raises FileNotFoundError if a driver is missing."""
    monkeypatch.setitem(_paths_module._EQUIL_PATHS, "indentation", tmp_path / "gone.py")
    with pytest.raises(FileNotFoundError, match="equil.py not found"):
        verify_equil_drivers()
