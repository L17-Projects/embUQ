"""Tests that compression static geometry files exist in compression/src/.

These files are required at runtime by posterior_compression.py
(prepare_simulation_parameters) and equil.py — they are copied into the
per-diameter working directories during Mirheo simulation setup.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPRESSION_SRC = REPO_ROOT / "compression" / "src"


@pytest.fixture(params=[
    "mesh/cantilever.off",
    "mesh/rigid_coords.txt",
    "mesh/rigid_coords_reflected.txt",
    "mesh/emb00001eq.off",
    "mesh/gv00001eq.off",
    "gas_vesicle/gv_working.off",
    "gas_vesicle/create_gv.py",
    "gas_vesicle/parameters.yaml",
    "posq.txt",
])
def required_file(request) -> Path:
    return COMPRESSION_SRC / request.param


def test_static_geometry_file_exists(required_file: Path) -> None:
    assert required_file.exists(), f"Missing required file: {required_file.relative_to(REPO_ROOT)}"


def test_static_geometry_file_not_empty(required_file: Path) -> None:
    assert required_file.stat().st_size > 0, f"File is empty: {required_file.relative_to(REPO_ROOT)}"


def test_cantilever_off_has_valid_off_header() -> None:
    path = COMPRESSION_SRC / "mesh" / "cantilever.off"
    first_line = path.read_text().split("\n")[0].strip()
    assert first_line == "OFF", f"Expected OFF header, got: {first_line}"


def test_rigid_coords_has_multiple_rows() -> None:
    path = COMPRESSION_SRC / "mesh" / "rigid_coords.txt"
    lines = path.read_text().strip().split("\n")
    assert len(lines) > 100, f"Expected many coordinate rows, got {len(lines)}"


def test_rigid_coords_reflected_has_multiple_rows() -> None:
    path = COMPRESSION_SRC / "mesh" / "rigid_coords_reflected.txt"
    lines = path.read_text().strip().split("\n")
    assert len(lines) > 100, f"Expected many coordinate rows, got {len(lines)}"
