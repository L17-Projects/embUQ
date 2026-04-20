"""Tests for meso_uq.surrogate.cli — make_tensors and read_wide_curve_table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from meso_uq.surrogate.cli import make_tensors, read_wide_curve_table


# ---------------------------------------------------------------------------
# make_tensors
# ---------------------------------------------------------------------------


def test_make_tensors_standardizes_correctly() -> None:
    df = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [10.0, 20.0, 30.0], "y": [100.0, 200.0, 300.0]})
    X, y, x_mu, x_sd, y_mu, y_sd = make_tensors(df, ["a", "b"], "y")

    assert X.shape == (3, 2)
    assert y.shape == (3, 1)

    # standardized values should have zero mean
    assert X[:, 0].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert X[:, 1].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert y.mean().item() == pytest.approx(0.0, abs=1e-6)

    assert x_mu == pytest.approx([3.0, 20.0])
    assert y_mu == pytest.approx([200.0])


def test_make_tensors_handles_zero_std() -> None:
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "y": [1.0, 2.0, 3.0]})
    X, y, x_mu, x_sd, y_mu, y_sd = make_tensors(df, ["a"], "y")

    # zero std should be replaced with 1.0
    assert x_sd == pytest.approx([1.0])
    # result should be (5-5)/1 = 0
    assert X[:, 0].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_make_tensors_returns_float32_tensors() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "y": [3.0, 4.0]})
    X, y, *_ = make_tensors(df, ["a"], "y")
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# read_wide_curve_table
# ---------------------------------------------------------------------------


def test_read_wide_curve_table_parses_valid_file(tmp_path: Path) -> None:
    # 8 header columns + 2 curve axis + 2 values = 12 total
    lines = [
        "1.0 0.0 2.0 0.1 0.2 0.3 0.4 0.0  10.0 20.0  100.0 200.0\n",
        "3.0 0.0 4.0 0.5 0.6 0.7 0.8 0.0  30.0 40.0  300.0 400.0\n",
    ]
    path = tmp_path / "curves.dat"
    path.write_text("".join(lines))

    df = read_wide_curve_table(str(path), curve_axis_name="force", value_name="displacement")

    assert "Yt" in df.columns
    assert "force" in df.columns
    assert "displacement" in df.columns
    # 2 rows × 2 curve points = 4 output rows
    assert len(df) == 4
    assert df["Yt"].iloc[0] == pytest.approx(1.0)


def test_read_wide_curve_table_rejects_too_few_columns(tmp_path: Path) -> None:
    path = tmp_path / "short.dat"
    path.write_text("1.0 2.0 3.0\n")

    with pytest.raises(ValueError, match="columns"):
        read_wide_curve_table(str(path), curve_axis_name="x", value_name="y")


def test_read_wide_curve_table_rejects_odd_curve_columns(tmp_path: Path) -> None:
    # 8 header + 3 extra = 11 total (odd remainder)
    path = tmp_path / "odd.dat"
    path.write_text("1.0 0.0 2.0 0.1 0.2 0.3 0.4 0.0  10.0 20.0 30.0\n")

    with pytest.raises(ValueError, match="split into curve axis"):
        read_wide_curve_table(str(path), curve_axis_name="x", value_name="y")


def test_read_wide_curve_table_drops_inf_and_nan(tmp_path: Path) -> None:
    lines = [
        "1.0 0.0 2.0 0.1 0.2 0.3 0.4 0.0  10.0 20.0  100.0 inf\n",
    ]
    path = tmp_path / "inf.dat"
    path.write_text("".join(lines))

    df = read_wide_curve_table(str(path), curve_axis_name="x", value_name="y")
    # 2 curve points, but 1 has inf → dropped → 1 row
    assert len(df) == 1
