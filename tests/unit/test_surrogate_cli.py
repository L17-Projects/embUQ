"""Tests for meso_uq.surrogate.cli — make_tensors and training-table readers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from meso_uq.surrogate import bnn_training
from meso_uq.surrogate.cli import (
    SOURCE_CURVE_COL,
    build_row_split_manifest,
    make_tensors,
    read_compression_training_table,
    read_indentation_table,
    read_wide_curve_table,
    train_tabular_surrogate,
)


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


def test_read_indentation_table_tracks_cleaned_source_curves(tmp_path: Path) -> None:
    path = tmp_path / "indentation.dat"
    rows = [
        [1.0, 0.0, 2.0, 0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 1.8, 1.6, 0.0, 0.5, 1.0],
        [1.5, 0.0, 2.5, 0.1, 0.2, 0.3, 0.4, 1.0, 0.2, 0.8, 2.0, 0.1, 0.2, 0.3],
    ]
    pd.DataFrame(rows).to_csv(path, sep=" ", header=False, index=False)

    df = read_indentation_table(path, disp_source="auto", rupture_ratio_threshold=2.0)

    assert SOURCE_CURVE_COL in df.columns
    assert set(df.columns) == {"Yt", "kb", "b1", "b2", "a3", "a4", "F", "disp", SOURCE_CURVE_COL}
    assert df[SOURCE_CURVE_COL].nunique() == 2
    assert df[SOURCE_CURVE_COL].tolist() == [0, 0, 0, 1, 1]
    assert df["F"].tolist() == pytest.approx([0.0, 0.5, 1.0, 0.0, 0.1])
    assert df["disp"].tolist() == pytest.approx([0.0, 0.2, 0.4, 0.0, 1.8])


def test_read_compression_training_table_applies_uqdpd_cleaning(tmp_path: Path) -> None:
    path = tmp_path / "compression.dat"
    # 8 header columns + 5 disp + 5 force.
    # Cleaning expectations:
    # - disp<=0 removed
    # - non-monotonic force point at disp=0.3 removed
    # - anchor inserted at disp=0 with force=max(50, 10% first real force)=50
    # - low-force tail at disp=0.4 with F=40 removed because disp>0.2 and F<50
    row = [
        1.0,
        0.0,
        2.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.0,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        200.0,
        20.0,
        30.0,
        10.0,
        40.0,
    ]
    np.savetxt(path, np.array([row]), fmt="%.6f")

    df = read_compression_training_table(str(path), curve_axis_name="disp", value_name="F")

    assert list(df.columns) == ["Yt", "kb", "b1", "b2", "a3", "a4", "disp", "F"]
    assert df["disp"].tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert df["F"].tolist() == pytest.approx([50.0, 20.0, 30.0])


def test_read_compression_training_table_uses_proportional_anchor_above_floor(tmp_path: Path) -> None:
    path = tmp_path / "compression_anchor.dat"
    row = [
        1.0,
        0.0,
        2.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.0,
        0.1,
        0.2,
        0.3,
        60.0,
        80.0,
        120.0,
    ]
    np.savetxt(path, np.array([row]), fmt="%.6f")

    df = read_compression_training_table(str(path), curve_axis_name="disp", value_name="F")
    assert df["disp"].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert df["F"].tolist() == pytest.approx([50.0, 60.0, 80.0, 120.0])


def test_build_row_split_manifest_matches_bnn_split_like_dnn() -> None:
    n_rows = 20
    seed = 1234
    manifest = build_row_split_manifest(n_rows, seed=seed, val_fraction=0.10)

    Xz = torch.arange(n_rows, dtype=torch.float32).reshape(n_rows, 1)
    yz = torch.arange(n_rows, dtype=torch.float32).reshape(n_rows, 1)
    X_phys = np.arange(n_rows, dtype=float).reshape(n_rows, 1)
    y_phys = np.arange(n_rows, dtype=float)
    split = bnn_training._split_like_dnn(Xz, yz, X_phys, y_phys, seed=seed)
    val_row_ids = set(split["X_val_phys"][:, 0].astype(int).tolist())

    expected = pd.DataFrame(
        {
            "row_id": np.arange(n_rows, dtype=int),
            "split": ["validation" if idx in val_row_ids else "train" for idx in range(n_rows)],
            "seed": [seed] * n_rows,
            "val_fraction": [0.10] * n_rows,
        }
    )

    pd.testing.assert_frame_equal(manifest, expected)


def test_train_tabular_surrogate_writes_report_and_is_seed_stable(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "x1": np.linspace(-1.0, 1.0, 16),
            "x2": np.linspace(2.0, 5.0, 16),
            "y": np.linspace(-0.5, 0.5, 16) * 3.0,
        }
    )
    run_a = train_tabular_surrogate(
        df,
        input_cols=["x1", "x2"],
        target_col="y",
        out_path=str(tmp_path / "model_a.pkl"),
        report_path=str(tmp_path / "report_a.json"),
        width=8,
        depth=2,
        batch_size=4,
        lr=1e-3,
        max_epoch=2,
        seed=77,
        val_fraction=0.25,
    )
    run_b = train_tabular_surrogate(
        df,
        input_cols=["x1", "x2"],
        target_col="y",
        out_path=str(tmp_path / "model_b.pkl"),
        report_path=str(tmp_path / "report_b.json"),
        width=8,
        depth=2,
        batch_size=4,
        lr=1e-3,
        max_epoch=2,
        seed=77,
        val_fraction=0.25,
    )

    report = json.loads((tmp_path / "report_a.json").read_text(encoding="utf-8"))
    assert report["seed"] == 77
    assert report["n_train"] == 12
    assert report["n_val"] == 4
    assert report["val_rmse_phys"] >= 0.0
    assert Path(report["loss_history_path"]).exists()
    assert Path(report["out"]).exists()
    assert run_a["train_loss"] == pytest.approx(run_b["train_loss"])
    assert run_a["val_loss"] == pytest.approx(run_b["val_loss"])
