from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from meso_uq.surrogate.cli import SOURCE_CURVE_COL, read_indentation_table as read_indentation_table_cli
from meso_uq.surrogate.group_holdout import (
    build_curve_split_manifest,
    build_holdout_outputs,
    find_representative_curve,
    predict_family_mean_std,
    read_compression_table,
    read_indentation_table,
    resolve_modality_spec,
    resolve_surrogate_family,
    split_curves,
)
from meso_uq.surrogate.model import save_model_states


def _toy_curve_df() -> pd.DataFrame:
    rows = []
    params = [
        (1.0, 2.0, 0.1, 0.2, 0.3, 0.4),
        (1.5, 2.5, 0.1, 0.2, 0.3, 0.4),
        (2.0, 3.0, 0.1, 0.2, 0.3, 0.4),
        (2.5, 3.5, 0.1, 0.2, 0.3, 0.4),
    ]
    for (Yt, kb, b1, b2, a3, a4) in params:
        for force in (0.0, 1.0, 2.0):
            rows.append(
                {
                    "Yt": Yt,
                    "kb": kb,
                    "b1": b1,
                    "b2": b2,
                    "a3": a3,
                    "a4": a4,
                    "F": force,
                    "disp": force + 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_split_curves_deterministic_and_curve_consistent() -> None:
    df = _toy_curve_df()
    split_a = split_curves(df, val_fraction=0.25, seed=42)
    split_b = split_curves(df, val_fraction=0.25, seed=42)

    mapping_a = (
        split_a[["curve_id", "split"]].drop_duplicates("curve_id").sort_values("curve_id").reset_index(drop=True)
    )
    mapping_b = (
        split_b[["curve_id", "split"]].drop_duplicates("curve_id").sort_values("curve_id").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(mapping_a, mapping_b)

    # No curve can appear in both splits.
    per_curve_split_count = split_a.groupby("curve_id")["split"].nunique()
    assert int(per_curve_split_count.max()) == 1
    assert int((mapping_a["split"] == "validation").sum()) == 1


def test_build_holdout_outputs_and_representative_curve() -> None:
    df_val = pd.DataFrame(
        {
            "curve_id": [0, 0, 1, 1],
            "F": [0.0, 1.0, 0.0, 1.0],
            "disp": [0.0, 1.0, 0.0, 2.0],
        }
    )
    pred_mean = np.array([0.0, 1.2, 0.1, 1.8], dtype=float)
    pred_std = np.array([0.05, 0.10, 0.20, 0.15], dtype=float)

    metrics_df, preds_df = build_holdout_outputs(
        df_val,
        axis_col="F",
        target_col="disp",
        truth_col="disp_true",
        pred_col="disp_pred",
        pred_mean=pred_mean,
        pred_std=pred_std,
    )

    assert set(metrics_df.columns) == {
        "curve_id",
        "n_points",
        "rmse",
        "rel_l2_pct",
        "max_abs_err",
        "pred_std_mean",
        "pred_std_p95",
    }
    assert set(preds_df.columns) == {"curve_id", "F", "disp_true", "disp_pred", "pred_std"}
    assert len(metrics_df) == 2

    rep_curve = find_representative_curve(metrics_df)
    assert rep_curve in {0, 1}


def test_predict_family_mean_std_dnn(tmp_path: Path) -> None:
    model = torch.nn.Linear(7, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 6] = 1.0

    model_path = tmp_path / "toy.pkl"
    save_model_states(
        model,
        xshift=[0.0] * 7,
        xscale=[1.0] * 7,
        yshift=[0.0],
        yscale=[1.0],
        path=str(model_path),
    )
    X = np.array(
        [
            [1.0, 2.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 2.0, 0.1, 0.2, 0.3, 0.4, 1.5],
        ],
        dtype=float,
    )

    mean, std = predict_family_mean_std(family="dnn", model_path=model_path, X_raw=X)
    assert mean.tolist() == [0.5, 1.5]
    assert std.tolist() == [0.0, 0.0]


def test_resolve_helpers_and_split_errors() -> None:
    assert resolve_modality_spec("compression").target_col == "F"
    assert resolve_modality_spec("Indentation").target_col == "disp"
    with pytest.raises(ValueError, match="Unsupported modality"):
        resolve_modality_spec("foo")

    assert resolve_surrogate_family("DNN") == "dnn"
    assert resolve_surrogate_family("bnn") == "bnn"
    with pytest.raises(ValueError, match="Unsupported surrogate family"):
        resolve_surrogate_family("xgb")

    df = _toy_curve_df()
    with pytest.raises(ValueError, match="val_fraction must be in"):
        split_curves(df, val_fraction=0.0, seed=1)
    with pytest.raises(KeyError, match="Missing parameter columns"):
        split_curves(df.drop(columns=["kb"]), val_fraction=0.2, seed=1)
    with pytest.raises(ValueError, match="Need at least 2 unique curves"):
        split_curves(df[df["Yt"] == 1.0], val_fraction=0.2, seed=1)


def test_build_curve_split_manifest_roundtrip_and_guards() -> None:
    df = split_curves(_toy_curve_df(), val_fraction=0.25, seed=7)
    manifest = build_curve_split_manifest(df, seed=7, val_fraction=0.25)
    assert list(manifest.columns) == ["curve_id", "split", "seed", "val_fraction"]
    assert manifest["curve_id"].is_monotonic_increasing
    assert set(manifest["split"]) == {"train", "validation"}
    assert set(manifest["seed"]) == {7}
    assert set(manifest["val_fraction"]) == {0.25}

    with pytest.raises(KeyError, match="curve_id"):
        build_curve_split_manifest(df.drop(columns=["curve_id"]), seed=7, val_fraction=0.25)


def test_split_curves_prefers_cleaned_source_curve_metadata() -> None:
    df = pd.DataFrame(
        {
            "Yt": [1.0, 1.0, 1.0, 1.0],
            "kb": [2.0, 2.0, 2.0, 2.0],
            "b1": [0.1, 0.1, 0.1, 0.1],
            "b2": [0.2, 0.2, 0.2, 0.2],
            "a3": [0.3, 0.3, 0.3, 0.3],
            "a4": [0.4, 0.4, 0.4, 0.4],
            "F": [0.0, 1.0, 0.0, 1.0],
            "disp": [0.0, 0.5, 0.0, 0.6],
            SOURCE_CURVE_COL: [11, 11, 29, 29],
        }
    )

    split_df = split_curves(df, val_fraction=0.5, seed=5)
    manifest = build_curve_split_manifest(split_df, seed=5, val_fraction=0.5)

    assert split_df["curve_id"].nunique() == 2
    assert manifest[SOURCE_CURVE_COL].tolist() == [11, 29]
    assert set(manifest["split"]) == {"train", "validation"}


def test_read_indentation_table_and_compression_table(tmp_path: Path) -> None:
    indentation_path = tmp_path / "indent.dat"
    # 8 scalar columns + 3 displacement/diameter columns + 3 force columns.
    indentation_rows = [
        # valid curve, auto mode infers diameter -> displacement conversion.
        [1.0, 0.0, 2.0, 0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 1.8, 1.6, 0.0, 0.5, 1.0],
        # rupture curve, should be filtered out.
        [1.5, 0.0, 2.5, 0.1, 0.2, 0.3, 0.4, 1.0, 0.2, 0.8, 2.0, 0.1, 0.2, 0.3],
    ]
    pd.DataFrame(indentation_rows).to_csv(indentation_path, sep=" ", header=False, index=False)

    parsed = read_indentation_table(indentation_path, disp_source="auto", rupture_ratio_threshold=2.0)
    parsed_cli = read_indentation_table_cli(
        indentation_path,
        disp_source="auto",
        rupture_ratio_threshold=2.0,
    )
    pd.testing.assert_frame_equal(parsed, parsed_cli)
    assert len(parsed) >= 3
    assert np.isfinite(parsed["disp"]).all()
    assert set(parsed.columns) == {"Yt", "kb", "b1", "b2", "a3", "a4", "F", "disp", SOURCE_CURVE_COL}
    assert parsed[SOURCE_CURVE_COL].nunique() == 2

    parsed_with_rupture_filter = read_indentation_table(
        indentation_path,
        disp_source="displacement",
        rupture_ratio_threshold=2.0,
    )
    assert len(parsed_with_rupture_filter) < len(parsed)

    parsed_no_rupture_filter = read_indentation_table(
        indentation_path,
        disp_source="displacement",
        rupture_ratio_threshold=None,
    )
    assert len(parsed_no_rupture_filter) > len(parsed_with_rupture_filter)

    compression_path = tmp_path / "compression.dat"
    compression_rows = [
        [1.0, 0.0, 2.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.5, 1.0, 60.0, 80.0, 100.0],
    ]
    pd.DataFrame(compression_rows).to_csv(compression_path, sep=" ", header=False, index=False)
    compression_df = read_compression_table(compression_path)
    assert set(compression_df.columns) == {"Yt", "kb", "b1", "b2", "a3", "a4", "disp", "F"}
    assert len(compression_df) == 3
    assert compression_df["disp"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert compression_df["F"].tolist() == pytest.approx([50.0, 80.0, 100.0])


def test_read_indentation_table_rejects_invalid_shapes(tmp_path: Path) -> None:
    too_few = tmp_path / "few.dat"
    pd.DataFrame([[1.0] * 9]).to_csv(too_few, sep=" ", header=False, index=False)
    with pytest.raises(ValueError, match="expected 8 scalars"):
        read_indentation_table(too_few)

    odd_after_header = tmp_path / "odd.dat"
    # 8 + 5 trailing columns -> cannot split equally.
    pd.DataFrame([[1.0] * 13]).to_csv(odd_after_header, sep=" ", header=False, index=False)
    with pytest.raises(ValueError, match="must split into coordinates and forces"):
        read_indentation_table(odd_after_header)


def test_predict_family_mean_std_bnn_branch(monkeypatch) -> None:
    class _Predictor:
        def __init__(self, artifact_path: str, device: str = "cpu") -> None:
            assert artifact_path.endswith(".pt")
            assert device in {"cpu", "gpu"}

        def predict_mean_std(
            self,
            values: np.ndarray,
            *,
            predictive_mc_samples: int,
            predictive_mc_chunk_size: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            assert predictive_mc_samples == 5
            assert predictive_mc_chunk_size == 2
            mean = values[:, 0] + 1.0
            std = np.full(values.shape[0], 0.25, dtype=float)
            return mean, std

    monkeypatch.setattr("meso_uq.surrogate.group_holdout.VariationalBNNPredictor", _Predictor)
    X = np.array([[0.0, 1.0], [1.5, 2.0]], dtype=float)
    mean, std = predict_family_mean_std(
        family="bnn",
        model_path="model.pt",
        X_raw=X,
        predictive_mc_samples=5,
        predictive_mc_chunk_size=2,
        device="gpu",
    )
    assert mean.tolist() == [1.0, 2.5]
    assert std.tolist() == [0.25, 0.25]


def test_build_holdout_outputs_guards() -> None:
    df_val = pd.DataFrame({"curve_id": [0], "F": [0.0], "disp": [0.0]})
    with pytest.raises(ValueError, match="same length"):
        build_holdout_outputs(
            df_val,
            axis_col="F",
            target_col="disp",
            truth_col="disp_true",
            pred_col="disp_pred",
            pred_mean=np.array([0.0, 1.0]),
            pred_std=np.array([0.0]),
        )
    with pytest.raises(KeyError, match="curve_id"):
        build_holdout_outputs(
            df_val.drop(columns=["curve_id"]),
            axis_col="F",
            target_col="disp",
            truth_col="disp_true",
            pred_col="disp_pred",
            pred_mean=np.array([0.0]),
            pred_std=np.array([0.0]),
        )
    with pytest.raises(ValueError, match="empty"):
        find_representative_curve(pd.DataFrame(columns=["curve_id", "rel_l2_pct"]))
