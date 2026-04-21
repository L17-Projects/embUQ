from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch

from .bnn import VariationalBNNPredictor
from .cli import read_wide_curve_table
from .model import load_model_states

SurrogateFamily = Literal["dnn", "bnn"]
Modality = Literal["indentation", "compression"]

PARAM_COLS: tuple[str, ...] = ("Yt", "kb", "b1", "b2", "a3", "a4")


@dataclass(frozen=True)
class ModalitySpec:
    modality: Modality
    input_cols: tuple[str, ...]
    axis_col: str
    target_col: str
    truth_col: str
    pred_col: str
    best_artifact_prefix: str


_MODALITY_SPECS: dict[Modality, ModalitySpec] = {
    "indentation": ModalitySpec(
        modality="indentation",
        input_cols=(*PARAM_COLS, "F"),
        axis_col="F",
        target_col="disp",
        truth_col="disp_true",
        pred_col="disp_pred",
        best_artifact_prefix="microbubble_displacement",
    ),
    "compression": ModalitySpec(
        modality="compression",
        input_cols=(*PARAM_COLS, "disp"),
        axis_col="disp",
        target_col="F",
        truth_col="F_true",
        pred_col="F_pred",
        best_artifact_prefix="microbubble_force",
    ),
}


def resolve_modality_spec(modality: str) -> ModalitySpec:
    key = modality.strip().lower()
    if key not in _MODALITY_SPECS:
        allowed = ", ".join(sorted(_MODALITY_SPECS.keys()))
        raise ValueError(f"Unsupported modality={modality!r}. Expected one of: {allowed}.")
    return _MODALITY_SPECS[key]  # type: ignore[index]


def resolve_surrogate_family(family: str) -> SurrogateFamily:
    key = family.strip().lower()
    if key not in {"dnn", "bnn"}:
        raise ValueError(f"Unsupported surrogate family={family!r}. Expected 'dnn' or 'bnn'.")
    return key  # type: ignore[return-value]


def _infer_displacement(values: np.ndarray, radp: np.ndarray, mode: str) -> np.ndarray:
    if mode == "displacement":
        return values
    if mode == "diameter":
        return (2.0 * radp[:, None]) - values
    median_val = float(np.nanmedian(values))
    median_radp = float(np.nanmedian(radp))
    if median_val > (median_radp * 1.5):
        return (2.0 * radp[:, None]) - values
    return values


def _is_rupture_curve(disp_sorted: np.ndarray, ratio_threshold: float, min_disp: float = 0.1) -> bool:
    for idx in range(1, len(disp_sorted)):
        prev = float(disp_sorted[idx - 1])
        curr = float(disp_sorted[idx])
        if prev > min_disp and (curr / prev) > ratio_threshold:
            return True
    return False


def read_indentation_table(
    path: str | Path,
    *,
    disp_source: str = "auto",
    rupture_ratio_threshold: float | None = 2.0,
) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", engine="python", header=None).dropna(axis=1, how="all")
    if raw.shape[1] < 10:
        raise ValueError(
            f"File {str(path)!r} has {raw.shape[1]} columns; expected 8 scalars + curve coordinates + forces."
        )
    n_after_hdr = raw.shape[1] - 8
    if n_after_hdr % 2 != 0:
        raise ValueError(
            f"After the first 8 columns, remaining {n_after_hdr} columns must split into coordinates and forces."
        )

    m = n_after_hdr // 2
    n = int(len(raw))
    Yt = raw.iloc[:, 0].to_numpy(float)
    kb = raw.iloc[:, 2].to_numpy(float)
    b1 = raw.iloc[:, 3].to_numpy(float)
    b2 = raw.iloc[:, 4].to_numpy(float)
    a3 = raw.iloc[:, 5].to_numpy(float)
    a4 = raw.iloc[:, 6].to_numpy(float)
    radp = raw.iloc[:, 7].to_numpy(float)
    disp_or_diam = raw.iloc[:, 8 : 8 + m].to_numpy(float)
    forc = raw.iloc[:, 8 + m : 8 + (2 * m)].to_numpy(float)
    disp = _infer_displacement(disp_or_diam, radp, disp_source)

    m_ext = m + 1
    disp_ext = np.full((n, m_ext), np.nan, dtype=float)
    forc_ext = np.full((n, m_ext), np.nan, dtype=float)
    for i in range(n):
        valid = np.isfinite(disp[i, :]) & np.isfinite(forc[i, :])
        valid &= forc[i, :] >= 0.0
        valid &= disp[i, :] >= 0.0
        disp_valid = disp[i, valid]
        force_valid = forc[i, valid]
        sort_idx = np.argsort(force_valid)
        disp_sorted = disp_valid[sort_idx]
        force_sorted = force_valid[sort_idx]
        if len(force_sorted) == 0:
            continue

        keep_indices = [0]
        for j in range(1, len(force_sorted)):
            if disp_sorted[j] > disp_sorted[keep_indices[-1]]:
                keep_indices.append(j)
        disp_clean = disp_sorted[keep_indices]
        force_clean = force_sorted[keep_indices]

        if rupture_ratio_threshold is not None and _is_rupture_curve(disp_clean, rupture_ratio_threshold):
            continue

        if len(disp_clean) > 0 and force_clean[0] > 0.0:
            disp_clean = np.insert(disp_clean, 0, 0.0)
            force_clean = np.insert(force_clean, 0, 0.0)

        disp_ext[i, : len(disp_clean)] = disp_clean
        forc_ext[i, : len(force_clean)] = force_clean

    out = pd.DataFrame(
        {
            "Yt": np.repeat(Yt, m_ext),
            "kb": np.repeat(kb, m_ext),
            "b1": np.repeat(b1, m_ext),
            "b2": np.repeat(b2, m_ext),
            "a3": np.repeat(a3, m_ext),
            "a4": np.repeat(a4, m_ext),
            "F": forc_ext.reshape(-1),
            "disp": disp_ext.reshape(-1),
        }
    ).dropna()
    return out.reset_index(drop=True)


def read_compression_table(path: str | Path) -> pd.DataFrame:
    return read_wide_curve_table(str(path), curve_axis_name="disp", value_name="F")


def split_curves(
    df: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
    param_cols: Sequence[str] = PARAM_COLS,
) -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("Cannot split empty dataframe.")
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in (0, 1).")

    missing = [col for col in param_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing parameter columns for curve split: {missing}.")

    result = df.copy()
    result["curve_id"] = result.groupby(list(param_cols), sort=False).ngroup()
    n_curves = int(result["curve_id"].nunique())
    if n_curves < 2:
        raise ValueError("Need at least 2 unique curves for grouped holdout split.")

    rng = np.random.default_rng(int(seed))
    curve_ids = np.arange(n_curves, dtype=int)
    rng.shuffle(curve_ids)
    n_val = max(1, int(val_fraction * n_curves))
    if n_val >= n_curves:
        n_val = n_curves - 1
    val_ids = set(curve_ids[:n_val].tolist())
    result["split"] = result["curve_id"].map(lambda cid: "validation" if int(cid) in val_ids else "train")
    return result


def build_curve_split_manifest(df_with_split: pd.DataFrame, *, seed: int, val_fraction: float) -> pd.DataFrame:
    if "curve_id" not in df_with_split.columns or "split" not in df_with_split.columns:
        raise KeyError("df_with_split must contain 'curve_id' and 'split' columns.")
    manifest = (
        df_with_split[["curve_id", "split"]]
        .drop_duplicates("curve_id")
        .sort_values("curve_id")
        .reset_index(drop=True)
    )
    manifest["seed"] = int(seed)
    manifest["val_fraction"] = float(val_fraction)
    return manifest


def _resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized == "cuda" and not torch.cuda.is_available():
        normalized = "cpu"
    return torch.device(normalized)


def predict_dnn_mean_std(
    *,
    model_path: str | Path,
    X_raw: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model, xshift, xscale, yshift, yscale = load_model_states(str(model_path))
    dev = _resolve_device(device)
    if hasattr(model, "to"):
        model = model.to(dev)
    if hasattr(model, "eval"):
        model.eval()

    xshift_arr = np.asarray(xshift, dtype=np.float64)
    xscale_arr = np.asarray(xscale, dtype=np.float64)
    yshift_arr = np.asarray(yshift, dtype=np.float64)
    yscale_arr = np.asarray(yscale, dtype=np.float64)

    X_norm = (X_raw - xshift_arr) / xscale_arr
    X_t = torch.as_tensor(X_norm, dtype=torch.float32, device=dev)
    with torch.inference_mode():
        pred_norm = model(X_t).detach().cpu().numpy().reshape(-1)
    pred = np.maximum(0.0, (pred_norm * yscale_arr[0]) + yshift_arr[0]).astype(np.float64)
    std = np.zeros_like(pred)
    return pred, std


def predict_bnn_mean_std(
    *,
    artifact_path: str | Path,
    X_raw: np.ndarray,
    predictive_mc_samples: int,
    predictive_mc_chunk_size: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    predictor = VariationalBNNPredictor(str(artifact_path), device=device)
    mean, std = predictor.predict_mean_std(
        np.asarray(X_raw, dtype=np.float32),
        predictive_mc_samples=int(predictive_mc_samples),
        predictive_mc_chunk_size=int(predictive_mc_chunk_size),
    )
    mean = np.maximum(0.0, np.asarray(mean, dtype=np.float64))
    std = np.asarray(std, dtype=np.float64)
    return mean, std


def predict_family_mean_std(
    *,
    family: str,
    model_path: str | Path,
    X_raw: np.ndarray,
    predictive_mc_samples: int = 64,
    predictive_mc_chunk_size: int = 8,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    resolved = resolve_surrogate_family(family)
    if resolved == "dnn":
        return predict_dnn_mean_std(model_path=model_path, X_raw=X_raw, device=device)
    return predict_bnn_mean_std(
        artifact_path=model_path,
        X_raw=X_raw,
        predictive_mc_samples=predictive_mc_samples,
        predictive_mc_chunk_size=predictive_mc_chunk_size,
        device=device,
    )


def build_holdout_outputs(
    df_val: pd.DataFrame,
    *,
    axis_col: str,
    target_col: str,
    truth_col: str,
    pred_col: str,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "curve_id" not in df_val.columns:
        raise KeyError("df_val must contain curve_id column.")
    if len(df_val) != int(len(pred_mean)) or len(df_val) != int(len(pred_std)):
        raise ValueError("Prediction arrays must have same length as df_val.")

    pred_df = df_val[["curve_id", axis_col, target_col]].copy().reset_index(drop=True)
    pred_df.rename(columns={target_col: truth_col}, inplace=True)
    pred_df[pred_col] = np.asarray(pred_mean, dtype=np.float64)
    pred_df["pred_std"] = np.asarray(pred_std, dtype=np.float64)

    rows: list[dict[str, float | int]] = []
    for curve_id, grp in pred_df.groupby("curve_id"):
        y_true = grp[truth_col].to_numpy(float)
        y_pred = grp[pred_col].to_numpy(float)
        y_std = grp["pred_std"].to_numpy(float)
        residuals = y_pred - y_true
        rmse = float(np.sqrt(np.mean(residuals**2)))
        denom = float(np.sqrt(np.sum(y_true**2)))
        rel_l2_pct = float(100.0 * np.sqrt(np.sum(residuals**2)) / denom) if denom > 0 else float("nan")
        max_abs_err = float(np.max(np.abs(residuals)))
        rows.append(
            {
                "curve_id": int(curve_id),
                "n_points": int(len(grp)),
                "rmse": rmse,
                "rel_l2_pct": rel_l2_pct,
                "max_abs_err": max_abs_err,
                "pred_std_mean": float(np.mean(y_std)),
                "pred_std_p95": float(np.percentile(y_std, 95)),
            }
        )
    metrics_df = pd.DataFrame(rows).sort_values("curve_id").reset_index(drop=True)
    return metrics_df, pred_df.sort_values(["curve_id", axis_col]).reset_index(drop=True)


def find_representative_curve(metrics_df: pd.DataFrame) -> int:
    if len(metrics_df) == 0:
        raise ValueError("metrics_df is empty.")
    med = float(metrics_df["rel_l2_pct"].median())
    idx = (metrics_df["rel_l2_pct"] - med).abs().idxmin()
    return int(metrics_df.loc[idx, "curve_id"])

