import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import MLP, init_weights, save_model_states
from .training import train_model

SOURCE_CURVE_COL = "source_curve_id"


def seed_training_runtime(seed: int | None) -> None:
    if seed is None:
        return
    resolved = int(seed)
    random.seed(resolved)
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def _read_wide_curve_arrays(path):
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None).dropna(axis=1, how="all")
    if df.shape[1] < 10:
        raise ValueError(
            f"File '{path}' has {df.shape[1]} columns; expected scalar header plus curve coordinates and values."
        )
    n_after_hdr = df.shape[1] - 8
    if n_after_hdr % 2 != 0:
        raise ValueError(
            f"After the first 8 columns, the remaining ({n_after_hdr}) must split into curve axis and values."
        )
    m = n_after_hdr // 2
    return {
        "Yt": df.iloc[:, 0].to_numpy(float),
        "kb": df.iloc[:, 2].to_numpy(float),
        "b1": df.iloc[:, 3].to_numpy(float),
        "b2": df.iloc[:, 4].to_numpy(float),
        "a3": df.iloc[:, 5].to_numpy(float),
        "a4": df.iloc[:, 6].to_numpy(float),
        "axis": df.iloc[:, 8 : 8 + m].to_numpy(float),
        "vals": df.iloc[:, 8 + m : 8 + 2 * m].to_numpy(float),
        "m": m,
    }


def read_wide_curve_table(path, *, curve_axis_name, value_name):
    payload = _read_wide_curve_arrays(path)
    m = payload["m"]
    out = pd.DataFrame(
        {
            "Yt": np.repeat(payload["Yt"], m),
            "kb": np.repeat(payload["kb"], m),
            "b1": np.repeat(payload["b1"], m),
            "b2": np.repeat(payload["b2"], m),
            "a3": np.repeat(payload["a3"], m),
            "a4": np.repeat(payload["a4"], m),
            curve_axis_name: payload["axis"].reshape(-1),
            value_name: payload["vals"].reshape(-1),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


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
            SOURCE_CURVE_COL: np.repeat(np.arange(n, dtype=np.int64), m_ext),
        }
    ).dropna()
    out[SOURCE_CURVE_COL] = out[SOURCE_CURVE_COL].astype(int)
    return out.reset_index(drop=True)


def _clean_compression_curve(disp_row, force_row):
    valid_mask = np.isfinite(disp_row) & np.isfinite(force_row)
    valid_mask &= force_row < 30000.0
    valid_mask &= disp_row > 0.0

    disp_valid = disp_row[valid_mask]
    force_valid = force_row[valid_mask]
    if len(force_valid) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    sort_idx = np.argsort(disp_valid)
    disp_sorted = disp_valid[sort_idx]
    force_sorted = force_valid[sort_idx]

    keep_indices = [0]
    for j in range(1, len(force_sorted)):
        if force_sorted[j] > force_sorted[keep_indices[-1]]:
            keep_indices.append(j)

    disp_clean = disp_sorted[keep_indices]
    force_clean = force_sorted[keep_indices]

    anchor_force = max(50.0, 0.10 * float(force_clean[0]))
    disp_clean = np.insert(disp_clean, 0, 0.0)
    force_clean = np.insert(force_clean, 0, anchor_force)

    nonzero_mask = force_clean > 0.0
    disp_clean = disp_clean[nonzero_mask]
    force_clean = force_clean[nonzero_mask]

    remove_mask = (disp_clean > 0.2) & (force_clean < 50.0)
    disp_clean = disp_clean[~remove_mask]
    force_clean = force_clean[~remove_mask]
    return disp_clean, force_clean


def read_compression_training_table(path, *, curve_axis_name="disp", value_name="F"):
    payload = _read_wide_curve_arrays(path)
    rows = []
    n = len(payload["Yt"])
    for i in range(n):
        disp_clean, force_clean = _clean_compression_curve(payload["axis"][i, :], payload["vals"][i, :])
        if len(disp_clean) == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "Yt": np.full(len(disp_clean), payload["Yt"][i], dtype=float),
                    "kb": np.full(len(disp_clean), payload["kb"][i], dtype=float),
                    "b1": np.full(len(disp_clean), payload["b1"][i], dtype=float),
                    "b2": np.full(len(disp_clean), payload["b2"][i], dtype=float),
                    "a3": np.full(len(disp_clean), payload["a3"][i], dtype=float),
                    "a4": np.full(len(disp_clean), payload["a4"][i], dtype=float),
                    curve_axis_name: disp_clean,
                    value_name: force_clean,
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=["Yt", "kb", "b1", "b2", "a3", "a4", curve_axis_name, value_name]
        )
    return pd.concat(rows, ignore_index=True)


def split_row_indices(
    n_rows: int,
    *,
    val_fraction: float = 0.10,
    seed: int | None = None,
) -> tuple[torch.Tensor, int]:
    if n_rows < 2:
        raise ValueError("Need at least 2 rows to create train/validation split.")
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in (0, 1).")

    if seed is None:
        perm = torch.randperm(int(n_rows))
    else:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        perm = torch.randperm(int(n_rows), generator=generator)

    n_val = max(1, int(float(val_fraction) * int(n_rows)))
    if n_val >= int(n_rows):
        n_val = int(n_rows) - 1
    return perm, int(n_val)


def build_row_split_manifest(
    n_rows: int,
    *,
    seed: int,
    val_fraction: float = 0.10,
) -> pd.DataFrame:
    perm, n_val = split_row_indices(n_rows, val_fraction=val_fraction, seed=seed)
    row_ids = perm.detach().cpu().numpy().astype(int)
    split_labels = np.array(["train"] * int(n_rows), dtype=object)
    split_labels[:n_val] = "validation"
    manifest = pd.DataFrame(
        {
            "row_id": row_ids,
            "split": split_labels,
        }
    ).sort_values("row_id").reset_index(drop=True)
    manifest["seed"] = int(seed)
    manifest["val_fraction"] = float(val_fraction)
    return manifest


def make_tensors(df, input_cols, target_col):
    X = df[input_cols].to_numpy(float)
    y = df[[target_col]].to_numpy(float)
    x_mu = X.mean(axis=0)
    x_sd = X.std(axis=0)
    x_sd[x_sd == 0] = 1.0
    y_mu = y.mean(axis=0)
    y_sd = y.std(axis=0)
    y_sd[y_sd == 0] = 1.0
    Xz = (X - x_mu) / x_sd
    yz = (y - y_mu) / y_sd
    return torch.tensor(Xz, dtype=torch.float32), torch.tensor(yz, dtype=torch.float32), x_mu.tolist(), x_sd.tolist(), y_mu.tolist(), y_sd.tolist()


def train_tabular_surrogate(
    df,
    *,
    input_cols,
    target_col,
    out_path,
    width=64,
    depth=3,
    batch_size=128,
    lr=5e-4,
    max_epoch=100,
    seed: int | None = None,
    val_fraction: float = 0.10,
    report_path: str | None = None,
):
    seed_training_runtime(seed)
    X, y, x_mu, x_sd, y_mu, y_sd = make_tensors(df, input_cols, target_col)
    y_phys = df[[target_col]].to_numpy(float).reshape(-1)
    n = len(X)
    perm, n_val = split_row_indices(n, val_fraction=val_fraction, seed=seed)
    X = X[perm]
    y = y[perm]
    perm_np = perm.detach().cpu().numpy()
    y_phys = y_phys[perm_np]
    Xv, yv = X[:n_val], y[:n_val]
    Xt, yt = X[n_val:], y[n_val:]
    yv_phys = y_phys[:n_val]
    ds = TensorDataset(Xt, yt)
    loader_generator = None
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed))
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        generator=loader_generator,
    )
    model = MLP(input_dims=len(input_cols), output_dims=1, hl_dims=[width] * depth)
    model.apply(init_weights)
    model, tr_hist, va_hist = train_model(model, loader, Xv, yv, lr=lr, max_epoch=max_epoch, info_every=5)
    model.eval()
    with torch.inference_mode():
        yv_pred_norm = model(Xv).detach().cpu().numpy().reshape(-1)
    yv_pred_phys = np.maximum(0.0, (yv_pred_norm * y_sd[0]) + y_mu[0])
    val_rmse_phys = float(np.sqrt(np.mean(np.square(yv_pred_phys - yv_phys))))
    out_file = Path(out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_model_states(model, xshift=x_mu, xscale=x_sd, yshift=y_mu, yscale=y_sd, path=str(out_file))
    loss_hist_path = out_file.with_name(f"{out_file.stem}_loss_hist.pkl")
    with loss_hist_path.open("wb") as f:
        pickle.dump({"train": tr_hist, "val": va_hist}, f)
    result = {
        "train_loss": float(tr_hist[-1]),
        "val_loss": float(va_hist[-1]),
        "out": str(out_file),
        "loss_history_path": str(loss_hist_path),
        "n_train": int(len(Xt)),
        "n_val": int(len(Xv)),
        "val_rmse_phys": float(val_rmse_phys),
        "seed": None if seed is None else int(seed),
        "val_fraction": float(val_fraction),
        "width": int(width),
        "depth": int(depth),
        "batch_size": int(min(batch_size, len(ds))),
        "lr": float(lr),
        "max_epoch": int(max_epoch),
    }
    if report_path is not None:
        report_file = Path(report_path).resolve()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
