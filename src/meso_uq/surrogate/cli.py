import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import MLP, init_weights, save_model_states
from .training import train_model


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


def train_tabular_surrogate(df, *, input_cols, target_col, out_path, width=64, depth=3, batch_size=128, lr=5e-4, max_epoch=100):
    X, y, x_mu, x_sd, y_mu, y_sd = make_tensors(df, input_cols, target_col)
    n = len(X)
    n_val = max(1, int(0.1 * n))
    perm = torch.randperm(n)
    X = X[perm]
    y = y[perm]
    Xv, yv = X[:n_val], y[:n_val]
    Xt, yt = X[n_val:], y[n_val:]
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    model = MLP(input_dims=len(input_cols), output_dims=1, hl_dims=[width] * depth)
    model.apply(init_weights)
    model, tr_hist, va_hist = train_model(model, loader, Xv, yv, lr=lr, max_epoch=max_epoch, info_every=5)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_model_states(model, xshift=x_mu, xscale=x_sd, yshift=y_mu, yscale=y_sd, path=out_path)
    with open(os.path.splitext(out_path)[0] + "_loss_hist.pkl", "wb") as f:
        pickle.dump({"train": tr_hist, "val": va_hist}, f)
    return {"train_loss": tr_hist[-1], "val_loss": va_hist[-1], "out": out_path}
