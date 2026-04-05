import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import MLP, init_weights, save_model_states
from .training import train_model


def read_wide_curve_table(path, *, curve_axis_name, value_name):
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None).dropna(axis=1, how="all")
    if df.shape[1] < 10:
        raise ValueError(f"File '{path}' has {df.shape[1]} columns; expected scalar header plus curve coordinates and values.")
    n_after_hdr = df.shape[1] - 8
    if n_after_hdr % 2 != 0:
        raise ValueError(f"After the first 8 columns, the remaining ({n_after_hdr}) must split into curve axis and values.")
    m = n_after_hdr // 2
    Yt = df.iloc[:, 0].to_numpy(float)
    kb = df.iloc[:, 2].to_numpy(float)
    b1 = df.iloc[:, 3].to_numpy(float)
    b2 = df.iloc[:, 4].to_numpy(float)
    a3 = df.iloc[:, 5].to_numpy(float)
    a4 = df.iloc[:, 6].to_numpy(float)
    axis = df.iloc[:, 8 : 8 + m].to_numpy(float)
    vals = df.iloc[:, 8 + m : 8 + 2 * m].to_numpy(float)
    out = pd.DataFrame(
        {
            "Yt": np.repeat(Yt, m),
            "kb": np.repeat(kb, m),
            "b1": np.repeat(b1, m),
            "b2": np.repeat(b2, m),
            "a3": np.repeat(a3, m),
            "a4": np.repeat(a4, m),
            curve_axis_name: axis.reshape(-1),
            value_name: vals.reshape(-1),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


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
