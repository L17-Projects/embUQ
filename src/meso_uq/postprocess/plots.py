import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_validation_overlay(reference_csv, prediction_csv, output_path, x_col=None, y_ref_col=None, y_pred_col=None, label_ref="reference", label_pred="prediction"):
    ref = pd.read_csv(reference_csv)
    pred = pd.read_csv(prediction_csv)
    if x_col is None:
        x_col = ref.columns[0]
    if y_ref_col is None:
        y_ref_col = ref.columns[1]
    if y_pred_col is None:
        y_pred_col = pred.columns[1]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(ref[x_col], ref[y_ref_col], marker="o", linestyle="-", label=label_ref)
    plt.plot(pred[x_col], pred[y_pred_col], marker="s", linestyle="--", label=label_pred)
    plt.xlabel(x_col)
    plt.ylabel(y_ref_col)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_d0_correlations(samples_csv, output_path, d0_col="d0"):
    df = pd.read_csv(samples_csv)
    params = [c for c in df.columns if c not in {d0_col, "logLikelihood", "logPrior", "logPosterior"}]
    n = len(params)
    if n == 0:
        raise ValueError("No parameter columns found for d0 correlation plot")
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, param in zip(axes[0], params):
        ax.scatter(df[param], df[d0_col], s=8, alpha=0.5)
        ax.set_xlabel(param)
        ax.set_ylabel(d0_col)
        ax.grid(alpha=0.2)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_posterior_marginals(samples_csv, output_path, exclude=None):
    df = pd.read_csv(samples_csv)
    exclude = set(exclude or ["logLikelihood", "logPrior", "logPosterior"])
    cols = [c for c in df.columns if c not in exclude]
    n = len(cols)
    if n == 0:
        raise ValueError("No posterior columns found to plot")
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col].dropna().to_numpy(), bins=40)
        ax.set_title(col)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
