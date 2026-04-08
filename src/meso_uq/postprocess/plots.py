import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _looks_numeric_label(value) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _read_reference_table(reference_csv):
    ref = pd.read_csv(reference_csv)
    if len(ref.columns) >= 2 and all(_looks_numeric_label(column) for column in ref.columns[:2]):
        ref = pd.read_csv(reference_csv, header=None)
    return ref


def plot_validation_overlay(reference_csv, prediction_csv, output_path, x_col=None, y_ref_col=None, y_pred_col=None, label_ref="reference", label_pred="prediction"):
    ref = _read_reference_table(reference_csv)
    pred = pd.read_csv(prediction_csv)
    x_ref_col = ref.columns[0] if x_col is None or x_col not in ref.columns else x_col
    x_pred_col = pred.columns[0] if x_col is None or x_col not in pred.columns else x_col
    if y_ref_col is None:
        y_ref_col = ref.columns[1]
    if y_pred_col is None:
        y_pred_col = pred.columns[1]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(ref[x_ref_col], ref[y_ref_col], marker="o", linestyle="-", label=label_ref)
    plt.plot(pred[x_pred_col], pred[y_pred_col], marker="s", linestyle="--", label=label_pred)
    plt.xlabel(str(x_col if x_col is not None else x_ref_col))
    plt.ylabel(y_ref_col)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_propagation_summary(
    prediction_csv,
    output_path,
    *,
    reference_csv=None,
    x_col="x",
    mean_col="mean",
    q05_col="q05",
    q95_col="q95",
    y_ref_col=None,
    label_ref="reference",
    label_mean="propagation mean",
    envelope_label="propagation q05-q95",
):
    pred = pd.read_csv(prediction_csv)
    ref = _read_reference_table(reference_csv) if reference_csv is not None else None
    if ref is not None and y_ref_col is None:
        y_ref_col = ref.columns[1]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    if ref is not None:
        plt.plot(ref.iloc[:, 0], ref[y_ref_col], marker="o", linestyle="-", label=label_ref)
    plt.plot(pred[x_col], pred[mean_col], marker="s", linestyle="--", label=label_mean)
    if q05_col in pred.columns and q95_col in pred.columns:
        plt.fill_between(pred[x_col], pred[q05_col], pred[q95_col], alpha=0.2, label=envelope_label)
    plt.xlabel(x_col)
    plt.ylabel(y_ref_col if y_ref_col is not None else mean_col)
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
