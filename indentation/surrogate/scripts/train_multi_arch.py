#!/usr/bin/env python3
"""
Multi-architecture training script for the indentation displacement surrogate.

This is a richer diagnostic utility than the lightweight public model-selection
wrapper: it keeps cleaning heuristics, summary tables, and a diagnostic summary
plot that are useful during refresh/debug loops.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from meso_uq.surrogate.model import MLP, init_weights, save_model_states
from meso_uq.surrogate.training import train_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SURROGATE_ROOT = PROJECT_ROOT / "indentation" / "surrogate"


def get_architectures():
    return [
        (32, 2, "w32_d2"),
        (32, 3, "w32_d3"),
        (32, 4, "w32_d4"),
        (64, 2, "w64_d2"),
        (64, 3, "w64_d3"),
        (64, 4, "w64_d4"),
        (64, 5, "w64_d5"),
        (128, 2, "w128_d2"),
        (128, 3, "w128_d3"),
        (128, 4, "w128_d4"),
        (256, 2, "w256_d2"),
        (256, 3, "w256_d3"),
    ]


def _infer_displacement(values, radp, mode):
    if mode == "displacement":
        return values
    if mode == "diameter":
        return (2.0 * radp[:, None]) - values
    median_val = np.nanmedian(values)
    median_radp = np.nanmedian(radp)
    if median_val > (median_radp * 1.5):
        return (2.0 * radp[:, None]) - values
    return values


def _is_rupture_curve(disp_sorted, ratio_threshold, min_disp=0.1):
    for j in range(1, len(disp_sorted)):
        if disp_sorted[j - 1] > min_disp and disp_sorted[j] / disp_sorted[j - 1] > ratio_threshold:
            return True
    return False


def read_new_dat_to_long(path, disp_source="auto", rupture_ratio_threshold=2.0):
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None).dropna(axis=1, how="all")
    if df.shape[1] < 10:
        raise ValueError(f"File '{path}' has {df.shape[1]} columns; expected 8 scalars + curve coordinates + forces.")
    n_after_hdr = df.shape[1] - 8
    if n_after_hdr % 2 != 0:
        raise ValueError(f"After the first 8 columns, the remaining ({n_after_hdr}) must split into coordinates and forces.")

    m = n_after_hdr // 2
    n = len(df)
    Yt = df.iloc[:, 0].to_numpy(float)
    kb = df.iloc[:, 2].to_numpy(float)
    b1 = df.iloc[:, 3].to_numpy(float)
    b2 = df.iloc[:, 4].to_numpy(float)
    a3 = df.iloc[:, 5].to_numpy(float)
    a4 = df.iloc[:, 6].to_numpy(float)
    radp = df.iloc[:, 7].to_numpy(float)
    disp_or_diam = df.iloc[:, 8 : 8 + m].to_numpy(float)
    forc = df.iloc[:, 8 + m : 8 + 2 * m].to_numpy(float)
    disp = _infer_displacement(disp_or_diam, radp, disp_source)

    m_ext = m + 1
    disp_ext = np.full((n, m_ext), np.nan)
    forc_ext = np.full((n, m_ext), np.nan)
    n_rupture = 0
    for i in range(n):
        valid_mask = np.isfinite(disp[i, :]) & np.isfinite(forc[i, :])
        valid_mask &= forc[i, :] >= 0.0
        valid_mask &= disp[i, :] >= 0.0
        disp_valid = disp[i, valid_mask]
        forc_valid = forc[i, valid_mask]
        sort_idx = np.argsort(forc_valid)
        disp_sorted = disp_valid[sort_idx]
        forc_sorted = forc_valid[sort_idx]
        if len(forc_sorted) > 0:
            keep_indices = [0]
            for j in range(1, len(forc_sorted)):
                if disp_sorted[j] > disp_sorted[keep_indices[-1]]:
                    keep_indices.append(j)
            disp_clean = disp_sorted[keep_indices]
            forc_clean = forc_sorted[keep_indices]
            if rupture_ratio_threshold is not None and _is_rupture_curve(disp_clean, rupture_ratio_threshold):
                n_rupture += 1
                continue
            if len(disp_clean) > 0 and forc_clean[0] > 0.0:
                disp_clean = np.insert(disp_clean, 0, 0.0)
                forc_clean = np.insert(forc_clean, 0, 0.0)
        else:
            disp_clean = np.array([])
            forc_clean = np.array([])
        if len(disp_clean) > 0:
            disp_ext[i, : len(disp_clean)] = disp_clean
            forc_ext[i, : len(forc_clean)] = forc_clean

    if rupture_ratio_threshold is not None:
        print(f"Rupture filter removed {n_rupture}/{n} curves ({100*n_rupture/max(n,1):.1f}%)")

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
    print(f"After cleaning: {len(out)} valid data points")
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
    return (
        torch.tensor(Xz, dtype=torch.float32),
        torch.tensor(yz, dtype=torch.float32),
        x_mu.tolist(),
        x_sd.tolist(),
        y_mu.tolist(),
        y_sd.tolist(),
    )


def check_negative_predictions(model, X_val, y_mu, y_sd, threshold=0.0):
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_val).cpu().numpy()
        y_pred = y_pred_norm * y_sd[0] + y_mu[0]
        num_negative = np.sum(y_pred < threshold)
        min_pred = np.min(y_pred)
    return bool(num_negative > 0), float(min_pred), int(num_negative), int(len(y_pred))


def train_single_architecture(arch_config, data_path, output_dir, disp_source, batch_size=1024, lr=5e-4, max_epoch=100, num_workers=4, rupture_ratio_threshold=2.0):
    width, depth, name = arch_config
    print(f"\n{'='*70}\nTraining architecture: {name} (width={width}, depth={depth})\n{'='*70}")
    df_long = read_new_dat_to_long(data_path, disp_source=disp_source, rupture_ratio_threshold=rupture_ratio_threshold)
    input_cols = ["Yt", "kb", "b1", "b2", "a3", "a4", "F"]
    target_col = "disp"
    X, y, x_mu, x_sd, y_mu, y_sd = make_tensors(df_long, input_cols, target_col)
    n = len(X)
    n_val = max(1, int(0.1 * n))
    perm = torch.randperm(n)
    X = X[perm]
    y = y[perm]
    Xv, yv = X[:n_val], y[:n_val]
    Xt, yt = X[n_val:], y[n_val:]
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, num_workers=num_workers, pin_memory=True)
    model = MLP(input_dims=len(input_cols), output_dims=1, hl_dims=[width] * depth)
    model.apply(init_weights)
    model, tr_hist, va_hist = train_model(model, loader, Xv, yv, lr=lr, max_epoch=max_epoch, info_every=5)
    has_neg, min_pred, num_neg, total = check_negative_predictions(model, Xv, y_mu, y_sd)
    out_path = os.path.join(output_dir, f"{name}.pkl")
    os.makedirs(output_dir, exist_ok=True)
    save_model_states(model, xshift=x_mu, xscale=x_sd, yshift=y_mu, yscale=y_sd, path=out_path)
    return {
        "name": name,
        "width": int(width),
        "depth": int(depth),
        "train_loss": float(tr_hist[-1]),
        "val_loss": float(va_hist[-1]),
        "has_negatives": bool(has_neg),
        "min_prediction": float(min_pred),
        "num_negatives": int(num_neg),
        "total_samples": int(total),
        "model_path": out_path,
    }


def plot_results_summary(results, output_path, diameter):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Indentation Surrogate Training Summary - {diameter}um", fontsize=16, fontweight="bold")
    names = [r["name"] for r in results]
    train_losses = [r["train_loss"] for r in results]
    val_losses = [r["val_loss"] for r in results]
    neg_pcts = [100 * r["num_negatives"] / max(r["total_samples"], 1) for r in results]
    min_preds = [r["min_prediction"] for r in results]
    ax = axes[0, 0]
    ax.bar(range(len(names)), train_losses)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Training loss")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[0, 1]
    ax.bar(range(len(names)), val_losses)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Validation loss")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1, 0]
    ax.bar(range(len(names)), neg_pcts)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Negative predictions (%)")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1, 1]
    ax.bar(range(len(names)), min_preds)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Minimum prediction")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _resolve_data_path(data_diameter, data_file):
    data_dir = SURROGATE_ROOT / f"diameters/{data_diameter}um/data"
    if data_file:
        return data_dir / data_file
    for candidate in ["samples_all.dat", "F_Delta_b1_b2_a3_a4.dat", "F_Delta.dat"]:
        path = data_dir / candidate
        if path.exists():
            return path
    return data_dir / "samples_all.dat"


def finalize_results(
    *,
    diameter: str,
    output_dir: str | os.PathLike[str],
    results: list[dict[str, object]],
    plots_dir: str | os.PathLike[str],
    report_json: str | os.PathLike[str] | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    if not results:
        raise ValueError("No architecture results were provided for finalization.")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    results_sorted = sorted(results, key=lambda x: float(x["val_loss"]))
    training_results_path = output_dir_path / "training_results.csv"
    pd.DataFrame(results_sorted).to_csv(training_results_path, index=False)

    plots_dir_path = Path(plots_dir)
    plots_dir_path.mkdir(parents=True, exist_ok=True)
    effective_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_plot_path = plots_dir_path / f"training_summary_{effective_timestamp}.png"
    plot_results_summary(results_sorted, str(summary_plot_path), diameter)

    best = results_sorted[0]
    best_dest = SURROGATE_ROOT / f"diameters/{diameter}um/trained/microbubble_displacement_BEST.pkl"
    best_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best["model_path"], best_dest)
    print(f"Best model: {best['name']} -> {best_dest}")

    payload = {
        "diameter": diameter,
        "output_dir": str(output_dir_path),
        "best": best,
        "training_results_csv": str(training_results_path),
        "summary_plot": str(summary_plot_path),
        "best_dest": str(best_dest),
        "results": results_sorted,
    }
    if report_json:
        with open(report_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return payload


def main():
    ap = argparse.ArgumentParser(description="Train multiple NN architectures for the indentation surrogate")
    ap.add_argument("--diameter", type=str, required=True)
    ap.add_argument("--data-diameter", type=str, default=None)
    ap.add_argument("--data-file", type=str, default=None)
    ap.add_argument("--disp-source", type=str, default="auto", choices=["auto", "diameter", "displacement"])
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epoch", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--rupture-ratio", type=float, default=2.0)
    ap.add_argument("--arch-index", type=int, default=None)
    ap.add_argument("--arch-name", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--result-json", type=str, default=None)
    ap.add_argument("--report-json", type=str, default=None)
    ap.add_argument("--collect-only", action="store_true", default=False)
    args = ap.parse_args()

    data_diameter = args.data_diameter if args.data_diameter else args.diameter
    data_path = _resolve_data_path(data_diameter, args.data_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(SURROGATE_ROOT / f"diameters/{args.diameter}um/trained/multi_arch_{timestamp}")
    plots_dir = SURROGATE_ROOT / f"diameters/{args.diameter}um/plots"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    if args.collect_only:
        result_files = sorted(Path(output_dir).glob("result_*.json"))
        if not result_files:
            raise FileNotFoundError(f"No per-architecture result JSONs found in {output_dir}")
        results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
        payload = finalize_results(
            diameter=args.diameter,
            output_dir=output_dir,
            results=results,
            plots_dir=plots_dir,
            report_json=args.report_json,
            timestamp=timestamp,
        )
        payload["data_path"] = str(data_path)
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        return

    architectures = get_architectures()
    if args.arch_index is not None or args.arch_name is not None:
        if args.arch_index is not None and args.arch_name is not None:
            raise ValueError("Use only one of --arch-index or --arch-name")
        if args.arch_index is not None:
            arch = architectures[args.arch_index]
        else:
            matches = [a for a in architectures if a[2] == args.arch_name]
            if not matches:
                raise ValueError(f"Unknown arch-name '{args.arch_name}'")
            arch = matches[0]
        result = train_single_architecture(arch, str(data_path), output_dir, args.disp_source, args.batch_size, args.lr, args.max_epoch, args.num_workers, args.rupture_ratio if args.rupture_ratio > 0 else None)
        result_path = args.result_json or os.path.join(output_dir, f"{arch[2]}_result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Saved single-arch result to: {result_path}")
        return

    results = []
    rupture_threshold = args.rupture_ratio if args.rupture_ratio > 0 else None
    for arch in architectures:
        results.append(train_single_architecture(arch, str(data_path), output_dir, args.disp_source, args.batch_size, args.lr, args.max_epoch, args.num_workers, rupture_threshold))
    payload = finalize_results(
        diameter=args.diameter,
        output_dir=output_dir,
        results=results,
        plots_dir=plots_dir,
        report_json=args.report_json,
        timestamp=timestamp,
    )
    payload["data_path"] = str(data_path)
    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
