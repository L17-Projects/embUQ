#!/usr/bin/env python3
"""Generate HUQ-EMB out-of-paper-scope diagnostic figures.

The script expects the paper-data layout produced by run_vega_50k_campaign.py
and run_exact_uqdpd_asset_port.py. It writes figures under
paper_data/figures/out_of_paper_scope by default.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.mirheo.radp import RADP_LOOKUP  # noqa: E402
from meso_uq.postprocess.paper_figures import (  # noqa: E402
    configure_matplotlib,
    convert_to_physical,
    emb_yaml_path,
    load_scaling,
)


@dataclass(frozen=True)
class DiameterSpec:
    symbol: str
    experiment: str
    diameter: str

    @property
    def dataset(self) -> str:
        return f"{self.experiment}_{self.diameter}um"

    @property
    def title(self) -> str:
        return f"{self.symbol}: {self.experiment}, {self.diameter} um"


DIAMETERS: tuple[DiameterSpec, ...] = (
    DiameterSpec("d1", "compression", "2.1"),
    DiameterSpec("d2", "compression", "2.9"),
    DiameterSpec("d3", "compression", "3.0"),
    DiameterSpec("d4", "indentation", "3.2"),
    DiameterSpec("d5", "indentation", "3.4"),
    DiameterSpec("d6", "indentation", "5.8"),
)
MODEL_FAMILIES = ("full-model", "reduced-model")
PARAMETER_ORDER = ("Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma")
SOBOL_PARAMETERS = ("Yt", "kb", "b1", "b2", "a3", "a4")
POSTERIOR_HISTOGRAM_BINS = 100
COLORS = {
    "d1": "#184E77",
    "d2": "#1E88B6",
    "d3": "#67B7DC",
    "d4": "#A25F00",
    "d5": "#D89400",
    "d6": "#F2C14E",
}
MODEL_COLORS = {"full-model": "#24364B", "reduced-model": "#D49414"}
MODEL_LABELS = {"full-model": "Full model", "reduced-model": "Reduced model"}


def _configure_style() -> None:
    configure_matplotlib()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "figure.titlesize": 10.0,
            "figure.dpi": 120,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _infer_campaign_id(paper_data_root: Path, requested: str | None) -> str:
    if requested:
        return requested
    last = paper_data_root / "LAST_CAMPAIGN_ID.txt"
    if last.is_file():
        return last.read_text(encoding="utf-8").strip()
    runs_root = paper_data_root / "runs"
    candidates = sorted(path.name for path in runs_root.iterdir() if path.is_dir()) if runs_root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Pass --campaign-id. Could not infer one from {last} or a single directory under {runs_root}."
    )


def _parse_lane_override(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            "Lane overrides must use selection=/path, for example "
            "compression:full-model:production=/tmp/run"
        )
    key, path = value.split("=", 1)
    parts = key.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid lane selection {key!r}; expected experiment:model-family:profile.")
    experiment, model_family, profile = parts
    if experiment not in {"compression", "indentation"} or model_family not in MODEL_FAMILIES or profile != "production":
        raise ValueError(f"Unsupported lane override selection: {key}")
    return key, _resolve_path(path)


def _lane_key(spec: DiameterSpec, model_family: str) -> str:
    return f"{spec.experiment}:{model_family}:production"


def _build_lane_roots(workflow_runs_root: Path, overrides: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for spec in DIAMETERS:
        for model_family in MODEL_FAMILIES:
            roots[_lane_key(spec, model_family)] = workflow_runs_root / spec.experiment / model_family / "production"
    for item in overrides:
        key, path = _parse_lane_override(item)
        roots[key] = path
    return roots


def _read_staging_roots(
    *,
    campaign_root: Path,
    group_holdout_root: Path | None,
    sobol_root: Path | None,
    staging_dirname: str,
) -> tuple[Path, Path]:
    if group_holdout_root is not None and sobol_root is not None:
        return group_holdout_root, sobol_root
    report_path = campaign_root / "postprocess_graph" / staging_dirname / "dnn_figure_input_staging_report.json"
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        group = group_holdout_root or _resolve_path(str(payload["group_holdout_root"]))
        sobol = sobol_root or _resolve_path(str(payload["sobol_root"]))
        return group, sobol
    staging_root = campaign_root / "postprocess_graph" / staging_dirname
    return group_holdout_root or (staging_root / "group_holdout"), sobol_root or (staging_root / "sobol")


def _save_figure(fig: plt.Figure, stem: Path, assets: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = Path(f"{stem}{suffix}")
        fig.savefig(path, bbox_inches="tight")
        assets.append(str(path))
    plt.close(fig)


def _axis_labels(experiment: str) -> tuple[str, str]:
    if experiment == "compression":
        return "Displacement [nm]", "Force [nN]"
    return "Force [nN]", "Displacement [nm]"


def _length_force_factors(experiment: str) -> tuple[float, float]:
    return load_scaling(emb_yaml_path(experiment, REPO_ROOT))


def _load_reference_curve(spec: DiameterSpec) -> tuple[np.ndarray, np.ndarray]:
    prefix = "compression_data" if spec.experiment == "compression" else "indentation_data"
    path = REPO_ROOT / "emb" / spec.experiment / "evalkit" / "data" / f"{prefix}_{spec.diameter}um.dat"
    data = np.loadtxt(path, skiprows=1)
    length_factor, force_factor = _length_force_factors(spec.experiment)
    return convert_to_physical(spec.experiment, data[:, 0], data[:, 1], length_factor, force_factor)


def _holdout_leaf(group_holdout_root: Path, spec: DiameterSpec) -> Path:
    dnn_leaf = group_holdout_root / spec.experiment / f"{spec.diameter}um" / "dnn"
    return dnn_leaf if dnn_leaf.exists() else group_holdout_root / spec.experiment / f"{spec.diameter}um"


def _load_holdout_curve(group_holdout_root: Path, spec: DiameterSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    leaf = _holdout_leaf(group_holdout_root, spec)
    curve_path = leaf / "representative_curve.csv"
    summary_path = leaf / "summary.json"
    if not curve_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing grouped-holdout outputs for {spec.symbol}: {curve_path}, {summary_path}")
    return pd.read_csv(curve_path), json.loads(summary_path.read_text(encoding="utf-8"))


def _holdout_xy(spec: DiameterSpec, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if spec.experiment == "compression":
        axis_col = "disp"
        truth_col = "F_true" if "F_true" in df.columns else "y_true"
        pred_col = "F_pred" if "F_pred" in df.columns else "y_pred"
        std_factor = _length_force_factors(spec.experiment)[1]
    else:
        axis_col = "F"
        truth_col = "disp_true" if "disp_true" in df.columns else "y_true"
        pred_col = "disp_pred" if "disp_pred" in df.columns else "y_pred"
        std_factor = _length_force_factors(spec.experiment)[0]
    length_factor, force_factor = _length_force_factors(spec.experiment)
    x, y_true = convert_to_physical(
        spec.experiment,
        df[axis_col].to_numpy(float),
        df[truth_col].to_numpy(float),
        length_factor,
        force_factor,
    )
    _, y_pred = convert_to_physical(
        spec.experiment,
        df[axis_col].to_numpy(float),
        df[pred_col].to_numpy(float),
        length_factor,
        force_factor,
    )
    y_std = df["pred_std"].to_numpy(float) * std_factor if "pred_std" in df.columns else None
    return x, y_true, y_pred, y_std


def _plot_holdout_one(ax: plt.Axes, spec: DiameterSpec, df: pd.DataFrame, summary: dict[str, Any], *, legend: bool) -> None:
    x, y_true, y_pred, y_std = _holdout_xy(spec, df)
    order = np.argsort(x)
    x, y_true, y_pred = x[order], y_true[order], y_pred[order]
    if y_std is not None:
        y_std = y_std[order]
        ax.fill_between(x, y_pred - y_std, y_pred + y_std, color=COLORS[spec.symbol], alpha=0.16, linewidth=0)
    ax.plot(x, y_pred, color=COLORS[spec.symbol], linewidth=1.9, label="DNN prediction")
    ax.plot(
        x,
        y_true,
        linestyle="None",
        marker="o",
        markersize=3.1,
        markerfacecolor="white",
        markeredgecolor="#222222",
        markeredgewidth=0.8,
        label="Held-out DPD",
    )
    metric = summary.get("representative_curve_rel_l2_pct", summary.get("best_median_curve_rel_l2_pct"))
    metric_text = f"rel. L2={float(metric):.2f}%" if metric is not None else "rel. L2=n/a"
    ax.text(0.03, 0.94, metric_text, transform=ax.transAxes, ha="left", va="top", fontsize=7.2)
    xlabel, ylabel = _axis_labels(spec.experiment)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(spec.title)
    ax.grid(alpha=0.22)
    if legend:
        ax.legend(frameon=False, loc="best")


def build_grouped_holdout_figures(group_holdout_root: Path, output_root: Path, assets: list[str]) -> None:
    out_dir = output_root / "grouped_holdout_validation"
    curves: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    for spec in DIAMETERS:
        curves[spec.symbol] = _load_holdout_curve(group_holdout_root, spec)
        fig, ax = plt.subplots(figsize=(3.7, 2.8))
        _plot_holdout_one(ax, spec, *curves[spec.symbol], legend=True)
        fig.tight_layout()
        _save_figure(fig, out_dir / f"grouped_holdout_validation_{spec.symbol}_{spec.dataset}", assets)

    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.1), constrained_layout=False)
    for idx, spec in enumerate(DIAMETERS):
        ax = axes.flat[idx]
        _plot_holdout_one(ax, spec, *curves[spec.symbol], legend=idx == 0)
    fig.tight_layout()
    _save_figure(fig, out_dir / "grouped_holdout_validation_all_diameters", assets)


def _sobol_csv(sobol_root: Path, spec: DiameterSpec) -> Path:
    name = (
        f"sobol_vs_disp_{spec.diameter}um.csv"
        if spec.experiment == "compression"
        else f"sobol_vs_force_{spec.diameter}um.csv"
    )
    return sobol_root / spec.experiment / "dnn" / name


def _load_sobol_rows(sobol_root: Path, spec: DiameterSpec, index_type: str) -> pd.DataFrame:
    path = _sobol_csv(sobol_root, spec)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Sobol CSV for {spec.symbol}: {path}")
    df = pd.read_csv(path)
    required = {"parameter", "index_type", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"Sobol CSV lacks required columns {sorted(required)}: {path}")
    axis_col = "axis"
    for candidate in ("displacement", "force"):
        if candidate in df.columns:
            axis_col = candidate
            break
    sub = df[(df["index_type"] == index_type) & (df["parameter"].isin(SOBOL_PARAMETERS))].copy()
    if sub.empty:
        raise ValueError(f"No {index_type} Sobol rows for {spec.symbol}: {path}")
    sub["axis_raw"] = sub[axis_col].astype(float)
    length_factor, force_factor = _length_force_factors(spec.experiment)
    sub["axis_physical"] = sub["axis_raw"] * (length_factor if spec.experiment == "compression" else force_factor)
    sub["source_csv"] = str(path)
    return sub


def _integrated_sobol(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parameter, sub in df.groupby("parameter"):
        sub = sub.sort_values("axis_physical")
        x = sub["axis_physical"].to_numpy(float)
        y = sub["value"].to_numpy(float)
        width = float(x.max() - x.min()) if len(x) > 1 else 0.0
        score = float(np.trapezoid(y, x) / width) if width > 0 else float(np.mean(y))
        rows.append({"parameter": parameter, "score": score})
    return pd.DataFrame(rows)


def _plot_sobol_one(ax: plt.Axes, spec: DiameterSpec, df: pd.DataFrame, *, legend: bool) -> None:
    color_cycle = {
        "Yt": "#203050",
        "kb": "#E0A010",
        "b1": "#4A8C6B",
        "b2": "#B64B55",
        "a3": "#7566A0",
        "a4": "#7A6A55",
    }
    for parameter in SOBOL_PARAMETERS:
        sub = df[df["parameter"] == parameter].sort_values("axis_physical")
        if sub.empty:
            continue
        x = sub["axis_physical"].to_numpy(float)
        y = sub["value"].to_numpy(float)
        ax.plot(x, y, label=parameter, color=color_cycle[parameter], linewidth=1.45)
        if "confidence" in sub.columns:
            c = sub["confidence"].to_numpy(float)
            ax.fill_between(x, np.maximum(0.0, y - c), np.minimum(1.0, y + c), color=color_cycle[parameter], alpha=0.08, linewidth=0)
    xlabel = "Displacement [nm]" if spec.experiment == "compression" else "Force [nN]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total Sobol index")
    y_top = max(1.06, float(df["value"].max()) * 1.08)
    ax.set_ylim(bottom=-0.03, top=min(1.45, y_top))
    ax.set_title(spec.title)
    ax.grid(alpha=0.22)
    if legend:
        ax.legend(frameon=False, ncol=2, loc="best")


def build_sobol_figures(sobol_root: Path, output_root: Path, assets: list[str], *, index_type: str) -> None:
    out_dir = output_root / "sobol_sensitivity"
    rows: dict[str, pd.DataFrame] = {}
    integrated_rows: list[pd.DataFrame] = []
    for spec in DIAMETERS:
        df = _load_sobol_rows(sobol_root, spec, index_type)
        rows[spec.symbol] = df
        integ = _integrated_sobol(df)
        integ["symbol"] = spec.symbol
        integ["dataset"] = spec.dataset
        integrated_rows.append(integ)
        fig, ax = plt.subplots(figsize=(3.9, 2.9))
        _plot_sobol_one(ax, spec, df, legend=True)
        fig.tight_layout()
        _save_figure(fig, out_dir / f"sobol_sensitivity_{index_type}_{spec.symbol}_{spec.dataset}", assets)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), constrained_layout=False, sharey=True)
    for idx, spec in enumerate(DIAMETERS):
        _plot_sobol_one(axes.flat[idx], spec, rows[spec.symbol], legend=idx == 0)
    fig.tight_layout()
    _save_figure(fig, out_dir / f"sobol_sensitivity_{index_type}_all_diameters", assets)

    summary = pd.concat(integrated_rows, ignore_index=True)
    summary.to_csv(out_dir / f"sobol_sensitivity_{index_type}_integrated_scores.csv", index=False)
    assets.append(str(out_dir / f"sobol_sensitivity_{index_type}_integrated_scores.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 4.8), constrained_layout=False, sharey=True)
    for idx, spec in enumerate(DIAMETERS):
        ax = axes.flat[idx]
        sub = summary[summary["symbol"] == spec.symbol].set_index("parameter").reindex(SOBOL_PARAMETERS).reset_index()
        ax.bar(sub["parameter"], sub["score"], color=COLORS[spec.symbol], alpha=0.88)
        ax.set_ylim(0, max(1.0, float(summary["score"].max()) * 1.12))
        ax.set_title(spec.title)
        ax.grid(axis="y", alpha=0.22)
        if idx % 3 == 0:
            ax.set_ylabel(f"Integrated {index_type}")
    fig.tight_layout()
    _save_figure(fig, out_dir / f"sobol_sensitivity_{index_type}_integrated_scores_all_diameters", assets)


def _phase1_json(lane_root: Path, spec: DiameterSpec) -> Path:
    direct = lane_root / "results_phase_1" / spec.dataset / "genLatest.json"
    if direct.is_file():
        return direct
    latest = lane_root / "results_phase_1" / spec.dataset / "latest"
    if latest.is_file():
        return latest
    raise FileNotFoundError(f"Missing Phase 1 posterior JSON for {spec.dataset}: {direct}")


def _clean_parameter_name(name: str) -> str:
    return "sigma" if name == "[Sigma]" else name


def _load_phase1_samples(path: Path, *, max_samples: int) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("Results", {}).get("Posterior Sample Database")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No posterior samples found in {path}")
    names = [_clean_parameter_name(item["Name"]) for item in payload.get("Variables", [])]
    if not names:
        names = list(PARAMETER_ORDER[: len(samples[0])])
    df = pd.DataFrame(samples, columns=names[: len(samples[0])])
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=20260317).sort_index()
    return df


def _format_axis(ax: plt.Axes, parameter: str) -> None:
    if parameter == "Yt":
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.18)


def _plot_100_bin_histogram(ax: plt.Axes, values: np.ndarray, *, color: str, label: str | None) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    pad = (0.5 if vmin == 0.0 else abs(vmin) * 0.05) if span <= 0.0 else 0.04 * span
    counts, edges = np.histogram(
        values,
        bins=POSTERIOR_HISTOGRAM_BINS,
        range=(vmin - pad, vmax + pad),
        density=False,
    )
    peak = float(np.max(counts))
    if peak <= 0.0:
        return
    ax.bar(
        edges[:-1],
        counts / peak,
        width=np.diff(edges),
        align="edge",
        color=color,
        alpha=0.74,
        edgecolor=color,
        linewidth=0.12,
        label=label,
    )


def _plot_posterior_grid(df: pd.DataFrame, spec: DiameterSpec, model_family: str, output_stem: Path, assets: list[str]) -> None:
    params = [param for param in PARAMETER_ORDER if param in df.columns]
    ncols = 4 if len(params) > 4 else 2
    nrows = int(math.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.45 * ncols, 2.05 * nrows), constrained_layout=False)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    color = MODEL_COLORS[model_family]
    for idx, (ax, param) in enumerate(zip(axes_arr, params)):
        values = df[param].dropna().to_numpy(float)
        _plot_100_bin_histogram(ax, values, color=color, label="Posterior histogram" if idx == 0 else None)
        q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
        ax.axvline(q50, color="#111111", linewidth=1.0, label="Median" if idx == 0 else None)
        ax.axvspan(q05, q95, color="#111111", alpha=0.08, linewidth=0, label="90% interval" if idx == 0 else None)
        ax.set_ylim(0, 1.08)
        if idx % ncols == 0:
            ax.set_ylabel("Scaled density")
        else:
            ax.set_yticklabels([])
        ax.set_title(param)
        _format_axis(ax, param)
    for ax in axes_arr[len(params) :]:
        ax.axis("off")
    fig.suptitle(
        f"Phase 1 posterior marginals - {MODEL_LABELS[model_family]} - {spec.title}",
        y=0.995,
        fontsize=10.0,
    )
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(fig, output_stem, assets)


def build_posterior_figures(
    lane_roots: dict[str, Path],
    output_root: Path,
    assets: list[str],
    *,
    max_samples: int,
) -> None:
    out_dir = output_root / "posterior_marginals_phase1"
    for model_family in MODEL_FAMILIES:
        for spec in DIAMETERS:
            lane_root = lane_roots[_lane_key(spec, model_family)]
            df = _load_phase1_samples(_phase1_json(lane_root, spec), max_samples=max_samples)
            _plot_posterior_grid(
                df,
                spec,
                model_family,
                out_dir / f"posterior_marginals_phase1_{model_family}_{spec.symbol}_{spec.dataset}",
                assets,
            )


def _map_manifest(lane_root: Path) -> Path:
    path = lane_root / "map_mirheo" / "map_mirheo_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MAP Mirheo manifest: {path}")
    return path


def _map_result_json(lane_root: Path, spec: DiameterSpec) -> Path:
    manifest_path = _map_manifest(lane_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent
    for item in payload.get("diameters", []):
        if item.get("dataset_name") != spec.dataset:
            continue
        if item.get("status") not in {None, "passed"}:
            continue
        result = item.get("result_json")
        if not result:
            continue
        result_path = Path(str(result))
        if not result_path.is_absolute():
            result_path = (manifest_dir / result_path).resolve()
        if result_path.is_file():
            return result_path
    raise FileNotFoundError(f"No passed MAP Mirheo result for {spec.dataset} in {manifest_path}")


def _load_map_curve(path: Path, spec: DiameterSpec) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grid = np.asarray(payload["displacement_points"], dtype=float)
    forces = np.asarray(payload["forces"], dtype=float)
    d0_offset = float(payload["grid_config"]["d0_offset"])
    length_factor, force_factor = _length_force_factors(spec.experiment)
    if spec.experiment == "compression":
        return (grid + d0_offset) * length_factor, forces * force_factor
    diameter_um = float(payload["diameter_um"])
    radp = RADP_LOOKUP["indentation"][diameter_um]
    displacement = (2.0 * radp) - forces + d0_offset
    return grid * force_factor, displacement * length_factor


def _plot_map_one(
    ax: plt.Axes,
    spec: DiameterSpec,
    lane_roots: dict[str, Path],
    *,
    legend: bool,
    warnings: list[str] | None = None,
) -> None:
    ref_x, ref_y = _load_reference_curve(spec)
    ax.plot(
        ref_x,
        ref_y,
        linestyle="None",
        marker="o",
        markersize=3.1,
        markerfacecolor="white",
        markeredgecolor="#222222",
        markeredgewidth=0.8,
        label="Experiment",
    )
    single_point = False
    for model_family in MODEL_FAMILIES:
        path = _map_result_json(lane_roots[_lane_key(spec, model_family)], spec)
        x, y = _load_map_curve(path, spec)
        order = np.argsort(x)
        marker = "s" if len(x) < 2 else None
        if len(x) < 2:
            single_point = True
            if warnings is not None:
                warnings.append(
                    f"{spec.symbol} {MODEL_LABELS[model_family]} MAP curve has {len(x)} point from {path}; "
                    "the final production MAP replay should contain multiple displacement samples."
                )
        ax.plot(
            x[order],
            y[order],
            color=MODEL_COLORS[model_family],
            linewidth=1.8,
            marker=marker,
            markersize=4.2 if marker else None,
            label=MODEL_LABELS[model_family],
        )
    if single_point:
        ax.text(
            0.97,
            0.06,
            "single MAP point",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color="#555555",
        )
    xlabel, ylabel = _axis_labels(spec.experiment)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(spec.title)
    ax.grid(alpha=0.22)
    if legend:
        ax.legend(frameon=False, loc="best")


def build_map_vs_simulation_figures(
    lane_roots: dict[str, Path],
    output_root: Path,
    assets: list[str],
    warnings: list[str],
) -> None:
    out_dir = output_root / "map_vs_simulation"
    for spec in DIAMETERS:
        fig, ax = plt.subplots(figsize=(3.8, 2.9))
        _plot_map_one(ax, spec, lane_roots, legend=True, warnings=warnings)
        fig.tight_layout()
        _save_figure(fig, out_dir / f"map_vs_simulation_{spec.symbol}_{spec.dataset}", assets)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), constrained_layout=False)
    for idx, spec in enumerate(DIAMETERS):
        _plot_map_one(axes.flat[idx], spec, lane_roots, legend=idx == 0)
    fig.tight_layout()
    _save_figure(fig, out_dir / "map_vs_simulation_all_diameters", assets)


def _write_manifest(
    *,
    output_root: Path,
    assets: list[str],
    paper_data_root: Path,
    campaign_id: str,
    workflow_runs_root: Path,
    group_holdout_root: Path,
    sobol_root: Path,
    lane_roots: dict[str, Path],
    warnings: list[str],
) -> Path:
    manifest = {
        "status": "passed",
        "paper_data_root": str(paper_data_root),
        "campaign_id": campaign_id,
        "output_root": str(output_root),
        "workflow_runs_root": str(workflow_runs_root),
        "group_holdout_root": str(group_holdout_root),
        "sobol_root": str(sobol_root),
        "diameter_mapping": [
            {
                "symbol": spec.symbol,
                "experiment": spec.experiment,
                "diameter_um": spec.diameter,
                "dataset": spec.dataset,
            }
            for spec in DIAMETERS
        ],
        "lane_roots": lane_roots,
        "data_quality_warnings": warnings,
        "assets": assets,
    }
    path = output_root / "out_of_scope_figures_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-data-root", required=True)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--workflow-runs-root", default=None)
    parser.add_argument("--group-holdout-root", default=None)
    parser.add_argument("--sobol-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--staging-dirname", default="dnn_figure_input_staging")
    parser.add_argument("--sobol-index-type", choices=["S1", "ST"], default="ST")
    parser.add_argument("--max-posterior-samples", type=int, default=50000)
    parser.add_argument(
        "--lane-root",
        action="append",
        default=[],
        help="Override one lane root as experiment:model-family:production=/abs/path. Repeat as needed.",
    )
    args = parser.parse_args(argv)

    _configure_style()
    paper_data_root = _resolve_path(args.paper_data_root)
    campaign_id = _infer_campaign_id(paper_data_root, args.campaign_id)
    campaign_root = paper_data_root / "runs" / campaign_id
    workflow_runs_root = (
        _resolve_path(args.workflow_runs_root)
        if args.workflow_runs_root is not None
        else campaign_root / "workflow_matrix" / "runs"
    )
    group_holdout_root, sobol_root = _read_staging_roots(
        campaign_root=campaign_root,
        group_holdout_root=_resolve_path(args.group_holdout_root) if args.group_holdout_root else None,
        sobol_root=_resolve_path(args.sobol_root) if args.sobol_root else None,
        staging_dirname=args.staging_dirname,
    )
    output_root = (
        _resolve_path(args.output_root)
        if args.output_root is not None
        else paper_data_root / "figures" / "out_of_paper_scope"
    )
    lane_roots = _build_lane_roots(workflow_runs_root, list(args.lane_root))

    print("Diameter mapping used by this script:")
    for spec in DIAMETERS:
        print(f"  {spec.symbol} -> {spec.diameter} um ({spec.experiment}; {spec.dataset})")

    assets: list[str] = []
    warnings: list[str] = []
    build_grouped_holdout_figures(group_holdout_root, output_root, assets)
    build_sobol_figures(sobol_root, output_root, assets, index_type=args.sobol_index_type)
    build_posterior_figures(
        lane_roots,
        output_root,
        assets,
        max_samples=int(args.max_posterior_samples),
    )
    build_map_vs_simulation_figures(lane_roots, output_root, assets, warnings)
    manifest = _write_manifest(
        output_root=output_root,
        assets=assets,
        paper_data_root=paper_data_root,
        campaign_id=campaign_id,
        workflow_runs_root=workflow_runs_root,
        group_holdout_root=group_holdout_root,
        sobol_root=sobol_root,
        lane_roots=lane_roots,
        warnings=warnings,
    )
    print(f"Wrote {len(assets)} out-of-paper-scope assets.")
    if warnings:
        print("Data-quality warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
