#!/usr/bin/env python3
"""
Generate paper v3 supplementary MAP figures and parameter table.

Produces (in _paper/v3/generated/supplementary/):
  - map_confirmation_{modality}_{diameter}um.pdf — experiments + full MAP + reduced MAP per panel
  - map_confirmation_all.pdf — 3×2 multi-panel combining all 6 diameter/modality pairs
  - full_vs_reduced_map_overlay_dpd.pdf — updated overlay using DPD-evaluated (Mirheo) MAP curves
  - map_parameter_comparison.csv — shared params (ka, kb, d0) full vs reduced, all cases
  - map_parameter_comparison.tex — LaTeX table

All MAP curves are from Mirheo DPD evaluations (not surrogate).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = Path(
    os.environ.get(
        "MESOUQ_PAPER_CAMPAIGN_ROOT",
        REPO_ROOT / "_runs" / "paper_exact" / "missing_campaign_root",
    )
).resolve()
PAPER_STAGE_ROOT = Path(
    os.environ.get(
        "MESOUQ_PAPER_STAGE_ROOT",
        CAMPAIGN_ROOT / "paper_exact_stage",
    )
).resolve()
V3_ROOT = PAPER_STAGE_ROOT
SUPP_DIR = V3_ROOT / "generated" / "supplementary"
FIGURES_DIR = V3_ROOT / "generated" / "figures"
DEFAULT_TEXDEPS_DIR = REPO_ROOT / "papers" / "huq_emb" / "_texdeps"
TEXDEPS_DIR = Path(
    os.environ.get(
        "MESOUQ_PAPER_TEXDEPS_DIR",
        str(DEFAULT_TEXDEPS_DIR),
    )
).resolve()
WORKFLOW_ROOT = CAMPAIGN_ROOT / "workflow_matrix" / "runs"

# MAP Mirheo roots: compression (new supplementary run), indentation (existing production)
MAP_MIRHEO_ROOTS = {
    "compression": {
        "full": WORKFLOW_ROOT / "compression" / "full-model" / "production" / "map_mirheo",
        "reduced": WORKFLOW_ROOT / "compression" / "reduced-model" / "production" / "map_mirheo",
    },
    "indentation": {
        "full": WORKFLOW_ROOT / "indentation" / "full-model" / "production" / "map_mirheo",
        "reduced": WORKFLOW_ROOT / "indentation" / "reduced-model" / "production" / "map_mirheo",
    },
}

DIAMETERS = {
    "compression": ["2.1", "2.9", "3.0"],
    "indentation": ["3.2", "3.4", "5.8"],
}

sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

# ---------------------------------------------------------------------------
# Paper style constants (matching generate_reduced_story_assets.py)
# ---------------------------------------------------------------------------
INK = "#203050"
INK_LIGHT = "#7c8ba3"
INK_FAINT = "#c8d0db"
GOLD = "#E0A010"
GOLD_LIGHT = "#efc86a"
GOLD_FAINT = "#f6e5b6"
NEUTRAL = "#6f6f6f"
TEXTWIDTH_IN = 6.75
CAPTION_FONTSIZE = 8.5
TICK_FONTSIZE = 7.75
PANEL_TITLE_FONTSIZE = 8.75
LEGEND_FONTSIZE = 7.5
_MATPLOTLIB_CONFIGURED = False

# Yt -> ka conversion (must match the original paper script exactly)
KA_FACTOR = 0.0005583725703247615


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def configure_matplotlib() -> None:
    global _MATPLOTLIB_CONFIGURED
    if _MATPLOTLIB_CONFIGURED:
        return
    texinputs_prefix = f".:{TEXDEPS_DIR}//:"
    current = os.environ.get("TEXINPUTS", "")
    if texinputs_prefix not in current:
        os.environ["TEXINPUTS"] = texinputs_prefix + current
    use_tex = os.environ.get("HUQ_PAPER_DISABLE_TEX", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }
    rc_params = {
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
        "font.size": CAPTION_FONTSIZE,
        "axes.labelsize": CAPTION_FONTSIZE,
        "axes.titlesize": PANEL_TITLE_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "figure.titlesize": PANEL_TITLE_FONTSIZE,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.8,
    }
    if use_tex:
        rc_params["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    mpl.rcParams.update(rc_params)
    _MATPLOTLIB_CONFIGURED = True


def display_modality_name(modality: str) -> str:
    return "Definity-compression" if modality == "compression" else "Sonovue-indentation"


def modality_palette(modality: str) -> dict[str, str]:
    if modality == "compression":
        return {"dark": INK, "mid": INK_LIGHT, "faint": INK_FAINT}
    return {"dark": GOLD, "mid": GOLD_LIGHT, "faint": GOLD_FAINT}


def diameter_styles(modality: str, diameters: list[str]) -> dict[str, dict]:
    palette = modality_palette(modality)
    colors = [palette["faint"], palette["mid"], palette["dark"]]
    markers = ["o", "s", "D"]
    return {d: {"color": c, "marker": m} for d, c, m in zip(diameters, colors, markers)}


def latex_diameter(diameter: str) -> str:
    return rf"{diameter} $\mu$m"


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = 0.965,
    y: float = 0.965,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=PANEL_TITLE_FONTSIZE,
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )


def add_panel_labels(
    axes: np.ndarray | list[plt.Axes],
    *,
    x: float = 0.965,
    y: float = 0.965,
) -> None:
    labels = [f"({chr(ord('a') + i)})" for i in range(np.ravel(np.asarray(axes, dtype=object)).size)]
    for ax, label in zip(np.ravel(np.asarray(axes, dtype=object)), labels):
        add_panel_label(ax, label, x=x, y=y)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_scaling(modality: str, diameter: str) -> tuple[float, float]:
    if modality == "compression":
        param_file = REPO_ROOT / f"_init_compression_{diameter}um" / "parameter"
        candidates = (
            list(param_file.glob("parameters-default*.yaml")) if param_file.exists() else []
        )
        if candidates:
            path = candidates[0]
        else:
            path = REPO_ROOT / "compression" / "src" / "parameters-default.emb.yaml"
    else:
        path = REPO_ROOT / "indentation" / "src" / "parameters-default.emb.yaml"

    with path.open("rb") as f:
        p = yaml.load(f, Loader=yaml.CLoader)

    rho_water = p["rho_water"]
    rhow = p["rhow"]
    energy_factor = p["energyFactor"]
    kbol = p["kbol"]
    t0 = p["t0"]
    ul = p["ul"]
    fscale = p.get("fscale", 1.0)

    ue = energy_factor * kbol * t0
    um = rho_water * ul**3 / rhow
    ut = np.sqrt(um * ul**2 / ue)
    length_factor = ul * 1e9
    force_factor = (um * ul / ut**2) / fscale * 1e9
    return length_factor, force_factor


def load_reference_curve(modality: str, diameter: str) -> tuple[np.ndarray, np.ndarray]:
    if modality == "compression":
        config_path = (
            REPO_ROOT / "inference" / "configs" / "production" / "inference_config_compression.yaml"
        )
    else:
        config_path = (
            REPO_ROOT / "inference" / "configs" / "production" / "inference_config_indentation.yaml"
        )
    with config_path.open("rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    data_dir = REPO_ROOT / config["data_dir"]
    prefix = config["data_prefix"]
    dat_path = data_dir / f"{prefix}{diameter}um.dat"
    data = np.loadtxt(dat_path, skiprows=1, ndmin=2)
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def indentation_initial_diameter_dpd(diameter: str) -> float:
    samples_path = (
        REPO_ROOT
        / "indentation"
        / "surrogate"
        / "diameters"
        / f"{diameter}um"
        / "data"
        / "samples_all.dat"
    )
    if samples_path.exists():
        samples = np.loadtxt(samples_path, ndmin=2)
        return 2.0 * float(np.median(samples[:, 7]))
    param_path = REPO_ROOT / "indentation" / "src" / "parameter" / "parameters-default00001.yaml"
    with param_path.open("rb") as f:
        p = yaml.load(f, Loader=yaml.CLoader)
    return 2.0 * float(p["radp"])


def load_map_mirheo(modality: str, model_kind: str, diameter: str) -> dict:
    base = MAP_MIRHEO_ROOTS[modality][model_kind]
    prefix = "compression" if modality == "compression" else "indentation"
    candidates = [
        base / f"{diameter}um_map_mirheo.json",
        base / "results" / f"{prefix}_{diameter}um_result.json",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    with path.open() as f:
        return json.load(f)


def convert_map_to_real(
    modality: str, diameter: str, map_result: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MAP Mirheo result to real units matching the paper axes."""
    length_factor, force_factor = load_scaling(modality, diameter)
    d0_offset = float(map_result["grid_config"]["d0_offset"])

    if modality == "compression":
        x_dpd = np.asarray(map_result["displacement_points"], dtype=float) + d0_offset
        y_dpd = np.asarray(map_result["forces"], dtype=float)
        x_real = x_dpd * length_factor
        y_real = y_dpd * force_factor
    else:
        # Indentation: forces list stores short diameters; convert to displacement observable
        short_diameter_dpd = np.asarray(map_result["forces"], dtype=float)
        force_grid_dpd = np.asarray(map_result["displacement_points"], dtype=float)
        initial_diameter_dpd = indentation_initial_diameter_dpd(diameter)
        displacement_dpd = initial_diameter_dpd - short_diameter_dpd + d0_offset
        x_real = force_grid_dpd * force_factor  # x-axis = Force [nN]
        y_real = displacement_dpd * length_factor  # y-axis = Displacement [nm]
    return x_real, y_real


def convert_ref_to_real(
    modality: str, diameter: str, x_dpd: np.ndarray, y_dpd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    length_factor, force_factor = load_scaling(modality, diameter)
    if modality == "compression":
        return x_dpd * length_factor, y_dpd * force_factor
    return x_dpd * force_factor, y_dpd * length_factor


# ---------------------------------------------------------------------------
# Panel drawing helper
# ---------------------------------------------------------------------------
def draw_panel(
    ax: plt.Axes,
    modality: str,
    diameter: str,
    full_result: dict,
    reduced_result: dict,
    show_legend: bool = True,
) -> None:
    palette = modality_palette(modality)
    x_ref_dpd, y_ref_dpd = load_reference_curve(modality, diameter)
    x_ref, y_ref = convert_ref_to_real(modality, diameter, x_ref_dpd, y_ref_dpd)

    full_x, full_y = convert_map_to_real(modality, diameter, full_result)
    reduced_x, reduced_y = convert_map_to_real(modality, diameter, reduced_result)

    # Sort by x for clean line plots
    idx_f = np.argsort(full_x)
    idx_r = np.argsort(reduced_x)

    ax.plot(
        x_ref,
        y_ref,
        linestyle="None",
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=1.35,
        markeredgecolor="black",
        color="black",
        label="Experiments",
        zorder=6,
    )
    ax.plot(
        full_x[idx_f],
        full_y[idx_f],
        linestyle="-",
        color=palette["dark"],
        linewidth=1.6,
        label="Full MAP",
        zorder=4,
    )
    ax.plot(
        full_x[idx_f],
        full_y[idx_f],
        linestyle="None",
        marker="s",
        markersize=3.2,
        markerfacecolor=palette["dark"],
        markeredgewidth=0.0,
        color=palette["dark"],
        zorder=4,
    )
    ax.plot(
        reduced_x[idx_r],
        reduced_y[idx_r],
        linestyle=(0, (5.0, 2.0)),
        color=palette["mid"],
        linewidth=1.6,
        label="Reduced MAP",
        zorder=3,
    )

    title = rf"{display_modality_name(modality)}, {latex_diameter(diameter)}"
    ax.set_title(title, pad=5)
    if modality == "compression":
        ax.set_xlabel(r"Displacement [nm]")
        ax.set_ylabel(r"Force [nN]")
    else:
        ax.set_xlabel(r"Force [nN]")
        ax.set_ylabel(r"Displacement [nm]")
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(frameon=False, loc="best", fontsize=LEGEND_FONTSIZE)


# ---------------------------------------------------------------------------
# Figure 1: 6 individual per-diameter figures
# ---------------------------------------------------------------------------
def plot_individual_confirmations() -> None:
    configure_matplotlib()
    for modality in ["compression", "indentation"]:
        for diameter in DIAMETERS[modality]:
            full_result = load_map_mirheo(modality, "full", diameter)
            reduced_result = load_map_mirheo(modality, "reduced", diameter)

            fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH_IN / 2.0, 2.8))
            draw_panel(ax, modality, diameter, full_result, reduced_result, show_legend=True)
            fig.tight_layout()

            out = SUPP_DIR / f"map_confirmation_{modality}_{diameter}um.pdf"
            fig.savefig(out, bbox_inches="tight")
            fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: 3×2 combined multi-panel (3 diameters × 2 modalities)
# ---------------------------------------------------------------------------
def plot_combined_confirmation() -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 5.2), constrained_layout=False)
    for row, modality in enumerate(["compression", "indentation"]):
        for col, diameter in enumerate(DIAMETERS[modality]):
            full_result = load_map_mirheo(modality, "full", diameter)
            reduced_result = load_map_mirheo(modality, "reduced", diameter)
            show_legend = row == 0 and col == 0
            draw_panel(
                axes[row, col],
                modality,
                diameter,
                full_result,
                reduced_result,
                show_legend=show_legend,
            )

    # Shared legend from first panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if axes[0, 0].get_legend():
        axes[0, 0].get_legend().remove()
    add_panel_labels(axes)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.0,
        columnspacing=1.2,
    )

    out = SUPP_DIR / "map_confirmation_all.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 3: updated full-vs-reduced overlay (DPD Mirheo curves, paper style)
# ---------------------------------------------------------------------------
def plot_overlay_dpd() -> None:
    configure_matplotlib()
    specs = [
        ("compression", ["2.1", "2.9", "3.0"], "Displacement [nm]", "Force [nN]"),
        ("indentation", ["3.2", "3.4", "5.8"], "Force [nN]", "Displacement [nm]"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.0), constrained_layout=False)

    neutral_marker_handles = (
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=4.0,
            markerfacecolor=NEUTRAL,
            markeredgewidth=0.0,
            color=NEUTRAL,
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="s",
            markersize=4.0,
            markerfacecolor=NEUTRAL,
            markeredgewidth=0.0,
            color=NEUTRAL,
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="D",
            markersize=4.0,
            markerfacecolor=NEUTRAL,
            markeredgewidth=0.0,
            color=NEUTRAL,
        ),
    )
    type_handles = [
        neutral_marker_handles,
        Line2D([0], [0], color=NEUTRAL, linewidth=1.15, linestyle="-"),
    ]
    type_labels = ["Full", "Reduced"]

    for ax, (modality, diameters, xlabel, ylabel) in zip(np.ravel(axes), specs):
        styles = diameter_styles(modality, diameters)
        for diameter in diameters:
            full_result = load_map_mirheo(modality, "full", diameter)
            reduced_result = load_map_mirheo(modality, "reduced", diameter)
            full_x, full_y = convert_map_to_real(modality, diameter, full_result)
            reduced_x, reduced_y = convert_map_to_real(modality, diameter, reduced_result)
            idx_f = np.argsort(full_x)
            idx_r = np.argsort(reduced_x)
            color = str(styles[diameter]["color"])
            marker = str(styles[diameter]["marker"])

            ax.plot(
                reduced_x[idx_r],
                reduced_y[idx_r],
                color=color,
                linewidth=1.15,
                linestyle="-",
                zorder=2,
            )
            ax.plot(
                full_x[idx_f],
                full_y[idx_f],
                linestyle="None",
                marker=marker,
                markersize=4.3,
                markerfacecolor=color,
                markeredgewidth=0.0,
                color=color,
                zorder=3,
            )

        diameter_handles = [
            Line2D([0], [0], color=str(styles[d]["color"]), linewidth=1.15, linestyle="-")
            for d in diameters
        ]
        diameter_labels = [latex_diameter(d) for d in diameters]
        ax.legend(
            diameter_handles,
            diameter_labels,
            frameon=False,
            loc="best",
            fontsize=LEGEND_FONTSIZE,
            handlelength=1.8,
            labelspacing=0.3,
        )
        ax.set_title(display_modality_name(modality), pad=6)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(xlabel, fontsize=CAPTION_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=CAPTION_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    add_panel_labels(axes)
    fig.tight_layout()
    top_legend = fig.legend(
        type_handles,
        type_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.3,
        columnspacing=1.4,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    out_pdf = SUPP_DIR / "full_vs_reduced_map_overlay_dpd.pdf"
    out_png = SUPP_DIR / "full_vs_reduced_map_overlay_dpd.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", bbox_extra_artists=(top_legend,))
    fig.savefig(out_pdf, bbox_inches="tight", bbox_extra_artists=(top_legend,))
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}")


# ---------------------------------------------------------------------------
# Table: shared MAP parameters (ka, kb, d0) full vs reduced
# ---------------------------------------------------------------------------
def generate_parameter_table() -> None:
    rows = []
    for modality in ["compression", "indentation"]:
        for diameter in DIAMETERS[modality]:
            for model_kind in ["full", "reduced"]:
                result = load_map_mirheo(modality, model_kind, diameter)
                if "map_parameters" in result:
                    params = result["map_parameters"]
                else:
                    params = result["parameters"]

                # Extract shared params
                names = result.get("parameter_names", [])
                pdict = dict(zip(names, params))

                yt = pdict.get("Yt", float("nan"))
                kb = pdict.get("kb", float("nan"))
                d0 = pdict.get("d0", float("nan"))
                ka = yt * KA_FACTOR

                rows.append(
                    {
                        "modality": modality,
                        "diameter_um": float(diameter),
                        "model_kind": model_kind,
                        "ka": ka,
                        "kb": kb,
                        "d0": d0,
                        "Yt": yt,
                        "logPosterior": result.get("logPosterior", float("nan")),
                    }
                )

    df = pd.DataFrame(rows)

    # Add relative difference columns (full vs reduced per modality+diameter)
    diff_rows = []
    for modality in ["compression", "indentation"]:
        for diameter in DIAMETERS[modality]:
            sub = df[(df["modality"] == modality) & (df["diameter_um"] == float(diameter))]
            full_row = sub[sub["model_kind"] == "full"].iloc[0]
            red_row = sub[sub["model_kind"] == "reduced"].iloc[0]
            for param in ["ka", "kb", "d0"]:
                vf, vr = full_row[param], red_row[param]
                rel = (vf - vr) / abs(vr) * 100 if vr != 0 else float("nan")
                diff_rows.append(
                    {
                        "modality": modality,
                        "diameter_um": float(diameter),
                        "param": param,
                        "full": vf,
                        "reduced": vr,
                        "rel_diff_pct": rel,
                    }
                )
    df_diff = pd.DataFrame(diff_rows)

    csv_out = SUPP_DIR / "map_parameter_comparison.csv"
    df_diff.to_csv(csv_out, index=False, float_format="%.6g")
    print(f"  Saved: {csv_out.name}")

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{MAP parameter comparison: full vs.\ reduced model}",
        r"\label{tab:map_comparison}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Modality & Diam.\ [$\mu$m] & Param & Full & Reduced & Rel.\ diff.\ [\%] \\",
        r"\midrule",
    ]
    for _, row in df_diff.iterrows():
        mod_name = "Definity" if row["modality"] == "compression" else "Sonovue"
        param_tex = {"ka": r"$k_a$", "kb": r"$k_b$", "d0": r"$d_0$"}.get(row["param"], row["param"])
        lines.append(
            rf"{mod_name} & {row['diameter_um']:.1f} & {param_tex}"
            rf" & {row['full']:.4g} & {row['reduced']:.4g} & {row['rel_diff_pct']:+.1f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_out = SUPP_DIR / "map_parameter_comparison.tex"
    tex_out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {tex_out.name}")


# ---------------------------------------------------------------------------
# Table: L² discrepancies for reduced-model MAP DPD vs reference data
# ---------------------------------------------------------------------------
def generate_l2_discrepancy_table() -> None:
    """Compute L² discrepancies between reduced MAP DPD curves and reference data.

    For each (modality, diameter):
      - Reduced MAP DPD curve is interpolated onto the reference x-grid
        (restricted to the overlapping range).
      - Relative L²: ||y_map - y_ref|| / ||y_ref|| × 100  [%]
      - RMSE in physical units of the y-axis.

    Output: CSV + LaTeX table in generated/figures/.
    """
    rows = []
    for modality in ["compression", "indentation"]:
        for diameter in DIAMETERS[modality]:
            # Reference data in real units
            x_ref_dpd, y_ref_dpd = load_reference_curve(modality, diameter)
            x_ref, y_ref = convert_ref_to_real(modality, diameter, x_ref_dpd, y_ref_dpd)

            # Reduced MAP DPD curve in real units
            reduced_result = load_map_mirheo(modality, "reduced", diameter)
            x_map, y_map = convert_map_to_real(modality, diameter, reduced_result)

            # Sort by x
            idx_ref = np.argsort(x_ref)
            x_ref_s, y_ref_s = x_ref[idx_ref], y_ref[idx_ref]
            idx_map = np.argsort(x_map)
            x_map_s, y_map_s = x_map[idx_map], y_map[idx_map]

            # Restrict reference to the x range spanned by the MAP curve
            x_lo = max(x_ref_s[0], x_map_s[0])
            x_hi = min(x_ref_s[-1], x_map_s[-1])
            mask = (x_ref_s >= x_lo) & (x_ref_s <= x_hi)
            if mask.sum() < 2:
                # Fallback: use full range with boundary extrapolation
                mask = np.ones(len(x_ref_s), dtype=bool)

            x_eval = x_ref_s[mask]
            y_ref_eval = y_ref_s[mask]

            # Interpolate MAP curve at reference x-grid
            y_map_interp = np.interp(x_eval, x_map_s, y_map_s)

            residuals = y_map_interp - y_ref_eval
            l2_rel = np.linalg.norm(residuals) / np.linalg.norm(y_ref_eval) * 100.0
            rmse = np.sqrt(np.mean(residuals**2))

            y_unit = "nN" if modality == "compression" else "nm"
            rows.append(
                {
                    "modality": modality,
                    "diameter_um": float(diameter),
                    "n_ref_points": int(mask.sum()),
                    "relative_L2_pct": l2_rel,
                    "RMSE": rmse,
                    "RMSE_unit": y_unit,
                }
            )

    df = pd.DataFrame(rows)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_out = FIGURES_DIR / "map_l2_discrepancy_reduced.csv"
    df.to_csv(csv_out, index=False, float_format="%.6g")
    print(f"  Saved: {_display_path(csv_out)}")

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{L$^2$ discrepancy between reduced-model MAP DPD response and reference data.}",
        r"\label{tab:l2_discrepancy_reduced}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Modality & Diam.\ [$\mu$m] & $n_\mathrm{ref}$ & Rel.\ L$^2$ [\%] & RMSE \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        mod_name = "Definity" if row["modality"] == "compression" else "Sonovue"
        lines.append(
            rf"{mod_name} & {row['diameter_um']:.1f} & {int(row['n_ref_points'])}"
            rf" & {row['relative_L2_pct']:.2f}"
            rf" & {row['RMSE']:.4g} [{row['RMSE_unit']}] \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_out = FIGURES_DIR / "map_l2_discrepancy_reduced.tex"
    tex_out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {_display_path(tex_out)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATE SUPPLEMENTARY MAP FIGURES")
    print("=" * 70)

    print("\n[1/4] Individual per-diameter confirmation figures...")
    plot_individual_confirmations()

    print("\n[2/4] Combined 3x2 confirmation figure...")
    plot_combined_confirmation()

    print("\n[3/4] Full-vs-reduced MAP overlay (DPD)...")
    plot_overlay_dpd()

    print("\n[4/4] Parameter comparison table...")
    generate_parameter_table()

    print("\n[5/5] L² discrepancy table (reduced MAP DPD vs reference)...")
    generate_l2_discrepancy_table()

    print("\n" + "=" * 70)
    print("Done. Figures in:")
    print(f"  {SUPP_DIR}")
    print(f"  {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
