"""
Generate surrogate uncertainty propagation vs reference data figures.

Reads summary.csv (DPD units) from propagation_phase3b outputs, converts to
physical units (nm, nN) using parameters-default.emb.yaml, and overlays on
experimental reference data.

Produces:
  figures/propagation_bands_reduced.{pdf,png}
  figures/propagation_bands_full.{pdf,png}

Usage:
  python3.8 scripts/postprocess/generate_propagation_bands_figure.py \
      --run-root _o369_paper_runs \
      --output-dir _o369_paper_runs/figures
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Physical unit conversion
# ---------------------------------------------------------------------------

def _load_scaling(emb_yaml: Path):
    with emb_yaml.open() as f:
        p = yaml.safe_load(f)
    ul = p["ul"]
    rho_water = p["rho_water"]
    rhow = p["rhow"]
    energy_factor = p["energyFactor"]
    kbol = p["kbol"]
    t0 = p["t0"]
    fscale = p.get("fscale", 1.0)
    ue = energy_factor * kbol * t0
    um = rho_water * ul ** 3 / rhow
    ut = math.sqrt(um * ul ** 2 / ue)
    length_factor = ul * 1e9           # DPD length → nm
    force_factor = (um * ul / ut ** 2) / fscale * 1e9  # DPD force → nN
    return length_factor, force_factor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_summary(csv_path: Path):
    """Return x, median, q05, q95 from summary_predictive.csv (DPD units, noise-inclusive)."""
    # Prefer noise-inclusive predictive CSV; fall back to parameter-only summary
    predictive = csv_path.with_name(csv_path.stem + "_predictive.csv")
    path = predictive if predictive.exists() else csv_path
    if not predictive.exists():
        print(f"  WARNING: no predictive CSV at {predictive}, using parameter-only summary")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 2], data[:, 3], data[:, 4]  # x, median, q05, q95


def _load_reference_indentation(data_dir: Path, diameter: str, length_factor: float, force_factor: float):
    """Return force_nN, deformation_nm from the filtered DPD dat file (actual inference data)."""
    path = data_dir / f"indentation_data_{diameter}um.dat"
    data = np.loadtxt(path, skiprows=1)
    force_nN = data[:, 0] * force_factor      # DPD force → nN
    deform_nm = data[:, 1] * length_factor    # DPD disp  → nm
    return force_nN, deform_nm


def _load_reference_compression(data_dir: Path, diameter: str, length_factor: float, force_factor: float):
    """Return deformation_nm, force_nN from the filtered DPD dat file (actual inference data)."""
    path = data_dir / f"compression_data_{diameter}um.dat"
    data = np.loadtxt(path, skiprows=1)
    deform_nm = data[:, 0] * length_factor    # DPD disp  → nm
    force_nN = data[:, 1] * force_factor      # DPD force → nN
    return deform_nm, force_nN  # x=deformation, y=force


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

COLORS = {
    "indentation": {"3.2": "#c07b00", "3.4": "#e09a00", "5.8": "#f5c842"},
    "compression": {"2.1": "#1a5a8a", "2.9": "#2a7fc0", "3.0": "#5aaee0"},
}
LINESTYLES = ["solid", "dashed", "dashdot"]


def _configure_matplotlib():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
    })


def _add_thousands(ax):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:g}"
        ))


def _add_panel_label(ax, label):
    ax.text(-0.10, 1.02, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def _plot_one_panel(ax, modality, diameters, model_family, prop_root, emb_yaml, ref_data_dir):
    length_factor, force_factor = _load_scaling(emb_yaml)
    colors = COLORS[modality]

    for i, diameter in enumerate(diameters):
        csv_path = (prop_root / f"{modality}_{diameter}um" / "summary.csv")
        if not csv_path.exists():
            print(f"  WARNING: missing {csv_path}")
            continue

        x_dpd, med_dpd, q05_dpd, q95_dpd = _load_summary(csv_path)

        if modality == "indentation":
            # x = Force [DPD→nN], y = Displacement [DPD→nm]
            x = x_dpd * force_factor
            med = med_dpd * length_factor
            q05 = q05_dpd * length_factor
            q95 = q95_dpd * length_factor
            ref_x, ref_y = _load_reference_indentation(ref_data_dir, diameter, length_factor, force_factor)
        else:
            # x = Displacement [DPD→nm], y = Force [DPD→nN]
            x = x_dpd * length_factor
            med = med_dpd * force_factor
            q05 = q05_dpd * force_factor
            q95 = q95_dpd * force_factor
            ref_x, ref_y = _load_reference_compression(ref_data_dir, diameter, length_factor, force_factor)

        color = colors[diameter]
        ls = LINESTYLES[i]
        label = f"{diameter} μm"

        ax.fill_between(x, q05, q95, color=color, alpha=0.18)
        ax.plot(x, med, color=color, linewidth=2.0, linestyle=ls, label=label)
        ax.plot(ref_x, ref_y, linestyle="None", marker="o",
                markersize=3.4, markerfacecolor="white",
                markeredgewidth=1.1, markeredgecolor="black")

    # Dummy handle for experiment legend entry
    from matplotlib.lines import Line2D
    exp_handle = Line2D([0], [0], linestyle="None", marker="o", markersize=3.8,
                        markerfacecolor="white", markeredgewidth=1.0,
                        markeredgecolor="black")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [exp_handle], labels + ["Experiments"],
              frameon=False, loc="best")


def _make_figure(model_family, run_root, output_dir, repo_root):
    _configure_matplotlib()

    specs = [
        ("compression", ["2.1", "2.9", "3.0"],
         "Displacement [nm]", "Force [nN]",
         "Definity"),
        ("indentation", ["3.2", "3.4", "5.8"],
         "Force [nN]", "Displacement [nm]",
         "SonoVue"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.1))

    for ax, (panel_label), (modality, diameters, xlabel, ylabel, title) in \
            zip(axes, ["(a)", "(b)"], specs):

        # run folders: e.g. compression_reduced, indentation_full
        short = model_family.replace("-model", "")  # reduced or full
        slug = f"{modality}_{short}"
        prop_root = run_root / slug / "runs" / modality / model_family / "production" / "propagation_phase3b"
        emb_yaml = repo_root / modality / "src" / "parameters-default.emb.yaml"
        ref_data_dir = repo_root / modality / "evalkit" / "data"

        _plot_one_panel(ax, modality, diameters, model_family,
                        prop_root, emb_yaml, ref_data_dir)

        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        _add_thousands(ax)
        _add_panel_label(ax, panel_label)

    fig.tight_layout()

    stem = f"propagation_bands_{model_family.replace('-', '_')}"
    for ext in ("pdf", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220 if ext == "png" else None)
        print(f"  Saved: {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="_o369_paper_runs",
                        help="Root directory containing the 4 run folders")
    parser.add_argument("--output-dir", default="_o369_paper_runs/figures",
                        help="Directory to write figures into")
    parser.add_argument("--model-family", choices=["reduced-model", "full-model", "both"],
                        default="both")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = repo_root / run_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    families = ["reduced-model", "full-model"] if args.model_family == "both" else [args.model_family]
    for family in families:
        print(f"Generating figure for {family}...")
        _make_figure(family, run_root, output_dir, repo_root)

    print("Done.")


if __name__ == "__main__":
    main()
