"""Generate MAP DPD overlay figures — Mirheo MAP curve vs experimental reference data.

Reads map_mirheo_manifest.json (written by run_map_mirheo.py), loads the
per-diameter result JSONs, converts DPD units to physical units, and overlays
the MAP simulation curves on the experimental reference data.

Produces:
  figures/map_overlay_{experiment}_{model_family}.{pdf,png}

Usage:
  python3.8 scripts/shared/postprocess/generate_map_overlay_figure.py \\
      --map-mirheo-manifest _o369_paper_runs/.../map_mirheo/map_mirheo_manifest.json \\
      --output-dir _o369_paper_runs/figures \\
      --experiment indentation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.mirheo.radp import RADP_LOOKUP
from meso_uq.postprocess.paper_figures import (
    COLORS,
    LINESTYLES,
    configure_matplotlib,
    convert_to_physical,
    emb_yaml_path,
    load_scaling,
)

_DIAMETERS = {
    "indentation": ["3.2", "3.4", "5.8"],
    "compression": ["2.1", "2.9", "3.0"],
}

_REF_DATA_PREFIX = {
    "indentation": "indentation_data_",
    "compression": "compression_data_",
}

_AXIS_LABELS = {
    "indentation": ("Force [nN]", "Displacement [nm]"),
    "compression": ("Displacement [nm]", "Force [nN]"),
}


def _load_reference(modality: str, diameter: str, length_factor: float, force_factor: float):
    """Return (x_phys, y_phys) for the experimental reference curve."""
    data_dir = REPO_ROOT / modality / "evalkit" / "data"
    prefix = _REF_DATA_PREFIX[modality]
    path = data_dir / f"{prefix}{diameter}um.dat"
    data = np.loadtxt(path, skiprows=1)
    return convert_to_physical(modality, data[:, 0], data[:, 1], length_factor, force_factor)


def _load_map_curve(result_json: Path, modality: str, length_factor: float, force_factor: float):
    """Return (x_phys, y_phys) for the MAP Mirheo simulation curve.

    For indentation:
      - result["displacement_points"] is the FORCE grid (x-axis, DPD force units).
      - result["forces"] is the SHORT DIAMETERS (DPD length units, model output).
      - displacement [nm] = (D_initial − short_diameter + d0_offset) * length_factor
        where D_initial = 2 * radp from RADP_LOOKUP.

    For compression:
      - displacement_points is the displacement grid (DPD length units).
      - forces is the force output (DPD force units).
      - d0_offset shifts the displacement before converting.
    """
    with open(result_json) as f:
        result = json.load(f)

    d0_offset = float(result["grid_config"]["d0_offset"])

    if modality == "indentation":
        force_grid_dpd = np.array(result["displacement_points"])  # force grid → x
        short_diameters_dpd = np.array(result["forces"])          # short diameters → y
        diameter_um = float(result["diameter_um"])
        radp = RADP_LOOKUP["indentation"][diameter_um]
        D_initial = 2.0 * radp
        displacement_dpd = D_initial - short_diameters_dpd + d0_offset
        return (
            force_grid_dpd * force_factor,      # x = Force [nN]
            displacement_dpd * length_factor,   # y = Displacement [nm]
        )
    else:
        # compression: displacement_points is the displacement grid (PRE-SHIFT)
        displ_dpd = np.array(result["displacement_points"])
        forces_dpd = np.array(result["forces"])
        displ_shifted = displ_dpd + d0_offset
        return (
            displ_shifted * length_factor,   # x = Displacement [nm]
            forces_dpd * force_factor,       # y = Force [nN]
        )


def make_overlay_figure(
    manifest_path: Path,
    experiment: str,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> None:
    with open(manifest_path) as f:
        manifest = json.load(f)

    model_family = manifest.get("model_family", "unknown")
    manifest_dir = manifest_path.parent

    emb_yaml = emb_yaml_path(experiment, repo_root)
    length_factor, force_factor = load_scaling(emb_yaml)

    diameters = _DIAMETERS[experiment]
    colors = COLORS[experiment]
    xlabel, ylabel = _AXIS_LABELS[experiment]

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    # Build a lookup: dataset_name → result_json path (resolve relative paths
    # against the manifest's directory so the script works regardless of cwd)
    result_by_diam: dict[str, Path] = {}
    for diam_result in manifest.get("diameters", []):
        if diam_result.get("status") != "passed":
            continue
        result_path = diam_result.get("result_json")
        if result_path:
            result_abs = Path(result_path)
            if not result_abs.is_absolute():
                result_abs = (manifest_dir / result_abs).resolve()
            if result_abs.exists():
                # extract diameter from dataset name, e.g. indentation_3.2um
                name = diam_result["dataset_name"]
                for diam in diameters:
                    if f"_{diam}um" in name:
                        result_by_diam[diam] = result_abs

    for i, diameter in enumerate(diameters):
        color = colors[diameter]
        ls = LINESTYLES[i]
        label = f"{diameter} μm"

        # MAP Mirheo curve (drawn first so ref data appears on top)
        if diameter in result_by_diam:
            try:
                map_x, map_y = _load_map_curve(result_by_diam[diameter], experiment,
                                               length_factor, force_factor)
                ax.plot(map_x, map_y, color=color, linewidth=2.0, linestyle=ls,
                        label=f"{label} (MAP DPD)")
            except Exception as exc:
                print(f"  WARNING: could not load MAP curve for {diameter} μm: {exc}")
        else:
            print(f"  WARNING: no passed MAP result for {experiment} {diameter} μm")

        # Reference data (drawn on top of MAP curve)
        try:
            ref_x, ref_y = _load_reference(experiment, diameter, length_factor, force_factor)
            ax.plot(ref_x, ref_y, linestyle="None", marker="o",
                    markersize=3.4, markerfacecolor="white",
                    markeredgewidth=1.1, markeredgecolor=color,
                    label=f"{label} (exp.)")
        except FileNotFoundError:
            print(f"  WARNING: no reference data for {experiment} {diameter} μm")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"MAP DPD vs experiments — {experiment}", pad=6)
    ax.legend(frameon=False, loc="best", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    stem = f"map_overlay_{experiment}_{model_family}"
    for ext in ("pdf", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220 if ext == "png" else None)
        print(f"  Saved: {path}")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-mirheo-manifest", required=True, type=Path,
                        help="Path to map_mirheo_manifest.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", choices=["indentation", "compression"],
                        default="indentation")
    args = parser.parse_args(argv)

    if not args.map_mirheo_manifest.is_file():
        print(f"ERROR: manifest not found: {args.map_mirheo_manifest}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_overlay_figure(args.map_mirheo_manifest, args.experiment, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
