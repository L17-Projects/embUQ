#!/usr/bin/env python3
"""
Generate reduced-first manuscript assets for paper v3.

This script creates:
  - experimental reference-curve figure
  - grouped-holdout surrogate-validation figure
  - representative reduced Phase 1 posterior figure
  - reduced Phase 1 vs Phase 3b summary figure
  - reduced-only posterior predictive figure
  - representative reduced MAP confirmation figure
  - direct full-vs-reduced MAP overlay figure
  - reduced Phase 3b MAP-parameter table

Outputs are written to _paper/v3/generated/figures/ by default, or to
HUQ_PAPER_FIGURES_DIR when that environment variable is set.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter

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
FIGURES_DIR = Path(os.environ.get("HUQ_PAPER_FIGURES_DIR", str(V3_ROOT / "generated" / "figures")))
GENERATED_DIR = V3_ROOT / "generated"
SCRATCH_MIRHEO_ROOT = V3_ROOT / "scratch" / "map_mirheo_quick"
DEFAULT_TEXDEPS_DIR = REPO_ROOT / "papers" / "huq_emb" / "_texdeps"
TEXDEPS_DIR = Path(
    os.environ.get(
        "MESOUQ_PAPER_TEXDEPS_DIR",
        str(DEFAULT_TEXDEPS_DIR),
    )
).resolve()
WORKFLOW_ROOT = CAMPAIGN_ROOT / "workflow_matrix" / "runs"
GROUP_HOLDOUT_ROOT = Path(
    os.environ.get("MESOUQ_PAPER_GROUP_HOLDOUT_ROOT", str(GENERATED_DIR / "surrogate_group_holdout"))
).resolve()
SOBOL_ROOT = Path(
    os.environ.get("MESOUQ_PAPER_SOBOL_ROOT", str(V3_ROOT / "sobol"))
).resolve()

import yaml

sys.path.insert(0, str(REPO_ROOT))

from meso_uq.predictive_statistics import compute_interval_statistics

# Linear Yt -> ka conversion used throughout the manuscript.
# Keep the exact UQ_DPD paper constant so the figure/tables match the paper outputs.
KA_FACTOR = 0.0005583725703247615
PHASE3B_MASKED_INDENTATION = set()
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
SUPTITLE_FONTSIZE = 9.75
LEGEND_FONTSIZE = 7.5
ANNOTATION_FONTSIZE = 7.0
_MATPLOTLIB_CONFIGURED = False


def configure_matplotlib() -> None:
    global _MATPLOTLIB_CONFIGURED
    if _MATPLOTLIB_CONFIGURED:
        return
    texinputs_prefix = f".:{TEXDEPS_DIR}//:"
    current_texinputs = os.environ.get("TEXINPUTS", "")
    if texinputs_prefix not in current_texinputs:
        os.environ["TEXINPUTS"] = texinputs_prefix + current_texinputs
    use_tex = os.environ.get("HUQ_PAPER_DISABLE_TEX", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }
    rc_params = {
        "text.usetex": use_tex,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.unicode_minus": False,
        "font.size": CAPTION_FONTSIZE,
        "axes.labelsize": CAPTION_FONTSIZE,
        "axes.titlesize": PANEL_TITLE_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "figure.titlesize": SUPTITLE_FONTSIZE,
        "lines.linewidth": 1.8,
        "axes.linewidth": 1.0,
        "grid.linewidth": 1.0,
    }
    if use_tex:
        rc_params["text.latex.preamble"] = (
            r"\usepackage{amsmath}"
            r"\usepackage{amssymb}"
            r"\renewcommand{\familydefault}{\sfdefault}"
            r"\usepackage{sansmath}"
            r"\sansmath"
        )
    mpl.rcParams.update(rc_params)
    _MATPLOTLIB_CONFIGURED = True


def latex_diameter(diameter: str) -> str:
    return diameter_symbol(diameter)


def comma_tick_formatter(value: float, _position: int) -> str:
    if np.isclose(value, round(value), atol=1e-9):
        return f"{int(round(value)):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def apply_thousands_separators(axes: np.ndarray | list[plt.Axes]) -> None:
    formatter = FuncFormatter(comma_tick_formatter)
    for ax in np.ravel(np.asarray(axes, dtype=object)):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if max(abs(x0), abs(x1)) >= 1_000:
            ax.xaxis.set_major_formatter(formatter)
        if max(abs(y0), abs(y1)) >= 1_000:
            ax.yaxis.set_major_formatter(formatter)


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


def percent_text() -> str:
    return r"\%" if mpl.rcParams.get("text.usetex", False) else "%"


DIAMETER_INDEX = {
    "2.1": 1,
    "2.9": 2,
    "3.0": 3,
    "3.2": 4,
    "3.4": 5,
    "5.8": 6,
}


def diameter_symbol(diameter: str) -> str:
    return rf"$d_{{{DIAMETER_INDEX[diameter]}}}$"


def surrogate_symbol(diameter: str, reduced: bool = False) -> str:
    suffix = "^0" if reduced else ""
    return rf"$\mathcal{{S}}_{{d_{DIAMETER_INDEX[diameter]}}}{suffix}$"


def map_symbol(diameter: str) -> str:
    return rf"$\mathcal{{M}}_{{d_{DIAMETER_INDEX[diameter]}}}(\tilde{{\theta}}_{{d_{DIAMETER_INDEX[diameter]}}})$"


def display_modality_name(modality: str) -> str:
    if modality == "compression":
        return "Definity"
    if modality == "indentation":
        return "SonoVue"
    raise ValueError(f"Unknown modality: {modality}")


def modality_palette(modality: str) -> dict[str, str]:
    if modality == "compression":
        return {"dark": INK, "mid": INK_LIGHT, "faint": INK_FAINT}
    if modality == "indentation":
        return {"dark": GOLD, "mid": GOLD_LIGHT, "faint": GOLD_FAINT}
    raise ValueError(f"Unknown modality: {modality}")


def diameter_styles(modality: str, diameters: list[str]) -> dict[str, dict[str, str | float]]:
    palette = modality_palette(modality)
    base = [palette["faint"], palette["mid"], palette["dark"]]
    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "D"]
    styles: dict[str, dict[str, str | float]] = {}
    for diameter, color, linestyle, marker in zip(diameters, base, linestyles, markers):
        styles[diameter] = {
            "color": color,
            "linestyle": linestyle,
            "marker": marker,
        }
    return styles


FULL_PROP_ROOTS = {
    "compression": WORKFLOW_ROOT / "compression" / "full-model" / "production",
    "indentation": WORKFLOW_ROOT / "indentation" / "full-model" / "production",
}
REDUCED_PROP_ROOTS = {
    "compression": WORKFLOW_ROOT / "compression" / "reduced-model" / "production",
    "indentation": WORKFLOW_ROOT / "indentation" / "reduced-model" / "production",
}
SUPP_MAP_MIRHEO_ROOTS = {
    "compression": {
        "full": FULL_PROP_ROOTS["compression"] / "map_mirheo",
        "reduced": REDUCED_PROP_ROOTS["compression"] / "map_mirheo",
    },
    "indentation": {
        "full": FULL_PROP_ROOTS["indentation"] / "map_mirheo",
        "reduced": REDUCED_PROP_ROOTS["indentation"] / "map_mirheo",
    },
}


def load_modality_config(modality: str) -> dict:
    if modality == "compression":
        config_path = (
            REPO_ROOT / "inference" / "configs" / "production" / "inference_config_compression.yaml"
        )
    else:
        config_path = (
            REPO_ROOT / "inference" / "configs" / "production" / "inference_config_indentation.yaml"
        )
    with config_path.open("rb") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def load_reference_curve(modality: str, diameter: str) -> tuple[np.ndarray, np.ndarray]:
    config = load_modality_config(modality)
    data_dir = REPO_ROOT / config["data_dir"]
    data_files = config.get("data_files")
    if data_files:
        data_name = data_files.get(float(diameter), data_files.get(diameter))
        data_path = data_dir / data_name
        if data_path.suffix == ".csv":
            data_path = data_dir / f"{config['data_prefix']}{diameter}um.dat"
    else:
        data_path = data_dir / f"{config['data_prefix']}{diameter}um.dat"
    data = np.loadtxt(data_path, skiprows=1, ndmin=2)
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def load_map_parameters(modality: str, model_kind: str, diameter: str) -> dict:
    path = map_phase3b_path(modality, model_kind, diameter)
    if path.suffix == ".csv":
        row = pd.read_csv(path).iloc[0]
        if model_kind == "full":
            parameter_order = ["Yt", "kb", "b1", "b2", "a3", "a4", "d0"]
        else:
            parameter_order = ["Yt", "kb", "d0"]
        return {
            "parameters": [float(row[name]) for name in parameter_order],
            "sigma": float(row["sigma"]),
            "logLikelihood": float(row["logLikelihood"]),
            "logPrior": float(row["logPrior"]),
            "logPosterior": float(row["logPosterior"]),
        }
    return load_latest(path)


def extract_map_surrogate_curve(
    modality: str,
    model_kind: str,
    diameter: str,
) -> tuple[np.ndarray, np.ndarray]:
    map_data = load_map_parameters(modality, model_kind, diameter)
    x_dpd, _ = load_reference_curve(modality, diameter)
    params = list(map_data["parameters"])
    sigma = float(map_data.get("sigma", 0.0))
    sample = {"Parameters": params + [sigma]}

    with inference_config_environment(workflow_config_path(modality, model_kind)):
        if modality == "compression":
            from compression.evalkit.posterior_compression import compute_compression_surrogate

            compute_compression_surrogate(sample, x_dpd.tolist(), float(diameter))
        else:
            from indentation.evalkit.posterior_indentation import compute_indentation_surrogate

            compute_indentation_surrogate(sample, x_dpd.tolist(), float(diameter))

    y_dpd = np.asarray(sample["Reference Evaluations"], dtype=float)
    return x_dpd, y_dpd


def _map_mirheo_override_env(modality: str, model_kind: str, diameter: str) -> str:
    diameter_token = diameter.replace(".", "p").replace("-", "m")
    return f"HUQ_MAP_MIRHEO_OVERRIDE_{modality.upper()}_{model_kind.upper()}_{diameter_token.upper()}"


def load_supp_map_mirheo(modality: str, model_kind: str, diameter: str) -> dict | None:
    override = os.environ.get(_map_mirheo_override_env(modality, model_kind, diameter))
    if override:
        path = Path(override)
    else:
        base = SUPP_MAP_MIRHEO_ROOTS[modality][model_kind]
        prefix = "compression" if modality == "compression" else "indentation"
        candidates = [
            base / f"{diameter}um_map_mirheo.json",
            base / "results" / f"{prefix}_{diameter}um_result.json",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def resolve_parameters_file(modality: str, diameter: str) -> Path:
    diameter_specific = (
        REPO_ROOT / f"_init_compression_{diameter}um" / "parameter" / "parameters-default00001.yaml"
    )
    if diameter_specific.exists():
        return diameter_specific

    if modality == "compression":
        return REPO_ROOT / "compression" / "src" / "parameters-default.emb.yaml"
    return REPO_ROOT / "indentation" / "src" / "parameters-default.emb.yaml"


def load_scaling(modality: str, diameter: str) -> tuple[float, float]:
    with resolve_parameters_file(modality, diameter).open("rb") as handle:
        params = yaml.load(handle, Loader=yaml.CLoader)

    rho_water = params["rho_water"]
    rhow = params["rhow"]
    energy_factor = params["energyFactor"]
    kbol = params["kbol"]
    t0 = params["t0"]
    ul = params["ul"]
    fscale = params.get("fscale", 1.0)

    ue = energy_factor * kbol * t0
    um = rho_water * ul**3 / rhow
    ut = np.sqrt(um * ul**2 / ue)

    length_factor = ul * 1e9
    force_factor = (um * ul / ut**2) / fscale * 1e9
    return length_factor, force_factor


def convert_map_mirheo_to_real(
    modality: str, diameter: str, map_result: dict
) -> tuple[np.ndarray, np.ndarray]:
    length_factor, force_factor = load_scaling(modality, diameter)
    d0_offset = float(map_result.get("grid_config", {}).get("d0_offset", 0.0))

    if modality == "compression":
        # Legacy schema: displacement_points + forces
        # New schema: force_points + short_diameters (indentation only)
        x_dpd = np.asarray(map_result["displacement_points"], dtype=float) + d0_offset
        y_dpd = np.asarray(map_result["forces"], dtype=float)
        return x_dpd * length_factor, y_dpd * force_factor

    # Indentation legacy schema:
    #   displacement_points = force grid, forces = short diameters
    # Indentation new schema:
    #   force_points = force grid, short_diameters = short diameters
    if "force_points" in map_result and "short_diameters" in map_result:
        force_grid_dpd = np.asarray(map_result["force_points"], dtype=float)
        short_diameter_dpd = np.asarray(map_result["short_diameters"], dtype=float)
    else:
        force_grid_dpd = np.asarray(map_result["displacement_points"], dtype=float)
        short_diameter_dpd = np.asarray(map_result["forces"], dtype=float)

    initial_diameter_dpd = indentation_initial_diameter_dpd(diameter)
    displacement_dpd = initial_diameter_dpd - short_diameter_dpd + d0_offset
    return force_grid_dpd * force_factor, displacement_dpd * length_factor


def indentation_reference_diameter_dpd() -> float:
    param_path = REPO_ROOT / "indentation" / "src" / "parameter" / "parameters-default00001.yaml"
    with param_path.open("rb") as handle:
        params = yaml.load(handle, Loader=yaml.CLoader)
    return 2.0 * float(params["radp"])


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
    return indentation_reference_diameter_dpd()


def convert_axis(
    values: np.ndarray, kind: str, length_factor: float, force_factor: float
) -> np.ndarray:
    if kind == "displacement":
        return values * length_factor
    if kind == "force":
        return values * force_factor
    raise ValueError(f"Unknown axis kind: {kind}")


def convert_to_real(
    modality: str, diameter: str, x_dpd: np.ndarray, y_dpd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    length_factor, force_factor = load_scaling(modality, diameter)
    if modality == "compression":
        x_kind, y_kind = "displacement", "force"
    else:
        x_kind, y_kind = "force", "displacement"
    x_real = convert_axis(x_dpd, x_kind, length_factor, force_factor)
    y_real = convert_axis(y_dpd, y_kind, length_factor, force_factor)
    return x_real, y_real


def output_scale(modality: str, diameter: str) -> float:
    length_factor, force_factor = load_scaling(modality, diameter)
    return force_factor if modality == "compression" else length_factor


def load_latest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


@contextmanager
def inference_config_environment(config_path: Path):
    previous = os.environ.get("HUQ_INFERENCE_CONFIG")
    os.environ["HUQ_INFERENCE_CONFIG"] = str(config_path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HUQ_INFERENCE_CONFIG", None)
        else:
            os.environ["HUQ_INFERENCE_CONFIG"] = previous


def paper_results_root(root: Path) -> Path:
    """Support both historical `.../results` roots and workstation `.../production/setup` roots."""
    setup_root = root / "setup"
    if setup_root.exists():
        return setup_root
    return root


def convert_samples(data: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    samples = np.asarray(data["Results"]["Posterior Sample Database"], dtype=float)
    log_like = np.asarray(data["Results"]["Posterior Sample LogLikelihood Database"], dtype=float)
    log_prior = np.asarray(data["Results"]["Posterior Sample LogPrior Database"], dtype=float)
    names = [v["Name"] for v in data["Variables"]]
    expected_names = ["Yt", "kb", "d0", "[Sigma]"]
    assert names == expected_names, (
        f"Korali variable order mismatch: got {names}, expected {expected_names}. "
        "Update pretty_names mapping if the config changed."
    )
    converted = samples.copy()
    converted[:, 0] = converted[:, 0] * KA_FACTOR
    pretty_names = ["ka", "kb", "d0", "sigma"]
    return converted, pretty_names, log_like + log_prior


def reduced_path(modality: str, phase: str, diameter: str) -> Path:
    if modality == "compression":
        root = paper_results_root(REDUCED_PROP_ROOTS["compression"])
        prefix = "compression"
    else:
        root = paper_results_root(REDUCED_PROP_ROOTS["indentation"])
        prefix = "indentation"
    return root / f"results_{phase}" / f"{prefix}_{diameter}um" / "latest"


def propagation_latest(modality: str, model_kind: str, diameter: str) -> Path:
    roots = FULL_PROP_ROOTS if model_kind == "full" else REDUCED_PROP_ROOTS
    prefix = "compression" if modality == "compression" else "indentation"
    root = paper_results_root(roots[modality])
    latest_dir = root / "propagation_phase3b" / f"{prefix}_{diameter}um" / "latest"
    if latest_dir.exists():
        return latest_dir
    csv_root = root / "propagation_phase3b" / f"{prefix}_{diameter}um"
    predictive_csv = csv_root / "summary_predictive.csv"
    if predictive_csv.exists():
        return predictive_csv
    return csv_root / "summary.csv"


def map_phase3b_path(modality: str, model_kind: str, diameter: str) -> Path:
    root = FULL_PROP_ROOTS if model_kind == "full" else REDUCED_PROP_ROOTS
    base = paper_results_root(root[modality])
    map_parameters_dir = base / "MAP" / "map_parameters"
    if map_parameters_dir.exists():
        return map_parameters_dir / f"{diameter}um_map.json"
    prefix = "compression" if modality == "compression" else "indentation"
    csv_candidate = base / "map_phase3b" / f"{prefix}_{diameter}um.csv"
    if csv_candidate.exists():
        return csv_candidate
    return base / "map_phase3b" / f"{diameter}um_map.json"


def workflow_config_path(modality: str, model_kind: str) -> Path:
    if model_kind == "full":
        if modality == "compression":
            return REPO_ROOT / "inference" / "configs" / "production" / "inference_config_compression.yaml"
        return REPO_ROOT / "inference" / "configs" / "production" / "inference_config_indentation.yaml"
    if modality == "compression":
        return REPO_ROOT / "reduced" / "configs" / "production" / "reduced_config_compression.yaml"
    return REPO_ROOT / "reduced" / "configs" / "production" / "reduced_config_indentation.yaml"


def map_mirheo_path(modality: str, model_kind: str, diameter: str) -> Path | None:
    root = FULL_PROP_ROOTS if model_kind == "full" else REDUCED_PROP_ROOTS
    base = paper_results_root(root[modality])
    map_mirheo_dir = base / "MAP" / "map_mirheo"
    prefix = "compression" if modality == "compression" else "indentation"
    candidates = [
        map_mirheo_dir / f"{diameter}um_map_mirheo.json",
        base / "map_mirheo" / f"{diameter}um_map_mirheo.json",
        base / "map_mirheo" / "results" / f"{prefix}_{diameter}um_result.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_map_curve(
    modality: str, model_kind: str, diameter: str, *, apply_indent_d0_offset: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    x_dpd, _ = load_reference_curve(modality, diameter)
    target_map = load_map_parameters(modality, model_kind, diameter)
    target_params = np.asarray(target_map["parameters"], dtype=float)
    propagation_path = propagation_latest(modality, model_kind, diameter)
    propagation = None
    if propagation_path.name == "latest":
        propagation = load_latest(propagation_path)

    best_sample = None
    best_err = None
    if propagation is not None:
        for sample in propagation["Samples"]:
            params = np.asarray(sample["Parameters"], dtype=float)
            if params.shape[0] < target_params.shape[0]:
                continue
            # Propagation stores an extra trailing sigma entry; the MAP parameter
            # files do not. Match on the common prefix so the paper overlay uses the
            # same MAP-aligned propagated surrogate curve as the historical figure.
            err = float(np.max(np.abs(params[: target_params.shape[0]] - target_params)))
            if best_err is None or err < best_err:
                best_err = err
                best_sample = sample

    if best_sample is not None and best_err is not None and best_err <= 1e-12:
        y_dpd = np.asarray(best_sample["Reference Evaluations"], dtype=float)
        return x_dpd, y_dpd

    map_mirheo = map_mirheo_path(modality, model_kind, diameter)
    if map_mirheo is not None:
        map_result = load_latest(map_mirheo)
        if modality == "compression":
            x_dpd = np.asarray(map_result["displacement_points"], dtype=float)
            x_dpd = x_dpd + float(map_result["grid_config"]["d0_offset"])
            y_dpd = np.asarray(map_result["forces"], dtype=float)
        else:
            # Fallback only: indentation MAP Mirheo writes a force grid together
            # with the direct short-diameter response, so convert back to the
            # paper's displacement observable.
            if "force_points" in map_result and "short_diameters" in map_result:
                x_dpd = np.asarray(map_result["force_points"], dtype=float)
                short_diameter_dpd = np.asarray(map_result["short_diameters"], dtype=float)
            else:
                x_dpd = np.asarray(map_result["displacement_points"], dtype=float)
                short_diameter_dpd = np.asarray(map_result["forces"], dtype=float)
            d0_offset = float(map_result["grid_config"]["d0_offset"])
            initial_diameter_dpd = indentation_initial_diameter_dpd(diameter)
            y_dpd = initial_diameter_dpd - short_diameter_dpd
            if apply_indent_d0_offset:
                y_dpd = y_dpd + d0_offset
        return x_dpd, y_dpd

    raise RuntimeError(
        f"Could not locate a MAP-matching curve for {modality} {model_kind} {diameter}um"
    )


def quantiles(x: np.ndarray) -> tuple[float, float, float]:
    return float(np.quantile(x, 0.05)), float(np.median(x)), float(np.quantile(x, 0.95))


def load_best_holdout_curve_metrics(modality: str, diameter: str) -> pd.DataFrame:
    holdout_root = resolve_holdout_root(modality, diameter)
    summary = json.loads((holdout_root / "summary.json").read_text())
    metrics_candidates = [
        holdout_root / "architectures" / summary["best_architecture"] / "per_curve_metrics.csv",
        holdout_root / "per_curve_metrics.csv",
    ]
    metrics_path = next((path for path in metrics_candidates if path.exists()), None)
    if metrics_path is None:
        raise FileNotFoundError(
            "Could not locate grouped-holdout per-curve metrics for "
            f"{modality} {diameter}um under {holdout_root}"
        )
    return pd.read_csv(metrics_path)


def load_group_holdout_summary() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(GROUP_HOLDOUT_ROOT.glob("*/*um/**/summary.json")):
        with summary_path.open() as handle:
            row = json.load(handle)
        row["diameter_um"] = float(row["diameter_um"])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No grouped-holdout summaries found under {GROUP_HOLDOUT_ROOT}"
        )
    summary = pd.DataFrame(rows).sort_values(["modality", "diameter_um"]).reset_index(drop=True)
    summary.to_csv(FIGURES_DIR / "surrogate_group_holdout_summary.csv", index=False)
    return summary


def resolve_holdout_root(modality: str, diameter: str) -> Path:
    direct_root = GROUP_HOLDOUT_ROOT / modality / f"{diameter}um"
    dnn_root = direct_root / "dnn"
    if dnn_root.exists():
        return dnn_root
    return direct_root


def load_representative_holdout_curve(
    modality: str, diameter: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rep_path = resolve_holdout_root(modality, diameter) / "representative_curve.csv"
    df = pd.read_csv(rep_path)
    if modality == "compression":
        x_dpd = df["disp"].to_numpy(dtype=float)
        truth_col = "F_true" if "F_true" in df.columns else "y_true"
        pred_col = "F_pred" if "F_pred" in df.columns else "y_pred"
    else:
        x_dpd = df["F"].to_numpy(dtype=float)
        truth_col = "disp_true" if "disp_true" in df.columns else "y_true"
        pred_col = "disp_pred" if "disp_pred" in df.columns else "y_pred"
    return (
        x_dpd,
        df[truth_col].to_numpy(dtype=float),
        df[pred_col].to_numpy(dtype=float),
    )


def plot_reference_curves() -> None:
    configure_matplotlib()
    specs = [
        (
            display_modality_name("compression"),
            "compression",
            ["2.1", "2.9", "3.0"],
            "Displacement [nm]",
            "Force [nN]",
        ),
        (
            display_modality_name("indentation"),
            "indentation",
            ["3.2", "3.4", "5.8"],
            "Force [nN]",
            "Displacement [nm]",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.0), constrained_layout=False)

    for ax, (title, modality, diameters, xlabel, ylabel) in zip(np.ravel(axes), specs):
        styles = diameter_styles(modality, diameters)
        for diameter in diameters:
            x_dpd, y_dpd = load_reference_curve(modality, diameter)
            x_real, y_real = convert_to_real(modality, diameter, x_dpd, y_dpd)
            style = styles[diameter]
            ax.plot(
                x_real,
                y_real,
                color=style["color"],
                linestyle="None",
                marker=style["marker"],
                markersize=3.3,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=latex_diameter(diameter),
            )
        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=1, loc="best")

    add_panel_labels(axes)
    apply_thousands_separators(axes)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experimental_reference_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "experimental_reference_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_surrogate_validation() -> None:
    configure_matplotlib()
    summary = load_group_holdout_summary()
    pct = percent_text()
    representative_cases = [
        ("compression", "2.9"),
        ("indentation", "3.2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 5.9), constrained_layout=False)
    for ax, (modality, diameter) in zip(axes[0], representative_cases):
        palette = modality_palette(modality)
        x_dpd, y_true_dpd, y_pred_dpd = load_representative_holdout_curve(modality, diameter)
        row = summary[
            (summary["modality"] == modality) & (summary["diameter_um"] == float(diameter))
        ].iloc[0]
        metrics = load_best_holdout_curve_metrics(modality, diameter)
        rel_l2 = metrics["rel_l2_pct"].to_numpy(dtype=float)
        q90 = float(np.quantile(rel_l2, 0.90))
        x_real, y_true_real = convert_to_real(modality, diameter, x_dpd, y_true_dpd)
        _, y_pred_real = convert_to_real(modality, diameter, x_dpd, y_pred_dpd)

        ax.plot(
            x_real,
            y_pred_real,
            color=palette["dark"],
            linewidth=2.4,
            label="Surrogate prediction",
            zorder=2,
        )
        ax.plot(
            x_real,
            y_true_real,
            linestyle="None",
            marker="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.25,
            markeredgecolor="black",
            color="black",
            label=r"Held-out \textsc{dpd} curve",
            zorder=3,
        )
        annotation = (
            rf"Representative curve rel. $L^2$: {row['representative_curve_rel_l2_pct']:.2f}{pct}"
        )
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONTSIZE,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": palette["faint"],
            },
        )
        ax.set_title(surrogate_symbol(diameter), pad=6)
        if modality == "compression":
            ax.set_xlabel(r"Displacement [nm]")
            ax.set_ylabel(r"Force [nN]")
        else:
            ax.set_xlabel(r"Force [nN]")
            ax.set_ylabel(r"Displacement [nm]")
        ax.grid(True, alpha=0.22)

    bottom_specs = [
        (axes[1, 0], "compression", display_modality_name("compression")),
        (axes[1, 1], "indentation", display_modality_name("indentation")),
    ]
    bottom_handles_by_modality = {}
    for ax, modality, title in bottom_specs:
        palette = modality_palette(modality)
        subset = summary[summary["modality"] == modality].copy()
        subset = subset.sort_values("diameter_um")
        x = np.arange(len(subset))
        med = subset["best_median_curve_rel_l2_pct"].to_numpy(dtype=float)
        q90 = np.asarray(
            [
                np.quantile(
                    load_best_holdout_curve_metrics(modality, f"{diameter:.1f}")[
                        "rel_l2_pct"
                    ].to_numpy(dtype=float),
                    0.90,
                )
                for diameter in subset["diameter_um"]
            ],
            dtype=float,
        )
        ax.vlines(x, med, q90, color=palette["faint"], linewidth=4.0, zorder=1)
        med_handle = ax.scatter(
            x,
            med,
            s=55,
            color=palette["dark"],
            label=r"Median held-out curve",
            zorder=3,
        )
        q90_handle = ax.scatter(
            x,
            q90,
            s=58,
            marker="D",
            facecolors="white",
            edgecolors=palette["mid"],
            linewidths=1.4,
            label=r"90th-percentile held-out curve",
            zorder=4,
        )
        bottom_handles_by_modality[modality] = (
            [med_handle, q90_handle],
            [r"Median held-out curve", r"90th-percentile held-out curve"],
        )
        for xi, yi in zip(x, med):
            ax.text(
                xi,
                yi + 0.14 * max(q90),
                rf"{yi:.2f}{pct}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONTSIZE,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([diameter_symbol(f"{d:.1f}") for d in subset["diameter_um"]])
        ax.margins(x=0.28)
        ax.set_title(title, pad=6)
        ax.set_ylabel(rf"Relative $L^2$ error [{pct}]")
        ax.set_xlabel(r"Diameter")
        ax.grid(True, alpha=0.22)

    handles, labels = bottom_handles_by_modality["compression"]
    axes[1, 0].legend(
        handles,
        labels,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        fontsize=5.7,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    handles, labels = bottom_handles_by_modality["indentation"]
    axes[1, 1].legend(
        handles,
        labels,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.80, 0.02),
        fontsize=5.7,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    add_panel_labels(axes)
    apply_thousands_separators(axes)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "surrogate_validation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "surrogate_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_sensitivity_summary() -> pd.DataFrame:
    source_specs = [
        (
            "compression",
            2.9,
            REPO_ROOT
            / "compression"
            / "surrogate"
            / "sensitivity"
            / "scripts"
            / "csv"
            / "sobol_vs_disp_2.9um.csv",
            "displacement",
        ),
        (
            "indentation",
            3.2,
            REPO_ROOT
            / "indentation"
            / "surrogate"
            / "sensitivity"
            / "scripts"
            / "csv"
            / "sobol_vs_force_3.2um.csv",
            "force",
        ),
    ]

    if SOBOL_ROOT.exists():
        source_specs = [
            ("compression", 2.9, SOBOL_ROOT / "compression" / "dnn" / "sobol_vs_disp_2.9um.csv", "displacement"),
            ("indentation", 3.2, SOBOL_ROOT / "indentation" / "dnn" / "sobol_vs_force_3.2um.csv", "force"),
        ]

    rows = []
    for modality, diameter_um, source_csv, axis_column in source_specs:
        df = pd.read_csv(source_csv)
        axis_key = axis_column if axis_column in df.columns else "axis"
        for (parameter, index_type), subset in df.groupby(["parameter", "index_type"], sort=False):
            x = subset[axis_key].to_numpy(dtype=float)
            y = subset["value"].to_numpy(dtype=float)
            yconf = subset["confidence"].to_numpy(dtype=float)
            width = float(x.max() - x.min())
            integrated_s1 = float(np.trapz(y, x) / width) if width > 0 else float(np.mean(y))
            integrated_confidence = (
                float(np.trapz(yconf, x) / width) if width > 0 else float(np.mean(yconf))
            )
            rows.append(
                {
                    "modality": modality,
                    "diameter_um": diameter_um,
                    "parameter": parameter,
                    "index_type": index_type,
                    "integrated_s1": integrated_s1,
                    "integrated_confidence": integrated_confidence,
                    "source_csv": str(source_csv),
                }
            )

    summary = pd.DataFrame(rows)
    s1_summary = summary[summary["index_type"] == "S1"].copy()
    s1_summary.to_csv(FIGURES_DIR / "sensitivity.csv", index=False)
    s1_summary.to_csv(FIGURES_DIR / "sensitivity_bar_scores.csv", index=False)
    return summary


def plot_sensitivity() -> None:
    configure_matplotlib()
    df = load_sensitivity_summary()
    df = df[df["index_type"] == "S1"].copy()
    df["parameter"] = df["parameter"].replace({"Yt": "ka"})

    specs = [
        ("compression", "2.9", rf"Sensitivity of {surrogate_symbol('2.9')}"),
        ("indentation", "3.2", rf"Sensitivity of {surrogate_symbol('3.2')}"),
    ]
    param_order = ["ka", "kb", "b1", "b2", "a3", "a4"]
    param_labels = {
        "ka": r"$k_a$",
        "kb": r"$k_b$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "a3": r"$a_3$",
        "a4": r"$a_4$",
    }

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.85), constrained_layout=False)

    for ax, (modality, diameter, title) in zip(axes, specs):
        palette = modality_palette(modality)
        subset = df[(df["modality"] == modality) & (df["diameter_um"] == float(diameter))].copy()
        subset["parameter"] = pd.Categorical(
            subset["parameter"], categories=param_order, ordered=True
        )
        subset = subset.sort_values("parameter")

        x = np.arange(len(subset))
        y = subset["integrated_s1"].to_numpy(dtype=float)
        yerr = subset["integrated_confidence"].to_numpy(dtype=float)

        ax.bar(
            x,
            y,
            color=palette["mid"],
            edgecolor=palette["dark"],
            linewidth=1.0,
            width=0.72,
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=palette["dark"],
            elinewidth=1.0,
            capsize=3.0,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([param_labels[p] for p in subset["parameter"]])
        ax.set_ylim(bottom=min(-0.02, float(np.min(y - yerr)) - 0.01))
        ax.set_ylabel(r"Range-averaged first-order Sobol index")
        ax.set_title(title, pad=6)

    add_panel_labels(axes)
    apply_thousands_separators(axes)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase1_representative() -> None:
    configure_matplotlib()
    cases = [
        ("compression", "2.9", reduced_path("compression", "phase_1", "2.9")),
        ("indentation", "3.2", reduced_path("indentation", "phase_1", "3.2")),
    ]
    labels = [r"$k_a$", r"$k_b$", r"$d_0$", r"$\sigma$"]

    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH_IN, 3.9), constrained_layout=False)
    sci_y_panels = {(0, 0), (0, 1), (1, 0), (1, 1)}

    for row, (modality, diameter, path) in enumerate(cases):
        data = load_latest(path)
        samples, _, _ = convert_samples(data)
        color = modality_palette(modality)["dark"]
        for col, label in enumerate(labels):
            ax = axes[row, col]
            vals = samples[:, col]
            if row == 1:
                q02, q10, q50, q90, q98 = np.quantile(vals, [0.02, 0.10, 0.50, 0.90, 0.98])
                half_width = max(q90 - q50, q50 - q10)
                half_width = max(half_width * 1.25, (q98 - q02) * 0.30)
            else:
                q01, q05, q50, q95, q99 = np.quantile(vals, [0.01, 0.05, 0.50, 0.95, 0.99])
                half_width = max(q95 - q50, q50 - q05)
                half_width = max(half_width * 1.65, (q99 - q01) * 0.48)
            if half_width > 0.0:
                xmin = q50 - half_width
                xmax = q50 + half_width
            else:
                xmin = float(np.min(vals))
                xmax = float(np.max(vals))
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
                span = max(abs(q50), 1.0) * 1.0e-6
                xmin = float(q50 - span)
                xmax = float(q50 + span)

            visible_vals = vals[(vals >= xmin) & (vals <= xmax)]
            if visible_vals.size == 0:
                visible_vals = vals
            bin_edges = np.linspace(xmin, xmax, 101)
            ax.hist(
                visible_vals,
                bins=bin_edges,
                density=True,
                color=color,
                alpha=0.35,
                edgecolor="none",
            )
            ax.set_xlim(xmin, xmax)
            ax.set_title(label, pad=3)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            if (row, col) in sci_y_panels:
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 2))
                ax.yaxis.set_major_formatter(formatter)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            if col == 0:
                ax.set_ylabel(r"Density")

    add_panel_labels(axes)
    apply_thousands_separators(axes)
    fig.tight_layout()
    out_pdf = FIGURES_DIR / "phase1_reduced_representative.pdf"
    out_png = FIGURES_DIR / "phase1_reduced_representative.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase1_vs_phase3b_summary() -> None:
    configure_matplotlib()
    modalities = [
        (display_modality_name("compression"), "compression", ["2.1", "2.9", "3.0"], INK),
        (display_modality_name("indentation"), "indentation", ["3.2", "3.4", "5.8"], GOLD),
    ]
    params = [("k_a", 0), ("k_b", 1)]

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 4.2), constrained_layout=False)

    csv_lines = [
        "modality,diameter_um,param,phase1_q05,phase1_median,phase1_q95,phase3b_q05,phase3b_median,phase3b_q95"
    ]

    for row, (title, modality_key, diameters, color) in enumerate(modalities):
        palette = modality_palette(modality_key)
        for col, (param_label, idx) in enumerate(params):
            ax = axes[row, col]
            x = np.arange(len(diameters))
            phase1_stats = []
            phase3b_stats = []
            for diameter in diameters:
                p1_data = load_latest(reduced_path(modality_key, "phase_1", diameter))
                p3_data = load_latest(reduced_path(modality_key, "phase_3b", diameter))
                p1_samples, _, _ = convert_samples(p1_data)
                p3_samples, _, _ = convert_samples(p3_data)
                q1 = quantiles(p1_samples[:, idx])
                q3 = quantiles(p3_samples[:, idx])
                if modality_key == "indentation" and diameter in PHASE3B_MASKED_INDENTATION:
                    q3 = (np.nan, np.nan, np.nan)
                phase1_stats.append(q1)
                phase3b_stats.append(q3)
                csv_lines.append(
                    f"{title},{diameter},{param_label},{q1[0]},{q1[1]},{q1[2]},{q3[0]},{q3[1]},{q3[2]}"
                )

            p1 = np.asarray(phase1_stats)
            p3 = np.asarray(phase3b_stats)
            ax.errorbar(
                x - 0.08,
                p1[:, 1],
                yerr=np.vstack((p1[:, 1] - p1[:, 0], p1[:, 2] - p1[:, 1])),
                fmt="o",
                color=palette["mid"],
                ecolor=palette["mid"],
                elinewidth=1.5,
                capsize=4,
                markerfacecolor="white",
                markeredgewidth=1.6,
                label="Independent inference" if row == 0 and col == 0 else None,
            )
            ax.errorbar(
                x + 0.08,
                p3[:, 1],
                yerr=np.vstack((p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 1])),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=4,
                markeredgewidth=1.2,
                label="Hierarchical inference" if row == 0 and col == 0 else None,
            )
            ax.set_xticks(x)
            ax.set_xticklabels([diameter_symbol(d) for d in diameters])
            ax.grid(True, alpha=0.25)
            ax.set_title(rf"${param_label}$", pad=5)
            if col == 0:
                ax.set_ylabel(title)
            if row == 1:
                ax.set_xlabel(r"Diameter")

    axes[0, 0].legend(frameon=False, loc="best")
    bottom_palette = modality_palette("indentation")
    bottom_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=bottom_palette["dark"],
            markeredgewidth=1.6,
            color=bottom_palette["dark"],
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=bottom_palette["dark"],
            markeredgecolor=bottom_palette["dark"],
            markeredgewidth=1.2,
            color=bottom_palette["dark"],
            markersize=7,
        ),
    ]
    bottom_labels = ["Independent inference", "Hierarchical inference"]
    axes[1, 0].legend(bottom_handles, bottom_labels, frameon=False, loc="best")
    add_panel_labels(axes)
    apply_thousands_separators(axes)
    fig.tight_layout()
    out_pdf = FIGURES_DIR / "phase1_vs_phase3b_reduced_summary.pdf"
    out_png = FIGURES_DIR / "phase1_vs_phase3b_reduced_summary.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    (FIGURES_DIR / "phase1_vs_phase3b_reduced_summary.csv").write_text("\n".join(csv_lines) + "\n")


def generate_reduced_map_parameter_table() -> None:
    specs = [
        ("Compression", "compression", ["2.1", "2.9", "3.0"]),
        ("Indentation", "indentation", ["3.2", "3.4", "5.8"]),
    ]
    csv_lines = ["modality,diameter_um,ka,kb"]
    tex_rows = []
    for title, modality, diameters in specs:
        for diameter in diameters:
            data = load_map_parameters(modality, "reduced", diameter)
            params = np.asarray(data["parameters"], dtype=float)
            ka = float(params[0] * KA_FACTOR)
            kb = float(params[1])
            csv_lines.append(f"{title},{diameter},{ka},{kb}")
            tex_rows.append(f"{title} & {diameter} & {ka:.1f} & {kb:.1f} \\\\")

    (GENERATED_DIR / "map_parameters_reduced_phase3b.csv").write_text("\n".join(csv_lines) + "\n")
    (GENERATED_DIR / "map_parameters_reduced_phase3b.tex").write_text(
        "\n".join(
            [
                "% Auto-generated by _paper/v3/scripts/generate_reduced_story_assets.py",
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{Reduced Phase~3b MAP values of the dominant elastic parameters across modalities and diameters (\\textsc{dpd} units).}",
                "\\label{tab:map_parameters_reduced}",
                "{",
                "\\footnotesize",
                "\\setlength{\\tabcolsep}{6pt}",
                "\\begin{tabular}{@{}lccc@{}}",
                "\\toprule",
                "\\textbf{Modality} & \\textbf{Diameter [$\\mu$m]} & \\textbf{$k_a$} & \\textbf{$k_b$} \\\\",
                "\\midrule",
                *tex_rows,
                "\\bottomrule",
                "\\end{tabular}",
                "}",
                "\\end{table}",
                "",
            ]
        )
    )


def load_propagation_stats(
    prop_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if prop_path.suffix == ".csv":
        df = pd.read_csv(prop_path)
        stats = {
            "ci_90_lower": df["q05"].to_numpy(dtype=float),
            "median": df["median"].to_numpy(dtype=float),
            "ci_90_upper": df["q95"].to_numpy(dtype=float),
        }
        empty = np.empty((0, len(df)), dtype=float)
        return stats, empty, empty
    data = load_latest(prop_path)
    evals = np.asarray([sample["Reference Evaluations"] for sample in data["Samples"]], dtype=float)
    stds = np.asarray(
        [
            sample.get("Standard Deviation", np.zeros_like(sample["Reference Evaluations"]))
            for sample in data["Samples"]
        ],
        dtype=float,
    )
    stats = compute_interval_statistics(
        evals,
        stds,
        percentiles=(90,),
        include_observation_noise=True,
    )
    return stats, evals, stds


def load_propagation_samples(
    modality: str, diameter: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats, _, _ = load_propagation_stats(propagation_latest(modality, "reduced", diameter))
    return stats["ci_90_lower"], stats["median"], stats["ci_90_upper"]


def plot_reduced_only_propagated_bands() -> None:
    configure_matplotlib()
    specs = [
        (
            display_modality_name("compression"),
            "compression",
            ["2.1", "2.9", "3.0"],
            "Displacement [nm]",
            "Force [nN]",
        ),
        (
            display_modality_name("indentation"),
            "indentation",
            ["3.2", "3.4", "5.8"],
            "Force [nN]",
            "Displacement [nm]",
        ),
    ]
    modality_alphas = {
        "compression": {"2.1": 0.16, "2.9": 0.18, "3.0": 0.20},
        "indentation": {"3.2": 0.16, "3.4": 0.18, "5.8": 0.20},
    }

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.1), constrained_layout=False)

    for ax, (title, modality, diameters, xlabel, ylabel) in zip(axes, specs):
        styles = diameter_styles(modality, diameters)
        legend_handles = []
        legend_labels = []
        for diameter in diameters:
            x_dpd, y_dpd = load_reference_curve(modality, diameter)
            q05_dpd, med_dpd, q95_dpd = load_propagation_samples(modality, diameter)
            x, y = convert_to_real(modality, diameter, x_dpd, y_dpd)
            _, q05 = convert_to_real(modality, diameter, x_dpd, q05_dpd)
            _, med = convert_to_real(modality, diameter, x_dpd, med_dpd)
            _, q95 = convert_to_real(modality, diameter, x_dpd, q95_dpd)
            style = styles[diameter]
            color = str(style["color"])
            ax.fill_between(x, q05, q95, color=color, alpha=modality_alphas[modality][diameter])
            (line,) = ax.plot(
                x,
                med,
                color=color,
                linewidth=2.0,
                linestyle=str(style["linestyle"]),
                label=diameter_symbol(diameter),
            )
            legend_handles.append(line)
            legend_labels.append(diameter_symbol(diameter))
            ax.plot(
                x,
                y,
                linestyle="None",
                marker="o",
                markersize=3.4,
                markerfacecolor="white",
                markeredgewidth=1.1,
                markeredgecolor="black",
                color="black",
            )
        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=1.0,
                markeredgecolor="black",
                color="black",
            )
        )
        legend_labels.append("Experiments")
        ax.legend(legend_handles, legend_labels, frameon=False, loc="best")

    out_pdf = FIGURES_DIR / "posterior_predictive_reduced_only.pdf"
    out_png = FIGURES_DIR / "posterior_predictive_reduced_only.png"
    add_panel_labels(axes)
    apply_thousands_separators(axes)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_quick_map_mirheo_result(
    modality: str,
    model_kind: str,
    diameter: str,
    run_name: str | None = None,
) -> dict | None:
    if run_name is None:
        run_name = f"quick_{modality}_{model_kind}_{diameter.replace('.', 'p')}um"
    json_path = SCRATCH_MIRHEO_ROOT / run_name / f"{diameter}um_map_mirheo.json"
    if not json_path.exists():
        return None
    return load_latest(json_path)


def load_first_existing_map_result(
    modality: str,
    model_kind: str,
    diameter: str,
    run_names: list[str],
) -> dict | None:
    for run_name in run_names:
        result = load_quick_map_mirheo_result(modality, model_kind, diameter, run_name=run_name)
        if result is not None:
            return result
    return None


def generate_coverage_table() -> None:
    specs = [
        ("Compression", "compression", ["2.1", "2.9", "3.0"]),
        ("Indentation", "indentation", ["3.2", "3.4", "5.8"]),
    ]

    csv_lines = [
        "modality,diameter_um,coverage_full_pct,coverage_reduced_pct,mean_band_width_full,mean_band_width_reduced,width_reduction_pct,n_points,n_samples_full,n_samples_reduced,full_latest,reduced_latest"
    ]
    tex_rows = []

    for title, modality, diameters in specs:
        for diameter in diameters:
            full_path = propagation_latest(modality, "full", diameter)
            reduced_path_latest = propagation_latest(modality, "reduced", diameter)

            full_stats, full_evals, _ = load_propagation_stats(full_path)
            reduced_stats, reduced_evals, _ = load_propagation_stats(reduced_path_latest)
            _, ref_y = load_reference_curve(modality, diameter)

            full_low = full_stats["ci_90_lower"]
            full_high = full_stats["ci_90_upper"]
            reduced_low = reduced_stats["ci_90_lower"]
            reduced_high = reduced_stats["ci_90_upper"]

            coverage_full = 100.0 * np.mean((ref_y >= full_low) & (ref_y <= full_high))
            coverage_reduced = 100.0 * np.mean((ref_y >= reduced_low) & (ref_y <= reduced_high))
            width_full = float(np.mean(full_high - full_low))
            width_reduced = float(np.mean(reduced_high - reduced_low))
            if abs(width_full) < 1e-12:
                width_reduction = np.nan
            else:
                width_reduction = 100.0 * (width_full - width_reduced) / width_full

            csv_lines.append(
                ",".join(
                    [
                        title,
                        diameter,
                        str(coverage_full),
                        str(coverage_reduced),
                        str(width_full),
                        str(width_reduced),
                        str(width_reduction),
                        str(len(ref_y)),
                        str(full_evals.shape[0]),
                        str(reduced_evals.shape[0]),
                        str(full_path),
                        str(reduced_path_latest),
                    ]
                )
            )
            tex_rows.append(
                f"{title} & {diameter} & {coverage_full:.1f} & {coverage_reduced:.1f} & {width_reduction:.1f} \\\\"
            )

    table_csv = GENERATED_DIR / "table_coverage_v2.csv"
    table_tex = GENERATED_DIR / "table_coverage_v2.tex"
    table_csv.write_text("\n".join(csv_lines) + "\n")
    table_tex.write_text(
        "\n".join(
            [
                "% Auto-generated by _paper/v3/scripts/generate_reduced_story_assets.py",
                f"% compression_full={FULL_PROP_ROOTS['compression']}",
                f"% compression_reduced={REDUCED_PROP_ROOTS['compression']}",
                f"% indentation_full={FULL_PROP_ROOTS['indentation']}",
                f"% indentation_reduced={REDUCED_PROP_ROOTS['indentation']}",
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{Coverage of experimental points under pointwise 90\\% \\sout{parameter-propagation} \\textcolor{red}{posterior predictive} bands (full vs reduced), per diameter and modality. Band width reduction is reported as $100\\,(w_\\mathrm{full}-w_\\mathrm{reduced})/w_\\mathrm{full}$, where $w$ is the mean pointwise 90\\% band width. \\textcolor{red}{These intervals include the inferred $\\sigma$ term under the current heteroscedastic Gaussian likelihood, but they still do not include an explicit surrogate-error model, so the reported coverage values should be read as empirical agreement diagnostics under the current statistical model.}}",
                "\\label{tab:coverage}",
                "\\begin{tabular}{lcccc}",
                "\\toprule",
                "\\textbf{Modality} & \\textbf{Diameter [$\\mu$m]} & \\textbf{Coverage (Full) [\\%]} & \\textbf{Coverage (Reduced) [\\%]} & \\textbf{Width Reduct. [\\%]} \\\\",
                "\\midrule",
                *tex_rows,
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ]
        )
    )


def empirical_ensemble_crps(samples: np.ndarray, observation: float) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    term1 = np.mean(np.abs(x - observation))
    coeff = 2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    term2 = np.sum(coeff * x) / (n * n)
    return float(term1 - term2)


def estimate_curve_mean_crps(
    modality: str,
    model_kind: str,
    diameter: str,
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> float:
    latest = propagation_latest(modality, model_kind, diameter)
    if latest.suffix == ".csv":
        df = pd.read_csv(latest)
        means = np.asarray([df["median"].to_numpy(dtype=float)], dtype=float)
        # Convert the pointwise 90% interval into an equivalent Gaussian std.
        std_equiv = (df["q95"].to_numpy(dtype=float) - df["q05"].to_numpy(dtype=float)) / (
            2.0 * 1.6448536269514722
        )
        stds = np.asarray([std_equiv], dtype=float)
    else:
        data = load_latest(latest)
        means = np.asarray([sample["Reference Evaluations"] for sample in data["Samples"]], dtype=float)
        stds = np.asarray(
            [
                sample.get("Standard Deviation", np.zeros_like(sample["Reference Evaluations"]))
                for sample in data["Samples"]
            ],
            dtype=float,
        )
    _, ref_y = load_reference_curve(modality, diameter)

    scale = output_scale(modality, diameter)
    means = means * scale
    stds = stds * scale
    ref_y = ref_y * scale

    curve_scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        draws = means + stds * rng.standard_normal(size=means.shape)
        point_scores = [
            empirical_ensemble_crps(draws[:, j], ref_y[j]) for j in range(draws.shape[1])
        ]
        curve_scores.append(float(np.mean(point_scores)))

    return float(np.mean(curve_scores))


def generate_crps_table() -> None:
    specs = [
        ("Compression", "compression", ["2.1", "2.9", "3.0"], "nN"),
        ("Indentation", "indentation", ["3.2", "3.4", "5.8"], "nm"),
    ]

    csv_lines = ["modality,diameter_um,reduced_mean_crps,full_mean_crps,units"]
    tex_rows = []
    for title, modality, diameters, units in specs:
        for diameter in diameters:
            reduced = estimate_curve_mean_crps(modality, "reduced", diameter)
            full = estimate_curve_mean_crps(modality, "full", diameter)
            csv_lines.append(f"{title},{diameter},{reduced},{full},{units}")
            tex_rows.append(f"{title} & {diameter} & {reduced:.3f} & {full:.3f} & {units} \\\\")

    table_csv = GENERATED_DIR / "table_crps_trusted.csv"
    table_tex = GENERATED_DIR / "table_crps_trusted.tex"
    table_csv.write_text("\n".join(csv_lines) + "\n")
    table_tex.write_text(
        "\n".join(
            [
                "% Auto-generated by _paper/v3/scripts/generate_reduced_story_assets.py",
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{Curve-averaged posterior-predictive CRPS for the reduced and full workflows across the six datasets reported in the paper. Lower values indicate better probabilistic agreement. Compression CRPS is reported in force units (nN), and indentation CRPS is reported in displacement units (nm).}",
                "\\label{tab:crps_trusted}",
                "{",
                "\\footnotesize",
                "\\setlength{\\tabcolsep}{5pt}",
                "\\begin{tabular}{@{}lcccc@{}}",
                "\\toprule",
                "\\textbf{Modality} & \\textbf{Diameter [$\\mu$m]} & \\textbf{Reduced mean CRPS} & \\textbf{Full mean CRPS} & \\textbf{Units} \\\\",
                "\\midrule",
                *tex_rows,
                "\\bottomrule",
                "\\end{tabular}",
                "}",
                "\\end{table}",
                "",
            ]
        )
    )


def _draw_placeholder_panel(ax: plt.Axes, text: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "edgecolor": "0.7"},
    )


def plot_representative_confirmation() -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.0), constrained_layout=False)

    ax = axes[0]
    modality = "compression"
    diameter = "2.9"
    palette = modality_palette(modality)
    x_ref_dpd, y_ref_dpd = load_reference_curve(modality, diameter)
    q05_dpd, med_dpd, q95_dpd = load_propagation_samples(modality, diameter)
    x_ref, y_ref = convert_to_real(modality, diameter, x_ref_dpd, y_ref_dpd)
    _, q05 = convert_to_real(modality, diameter, x_ref_dpd, q05_dpd)
    _, med = convert_to_real(modality, diameter, x_ref_dpd, med_dpd)
    _, q95 = convert_to_real(modality, diameter, x_ref_dpd, q95_dpd)
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
    map_result = load_supp_map_mirheo(modality, "reduced", diameter)
    if map_result is not None:
        map_x, map_y = convert_map_mirheo_to_real(modality, diameter, map_result)
    else:
        try:
            map_x_dpd, map_y_dpd = extract_map_surrogate_curve(modality, "reduced", diameter)
            map_x, map_y = convert_to_real(modality, diameter, map_x_dpd, map_y_dpd)
        except RuntimeError:
            map_x, map_y = None, None
    if map_x is not None and map_y is not None:
        ax.plot(
            map_x,
            map_y,
            color=palette["mid"],
            linewidth=2.0,
            linestyle=(0, (5.0, 2.0)),
            label=rf"{map_symbol(diameter)} prediction",
            zorder=5,
        )
    ax.set_title(rf"{display_modality_name('compression')}, {map_symbol(diameter)}", pad=6)
    ax.set_xlabel(r"Displacement [nm]")
    ax.set_ylabel(r"Force [nN]")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    modality = "indentation"
    diameter = "3.2"
    palette = modality_palette(modality)
    x_ref_dpd, y_ref_dpd = load_reference_curve(modality, diameter)
    q05_dpd, med_dpd, q95_dpd = load_propagation_samples(modality, diameter)
    x_ref, y_ref = convert_to_real(modality, diameter, x_ref_dpd, y_ref_dpd)
    _, q05 = convert_to_real(modality, diameter, x_ref_dpd, q05_dpd)
    _, med = convert_to_real(modality, diameter, x_ref_dpd, med_dpd)
    _, q95 = convert_to_real(modality, diameter, x_ref_dpd, q95_dpd)
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
    map_result = load_supp_map_mirheo(modality, "reduced", diameter)
    if map_result is not None:
        map_x, map_y = convert_map_mirheo_to_real(modality, diameter, map_result)
    else:
        try:
            map_x_dpd, map_y_dpd = extract_map_surrogate_curve(modality, "reduced", diameter)
            map_x, map_y = convert_to_real(modality, diameter, map_x_dpd, map_y_dpd)
        except RuntimeError:
            map_x, map_y = None, None
    if map_x is not None and map_y is not None:
        ax.plot(
            map_x,
            map_y,
            color=palette["mid"],
            linewidth=2.0,
            linestyle=(0, (5.0, 2.0)),
            label=rf"{map_symbol(diameter)} prediction",
            zorder=5,
        )
    ax.set_title(rf"{display_modality_name('indentation')}, {map_symbol(diameter)}", pad=6)
    ax.set_xlabel(r"Force [nN]")
    ax.set_ylabel(r"Displacement [nm]")
    ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")
    add_panel_labels(axes)
    apply_thousands_separators(axes)
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "map_confirmation_reduced_representative.png", dpi=220, bbox_inches="tight"
    )
    fig.savefig(FIGURES_DIR / "map_confirmation_reduced_representative.pdf", bbox_inches="tight")
    plt.close(fig)


def merge_full_vs_reduced_map_overlays() -> None:
    configure_matplotlib()
    specs = [
        ("compression", ["2.1", "2.9", "3.0"], "Displacement [nm]", "Force [nN]"),
        ("indentation", ["3.2", "3.4", "5.8"], "Force [nN]", "Displacement [nm]"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.0), constrained_layout=False)
    type_handles = [
        (
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
        ),
        Line2D([0], [0], color=NEUTRAL, linewidth=1.15, linestyle="-"),
    ]
    type_labels = [r"Full: $\mathcal{S}_{d_i}$", r"Reduced: $\mathcal{S}_{d_i}^0$"]

    for ax, (modality, diameters, xlabel, ylabel) in zip(np.ravel(axes), specs):
        title = display_modality_name(modality)
        styles = diameter_styles(modality, diameters)

        for diameter in diameters:
            full_x_dpd, full_y_dpd = extract_map_surrogate_curve(modality, "full", diameter)
            reduced_x_dpd, reduced_y_dpd = extract_map_surrogate_curve(modality, "reduced", diameter)

            full_x, full_y = convert_to_real(modality, diameter, full_x_dpd, full_y_dpd)
            reduced_x, reduced_y = convert_to_real(modality, diameter, reduced_x_dpd, reduced_y_dpd)
            style = styles[diameter]
            color = str(style["color"])
            marker = str(style["marker"])

            ax.plot(
                reduced_x,
                reduced_y,
                color=color,
                linewidth=1.15,
                linestyle="-",
                zorder=2,
            )
            ax.plot(
                full_x,
                full_y,
                linestyle="None",
                marker=marker,
                markersize=4.3,
                markerfacecolor=color,
                markeredgewidth=0.0,
                color=color,
                zorder=3,
            )

        diameter_handles = [
            Line2D(
                [0],
                [0],
                color=str(styles[diameter]["color"]),
                linewidth=1.15,
                linestyle="-",
            )
            for diameter in diameters
        ]
        diameter_labels = [latex_diameter(diameter) for diameter in diameters]
        ax.legend(
            diameter_handles,
            diameter_labels,
            frameon=False,
            loc="best",
            fontsize=LEGEND_FONTSIZE,
            handlelength=1.8,
            labelspacing=0.3,
        )

        ax.set_title(title, pad=6, fontsize=PANEL_TITLE_FONTSIZE)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(xlabel, fontsize=CAPTION_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=CAPTION_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    add_panel_labels(axes)
    apply_thousands_separators(axes)
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
    fig.savefig(
        FIGURES_DIR / "full_vs_reduced_map_overlay.png",
        dpi=220,
        bbox_inches="tight",
        bbox_extra_artists=(top_legend,),
    )
    fig.savefig(
        FIGURES_DIR / "full_vs_reduced_map_overlay.pdf",
        bbox_inches="tight",
        bbox_extra_artists=(top_legend,),
    )
    plt.close(fig)


def plot_group_holdout_examples() -> None:
    configure_matplotlib()
    specs = [
        ("compression", ["2.1", "2.9", "3.0"], "Displacement [nm]", "Force [nN]"),
        ("indentation", ["3.2", "3.4", "5.8"], "Force [nN]", "Displacement [nm]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 4.4), constrained_layout=False)

    for row, (modality, diameters, xlabel, ylabel) in enumerate(specs):
        palette = modality_palette(modality)
        pred_color = palette["dark"]
        for col, diameter in enumerate(diameters):
            ax = axes[row, col]
            x_dpd, y_true_dpd, y_pred_dpd = load_representative_holdout_curve(modality, diameter)
            x_real, y_true_real = convert_to_real(modality, diameter, x_dpd, y_true_dpd)
            _, y_pred_real = convert_to_real(modality, diameter, x_dpd, y_pred_dpd)

            ax.plot(
                x_real,
                y_pred_real,
                color=pred_color,
                linewidth=1.15,
                label="Surrogate prediction" if row == 0 and col == 0 else None,
            )
            ax.plot(
                x_real,
                y_true_real,
                linestyle="None",
                marker="o",
                markersize=2.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                markeredgecolor="black",
                label=r"Held-out \textsc{dpd} curve" if row == 0 and col == 0 else None,
            )

            ax.set_title(latex_diameter(diameter), pad=5)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)

    axes[0, 0].legend(frameon=False, loc="best", fontsize=6.6)
    add_panel_labels(axes)
    apply_thousands_separators(axes)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "surrogate_group_holdout_examples.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "surrogate_group_holdout_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    plot_reference_curves()
    plot_surrogate_validation()
    plot_sensitivity()
    plot_phase1_representative()
    plot_phase1_vs_phase3b_summary()
    generate_reduced_map_parameter_table()
    plot_reduced_only_propagated_bands()
    generate_coverage_table()
    generate_crps_table()
    plot_representative_confirmation()
    merge_full_vs_reduced_map_overlays()
    plot_group_holdout_examples()
    print("Generated reduced-story assets in", FIGURES_DIR)


def generate_figures_only() -> None:
    configure_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    plot_reference_curves()
    plot_surrogate_validation()
    plot_sensitivity()
    plot_phase1_representative()
    plot_phase1_vs_phase3b_summary()
    plot_reduced_only_propagated_bands()
    plot_representative_confirmation()
    merge_full_vs_reduced_map_overlays()
    plot_group_holdout_examples()
    print("Generated reduced-story figures in", FIGURES_DIR)


if __name__ == "__main__":
    main()
