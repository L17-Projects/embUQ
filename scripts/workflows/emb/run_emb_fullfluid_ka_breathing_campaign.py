#!/usr/bin/env python3
"""Prepare, run, and summarize the full-fluid EMB frequency-versus-ka campaign.

Each production row is one full DPD realization. The campaign uses the
original full water-plus-gas ingredients, a smooth 5% post-equilibration
deflation, a fixed-coordinate fluid-settling hold, and free release from rest.
The controlled measurement inertia is recorded per row and the reported
physical frequency is ``f_raw * sqrt(mass_scale)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import emb_deflate_only_analysis as qualified_analysis
import run_emb_free_shell_breathing_protocol as protocol
from emb_particle_staging import (
    particle_staging_namespace,
    release_particle_staging,
    reserve_particle_staging,
)


BASE_MASS_SCALE = {
    "d1": 1.0,
    "d2": 1.0,
    "d3": 1.0,
    "d4": 5.0,
    "d5": 5.0,
    "d6": 20.0,
}

AGENT_DIAMETER_GRID_LABELS_UM = {
    "Definity": ("2.10", "2.30", "2.50", "2.70", "2.90", "3.00", "3.10", "3.20", "3.30", "3.40"),
    "SonoVue": ("3.20", "3.40", "3.70", "4.00", "4.30", "4.60", "4.90", "5.20", "5.50", "5.80"),
}
AGENT_DIAMETER_GRIDS_UM = {
    agent: tuple(float(value) for value in values)
    for agent, values in AGENT_DIAMETER_GRID_LABELS_UM.items()
}
AGENT_MODALITY = {
    "Definity": "compression",
    "SonoVue": "indentation",
}
AGENT_DEFAULT_MASS_SCALE = {
    "Definity": 1.0,
    "SonoVue": 5.0,
}
AGENT_KB_REF = {
    "Definity": 389.2325564501215,
    "SonoVue": 7850.288865935088,
}
AGENT_KB_SENSITIVITY_DIAMETER_UM = {
    "Definity": 2.90,
    "SonoVue": 4.60,
}
# Calibrate both the small-radius and large-radius SonoVue limits because the
# latter is the observability boundary that can require additional inertia.
MASS_CALIBRATION_DIAMETERS_UM = {
    "Definity": (3.00,),
    "SonoVue": (3.20, 5.80),
}
KB_SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)
DESIGN_PHASE_FILENAMES = {
    "production": "production_design.csv",
    "mass_rescue": "mass_rescue_design.csv",
    "calibration": "mass_calibration_design.csv",
    "kb_sensitivity": "kb_sensitivity_design.csv",
}
MAP_VALUES_FIELDNAMES = (
    "agent",
    "diameter_symbol",
    "modality",
    "dataset",
    "diameter_um",
    "radius_dpd",
    "radius_source",
    "source_label",
    "sample_index",
    "sample_count",
    "ka",
    "kb",
    "d0",
    "sigma",
    "legacy_Yt",
    "logLikelihood",
    "logPrior",
    "logPosterior",
    "state_path",
    "map_template_state_path",
)
PRODUCTION_POINT_COUNT = 200
CALIBRATION_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
CAMPAIGN_DT = 1.0e-4
CAMPAIGN_SAMPLE_EVERY = 150
CAMPAIGN_INITIAL_RADIUS_SCALE = 0.95
CAMPAIGN_POST_DEFLATION_RAMP_STEPS = 500
CAMPAIGN_POST_DEFLATION_HOLD_STEPS = 5000
CAMPAIGN_POST_DEFLATION_HOLD_UPDATE_EVERY_STEPS = 10
# The canonical inertia is fixed; one deterministic cycle-only rescue is
# frozen below for otherwise-qualified traces.
CAMPAIGN_MEASUREMENT_MASS_SCALE = 160.0
CAMPAIGN_RESCUE_MASS_SCALE = 320.0
CAMPAIGN_RESCUE_RELAX_STEPS = 240000
CAMPAIGN_RESCUE_SAMPLE_EVERY = 300
CAMPAIGN_RELAX_STEPS = 120000
CAMPAIGN_RADIAL_VELOCITY_KICK = 0.0
CAMPAIGN_RADIAL_VELOCITY_KICK_MODE = "add"
CAMPAIGN_KA_MIN = 6000.0
CAMPAIGN_KA_MAX = 30000.0
CAMPAIGN_SCREENING_MIN_R2 = qualified_analysis.MIN_FIT_R2
CAMPAIGN_SCREENING_MAX_FFT_DAMPED_RELATIVE_DELTA = qualified_analysis.MAX_FFT_DAMPED_RELATIVE_DELTA
CAMPAIGN_SCREENING_MIN_VISIBLE_CYCLES = qualified_analysis.MIN_OBSERVED_CYCLES
CAMPAIGN_STRICT_MIN_R2 = CAMPAIGN_SCREENING_MIN_R2
CAMPAIGN_STRICT_MAX_FFT_DAMPED_RELATIVE_DELTA = CAMPAIGN_SCREENING_MAX_FFT_DAMPED_RELATIVE_DELTA
CAMPAIGN_STRICT_MIN_VISIBLE_CYCLES = CAMPAIGN_SCREENING_MIN_VISIBLE_CYCLES
REPRESENTATIVE_PRODUCTION_KA_INDICES = frozenset((0, PRODUCTION_POINT_COUNT // 2, PRODUCTION_POINT_COUNT - 1))
DEFAULT_PARTICLE_STAGING_LIMIT_BYTES = 30 * 1024**3
DEFAULT_PARTICLE_STAGING_RESERVATION_BYTES = 512 * 1024**2


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _default_map_values() -> Path:
    return protocol.repo_root() / "papers" / "huq_emb" / "resonance_breathing_mode_handoff" / "latest_50k_map_values.csv"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _diameter_label(value: float) -> str:
    return f"{float(value):.2f}"


def _diameter_key(value: float) -> float:
    return round(float(value), 2)


def _agent_slug(agent: str) -> str:
    return str(agent).strip().lower()


def _node_id(agent: str, diameter_um: float) -> str:
    return f"{_agent_slug(agent)}_{_diameter_label(diameter_um).replace('.', 'p')}um"


def _factor_token(value: float) -> str:
    return f"{float(value):.3g}".replace(".", "p")


def _dataset_for(agent: str, diameter_um: float, source_bubble: Any | None) -> str:
    if source_bubble is not None:
        return str(source_bubble.dataset)
    return f"{AGENT_MODALITY[agent]}_{_diameter_label(diameter_um)}um"


def _source_bubbles_by_agent(bubbles: list[Any]) -> dict[str, list[Any]]:
    by_agent: dict[str, list[Any]] = {}
    for bubble in bubbles:
        by_agent.setdefault(str(bubble.agent), []).append(bubble)
    return by_agent


def _source_bubbles_by_agent_diameter(bubbles: list[Any]) -> dict[tuple[str, float], Any]:
    return {
        (str(bubble.agent), _diameter_key(float(bubble.diameter_um))): bubble
        for bubble in bubbles
    }


def _validate_kb_reference_medians(bubbles: list[Any]) -> dict[str, float]:
    by_agent = _source_bubbles_by_agent(bubbles)
    medians: dict[str, float] = {}
    for agent, expected in AGENT_KB_REF.items():
        values = sorted(float(bubble.kb) for bubble in by_agent.get(agent, []))
        if len(values) != 3:
            raise RuntimeError(f"{agent} kb reference requires exactly three accepted source-MAP entries, found {len(values)}.")
        median = values[1]
        if median != float(expected):
            raise RuntimeError(f"{agent} kb reference median drifted: {median!r} != approved {expected!r}.")
        medians[agent] = median
    return medians


def _agent_template_bubbles(bubbles: list[Any]) -> dict[str, Any]:
    templates: dict[str, Any] = {}
    for agent, kb_ref in AGENT_KB_REF.items():
        matches = [
            bubble
            for bubble in bubbles
            if str(bubble.agent) == agent and float(bubble.kb) == float(kb_ref)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {agent} MAP template at kb_ref={kb_ref!r}, found {len(matches)}.")
        templates[agent] = matches[0]
    return templates


def _radius_dpd_metadata(agent: str, diameter_um: float, source_bubble: Any | None) -> tuple[float, str]:
    if source_bubble is not None:
        return float(source_bubble.radius_dpd), str(source_bubble.radius_source)
    return float(diameter_um) / 0.5, "fallback_diameter_um_over_0p5"


def _base_mass_scale_for_surface_node(agent: str, diameter_um: float) -> tuple[float, str]:
    del agent, diameter_um
    return CAMPAIGN_MEASUREMENT_MASS_SCALE, "uniform_controlled_l0_measurement_inertia"


def _release_steps_for_surface_node(agent: str, diameter_um: float) -> int:
    del agent, diameter_um
    return CAMPAIGN_RELAX_STEPS


def _source_payload_from_template(template: Any) -> dict[str, Any]:
    return {
        "source_label": template.source_label,
        "sample_index": int(template.sample_index),
        "sample_count": int(template.sample_count),
        "d0": float(template.d0),
        "sigma": float(template.sigma),
        "legacy_Yt": float(template.legacy_yt),
        "logLikelihood": float(template.log_likelihood),
        "logPrior": float(template.log_prior),
        "logPosterior": float(template.log_posterior),
        "map_template_state_path": template.state_path,
    }


def _surface_node_metadata(
    *,
    agent: str,
    node_index: int,
    diameter_um: float,
    template: Any,
    source_by_agent_diameter: dict[tuple[str, float], Any],
) -> dict[str, Any]:
    source_bubble = source_by_agent_diameter.get((agent, _diameter_key(diameter_um)))
    radius_dpd, radius_dpd_source = _radius_dpd_metadata(agent, diameter_um, source_bubble)
    base_mass_scale, mass_scale_policy = _base_mass_scale_for_surface_node(agent, diameter_um)
    return {
        "agent": agent,
        "surface_node_index": int(node_index),
        "surface_node_id": _node_id(agent, diameter_um),
        "symbol": _node_id(agent, diameter_um),
        "historical_symbol": "" if source_bubble is None else source_bubble.symbol,
        "map_template_symbol": template.symbol,
        "map_template_dataset": template.dataset,
        "modality": AGENT_MODALITY[agent],
        "dataset": _dataset_for(agent, diameter_um, source_bubble),
        "diameter_um": float(diameter_um),
        "diameter_label_um": _diameter_label(diameter_um),
        "radius_um": float(diameter_um) / 2.0,
        "radius_label_um": _diameter_label(float(diameter_um) / 2.0),
        "radius_dpd": float(radius_dpd),
        "radius_dpd_source": radius_dpd_source,
        "base_mass_scale": float(base_mass_scale),
        "mass_scale_policy": mass_scale_policy,
        "state_path": f"generated_radius_node:{_node_id(agent, diameter_um)}",
        **_source_payload_from_template(template),
    }


def _surface_case_base(
    *,
    phase: str,
    case_index: int,
    node: dict[str, Any],
    ka_index: int,
    ka: float,
    kb: float,
    kb_role: str,
    kb_sensitivity_factor: float,
    mass_calibration_multiplier: float,
    mass_scale: float,
    equil_steps: int,
    hold_steps: int,
    hold_update_every_steps: int,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "case_index": int(case_index),
        **node,
        "ka_index": int(ka_index),
        "ka": float(ka),
        "kb": float(kb),
        "kb_ref": float(AGENT_KB_REF[str(node["agent"])]),
        "kb_role": kb_role,
        "kb_sensitivity_factor": float(kb_sensitivity_factor),
        "mass_scale": float(mass_scale),
        "mass_calibration_multiplier": float(mass_calibration_multiplier),
        "dt": CAMPAIGN_DT,
        "equil_steps": int(equil_steps),
        "relax_steps": int(_release_steps_for_surface_node(str(node["agent"]), float(node["diameter_um"]))),
        "sample_every": CAMPAIGN_SAMPLE_EVERY,
        "initial_radius_scale": CAMPAIGN_INITIAL_RADIUS_SCALE,
        "hold_steps": int(hold_steps),
        "hold_update_every_steps": int(hold_update_every_steps),
        "ramp_steps": CAMPAIGN_POST_DEFLATION_RAMP_STEPS,
        "excitation_mode": "prestrain",
        "radial_velocity_kick": CAMPAIGN_RADIAL_VELOCITY_KICK,
        "radial_velocity_kick_mode": CAMPAIGN_RADIAL_VELOCITY_KICK_MODE,
        "solvent_mode": "full",
        "water_shell_fsi_scale": protocol.WATER_SHELL_FSI_CONSERVATIVE_SCALE,
        "bouncer_mode": "on",
        "preparation": "half_cosine_deflation_fixed_hold_release_from_rest",
    }


def _row_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _write_design_plot(path: Path, production_rows: list[dict[str, Any]], *, design_mode: str) -> None:
    """Render the frozen run layout only; this is not a frequency-result plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    if design_mode == "agent-radius-surface":
        styles = {"Definity": ("tab:blue", "o"), "SonoVue": ("tab:orange", "s")}
        for agent, (color, marker) in styles.items():
            rows = [row for row in production_rows if row["agent"] == agent]
            ax.scatter(
                [float(row["ka"]) for row in rows],
                [float(row["radius_um"]) for row in rows],
                s=6,
                alpha=0.45,
                marker=marker,
                color=color,
                label=f"{agent}: {len(AGENT_DIAMETER_GRIDS_UM[agent])} radii x {PRODUCTION_POINT_COUNT} ka",
            )
        ax.set_ylabel("physical radius [um]")
        ax.set_title("Frozen full-fluid EMB DPD frequency campaign design (not measured frequencies)")
    else:
        ax.scatter(
            [float(row["ka"]) for row in production_rows],
            [float(row["diameter_um"]) for row in production_rows],
            s=6,
            alpha=0.45,
            color="tab:blue",
        )
        ax.set_ylabel("reported diameter [um]")
        ax.set_title("Legacy six-bubble EMB ka campaign design (not measured frequencies)")
    ax.set_xlabel("Lim ka [Mirheo parameter]")
    ax.set_xlim(CAMPAIGN_KA_MIN, CAMPAIGN_KA_MAX)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _case_root(campaign_root: Path, row: dict[str, Any]) -> Path:
    if row.get("surface_node_id"):
        agent = _agent_slug(str(row["agent"]))
        node_id = str(row["surface_node_id"])
        phase = str(row["phase"])
        if phase == "production":
            leaf = f"ka-index-{int(row['ka_index']):03d}"
        elif phase == "mass_rescue":
            leaf = f"ka-index-{int(row['ka_index']):03d}"
        elif phase == "calibration":
            leaf = f"mass-multiplier-{_factor_token(float(row['mass_calibration_multiplier']))}"
        elif phase == "kb_sensitivity":
            leaf = f"kb-factor-{_factor_token(float(row['kb_sensitivity_factor']))}"
        else:
            leaf = f"case-{int(row['case_index']):04d}"
        return campaign_root / phase / agent / node_id / leaf
    return (
        campaign_root
        / str(row["phase"])
        / str(row["symbol"])
        / f"ka-index-{int(row['ka_index']):03d}"
    )


def _case_metrics_path(campaign_root: Path, row: dict[str, Any]) -> Path:
    return _case_root(campaign_root, row) / str(row["symbol"]) / "seed-000" / "campaign_case_metrics.json"


def _replacement_root(campaign_root: Path, row: dict[str, Any], replacement_id: str) -> Path:
    """Return an append-only evidence root for a measurement-only rerun.

    A replacement never occupies the canonical production path.  The canonical
    case remains the record of the frozen 24,000-step screening protocol, while
    this separate root records a longer observation window when that window is
    demonstrably too short to resolve a slow l=0 ringdown.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", replacement_id):
        raise ValueError("replacement_id must contain only letters, digits, '.', '_', or '-' and begin alphanumerically.")
    return (
        campaign_root
        / "targeted_replacements"
        / str(row["phase"])
        / _agent_slug(str(row["agent"]))
        / str(row["surface_node_id"])
        / f"ka-index-{int(row['ka_index']):03d}"
        / replacement_id
    )


def _replacement_row(
    row: dict[str, Any],
    *,
    replacement_id: str,
    relax_steps: int,
    sample_every: int,
    reason: str,
) -> dict[str, Any]:
    """Create a measurement-only variation without altering physical inputs."""

    if str(row.get("phase")) != "production":
        raise ValueError("Targeted replacements are only supported for production surface rows.")
    if relax_steps <= int(row["relax_steps"]):
        raise ValueError("replacement relax_steps must be greater than the canonical production duration.")
    if sample_every <= 0 or sample_every > int(row["sample_every"]):
        raise ValueError("replacement sample_every must be positive and no coarser than the canonical cadence.")
    if not reason.strip():
        raise ValueError("replacement reason must be nonempty.")

    replacement = dict(row)
    replacement.update(
        {
            "replacement_id": replacement_id,
            "replacement_of_phase": str(row["phase"]),
            "replacement_of_case_index": int(row["case_index"]),
            "replacement_reason": reason.strip(),
            "relax_steps": int(relax_steps),
            "sample_every": int(sample_every),
        }
    )
    return replacement


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_design(campaign_root: Path, phase: str) -> list[dict[str, Any]]:
    filename = DESIGN_PHASE_FILENAMES[phase]
    path = campaign_root / "design" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing campaign design: {path}. Run prepare first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    numeric_int = {
        "case_index",
        "surface_node_index",
        "sample_index",
        "sample_count",
        "ka_index",
        "equil_steps",
        "relax_steps",
        "sample_every",
        "hold_steps",
        "hold_update_every_steps",
        "ramp_steps",
        "source_case_index",
    }
    numeric_float = {
        "diameter_um",
        "radius_um",
        "radius_dpd",
        "ka",
        "kb",
        "kb_ref",
        "kb_sensitivity_factor",
        "base_mass_scale",
        "mass_scale",
        "mass_calibration_multiplier",
        "canonical_mass_scale",
        "dt",
        "initial_radius_scale",
        "radial_velocity_kick",
        "water_shell_fsi_scale",
        "d0",
        "sigma",
        "legacy_Yt",
        "logLikelihood",
        "logPrior",
        "logPosterior",
    }
    converted: dict[str, Any] = dict(row)
    for key in numeric_int:
        if key in converted and converted[key] != "":
            converted[key] = int(converted[key])
    for key in numeric_float:
        if key in converted and converted[key] != "":
            converted[key] = float(converted[key])
    return converted


def _mass_rescue_eligible(metric: dict[str, Any]) -> tuple[bool, list[str]]:
    """Allow more inertia only for a finite, clean l=0 trace lacking cycles."""

    checks = metric.get("checks") if isinstance(metric.get("checks"), dict) else {}
    required = (
        "finite_observable",
        "finite_volume",
        "finite_raw_frequency",
        "damped_sine_r2",
        "fft_damped_agreement",
        "amplitude_to_residual",
        "integrated_l0_mode_purity",
        "time_series_csv",
        "time_series_png",
        "spectrum_png",
        "mode_purity_png",
    )
    reasons = [f"{name}=false" for name in required if checks.get(name) is not True]
    cycles = metric.get("visible_alternating_cycles")
    if cycles is None or not math.isfinite(float(cycles)):
        reasons.append("visible cycle count is unavailable")
    elif float(cycles) >= CAMPAIGN_SCREENING_MIN_VISIBLE_CYCLES:
        reasons.append("canonical trace already contains at least three cycles")
    case = metric.get("case") if isinstance(metric.get("case"), dict) else {}
    if not math.isclose(float(case.get("mass_scale", math.nan)), CAMPAIGN_MEASUREMENT_MASS_SCALE):
        reasons.append("canonical run did not use the frozen 160x measurement inertia")
    if metric.get("status") in {"runner_error", "missing_result"}:
        reasons.append(f"canonical status is {metric.get('status')}")
    return not reasons, reasons


def plan_mass_rescues(args: argparse.Namespace) -> int:
    """Freeze the deterministic 320x reruns after all canonical rows finish."""

    campaign_root = args.campaign_root.resolve()
    production = [_coerce_row(row) for row in _read_design(campaign_root, "production")]
    rescue_rows: list[dict[str, Any]] = []
    incomplete: list[int] = []
    rejected: list[dict[str, Any]] = []
    for row in production:
        metric_path = _case_metrics_path(campaign_root, row)
        if not metric_path.is_file():
            incomplete.append(int(row["case_index"]))
            continue
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        eligible, reasons = _mass_rescue_eligible(metric)
        if not eligible:
            rejected.append({"case_index": int(row["case_index"]), "reasons": reasons})
            continue
        rescue = dict(row)
        rescue.update(
            {
                "phase": "mass_rescue",
                "case_index": len(rescue_rows),
                "source_case_index": int(row["case_index"]),
                "canonical_mass_scale": float(row["mass_scale"]),
                "mass_scale": CAMPAIGN_RESCUE_MASS_SCALE,
                "mass_calibration_multiplier": CAMPAIGN_RESCUE_MASS_SCALE
                / CAMPAIGN_MEASUREMENT_MASS_SCALE,
                "relax_steps": CAMPAIGN_RESCUE_RELAX_STEPS,
                "sample_every": CAMPAIGN_RESCUE_SAMPLE_EVERY,
                "mass_scale_policy": "deterministic_cycle_only_rescue",
                "rescue_reason": "canonical 160x trace passed finite, fit, FFT, amplitude, and l0-purity gates but had fewer than three visible cycles",
                "canonical_metric_path": str(metric_path),
            }
        )
        rescue_rows.append(rescue)
    if incomplete:
        raise RuntimeError(
            f"Refusing to freeze rescues while {len(incomplete)} canonical production rows lack metrics."
        )

    design_path = campaign_root / "design" / DESIGN_PHASE_FILENAMES["mass_rescue"]
    manifest_path = campaign_root / "design" / "mass_rescue_manifest.json"
    if (design_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError("Mass-rescue design already exists; it is immutable after submission.")
    fieldnames = _row_fieldnames(rescue_rows) if rescue_rows else _row_fieldnames(production) + [
        "source_case_index",
        "canonical_mass_scale",
        "rescue_reason",
        "canonical_metric_path",
    ]
    _write_csv(design_path, rescue_rows, list(dict.fromkeys(fieldnames)))
    manifest = {
        "schema": "mesouq.emb_fullfluid_mass_rescue_design.v1",
        "created_utc": _utc_stamp(),
        "canonical_case_count": len(production),
        "rescue_case_count": len(rescue_rows),
        "canonical_mass_scale": CAMPAIGN_MEASUREMENT_MASS_SCALE,
        "rescue_mass_scale": CAMPAIGN_RESCUE_MASS_SCALE,
        "rescue_relax_steps": CAMPAIGN_RESCUE_RELAX_STEPS,
        "rescue_sample_every": CAMPAIGN_RESCUE_SAMPLE_EVERY,
        "frequency_correction": "f_physical_mhz = f_raw_mhz * sqrt(membrane_mass_scale)",
        "eligibility": (
            "finite radius and volume; finite damped-sine frequency; R2, FFT, amplitude, "
            "integrated l0 purity, and artifact gates pass; visible cycles < 3"
        ),
        "ineligible_case_count": len(rejected),
        "ineligible_cases": rejected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rescue_cases": len(rescue_rows), "design": str(design_path)}, indent=2))
    return 0


def _release_steps(symbol: str) -> int:
    # d6 uses a 20x physical vertex mass, so it needs a longer raw-time window
    # to contain at least four periods before mass correction.
    return 20000 if symbol == "d6" else 16000


def _validate_surface_design(
    *,
    production_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    kb_sensitivity_rows: list[dict[str, Any]],
    ka_values: np.ndarray,
) -> None:
    expected_production = sum(len(grid) for grid in AGENT_DIAMETER_GRIDS_UM.values()) * PRODUCTION_POINT_COUNT
    if len(production_rows) != expected_production:
        raise RuntimeError(f"Expected {expected_production} production rows, found {len(production_rows)}.")
    if sorted(int(row["case_index"]) for row in production_rows) != list(range(expected_production)):
        raise RuntimeError("Production case_index values must be contiguous from zero.")

    for agent, grid in AGENT_DIAMETER_GRIDS_UM.items():
        agent_rows = [row for row in production_rows if row["agent"] == agent]
        if len(agent_rows) != len(grid) * PRODUCTION_POINT_COUNT:
            raise RuntimeError(f"{agent} production row count is not rectangular.")
        for node_index, diameter_um in enumerate(grid):
            node_rows = [
                row
                for row in agent_rows
                if int(row["surface_node_index"]) == node_index
                and _diameter_key(float(row["diameter_um"])) == _diameter_key(diameter_um)
            ]
            if len(node_rows) != PRODUCTION_POINT_COUNT:
                raise RuntimeError(f"{agent} {_diameter_label(diameter_um)} um has {len(node_rows)} rows.")
            node_rows = sorted(node_rows, key=lambda row: int(row["ka_index"]))
            for index, (row, ka) in enumerate(zip(node_rows, ka_values, strict=True)):
                if int(row["ka_index"]) != index:
                    raise RuntimeError(f"{agent} {_diameter_label(diameter_um)} um has non-contiguous ka_index values.")
                if float(row["ka"]) != float(ka):
                    raise RuntimeError(f"{agent} {_diameter_label(diameter_um)} um ka grid drifted at index {index}.")
                if float(row["kb"]) != float(AGENT_KB_REF[agent]):
                    raise RuntimeError(f"{agent} production kb must equal the approved kb_ref.")
                if float(row["radius_um"]) != float(row["diameter_um"]) / 2.0:
                    raise RuntimeError(f"{agent} production radius_um must be diameter_um / 2.")

    expected_calibration = sum(len(nodes) for nodes in MASS_CALIBRATION_DIAMETERS_UM.values()) * len(
        CALIBRATION_MULTIPLIERS
    )
    if len(calibration_rows) != expected_calibration:
        raise RuntimeError(f"Expected {expected_calibration} mass calibration rows, found {len(calibration_rows)}.")
    for agent, diameters_um in MASS_CALIBRATION_DIAMETERS_UM.items():
        rows = [row for row in calibration_rows if row["agent"] == agent]
        expected_rows = len(diameters_um) * len(CALIBRATION_MULTIPLIERS)
        if len(rows) != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} {agent} mass calibration rows, found {len(rows)}.")
        for diameter_um in diameters_um:
            node_rows = [
                row
                for row in rows
                if _diameter_key(float(row["diameter_um"])) == _diameter_key(diameter_um)
            ]
            if len(node_rows) != len(CALIBRATION_MULTIPLIERS):
                raise RuntimeError(f"{agent} {diameter_um:.2f} um mass calibration is incomplete.")
            if {float(row["mass_calibration_multiplier"]) for row in node_rows} != set(CALIBRATION_MULTIPLIERS):
                raise RuntimeError(f"{agent} {diameter_um:.2f} um mass calibration multiplier drifted.")
    expected_kb_rows = len(AGENT_DIAMETER_GRIDS_UM) * len(KB_SENSITIVITY_FACTORS)
    if len(kb_sensitivity_rows) != expected_kb_rows:
        raise RuntimeError(f"Expected {expected_kb_rows} kb sensitivity rows, found {len(kb_sensitivity_rows)}.")
    for agent in AGENT_DIAMETER_GRIDS_UM:
        rows = sorted(
            [row for row in kb_sensitivity_rows if row["agent"] == agent],
            key=lambda row: float(row["kb_sensitivity_factor"]),
        )
        if [float(row["kb_sensitivity_factor"]) for row in rows] != list(KB_SENSITIVITY_FACTORS):
            raise RuntimeError(f"{agent} kb sensitivity factors drifted.")
        for row in rows:
            if float(row["ka"]) != 15000.0:
                raise RuntimeError(f"{agent} kb sensitivity rows must use ka=15000.")
            if _diameter_key(float(row["diameter_um"])) != _diameter_key(AGENT_KB_SENSITIVITY_DIAMETER_UM[agent]):
                raise RuntimeError(f"{agent} kb sensitivity central diameter drifted.")
            expected_kb = float(AGENT_KB_REF[agent]) * float(row["kb_sensitivity_factor"])
            if float(row["kb"]) != expected_kb:
                raise RuntimeError(f"{agent} kb sensitivity kb value drifted.")


def _build_surface_design(
    *,
    bubbles: list[Any],
    ka_values: np.ndarray,
    hold_steps: int,
    hold_update_every_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    kb_medians = _validate_kb_reference_medians(bubbles)
    templates = _agent_template_bubbles(bubbles)
    source_by_agent_diameter = _source_bubbles_by_agent_diameter(bubbles)

    production_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    kb_sensitivity_rows: list[dict[str, Any]] = []
    equil_steps_by_agent: dict[str, int] = {}
    nodes_by_agent: dict[str, list[dict[str, Any]]] = {}
    case_index = 0
    calibration_index = 0
    kb_sensitivity_index = 0
    for agent, diameters_um in AGENT_DIAMETER_GRIDS_UM.items():
        template = templates[agent]
        equil_steps, _ = protocol._default_protocol_steps(template)
        equil_steps_by_agent[agent] = int(equil_steps)
        nodes_by_agent[agent] = []
        for node_index, diameter_um in enumerate(diameters_um):
            node = _surface_node_metadata(
                agent=agent,
                node_index=node_index,
                diameter_um=diameter_um,
                template=template,
                source_by_agent_diameter=source_by_agent_diameter,
            )
            nodes_by_agent[agent].append(node)
            for ka_index, ka in enumerate(ka_values):
                production_rows.append(
                    _surface_case_base(
                        phase="production",
                        case_index=case_index,
                        node=node,
                        ka_index=ka_index,
                        ka=float(ka),
                        kb=float(AGENT_KB_REF[agent]),
                        kb_role="reference",
                        kb_sensitivity_factor=1.0,
                        mass_calibration_multiplier=1.0,
                        mass_scale=float(node["base_mass_scale"]),
                        equil_steps=equil_steps,
                        hold_steps=hold_steps,
                        hold_update_every_steps=hold_update_every_steps,
                    )
                )
                case_index += 1

        central_diameter = AGENT_KB_SENSITIVITY_DIAMETER_UM[agent]
        central_node = nodes_by_agent[agent][diameters_um.index(central_diameter)]
        for calibration_diameter in MASS_CALIBRATION_DIAMETERS_UM[agent]:
            calibration_node = nodes_by_agent[agent][diameters_um.index(calibration_diameter)]
            for multiplier in CALIBRATION_MULTIPLIERS:
                calibration_rows.append(
                    _surface_case_base(
                        phase="calibration",
                        case_index=calibration_index,
                        node=calibration_node,
                        ka_index=-1,
                        ka=15000.0,
                        kb=float(AGENT_KB_REF[agent]),
                        kb_role="reference",
                        kb_sensitivity_factor=1.0,
                        mass_calibration_multiplier=float(multiplier),
                        mass_scale=float(calibration_node["base_mass_scale"]) * float(multiplier),
                        equil_steps=equil_steps,
                        hold_steps=hold_steps,
                        hold_update_every_steps=hold_update_every_steps,
                    )
                )
                calibration_index += 1
        for factor in KB_SENSITIVITY_FACTORS:
            kb_sensitivity_rows.append(
                _surface_case_base(
                    phase="kb_sensitivity",
                    case_index=kb_sensitivity_index,
                    node=central_node,
                    ka_index=-1,
                    ka=15000.0,
                    kb=float(AGENT_KB_REF[agent]) * float(factor),
                    kb_role="sensitivity",
                    kb_sensitivity_factor=float(factor),
                    mass_calibration_multiplier=1.0,
                    mass_scale=float(central_node["base_mass_scale"]),
                    equil_steps=equil_steps,
                    hold_steps=hold_steps,
                    hold_update_every_steps=hold_update_every_steps,
                )
            )
            kb_sensitivity_index += 1

    _validate_surface_design(
        production_rows=production_rows,
        calibration_rows=calibration_rows,
        kb_sensitivity_rows=kb_sensitivity_rows,
        ka_values=ka_values,
    )
    metadata = {
        "source_map_kb_medians": kb_medians,
        "agent_templates": {
            agent: {
                "symbol": template.symbol,
                "dataset": template.dataset,
                "source_label": template.source_label,
                "diameter_um": float(template.diameter_um),
                "legacy_Yt": float(template.legacy_yt),
            }
            for agent, template in templates.items()
        },
        "equil_steps_by_agent": equil_steps_by_agent,
    }
    return production_rows, calibration_rows, kb_sensitivity_rows, metadata


def _build_six_bubble_design(
    *,
    bubbles: list[Any],
    ka_values: np.ndarray,
    hold_steps: int,
    hold_update_every_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    production_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    case_index = 0
    calibration_index = 0
    for bubble in bubbles:
        equil_steps, _ = protocol._default_protocol_steps(bubble)
        for ka_index, ka in enumerate(ka_values):
            production_rows.append(
                {
                    "phase": "production",
                    "case_index": case_index,
                    "symbol": bubble.symbol,
                    "agent": bubble.agent,
                    "modality": bubble.modality,
                    "dataset": bubble.dataset,
                    "diameter_um": float(bubble.diameter_um),
                    "radius_um": float(bubble.diameter_um) / 2.0,
                    "radius_dpd": float(bubble.radius_dpd),
                    "radius_dpd_source": bubble.radius_source,
                    "ka_index": ka_index,
                    "ka": float(ka),
                    "kb": float(bubble.kb),
                    "base_mass_scale": CAMPAIGN_MEASUREMENT_MASS_SCALE,
                    "mass_scale": CAMPAIGN_MEASUREMENT_MASS_SCALE,
                    "mass_calibration_multiplier": 1.0,
                    "dt": CAMPAIGN_DT,
                    "equil_steps": int(equil_steps),
                    "relax_steps": CAMPAIGN_RELAX_STEPS,
                    "sample_every": CAMPAIGN_SAMPLE_EVERY,
                    "initial_radius_scale": CAMPAIGN_INITIAL_RADIUS_SCALE,
                    "hold_steps": int(hold_steps),
                    "hold_update_every_steps": int(hold_update_every_steps),
                    "ramp_steps": CAMPAIGN_POST_DEFLATION_RAMP_STEPS,
                    "excitation_mode": "prestrain",
                    "radial_velocity_kick": CAMPAIGN_RADIAL_VELOCITY_KICK,
                    "radial_velocity_kick_mode": CAMPAIGN_RADIAL_VELOCITY_KICK_MODE,
                    "solvent_mode": "full",
                    "water_shell_fsi_scale": protocol.WATER_SHELL_FSI_CONSERVATIVE_SCALE,
                    "bouncer_mode": "on",
                    "preparation": "half_cosine_deflation_fixed_hold_release_from_rest",
                }
            )
            case_index += 1
        for multiplier in CALIBRATION_MULTIPLIERS:
            calibration_rows.append(
                {
                    "phase": "calibration",
                    "case_index": calibration_index,
                    "symbol": bubble.symbol,
                    "agent": bubble.agent,
                    "modality": bubble.modality,
                    "dataset": bubble.dataset,
                    "diameter_um": float(bubble.diameter_um),
                    "radius_um": float(bubble.diameter_um) / 2.0,
                    "radius_dpd": float(bubble.radius_dpd),
                    "radius_dpd_source": bubble.radius_source,
                    "ka_index": -1,
                    "ka": 15000.0,
                    "kb": float(bubble.kb),
                    "base_mass_scale": CAMPAIGN_MEASUREMENT_MASS_SCALE,
                    "mass_scale": CAMPAIGN_MEASUREMENT_MASS_SCALE * float(multiplier),
                    "mass_calibration_multiplier": float(multiplier),
                    "dt": CAMPAIGN_DT,
                    "equil_steps": int(equil_steps),
                    "relax_steps": CAMPAIGN_RELAX_STEPS,
                    "sample_every": CAMPAIGN_SAMPLE_EVERY,
                    "initial_radius_scale": CAMPAIGN_INITIAL_RADIUS_SCALE,
                    "hold_steps": int(hold_steps),
                    "hold_update_every_steps": int(hold_update_every_steps),
                    "ramp_steps": CAMPAIGN_POST_DEFLATION_RAMP_STEPS,
                    "excitation_mode": "prestrain",
                    "radial_velocity_kick": CAMPAIGN_RADIAL_VELOCITY_KICK,
                    "radial_velocity_kick_mode": CAMPAIGN_RADIAL_VELOCITY_KICK_MODE,
                    "solvent_mode": "full",
                    "water_shell_fsi_scale": protocol.WATER_SHELL_FSI_CONSERVATIVE_SCALE,
                    "bouncer_mode": "on",
                    "preparation": "half_cosine_deflation_fixed_hold_release_from_rest",
                }
            )
            calibration_index += 1
    metadata = {
        "legacy_six_bubble_mode": True,
        "source_symbols": [bubble.symbol for bubble in bubbles],
    }
    return production_rows, calibration_rows, [], metadata


def prepare(args: argparse.Namespace) -> int:
    if int(args.point_count) != PRODUCTION_POINT_COUNT:
        raise ValueError(f"This approved campaign requires exactly {PRODUCTION_POINT_COUNT} production points per radius node.")
    if not math.isclose(float(args.ka_min), CAMPAIGN_KA_MIN, abs_tol=0.0):
        raise ValueError(f"The qualified ka grid must begin at {CAMPAIGN_KA_MIN:g}.")
    if not math.isclose(float(args.ka_max), CAMPAIGN_KA_MAX, abs_tol=0.0):
        raise ValueError(f"The qualified ka grid must end at {CAMPAIGN_KA_MAX:g}.")
    if int(args.hold_steps) <= 0:
        raise ValueError("The full-fluid campaign requires a positive post-deflation hold duration.")
    if int(args.hold_update_every_steps) != CAMPAIGN_POST_DEFLATION_HOLD_UPDATE_EVERY_STEPS:
        raise ValueError(
            "The qualified fixed-shape hold must reset coordinates every "
            f"{CAMPAIGN_POST_DEFLATION_HOLD_UPDATE_EVERY_STEPS} integration steps."
        )

    campaign_root = args.campaign_root.resolve()
    design_root = campaign_root / "design"
    if design_root.exists() and not args.force:
        raise FileExistsError(f"Design already exists: {design_root}. Use --force only before jobs are submitted.")
    map_values = args.map_values.resolve()
    bubbles = protocol.load_bubbles(map_values)
    symbols = [bubble.symbol for bubble in bubbles]
    if set(symbols) != set(BASE_MASS_SCALE):
        raise RuntimeError(f"Unexpected bubble set in {map_values}: {symbols}")

    ka_values = np.linspace(float(args.ka_min), float(args.ka_max), int(args.point_count), dtype=np.float64)
    if args.design_mode == "agent-radius-surface":
        production_rows, calibration_rows, kb_sensitivity_rows, design_metadata = _build_surface_design(
            bubbles=bubbles,
            ka_values=ka_values,
            hold_steps=int(args.hold_steps),
            hold_update_every_steps=int(args.hold_update_every_steps),
        )
        schema = "mesouq.emb_fullfluid_ka_breathing_campaign.v2"
        production_count_key = "production_point_count_per_radius_node"
    elif args.design_mode == "six-bubble":
        production_rows, calibration_rows, kb_sensitivity_rows, design_metadata = _build_six_bubble_design(
            bubbles=bubbles,
            ka_values=ka_values,
            hold_steps=int(args.hold_steps),
            hold_update_every_steps=int(args.hold_update_every_steps),
        )
        schema = "mesouq.emb_fullfluid_ka_breathing_campaign.v1"
        production_count_key = "production_point_count_per_bubble"
    else:  # pragma: no cover - argparse guards choices
        raise ValueError(f"Unsupported design mode: {args.design_mode}")

    _write_csv(design_root / DESIGN_PHASE_FILENAMES["production"], production_rows, _row_fieldnames(production_rows))
    _write_csv(design_root / DESIGN_PHASE_FILENAMES["calibration"], calibration_rows, _row_fieldnames(calibration_rows))
    if kb_sensitivity_rows:
        _write_csv(
            design_root / DESIGN_PHASE_FILENAMES["kb_sensitivity"],
            kb_sensitivity_rows,
            _row_fieldnames(kb_sensitivity_rows),
        )
    design_plot = design_root / "campaign_design.png"
    _write_design_plot(design_plot, production_rows, design_mode=args.design_mode)

    map_sha256 = _sha256_file(map_values)
    if args.design_mode == "agent-radius-surface":
        design_contract = {
            "diameter_grids_reported_um": {
                agent: list(values)
                for agent, values in AGENT_DIAMETER_GRIDS_UM.items()
            },
            "diameter_grid_labels_reported_um": {
                agent: list(values)
                for agent, values in AGENT_DIAMETER_GRID_LABELS_UM.items()
            },
            "radius_grids_um": {
                agent: [float(value) / 2.0 for value in values]
                for agent, values in AGENT_DIAMETER_GRIDS_UM.items()
            },
            "kb_reference": {
                agent: {
                    "value": float(value),
                    "source": "median_of_three_accepted_source_map_entries",
                    "source_map_sha256": map_sha256,
                }
                for agent, value in AGENT_KB_REF.items()
            },
            "kb_sensitivity_design": {
                "filename": DESIGN_PHASE_FILENAMES["kb_sensitivity"],
                "central_diameter_um": AGENT_KB_SENSITIVITY_DIAMETER_UM,
                "ka": 15000.0,
                "factors": list(KB_SENSITIVITY_FACTORS),
                "case_count": len(kb_sensitivity_rows),
            },
            "historical_base_mass_scale_relative_to_physical_mass": BASE_MASS_SCALE,
            "surface_node_mass_scale_policy": {
                "all_nodes": (
                    f"all frozen grid nodes initially use {CAMPAIGN_MEASUREMENT_MASS_SCALE:g}x physical vertex mass "
                    "during the controlled l=0 measurement; raw frequencies are corrected by sqrt(mass_scale)"
                ),
                "cycle_only_rescue": (
                    f"rerun at {CAMPAIGN_RESCUE_MASS_SCALE:g}x only when the canonical trace passes finite, fit, "
                    "FFT, amplitude, and integrated l0-purity gates but contains fewer than three visible cycles"
                ),
                "forbidden_rescue_uses": "nonfinite, unstable, weak, fit-inconsistent, or mode-contaminated traces",
            },
            "mass_calibration_design": {
                "ka": 15000.0,
                "total_mass_multipliers_relative_to_each_node_base": list(CALIBRATION_MULTIPLIERS),
                "diameters_um_by_agent": {
                    agent: list(values) for agent, values in MASS_CALIBRATION_DIAMETERS_UM.items()
                },
                "reason": (
                    "test corrected-frequency invariance and raw damping across 80x, 160x, 320x, and 640x "
                    "controlled measurement inertia"
                ),
            },
        }
    else:
        design_contract = {
            "legacy_six_bubble_symbols": sorted(BASE_MASS_SCALE),
            "kb_sensitivity_design": {"case_count": 0},
        }
    manifest = {
        "schema": schema,
        "created_utc": _utc_stamp(),
        "campaign_root": str(campaign_root),
        "design_mode": args.design_mode,
        "map_values": {
            "path": str(map_values),
            "sha256": map_sha256,
        },
        "git_commit": protocol.git_commit(),
        production_count_key: PRODUCTION_POINT_COUNT,
        "production_case_count": len(production_rows),
        "calibration_case_count": len(calibration_rows),
        "kb_sensitivity_case_count": len(kb_sensitivity_rows),
        "ka_grid": {
            "minimum": CAMPAIGN_KA_MIN,
            "maximum": CAMPAIGN_KA_MAX,
            "point_count": PRODUCTION_POINT_COUNT,
            "requested_prior_support": [0.0, CAMPAIGN_KA_MAX],
            "excluded_lower_support": [0.0, CAMPAIGN_KA_MIN],
            "exclusion_basis": (
                "full-fluid deflate-only qualification found no common clean three-cycle l0 response below ka=6000"
            ),
        },
        "surface_design": {
            "variables": ["ka", "radius_um"],
            "layout": "rectangular_radius_by_ka" if args.design_mode == "agent-radius-surface" else "legacy_six_bubble_ka_curves",
            "agent_count": len(AGENT_DIAMETER_GRIDS_UM) if args.design_mode == "agent-radius-surface" else None,
            "radius_nodes_per_agent": 10 if args.design_mode == "agent-radius-surface" else None,
            "rows_per_agent": 2000 if args.design_mode == "agent-radius-surface" else None,
            "diameter_unit": "reported micrometre",
            "radius_unit": "physical micrometre",
            "radius_definition": "radius_um = diameter_um / 2",
        },
        "design_evidence": {
            "campaign_design_png": str(design_plot),
            "label": "frozen input layout only; contains no measured DPD frequencies",
        },
        **design_contract,
        "design_metadata": design_metadata,
        "historical_base_mass_scale_relative_to_physical_mass": BASE_MASS_SCALE,
        "measurement_mass_scale_relative_to_physical_mass": CAMPAIGN_MEASUREMENT_MASS_SCALE,
        "cycle_only_rescue_mass_scale_relative_to_physical_mass": CAMPAIGN_RESCUE_MASS_SCALE,
        "preparation": {
            "initial_radius_scale": CAMPAIGN_INITIAL_RADIUS_SCALE,
            "post_deflation_half_cosine_ramp_steps": CAMPAIGN_POST_DEFLATION_RAMP_STEPS,
            "post_deflation_coordinate_hold_steps": int(args.hold_steps),
            "post_deflation_coordinate_hold_update_every_steps": int(args.hold_update_every_steps),
            "post_deflation_coordinate_hold_velocity_reset": True,
            "release_constraint": "none; coordinate hold stops before the first dump/release step",
            "post_hold_excitation": {
                "mode": "free-release-from-rest",
                "radial_velocity_kick": CAMPAIGN_RADIAL_VELOCITY_KICK,
                "radial_velocity_kick_mode": CAMPAIGN_RADIAL_VELOCITY_KICK_MODE,
                "mass_correction": "f_physical_mhz = f_raw_mhz * sqrt(membrane_mass_scale)",
            },
        },
        "full_fluid_policy": {
            "solvent_mode": "full",
            "water_shell_fsi_conservative_scale": protocol.WATER_SHELL_FSI_CONSERVATIVE_SCALE,
            "water_shell_afsi_with_aii_100": 40.0,
            "bouncer_mode": "on",
            "thermal_noise": "generated kBT preserved",
            "gas_shell_conservative_a": 0.0,
            "direct_membrane_bpress": "generated original parameter value preserved",
            "lim_mu_policy": "derive-from-ka",
            "lim_mu_relation": "mu = ka * (1 - nu) / (1 + nu), matching the original shared-Yt generators",
        },
        "acceptance": {
            "qualified_signal": {
                "minimum_damped_sine_r2": CAMPAIGN_SCREENING_MIN_R2,
                "maximum_fft_damped_relative_delta": CAMPAIGN_SCREENING_MAX_FFT_DAMPED_RELATIVE_DELTA,
                "minimum_visible_alternating_cycles": CAMPAIGN_SCREENING_MIN_VISIBLE_CYCLES,
                "require_finite_radius_and_volume": True,
                "require_damped_sine_fit": True,
                "minimum_amplitude_to_residual_std": 5.0,
                "require_l0_mode_purity": True,
                "mode_purity_basis": "spatiotemporal RMS over the selected early ringdown fit window",
                "purpose": "only qualified signals can become surrogate labels",
            },
            "strict_quality_annotation": {
                "minimum_damped_sine_r2": CAMPAIGN_STRICT_MIN_R2,
                "maximum_fft_damped_relative_delta": CAMPAIGN_STRICT_MAX_FFT_DAMPED_RELATIVE_DELTA,
                "minimum_visible_alternating_cycles": CAMPAIGN_STRICT_MIN_VISIBLE_CYCLES,
                "require_selected_early_window_frequency": True,
                "require_integrated_l0_mode_purity": True,
                "purpose": "same scientific gate as qualified_signal; retained for schema compatibility",
            },
        },
        "storage_policy": {
            "production_membrane_hdf5": "stage transiently outside project data and remove after CSV/JSON/PNG extraction",
            "representative_production_ka_indices": sorted(REPRESENTATIVE_PRODUCTION_KA_INDICES),
            "calibration_membrane_hdf5": "remove after CSV/JSON/PNG extraction; retained trajectory reruns are separate",
            "failed_production_membrane_hdf5": "remove after failure classification and finite artifact extraction",
            "fluid_particle_hdf5": "disabled",
        },
        "validation": {
            "status": "passed",
            "offline_design_validation": True,
        },
    }
    (design_root / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "design_mode": args.design_mode,
                "production_cases": len(production_rows),
                "calibration_cases": len(calibration_rows),
                "kb_sensitivity_cases": len(kb_sensitivity_rows),
            },
            indent=2,
        )
    )
    return 0


def _parameter_snapshot(symbol_root: Path) -> dict[str, Any]:
    parameter_root = symbol_root / "simulation" / "parameter"
    params = protocol._load_yaml(parameter_root / "parameters00001.yaml")
    defaults = protocol._load_yaml(parameter_root / "parameters-default00001.yaml")
    prms = protocol._load_yaml(parameter_root / "parameters.prms00001.yaml")
    return {
        "parameters00001_yaml": str(parameter_root / "parameters00001.yaml"),
        "parameters_prms00001_yaml": str(parameter_root / "parameters.prms00001.yaml"),
        "mvert": float(params["mvert"]),
        "mvert_unscaled": float(params["mvert_unscaled"]),
        "mvert_raw_generated": float(params.get("mvert_raw_generated", params["mvert"])),
        "membrane_mass_scale": float(params["membrane_mass_scale"]),
        "membrane_mass_policy": str(params.get("membrane_mass_policy", "unknown")),
        "kbt": float(params["kbt"]),
        "mw": float(params["mw"]),
        "mg": float(params["mg"]),
        "gamma_fsi": float(params["gamma_fsi"]),
        "gamma_fsi_gas": float(params["gamma_fsi_gas"]),
        "rhow": float(defaults["rhow"]),
        "rhog": float(defaults["rhog"]),
        "aii": float(defaults["aii"]),
        "bpress": float(defaults["bpress"]),
        "dt": float(defaults["dt"]),
        "dt_eq": float(defaults["dt_eq"]),
        "Yt": float(defaults["Yt"]),
        "ka": float(prms["ka"]),
        "mu": float(prms["mu"]),
        "kb": float(prms["kb"]),
        "nu": float(defaults["nu"]),
        "lim_mu_policy": str(params.get("lim_mu_policy", "legacy-generated")),
        "lim_mu_to_ka_ratio": float(
            params.get("lim_mu_to_ka_ratio", float(prms["mu"]) / float(prms["ka"]) if float(prms["ka"]) else 0.0)
        ),
    }


def _file_exists(path_value: Any) -> bool:
    return isinstance(path_value, str) and Path(path_value).is_file()


def _retain_production_hdf5(row: dict[str, Any], *, phase: str) -> bool:
    """Production arrays never retain transient particle trajectories."""
    del row, phase
    return False


def _delete_nonrepresentative_production_hdf5(
    row: dict[str, Any],
    *,
    phase: str,
    enabled: bool,
) -> bool:
    """Compatibility predicate; transient staging removes every particle dump."""
    del row, phase
    return bool(enabled)


def _runtime_provenance() -> dict[str, str | None]:
    """Capture scheduler and MPI metadata without changing DPD physics."""

    return {
        "hostname": os.environ.get("MESOUQ_RUNTIME_HOST") or socket.getfqdn(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "mpi_transport": os.environ.get("MESOUQ_MPI_TRANSPORT"),
        "mpi_pml": os.environ.get("OMPI_MCA_pml"),
        "mpi_btl": os.environ.get("OMPI_MCA_btl"),
        "mpi_coll": os.environ.get("OMPI_MCA_coll"),
        "mpi_osc": os.environ.get("OMPI_MCA_osc"),
        "runtime_shim": os.environ.get("MESOUQ_RUNTIME_SHIM"),
        "mesouq_git_commit": os.environ.get("MESOUQ_GIT_COMMIT"),
    }


def _case_metrics(
    row: dict[str, Any],
    symbol_root: Path,
    *,
    phase: str,
    delete_nonrepresentative_production_hdf5: bool,
) -> dict[str, Any]:
    result_path = symbol_root / "result.json"
    if not result_path.is_file():
        return {
            "schema": "mesouq.emb_fullfluid_ka_breathing_case_metrics.v1",
            "status": "missing_result",
            "phase": phase,
            "case": row,
            "symbol_root": str(symbol_root),
            "runtime_provenance": _runtime_provenance(),
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_checks = result.get("checks") or {}
    qualified = qualified_analysis.qualified_early_window_row(
        {
            "case": row,
            "finite_observable": bool(result_checks.get("finite_observable")),
            "finite_volume": bool(result_checks.get("finite_volume")),
            "time_series_csv": result.get("time_series_csv"),
            "result_json": str(result_path),
        }
    )
    raw_frequency = qualified.get("raw_frequency_mhz")
    raw_lambda = qualified.get("raw_damping_lambda_dpd_inv")
    fft_delta = qualified.get("fft_damped_relative_delta")
    r2 = qualified.get("damped_sine_r2")
    cycles = qualified.get("visible_alternating_cycles")
    checks = {
        "finite_observable": bool(result_checks.get("finite_observable")),
        "finite_volume": bool(result_checks.get("finite_volume")),
        "finite_raw_frequency": raw_frequency is not None and math.isfinite(float(raw_frequency)),
        "damped_sine_r2": r2 is not None and float(r2) >= CAMPAIGN_SCREENING_MIN_R2,
        "fft_damped_agreement": (
            fft_delta is not None and float(fft_delta) <= CAMPAIGN_SCREENING_MAX_FFT_DAMPED_RELATIVE_DELTA
        ),
        "visible_cycles": cycles is not None and float(cycles) >= CAMPAIGN_SCREENING_MIN_VISIBLE_CYCLES,
        "amplitude_to_residual": (
            qualified.get("amplitude_to_residual_std") is not None
            and float(qualified["amplitude_to_residual_std"])
            >= qualified_analysis.MIN_AMPLITUDE_TO_RESIDUAL
        ),
        "integrated_l0_mode_purity": bool(qualified.get("mode_purity_accepted")),
        "time_series_csv": _file_exists(result.get("time_series_csv")),
        "time_series_png": _file_exists((result.get("plots") or {}).get("time_series_png")),
        "spectrum_png": _file_exists((result.get("plots") or {}).get("spectrum_png")),
        "mode_purity_png": _file_exists((result.get("plots") or {}).get("mode_purity_png")),
    }
    strict_checks = {
        "qualified_signal": all(checks.values()),
        "damped_sine_r2": r2 is not None and float(r2) >= CAMPAIGN_STRICT_MIN_R2,
        "fft_damped_agreement": (
            fft_delta is not None and float(fft_delta) <= CAMPAIGN_STRICT_MAX_FFT_DAMPED_RELATIVE_DELTA
        ),
        "visible_cycles": cycles is not None and float(cycles) >= CAMPAIGN_STRICT_MIN_VISIBLE_CYCLES,
        "selected_early_window_frequency": bool(qualified.get("fit_signal_accepted")),
        "integrated_l0_mode_purity": bool(qualified.get("mode_purity_accepted")),
    }
    accepted = bool(qualified.get("accepted") and all(checks.values()))
    strict_accepted = all(strict_checks.values())
    corrected_frequency = qualified.get("mass_corrected_frequency_mhz")
    particle_paths = [Path(path) for path in result.get("particle_dump_paths", []) if isinstance(path, str)]
    return {
        "schema": "mesouq.emb_fullfluid_ka_breathing_case_metrics.v1",
        "status": "accepted" if accepted else str(qualified.get("status") or "qualified_gate_failed"),
        "phase": phase,
        "case": row,
        "symbol_root": str(symbol_root),
        "runtime_provenance": _runtime_provenance(),
        "result_json": str(result_path),
        "parameter_snapshot": _parameter_snapshot(symbol_root),
        "raw_frequency_mhz": raw_frequency,
        "mass_corrected_frequency_mhz": corrected_frequency,
        "raw_damping_lambda_dpd_inv": raw_lambda,
        "mass_weighted_damping_lambda_dpd_inv": qualified.get(
            "mass_weighted_damping_lambda_dpd_inv"
        ),
        "raw_undamped_frequency_mhz": qualified.get("raw_undamped_frequency_mhz"),
        "mass_corrected_undamped_frequency_mhz": qualified.get(
            "mass_corrected_undamped_frequency_mhz"
        ),
        "damping_ratio": qualified.get("damping_ratio"),
        "fit_amplitude_dpd": qualified.get("fit_amplitude_dpd"),
        "damped_sine_r2": r2,
        "amplitude_to_residual_std": qualified.get("amplitude_to_residual_std"),
        "fft_damped_relative_delta": fft_delta,
        "visible_alternating_cycles": cycles,
        "fit_start_dpd": qualified.get("fit_start_dpd"),
        "fit_end_dpd": qualified.get("fit_end_dpd"),
        "fit_sample_count": qualified.get("fit_sample_count"),
        "fit": qualified.get("selected_fit") or {},
        "mode_purity_status": qualified.get("mode_purity_status"),
        "mode_purity_frame_count": qualified.get("mode_purity_frame_count"),
        "l1_over_l0": qualified.get("l1_over_l0"),
        "l2_over_l0": qualified.get("l2_over_l0"),
        "radial_residual_over_l0": qualified.get("radial_residual_over_l0"),
        "tangential_over_l0": qualified.get("tangential_over_l0"),
        "checks": checks,
        "strict_checks": strict_checks,
        "accepted": accepted,
        "strict_accepted": strict_accepted,
        "plots": result.get("plots") or {},
        "time_series_csv": result.get("time_series_csv"),
        "particle_hdf5_transient_count": len(particle_paths),
        "particle_hdf5_transient_first_path": str(particle_paths[0]) if particle_paths else None,
        "particle_hdf5_transient_last_path": str(particle_paths[-1]) if particle_paths else None,
        "particle_hdf5_transient_paths": [],
        "particle_hdf5_deleted": [],
        "particle_hdf5_deleted_count": 0,
        "particle_hdf5_retained": [],
        "representative_production_hdf5_retained": False,
        "particle_hdf5_retention_policy": "transient_staging_removed_after_analysis",
        "delete_nonrepresentative_production_hdf5_compatibility_flag": bool(
            delete_nonrepresentative_production_hdf5
        ),
    }


def _write_case_map_values(path: Path, row: dict[str, Any]) -> None:
    payload = {
        "agent": str(row["agent"]),
        "diameter_symbol": str(row["symbol"]),
        "modality": str(row["modality"]),
        "dataset": str(row["dataset"]),
        "diameter_um": float(row["diameter_um"]),
        "radius_dpd": float(row["radius_dpd"]),
        "radius_source": str(row["radius_dpd_source"]),
        "source_label": str(row["source_label"]),
        "sample_index": int(row["sample_index"]),
        "sample_count": int(row["sample_count"]),
        "ka": float(row["ka"]),
        "kb": float(row["kb"]),
        "d0": float(row["d0"]),
        "sigma": float(row["sigma"]),
        "legacy_Yt": float(row["legacy_Yt"]),
        "logLikelihood": float(row["logLikelihood"]),
        "logPrior": float(row["logPrior"]),
        "logPosterior": float(row["logPosterior"]),
        "state_path": str(row["state_path"]),
        "map_template_state_path": str(row.get("map_template_state_path", "")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, [payload], list(MAP_VALUES_FIELDNAMES))


def _run_case_row(
    args: argparse.Namespace,
    *,
    row: dict[str, Any],
    case_root: Path,
    metrics_path: Path,
    phase: str,
    replacement_metadata: dict[str, Any] | None = None,
) -> int:
    symbol_root = case_root / str(row["symbol"]) / "seed-000"

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    staging_allocation: dict[str, Any] | None = None
    map_values = args.map_values.resolve()
    if row.get("surface_node_id"):
        map_values = case_root / "campaign_case_map_values.csv"
        if rank == 0:
            _write_case_map_values(map_values, row)
        comm.Barrier()

    runner_args = [
        "--map-values", str(map_values),
        "--run-root", str(case_root),
        "--symbols", str(row["symbol"]),
        "--seed-indices", "0",
        "--dt", str(row["dt"]),
        "--equil-steps", str(row["equil_steps"]),
        "--pulse-steps", "0",
        "--relax-steps", str(row["relax_steps"]),
        "--sample-every", str(row["sample_every"]),
        # Mirheo's rank-0 host coordinate accessor is not reliable on the Vega
        # GPU build. Use the validated particle-dump reader; production dumps are
        # removed after analysis by the bounded retention policy below.
        "--trajectory-capture", "particle-dump",
        "--excitation-mode", str(row["excitation_mode"]),
        "--initial-radius-scale", str(row["initial_radius_scale"]),
        "--prestrain-placement", "post-equilibration",
        "--post-deflation-ramp-steps", str(row["ramp_steps"]),
        "--post-deflation-hold-steps", str(row["hold_steps"]),
        "--post-deflation-hold-update-every-steps", str(row["hold_update_every_steps"]),
        "--post-deflation-hold-reset-velocities",
        "--radial-velocity-kick", str(row["radial_velocity_kick"]),
        "--radial-velocity-kick-mode", str(row["radial_velocity_kick_mode"]),
        "--solvent-mode", str(row["solvent_mode"]),
        "--water-shell-fsi-scale", str(row["water_shell_fsi_scale"]),
        "--water-shell-gamma-scale", "1.0",
        "--bouncer-mode", str(row["bouncer_mode"]),
        "--membrane-mass-scale", str(row["mass_scale"]),
        "--primary-observable", "rms_radius_dpd",
        "--ka-override", str(row["ka"]),
        "--kb-override", str(row["kb"]),
        "--lim-mu-policy", "derive-from-ka",
        "--fit-start-dpd", "0.0",
        "--fit-end-dpd", "-1.0",
        "--pin-com",
        "--no-dump-fluid-particles",
        "--no-dump-fluid-density-grid",
        "--particle-checker-every", str(max(0, int(args.particle_checker_every))),
    ]
    if args.resume:
        runner_args.append("--resume")
    if args.force_prepare:
        runner_args.append("--force-prepare")

    preflight_error: str | None = None
    if rank == 0:
        try:
            staging_allocation = reserve_particle_staging(
                args.particle_staging_root,
                case_index=int(row["case_index"]),
                stage=phase,
                limit_bytes=int(args.particle_staging_limit_bytes),
                reservation_bytes=int(args.particle_staging_reservation_bytes),
                namespace=particle_staging_namespace(args.campaign_root),
            )
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"
            if staging_allocation is not None:
                try:
                    release_particle_staging(staging_allocation)
                except Exception as cleanup_exc:
                    preflight_error += (
                        "; particle staging cleanup also failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                staging_allocation = None
    preflight_error = comm.bcast(preflight_error, root=0)
    if preflight_error is not None:
        raise RuntimeError(f"Case preflight failed: {preflight_error}")
    staging_allocation = comm.bcast(staging_allocation, root=0)
    runner_args.extend(["--particle-dump-root", str(staging_allocation["case_root"])])

    exit_code = 0
    error: str | None = None
    try:
        try:
            exit_code = int(protocol.main(runner_args))
        except Exception as exc:  # pragma: no cover - only reachable inside an MPI job
            exit_code = 1
            error = f"{type(exc).__name__}: {exc}"
        rank_errors = comm.allgather(error)
        error = next((rank_error for rank_error in rank_errors if rank_error is not None), None)
        exit_code = max(comm.allgather(exit_code))
        if rank == 0:
            if error is not None:
                metrics = {
                    "schema": "mesouq.emb_fullfluid_ka_breathing_case_metrics.v1",
                    "status": "runner_error",
                    "phase": phase,
                    "case": row,
                    "symbol_root": str(symbol_root),
                    "error": error,
                    "runtime_provenance": _runtime_provenance(),
                }
            else:
                metrics = _case_metrics(
                    row,
                    symbol_root,
                    phase=phase,
                    delete_nonrepresentative_production_hdf5=bool(
                        args.delete_nonrepresentative_production_hdf5
                    ),
                )
            metrics["particle_staging"] = staging_allocation
            if replacement_metadata is not None:
                metrics["measurement_replacement"] = replacement_metadata
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "case_index": row["case_index"],
                        "symbol": row["symbol"],
                        "status": metrics["status"],
                        "metrics": str(metrics_path),
                    },
                    indent=2,
                ),
                flush=True,
            )
        comm.Barrier()
    finally:
        cleanup_error: str | None = None
        if rank == 0:
            try:
                staging_cleanup = release_particle_staging(staging_allocation)
                if metrics_path.is_file():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    metrics["particle_staging_cleanup"] = staging_cleanup
                    metrics["particle_hdf5_deleted_count"] = int(
                        metrics.get("particle_hdf5_transient_count")
                        or len(metrics.get("particle_hdf5_transient_paths", []))
                    )
                    metrics["particle_hdf5_deleted"] = []
                    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
        cleanup_error = comm.bcast(cleanup_error, root=0)
        comm.Barrier()
        if cleanup_error is not None:
            raise RuntimeError(f"Particle staging cleanup failed: {cleanup_error}")
    return exit_code


def _load_case_row(args: argparse.Namespace) -> dict[str, Any]:
    rows = [_coerce_row(row) for row in _read_design(args.campaign_root, args.phase)]
    matches = [row for row in rows if int(row["case_index"]) == int(args.case_index)]
    if len(matches) != 1:
        raise ValueError(f"No unique {args.phase} case index {args.case_index}.")
    return matches[0]


def run_case(args: argparse.Namespace) -> int:
    row = _load_case_row(args)
    case_root = _case_root(args.campaign_root, row)
    return _run_case_row(
        args,
        row=row,
        case_root=case_root,
        metrics_path=_case_metrics_path(args.campaign_root, row),
        phase=str(args.phase),
    )


def run_replacement(args: argparse.Namespace) -> int:
    row = _load_case_row(args)
    replacement = _replacement_row(
        row,
        replacement_id=str(args.replacement_id),
        relax_steps=int(args.relax_steps),
        sample_every=int(args.sample_every),
        reason=str(args.reason),
    )
    attempt_root = _replacement_root(args.campaign_root, row, str(args.replacement_id))

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    preflight_error: str | None = None
    if rank == 0:
        try:
            if attempt_root.exists() and not args.resume:
                raise FileExistsError(
                    f"Replacement evidence already exists: {attempt_root}. Use --resume only to continue that exact attempt."
                )
            attempt_root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema": "mesouq.emb_fullfluid_ka_breathing_targeted_replacement.v1",
                "purpose": "Extend only the free l=0 observation duration after a canonical measurement was under-resolved.",
                "source_case": row,
                "replacement_row": replacement,
                "allowed_measurement_deltas": {
                    "relax_steps": {
                        "canonical": int(row["relax_steps"]),
                        "replacement": int(replacement["relax_steps"]),
                    },
                    "sample_every": {
                        "canonical": int(row["sample_every"]),
                        "replacement": int(replacement["sample_every"]),
                    },
                },
                "physical_inputs_preserved": {
                    key: replacement[key]
                    for key in (
                        "ka",
                        "kb",
                        "mass_scale",
                        "dt",
                        "equil_steps",
                        "initial_radius_scale",
                        "hold_steps",
                        "hold_update_every_steps",
                        "excitation_mode",
                        "radial_velocity_kick",
                        "radial_velocity_kick_mode",
                        "solvent_mode",
                        "water_shell_fsi_scale",
                        "bouncer_mode",
                    )
                },
            }
            (attempt_root / "replacement_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"
    preflight_error = comm.bcast(preflight_error, root=0)
    if preflight_error is not None:
        raise RuntimeError(preflight_error)
    comm.Barrier()
    replacement_metadata = {
        "source_case_index": int(row["case_index"]),
        "replacement_id": str(args.replacement_id),
        "replacement_reason": str(args.reason),
        "replacement_manifest": str(attempt_root / "replacement_manifest.json"),
        "canonical_relax_steps": int(row["relax_steps"]),
        "replacement_relax_steps": int(replacement["relax_steps"]),
        "canonical_sample_every": int(row["sample_every"]),
        "replacement_sample_every": int(replacement["sample_every"]),
    }
    return _run_case_row(
        args,
        row=replacement,
        case_root=attempt_root,
        metrics_path=attempt_root / str(row["symbol"]) / "seed-000" / "campaign_case_metrics.json",
        phase="targeted_replacement",
        replacement_metadata=replacement_metadata,
    )


def summarize(args: argparse.Namespace) -> int:
    campaign_root = args.campaign_root.resolve()
    metric_paths = sorted(campaign_root.glob("**/campaign_case_metrics.json"))
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        case = metric.get("case") or {}
        rows.append(
            {
                "phase": metric.get("phase"),
                "case_index": case.get("case_index"),
                "symbol": case.get("symbol"),
                "surface_node_id": case.get("surface_node_id"),
                "agent": case.get("agent"),
                "modality": case.get("modality"),
                "dataset": case.get("dataset"),
                "diameter_um": case.get("diameter_um"),
                "radius_um": case.get("radius_um"),
                "radius_dpd": case.get("radius_dpd"),
                "ka_index": case.get("ka_index"),
                "ka": case.get("ka"),
                "kb": case.get("kb"),
                "kb_ref": case.get("kb_ref"),
                "kb_role": case.get("kb_role"),
                "kb_sensitivity_factor": case.get("kb_sensitivity_factor"),
                "mass_scale": case.get("mass_scale"),
                "mass_calibration_multiplier": case.get("mass_calibration_multiplier"),
                "status": metric.get("status"),
                "accepted": metric.get("accepted"),
                "strict_accepted": metric.get("strict_accepted"),
                "raw_frequency_mhz": metric.get("raw_frequency_mhz"),
                "mass_corrected_frequency_mhz": metric.get("mass_corrected_frequency_mhz"),
                "raw_damping_lambda_dpd_inv": metric.get("raw_damping_lambda_dpd_inv"),
                "damped_sine_r2": metric.get("damped_sine_r2"),
                "fft_damped_relative_delta": metric.get("fft_damped_relative_delta"),
                "visible_alternating_cycles": metric.get("visible_alternating_cycles"),
                "low_level_clean_frequency": (metric.get("strict_checks") or {}).get("low_level_clean_frequency"),
                "low_level_clean_mode_purity": (metric.get("strict_checks") or {}).get("low_level_clean_mode_purity"),
                "time_series_png": (metric.get("plots") or {}).get("time_series_png"),
                "spectrum_png": (metric.get("plots") or {}).get("spectrum_png"),
                "mode_purity_png": (metric.get("plots") or {}).get("mode_purity_png"),
                "time_series_csv": metric.get("time_series_csv"),
                "result_json": metric.get("result_json"),
            }
        )
    summary_root = campaign_root / "evidence"
    _write_csv(summary_root / "campaign_summary.csv", rows, list(rows[0].keys()) if rows else ["phase"])
    (summary_root / "campaign_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_frequency_plots(summary_root, rows)
    print(json.dumps({"metric_count": len(metrics), "summary_csv": str(summary_root / "campaign_summary.csv")}, indent=2))
    return 0


def _write_frequency_plots(output_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_root / "plot_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    production = [row for row in rows if row.get("phase") == "production"]
    for symbol in sorted({str(row.get("symbol")) for row in production if row.get("symbol")}):
        selected = [row for row in production if row.get("symbol") == symbol]
        accepted = [row for row in selected if row.get("accepted") and row.get("mass_corrected_frequency_mhz") is not None]
        strict_accepted = [
            row for row in selected if row.get("strict_accepted") and row.get("mass_corrected_frequency_mhz") is not None
        ]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        if selected:
            ax.scatter(
                [float(row["ka"]) for row in selected],
                [float(row["raw_frequency_mhz"]) if row.get("raw_frequency_mhz") is not None else np.nan for row in selected],
                s=11,
                color="0.75",
                label="raw fitted frequency",
            )
        if accepted:
            ax.scatter(
                [float(row["ka"]) for row in accepted],
                [float(row["mass_corrected_frequency_mhz"]) for row in accepted],
                s=14,
                color="tab:blue",
                label="screening-signal mass-corrected frequency",
            )
        if strict_accepted:
            ax.scatter(
                [float(row["ka"]) for row in strict_accepted],
                [float(row["mass_corrected_frequency_mhz"]) for row in strict_accepted],
                s=18,
                marker="o",
                facecolors="none",
                edgecolors="tab:green",
                linewidths=0.9,
                label="strict-quality subset",
            )
        ax.set_xlabel("ka [Mirheo parameter]")
        ax.set_ylabel("frequency [MHz]")
        ax.set_title(f"{symbol}: full-fluid breathing frequency versus ka")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_root / f"{symbol}_fullfluid_frequency_vs_ka.png", dpi=180)
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Write the immutable production and calibration design tables.")
    prepare_parser.add_argument("--campaign-root", type=Path, required=True)
    prepare_parser.add_argument("--map-values", type=Path, default=_default_map_values())
    prepare_parser.add_argument("--point-count", type=int, default=PRODUCTION_POINT_COUNT)
    prepare_parser.add_argument("--ka-min", type=float, default=CAMPAIGN_KA_MIN)
    prepare_parser.add_argument("--ka-max", type=float, default=CAMPAIGN_KA_MAX)
    prepare_parser.add_argument("--hold-steps", type=int, default=CAMPAIGN_POST_DEFLATION_HOLD_STEPS)
    prepare_parser.add_argument("--hold-update-every-steps", type=int, default=CAMPAIGN_POST_DEFLATION_HOLD_UPDATE_EVERY_STEPS)
    prepare_parser.add_argument(
        "--design-mode",
        choices=["agent-radius-surface", "six-bubble"],
        default="agent-radius-surface",
        help="Default writes the approved 10-size-per-agent rectangular surface design; six-bubble preserves the legacy pilot layout.",
    )
    prepare_parser.add_argument("--force", action="store_true")

    rescue_plan_parser = subparsers.add_parser(
        "plan-mass-rescues",
        help="Freeze 320x reruns only for clean l0 traces with fewer than three cycles.",
    )
    rescue_plan_parser.add_argument("--campaign-root", type=Path, required=True)
    rescue_plan_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run one MPI campaign case from a prepared design table.")
    run_parser.add_argument("--campaign-root", type=Path, required=True)
    run_parser.add_argument("--map-values", type=Path, default=_default_map_values())
    run_parser.add_argument("--phase", choices=list(DESIGN_PHASE_FILENAMES), required=True)
    run_parser.add_argument("--case-index", type=int, required=True)
    run_parser.add_argument("--particle-checker-every", type=int, default=0)
    run_parser.add_argument(
        "--particle-staging-root",
        type=Path,
        default=Path(os.environ.get("MESOUQ_PARTICLE_STAGING_ROOT", Path.home() / "mesouq-dpd-staging")),
    )
    run_parser.add_argument(
        "--particle-staging-limit-bytes",
        type=int,
        default=DEFAULT_PARTICLE_STAGING_LIMIT_BYTES,
    )
    run_parser.add_argument(
        "--particle-staging-reservation-bytes",
        type=int,
        default=DEFAULT_PARTICLE_STAGING_RESERVATION_BYTES,
    )
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--force-prepare", action="store_true")
    run_parser.add_argument(
        "--delete-nonrepresentative-production-hdf5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove all nonrepresentative production membrane HDF5 after analysis; calibration HDF5 is retained.",
    )
    run_parser.add_argument(
        "--delete-accepted-production-hdf5",
        dest="delete_nonrepresentative_production_hdf5",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Deprecated compatibility alias for queued Vega batch scripts.",
    )

    replacement_parser = subparsers.add_parser(
        "run-replacement",
        help="Run an append-only, longer-record replacement for an under-resolved production case.",
    )
    replacement_parser.add_argument("--campaign-root", type=Path, required=True)
    replacement_parser.add_argument("--map-values", type=Path, default=_default_map_values())
    replacement_parser.add_argument("--phase", choices=["production"], default="production")
    replacement_parser.add_argument("--case-index", type=int, required=True)
    replacement_parser.add_argument("--replacement-id", required=True)
    replacement_parser.add_argument("--reason", required=True)
    replacement_parser.add_argument("--relax-steps", type=int, required=True)
    replacement_parser.add_argument("--sample-every", type=int, default=CAMPAIGN_SAMPLE_EVERY)
    replacement_parser.add_argument("--particle-checker-every", type=int, default=0)
    replacement_parser.add_argument(
        "--particle-staging-root",
        type=Path,
        default=Path(os.environ.get("MESOUQ_PARTICLE_STAGING_ROOT", Path.home() / "mesouq-dpd-staging")),
    )
    replacement_parser.add_argument(
        "--particle-staging-limit-bytes",
        type=int,
        default=DEFAULT_PARTICLE_STAGING_LIMIT_BYTES,
    )
    replacement_parser.add_argument(
        "--particle-staging-reservation-bytes",
        type=int,
        default=DEFAULT_PARTICLE_STAGING_RESERVATION_BYTES,
    )
    replacement_parser.add_argument("--resume", action="store_true")
    replacement_parser.add_argument("--force-prepare", action="store_true")
    replacement_parser.add_argument(
        "--delete-nonrepresentative-production-hdf5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove replacement particle dumps after analysis while retaining YAML, CSV, JSON, and PNG evidence.",
    )

    summarize_parser = subparsers.add_parser("summarize", help="Aggregate completed case metrics and regenerate frequency PNGs.")
    summarize_parser.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    if args.command == "plan-mass-rescues":
        return plan_mass_rescues(args)
    if args.command == "run":
        return run_case(args)
    if args.command == "run-replacement":
        return run_replacement(args)
    if args.command == "summarize":
        return summarize(args)
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
