#!/usr/bin/env python3
"""Select one PCA breathing candidate by volume sensitivity, blind to frequency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path("."))
    parser.add_argument(
        "--scores", default="individual_mode_audit_300_mode_scores.txt"
    )
    parser.add_argument("--eigenvalues", default="eigvalues_new.txt")
    parser.add_argument("--minimum-modes", type=int, default=300)
    args = parser.parse_args()

    case = args.case.resolve()
    score_path = case / args.scores
    eigenvalue_path = case / args.eigenvalues
    parameter_path = case / "parameter" / "parameters00001.yaml"
    scores = np.loadtxt(score_path, dtype=np.float64)
    eigenvalues = np.loadtxt(eigenvalue_path, dtype=np.float64).reshape(-1)
    with parameter_path.open("rb") as handle:
        parameters = yaml.load(handle, Loader=yaml.CLoader)

    if scores.ndim != 2 or scores.shape[1] != 10:
        raise ValueError(f"Expected a 10-column score table, got {scores.shape}")
    if len(scores) < args.minimum_modes or len(eigenvalues) < args.minimum_modes:
        raise ValueError(
            f"Need at least {args.minimum_modes} modes, got "
            f"{len(scores)} scores and {len(eigenvalues)} eigenvalues"
        )
    modes = scores[:, 0].astype(int)
    if not np.array_equal(modes, np.arange(len(scores))):
        raise ValueError("Score-table modes are not contiguous and zero-based")
    if np.any(~np.isfinite(eigenvalues[: len(scores)])) or np.any(
        eigenvalues[: len(scores)] <= 0.0
    ):
        raise ValueError("Eigenvalues must be finite and positive")

    # Score columns are defined by identify_breathing_by_volume.py:
    # mode, volume_score, dV_dq, radial_fraction, same_sign_fraction,
    # combined_score, rms_volume_fraction, V_plus, V_minus, second_order_term.
    volume_scores = scores[:, 1]
    order = np.argsort(volume_scores)[::-1]
    selected = int(order[0])
    runner_up = int(order[1])
    kbt = float(parameters["kbt"])
    mass_factor = float(parameters["mass_factor"])
    unit_time = float(parameters["ut"])
    omega = np.sqrt(kbt / eigenvalues[: len(scores)])
    frequency_hz = omega * math.sqrt(mass_factor) / (2.0 * math.pi * unit_time)

    fieldnames = [
        "mode_zero_based",
        "eigenvalue_mass_weighted",
        "volume_score",
        "dV_dq",
        "radial_fraction",
        "same_sign_fraction",
        "combined_score_diagnostic",
        "rms_volume_fraction",
        "omega_dpd_sim",
        "frequency_mhz_mass_corrected",
    ]
    table_path = case / "individual_mode_frequencies.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode in modes:
            writer.writerow(
                {
                    "mode_zero_based": int(mode),
                    "eigenvalue_mass_weighted": float(eigenvalues[mode]),
                    "volume_score": float(scores[mode, 1]),
                    "dV_dq": float(scores[mode, 2]),
                    "radial_fraction": float(scores[mode, 3]),
                    "same_sign_fraction": float(scores[mode, 4]),
                    "combined_score_diagnostic": float(scores[mode, 5]),
                    "rms_volume_fraction": float(scores[mode, 6]),
                    "omega_dpd_sim": float(omega[mode]),
                    "frequency_mhz_mass_corrected": float(
                        frequency_hz[mode] / 1.0e6
                    ),
                }
            )

    selected_row = {
        "mode_zero_based": selected,
        "runner_up_mode_zero_based": runner_up,
        "selection_metric": "maximum abs(dV/dq)/V0 for a unit-norm PCA eigenvector",
        "selection_uses_frequency": False,
        "volume_score": float(scores[selected, 1]),
        "volume_score_ratio_to_runner_up": float(
            scores[selected, 1] / scores[runner_up, 1]
        ),
        "radial_fraction": float(scores[selected, 3]),
        "same_sign_fraction": float(scores[selected, 4]),
        "combined_score_diagnostic": float(scores[selected, 5]),
        "rms_volume_fraction": float(scores[selected, 6]),
        "eigenvalue_mass_weighted": float(eigenvalues[selected]),
        "omega_dpd_sim": float(omega[selected]),
        "frequency_mhz_mass_corrected": float(frequency_hz[selected] / 1.0e6),
        "clean_spatial_candidate": bool(
            scores[selected, 3] >= 0.80 and scores[selected, 4] >= 0.95
        ),
        "clean_spatial_thresholds": {
            "radial_fraction_at_least": 0.80,
            "same_sign_fraction_at_least": 0.95,
        },
    }
    payload = {
        "schema": "rio.individual_pca_breathing_mode.v1",
        "case": json.loads((case.parent / "selected_map_row.json").read_text()),
        "mode_count": int(len(scores)),
        "mode_indexing": "zero-based",
        "selected": selected_row,
        "conversion": {
            "formula": "omega_dpd_sim*sqrt(mass_factor)/(2*pi*ut)",
            "mass_factor": mass_factor,
            "ut_seconds": unit_time,
            "extra_sqrt_fscale_applied": False,
        },
        "input_hashes": {
            str(score_path.name): sha256(score_path),
            str(eigenvalue_path.name): sha256(eigenvalue_path),
            str(parameter_path.relative_to(case)): sha256(parameter_path),
        },
        "table": str(table_path),
    }
    output = case / "provenance" / "individual_breathing_mode.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
