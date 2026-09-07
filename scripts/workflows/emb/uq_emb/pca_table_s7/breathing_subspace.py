#!/usr/bin/env python3
"""Measure a geometry-defined PCA breathing coordinate without mode picking.

The primary template is the discrete gradient of enclosed mesh volume at the
mean configuration.  Its covariance is evaluated both from the retained PCA
eigensystem and by direct projection of the aligned trajectory.  This keeps
the breathing definition independent of both the target frequency and the
arbitrary orientation of nearby PCA eigenvectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import trimesh
import yaml
from MDAnalysis.analysis import align


def relative_error(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / abs(float(reference))


def volume_gradient(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return the exact gradient of the oriented triangular-mesh volume."""
    gradient = np.zeros_like(vertices, dtype=np.float64)
    vi = vertices[faces[:, 0]]
    vj = vertices[faces[:, 1]]
    vk = vertices[faces[:, 2]]
    np.add.at(gradient, faces[:, 0], np.cross(vj, vk) / 6.0)
    np.add.at(gradient, faces[:, 1], np.cross(vk, vi) / 6.0)
    np.add.at(gradient, faces[:, 2], np.cross(vi, vj) / 6.0)
    return gradient


def omega_from_variance(variance: float, kbt: float) -> float:
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(f"Variance must be finite and positive, got {variance!r}")
    return math.sqrt(float(kbt) / float(variance))


def frequency_mhz(omega: float, mass_factor: float, unit_time: float) -> float:
    return omega * math.sqrt(mass_factor) / (2.0 * math.pi * unit_time) / 1.0e6


def integrated_autocorrelation_frames(values: np.ndarray) -> float:
    """Initial-positive-pair estimate, used only as a sampling diagnostic."""
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    n = len(centered)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)[:n]
    acov /= np.arange(n, 0, -1, dtype=np.float64)
    if acov[0] <= 0.0:
        return 0.5
    acf = acov / acov[0]
    pair_sums = acf[1:-1:2] + acf[2::2]
    positive = pair_sums[pair_sums > 0.0]
    if len(positive) < len(pair_sums):
        first_nonpositive = int(np.argmax(pair_sums <= 0.0))
        positive = pair_sums[:first_nonpositive]
    return max(0.5, 0.5 + float(np.sum(positive)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path("."))
    parser.add_argument("--retained-eigenvectors", default="eigvectors_retained.txt")
    parser.add_argument("--retained-eigenvalues", default="eigvalues_retained.txt")
    parser.add_argument("--trajectory", default="output/positions.xyz")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    case = args.case.resolve()
    metadata = json.loads((case.parent / "selected_map_row.json").read_text())
    with (case / "parameter" / "parameters00001.yaml").open("rb") as handle:
        parameters = yaml.load(handle, Loader=yaml.CLoader)

    mean = np.loadtxt(case / "output" / "ref1_positions.txt", dtype=np.float64)
    mesh = trimesh.load(case / "mesh" / "emb00001.off", process=False)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if mean.shape != np.asarray(mesh.vertices).shape:
        raise ValueError(f"Mean/mesh shape mismatch: {mean.shape} vs {mesh.vertices.shape}")

    gradient = volume_gradient(mean, faces)
    center = np.mean(mean, axis=0)
    radial = mean - center
    radial_hat = radial / np.linalg.norm(radial, axis=1)[:, None]
    if float(np.sum(gradient * radial_hat)) < 0.0:
        gradient *= -1.0
    radial_components = np.einsum("ij,ij->i", gradient, radial_hat)
    template = gradient.reshape(-1)
    template /= np.linalg.norm(template)

    eigenvectors = np.loadtxt(case / args.retained_eigenvectors, dtype=np.float64)
    eigenvalues = np.loadtxt(case / args.retained_eigenvalues, dtype=np.float64).reshape(-1)
    if eigenvectors.ndim == 1:
        eigenvectors = eigenvectors.reshape(1, -1)
    if eigenvectors.shape[1] != template.size:
        raise ValueError(
            f"Eigenvector width {eigenvectors.shape[1]} != template size {template.size}"
        )
    if len(eigenvalues) != len(eigenvectors):
        raise ValueError("Retained eigenvalue/eigenvector counts differ")
    eigenvectors /= np.linalg.norm(eigenvectors, axis=1)[:, None]

    overlaps = eigenvectors @ template
    weights = overlaps**2
    captured = float(np.sum(weights))
    spectral_variance = float(np.dot(weights, eigenvalues))
    kbt = float(parameters["kbt"])
    spectral_omega = omega_from_variance(spectral_variance, kbt)

    order = np.argsort(weights)[::-1]
    cumulative = np.cumsum(weights[order])
    rows = []
    rank_by_mode = np.empty(len(order), dtype=int)
    rank_by_mode[order] = np.arange(1, len(order) + 1)
    for mode in range(len(eigenvalues)):
        rows.append(
            {
                "mode": mode,
                "eigenvalue": float(eigenvalues[mode]),
                "omega_dpd_mass_scaled": omega_from_variance(eigenvalues[mode], kbt),
                "template_overlap": float(overlaps[mode]),
                "template_weight": float(weights[mode]),
                "weight_rank": int(rank_by_mode[mode]),
            }
        )
    with (case / "breathing_subspace_mode_weights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    universe = mda.Universe(
        str(case / "xyz0.xyz"), str(case / args.trajectory), format="XYZ", dt=1
    )
    reference = mda.Universe(str(case / "xyz0.xyz"), format="XYZ")
    align.AlignTraj(universe, reference, in_memory=True).run()
    q = np.empty(len(universe.trajectory), dtype=np.float64)
    mass_scale = math.sqrt(float(parameters["mvert"]))
    for index, timestep in enumerate(universe.trajectory):
        displacement = np.asarray(timestep.positions, dtype=np.float64) - mean
        q[index] = mass_scale * float(np.dot(template, displacement.reshape(-1)))

    full_variance = float(np.var(q, ddof=1))
    full_omega = omega_from_variance(full_variance, kbt)
    mass_factor = float(parameters["mass_factor"])
    unit_time = float(parameters["ut"])
    full_frequency = frequency_mhz(full_omega, mass_factor, unit_time)

    prefix_sizes = sorted({min(n, len(q)) for n in (10_000, 20_000, 30_000, len(q))})
    prefixes = []
    for stop in prefix_sizes:
        variance = float(np.var(q[:stop], ddof=1))
        omega = omega_from_variance(variance, kbt)
        prefixes.append(
            {
                "stop_frame_exclusive": int(stop),
                "variance": variance,
                "omega_dpd_mass_scaled": omega,
                "frequency_mhz_mass_corrected": frequency_mhz(
                    omega, mass_factor, unit_time
                ),
            }
        )

    edges = np.linspace(0, len(q), 5, dtype=int)
    blocks = []
    for block, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        variance = float(np.var(q[start:stop], ddof=1))
        omega = omega_from_variance(variance, kbt)
        blocks.append(
            {
                "block": block,
                "start_frame": int(start),
                "stop_frame_exclusive": int(stop),
                "variance": variance,
                "omega_dpd_mass_scaled": omega,
                "frequency_mhz_mass_corrected": frequency_mhz(
                    omega, mass_factor, unit_time
                ),
            }
        )
    first_omega = omega_from_variance(float(np.var(q[: len(q) // 2], ddof=1)), kbt)
    second_omega = omega_from_variance(float(np.var(q[len(q) // 2 :], ddof=1)), kbt)
    split_spread = abs(first_omega - second_omega) / (0.5 * (first_omega + second_omega))
    tau_frames = integrated_autocorrelation_frames(q)

    target = float(metadata["analytic_map_frequency_mhz"])
    checks = {
        "retained_template_weight_at_least_0p99": captured >= 0.99,
        "direct_projection_matches_spectral_variance_within_1_percent": relative_error(
            full_variance, spectral_variance
        )
        <= 0.01,
        "first_second_half_frequency_spread_at_most_10_percent": split_spread <= 0.10,
        "effective_frequency_within_10_percent_of_analytic_target": relative_error(
            full_frequency, target
        )
        <= 0.10,
    }
    top_modes = [
        {
            "mode": int(mode),
            "template_overlap": float(overlaps[mode]),
            "template_weight": float(weights[mode]),
            "cumulative_weight_in_rank_order": float(cumulative[rank]),
            "omega_dpd_mass_scaled": omega_from_variance(eigenvalues[mode], kbt),
        }
        for rank, mode in enumerate(order[: args.top])
    ]
    payload = {
        "schema": "rio.geometry_defined_breathing_subspace.v1",
        "case": metadata,
        "definition": {
            "template": "normalized discrete gradient of signed enclosed mesh volume at the PCA mean",
            "frequency": "static-susceptibility quasiharmonic frequency sqrt(kBT/Var(q_volume_gradient))",
            "selection_uses_target_frequency": False,
            "extra_sqrt_fscale_applied": False,
        },
        "template": {
            "retained_modes": int(len(eigenvalues)),
            "captured_squared_norm": captured,
            "radial_component_same_sign_fraction": float(
                max(np.mean(radial_components > 0.0), np.mean(radial_components < 0.0))
            ),
            "top_modes": top_modes,
        },
        "full_trajectory": {
            "frames": int(len(q)),
            "spectral_variance": spectral_variance,
            "direct_projection_variance": full_variance,
            "projection_spectral_relative_error": relative_error(
                full_variance, spectral_variance
            ),
            "omega_dpd_mass_scaled": full_omega,
            "frequency_mhz_mass_corrected": full_frequency,
            "analytic_map_frequency_mhz": target,
            "relative_error_vs_analytic_target": relative_error(full_frequency, target),
            "integrated_autocorrelation_frames_initial_positive_pairs": tau_frames,
            "effective_sample_size_diagnostic": float(len(q) / (2.0 * tau_frames)),
        },
        "prefix_convergence": prefixes,
        "quarter_blocks": blocks,
        "first_second_half_frequency_relative_spread": split_spread,
        "checks": checks,
        "scientific_acceptance": all(checks.values()),
        "interpretation_guard": (
            "This is a geometry-defined static-susceptibility PCA frequency, not proof "
            "that one individual covariance eigenvector is a pure breathing normal mode."
        ),
    }
    output = case / "provenance" / "breathing_subspace.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["scientific_acceptance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
