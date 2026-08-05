#!/usr/bin/env python3
"""Run and analyze full free-shell EMB breathing-mode Mirheo simulations.

This workflow is intentionally site-neutral. Site policy, module loading, and
Slurm placement belong in ``scripts/platforms/*`` wrappers.

The protocol is measurement-first: it uses the original generated EMB DPD
ingredients, applies a tiny uniform membrane-radius perturbation after thermal
equilibration, and reports the measured breathing frequency from damped-sine
and FFT fits. No analytical target is used to accept or reject a frequency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


FSCALE = 0.0074
UNIT_TIME_SECONDS = 8.86564957825223e-06

RADIUS_DPD_BY_DATASET: dict[str, tuple[float, str]] = {
    "compression_2.1um": (4.2, "configured"),
    "compression_2.9um": (5.8, "configured"),
    "compression_3.0um": (6.0, "configured"),
    "indentation_3.2um": (6.38, "configured"),
    "indentation_3.4um": (6.8, "fallback_diameter_um_over_0p5"),
    "indentation_5.8um": (11.6, "fallback_diameter_um_over_0p5"),
}

EXPECTED_SOURCE_LABELS = {
    "Definity": "attempt081_seed00_50k",
    "SonoVue": "attempt066_seed00_50k",
}

OBSERVABLE_PLOT_METADATA = {
    "rms_radius_dpd": ("rms_radius", "RMS membrane radius [DPD]"),
    "mean_radius_dpd": ("mean_radius", "Mean membrane radius [DPD]"),
    "equivalent_radius_dpd": ("equivalent_radius", "Volume-equivalent radius [DPD]"),
    "volume_dpd3": ("volume", "Membrane volume [DPD^3]"),
}

CLEAN_FFT_DAMPED_REL_TOL = 0.03
CLEAN_PEAK_TO_PEAK_DAMPED_REL_TOL = 0.03
MIN_DAMPED_FIT_R2 = 0.995
MIN_AMPLITUDE_TO_RESIDUAL_STD = 50.0
MIN_DOMINANT_TO_SECOND_FFT_POWER_RATIO = 20.0
# observed_alternating_cycles is a full-cycle count inferred from alternating
# extrema. The production protocol requires at least four visible oscillations.
MIN_OBSERVED_ALTERNATING_CYCLES = 4.0
MIN_TAIL_QUALIFIED_CYCLES = 4.0
MAX_ALTERNATING_HALF_PERIOD_CV = 0.20
MIN_TAIL_EXTREMUM_TO_RESIDUAL_RMS = 3.0
MAX_TAIL_GAP_HALF_PERIODS = 1.5
MAX_MODE_RATIO_FOR_CLEAN_L0 = 0.20
WATER_SHELL_FSI_CONSERVATIVE_SCALE = 0.4


@dataclass(frozen=True)
class Bubble:
    agent: str
    symbol: str
    modality: str
    dataset: str
    diameter_um: float
    source_label: str
    sample_index: int
    sample_count: int
    ka: float
    kb: float
    d0: float
    sigma: float
    legacy_yt: float
    log_likelihood: float
    log_prior: float
    log_posterior: float
    state_path: str
    radius_dpd: float
    radius_source: str


@dataclass(frozen=True)
class CapturedTrajectory:
    frames: list[np.ndarray]
    steps_after_release: list[int]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_commit() -> str:
    if os.environ.get("MESOUQ_GIT_COMMIT"):
        return os.environ["MESOUQ_GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def _optional_positive_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{key} must be a finite positive value when provided, got {value!r}.")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_bound_map_source(row: dict[str, str]) -> bool:
    expected_path = os.environ.get("MESOUQ_BOUND_MAP_SOURCE_PATH", "")
    expected_sha256 = os.environ.get("MESOUQ_BOUND_MAP_SOURCE_SHA256", "").lower()
    if not expected_path or len(expected_sha256) != 64:
        return False
    state_path = Path(row.get("state_path", "")).expanduser().absolute()
    if state_path != Path(expected_path).expanduser().absolute() or not state_path.is_file():
        return False
    return _sha256_file(state_path) == expected_sha256


def load_bubbles(map_values: Path) -> list[Bubble]:
    with map_values.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    bubbles: list[Bubble] = []
    for row in rows:
        source_label = row["source_label"]
        expected_label = EXPECTED_SOURCE_LABELS.get(row["agent"])
        if expected_label is None or (source_label != expected_label and not _has_bound_map_source(row)):
            raise ValueError(
                f"Unexpected MAP source for {row['diameter_symbol']}: "
                f"{source_label!r}, expected {expected_label!r}."
            )
        dataset = row["dataset"]
        explicit_radius_dpd = _optional_positive_float(row, "radius_dpd")
        if explicit_radius_dpd is not None:
            radius_dpd = explicit_radius_dpd
            radius_source = row.get("radius_source", "").strip() or "map_explicit"
        else:
            if dataset not in RADIUS_DPD_BY_DATASET:
                raise ValueError(
                    f"No radius mapping for dataset {dataset!r}; provide a positive radius_dpd column for generated nodes."
                )
            radius_dpd, radius_source = RADIUS_DPD_BY_DATASET[dataset]
        ka = _float(row, "ka")
        bubbles.append(
            Bubble(
                agent=row["agent"],
                symbol=row["diameter_symbol"],
                modality=row["modality"],
                dataset=dataset,
                diameter_um=_float(row, "diameter_um"),
                source_label=source_label,
                sample_index=_int(row, "sample_index"),
                sample_count=_int(row, "sample_count"),
                ka=ka,
                kb=_float(row, "kb"),
                d0=_float(row, "d0"),
                sigma=_float(row, "sigma"),
                legacy_yt=_float(row, "legacy_Yt"),
                log_likelihood=_float(row, "logLikelihood"),
                log_prior=_float(row, "logPrior"),
                log_posterior=_float(row, "logPosterior"),
                state_path=row["state_path"],
                radius_dpd=radius_dpd,
                radius_source=radius_source,
            )
        )
    return bubbles


def select_bubbles(bubbles: Iterable[Bubble], symbols: Iterable[str]) -> list[Bubble]:
    requested = [symbol.strip() for symbol in symbols if symbol.strip()]
    if not requested:
        raise ValueError("At least one diameter symbol is required.")
    by_symbol = {bubble.symbol: bubble for bubble in bubbles}
    missing = sorted(set(requested) - set(by_symbol))
    if missing:
        raise ValueError(f"Unknown diameter symbol(s): {', '.join(missing)}")
    return [by_symbol[symbol] for symbol in requested]


def _seed_label(seed_index: int) -> str:
    if int(seed_index) < 0:
        raise ValueError(f"Seed index must be non-negative, got {seed_index}.")
    return f"seed-{int(seed_index):03d}"


def _coerce_seed_indices(seed_indices: Iterable[int]) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for seed_index in seed_indices:
        value = int(seed_index)
        if value < 0:
            raise ValueError(f"Seed index must be non-negative, got {value}.")
        if value not in seen:
            seen.add(value)
            indices.append(value)
    if not indices:
        raise ValueError("At least one seed index is required.")
    return indices


def _result_roots_for_bubble(run_root: Path, bubble: Bubble) -> list[Path]:
    for symbol_root in (run_root / bubble.symbol, run_root / bubble.symbol / bubble.symbol):
        seed_roots = sorted(
            path
            for path in symbol_root.glob("seed-*")
            if path.is_dir() and (path / "result.json").exists()
        )
        if seed_roots:
            return seed_roots
        if (symbol_root / "result.json").exists():
            return [symbol_root]
    return []


def _default_protocol_steps(bubble: Bubble) -> tuple[int, int]:
    defaults_path = repo_root() / "emb" / bubble.modality / "src" / "parameters-default.emb.yaml"
    defaults = _load_yaml(defaults_path)
    return int(defaults["numsteps_eq"]), int(defaults["numsteps"])


def _resolve_protocol_steps(
    bubble: Bubble,
    *,
    equil_steps: int,
    relax_steps: int,
) -> tuple[int, int, dict[str, Any]]:
    default_equil_steps, default_relax_steps = _default_protocol_steps(bubble)
    resolved_equil_steps = default_equil_steps if int(equil_steps) < 0 else int(equil_steps)
    resolved_relax_steps = default_relax_steps if int(relax_steps) < 0 else int(relax_steps)
    return (
        resolved_equil_steps,
        resolved_relax_steps,
        {
            "requested_equil_steps": int(equil_steps),
            "requested_relax_steps": int(relax_steps),
            "default_equil_steps": int(default_equil_steps),
            "default_relax_steps": int(default_relax_steps),
            "resolved_equil_steps": int(resolved_equil_steps),
            "resolved_relax_steps": int(resolved_relax_steps),
        },
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = yaml.load(handle, Loader=yaml.CLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, Dumper=yaml.CDumper, default_flow_style=False, sort_keys=False)


def _physical_membrane_vertex_mass(*, defaults: dict[str, Any], mesh_path: Path) -> float:
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, process=False)
    nverts = len(mesh.vertices)
    if nverts <= 0:
        raise RuntimeError(f"Generated membrane mesh has no vertices: {mesh_path}")
    ul = float(defaults["ul"])
    um = float(defaults["rho_water"]) * ul**3 / float(defaults["rhow"])
    shell_th = float(defaults["th_fac"]) * float(defaults["shell_th"])
    return float(defaults["rho_shell"]) * shell_th * float(mesh.area) * ul**2 / um / float(nverts)


def _scale_about_centroid(vertices: np.ndarray, scale: float) -> np.ndarray:
    center = vertices.mean(axis=0)
    return center[None, :] + (vertices - center[None, :]) * float(scale)


def _prestrain_geometry(
    vertices: np.ndarray,
    *,
    initial_radius_scale: float,
    stress_free_radius_scale: float,
    prestrain_placement: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if prestrain_placement not in {"post-equilibration", "initial-geometry"}:
        raise ValueError("prestrain_placement must be 'post-equilibration' or 'initial-geometry'.")
    original_vertices = np.asarray(vertices, dtype=np.float64)
    stress_free_vertices = _scale_about_centroid(original_vertices, stress_free_radius_scale)
    if prestrain_placement == "initial-geometry":
        initial_vertices = _scale_about_centroid(original_vertices, initial_radius_scale)
        post_equilibration_scale = 1.0
    else:
        initial_vertices = np.array(original_vertices, copy=True)
        post_equilibration_scale = float(initial_radius_scale)
    return initial_vertices, stress_free_vertices, post_equilibration_scale


def _initial_geometry_prestrain_payload(
    original_vertices: np.ndarray,
    initial_vertices: np.ndarray,
    stress_free_vertices: np.ndarray,
    *,
    initial_radius_scale: float,
    stress_free_radius_scale: float,
) -> dict[str, Any]:
    def _radius_stats(values: np.ndarray) -> dict[str, float]:
        center = values.mean(axis=0)
        radii = np.linalg.norm(values - center[None, :], axis=1)
        return {
            "mean_radius_dpd": float(np.mean(radii)),
            "rms_radius_dpd": float(np.sqrt(np.mean(radii * radii))),
        }

    status = (
        "not_requested_initial_geometry"
        if abs(float(initial_radius_scale) - 1.0) <= 1.0e-14
        else "applied_initial_geometry"
    )
    return {
        "status": status,
        "prestrain_placement": "initial-geometry",
        "initial_radius_scale": float(initial_radius_scale),
        "stress_free_radius_scale": float(stress_free_radius_scale),
        "original_geometry": _radius_stats(np.asarray(original_vertices, dtype=np.float64)),
        "initial_geometry": _radius_stats(np.asarray(initial_vertices, dtype=np.float64)),
        "stress_free_geometry": _radius_stats(np.asarray(stress_free_vertices, dtype=np.float64)),
        "post_equilibration_coordinate_jump": False,
    }


def _dpd_frequency_to_mhz(frequency_dpd: float) -> float:
    return float(frequency_dpd) / UNIT_TIME_SECONDS / 1.0e6


def _mhz_to_dpd_frequency(frequency_mhz: float) -> float:
    return float(frequency_mhz) * 1.0e6 * UNIT_TIME_SECONDS


def _tone_burst_envelope(time_dpd: float, duration_dpd: float, ramp_dpd: float) -> float:
    if duration_dpd <= 0.0 or time_dpd < 0.0 or time_dpd > duration_dpd:
        return 0.0
    ramp = min(max(0.0, float(ramp_dpd)), 0.5 * float(duration_dpd))
    if ramp <= 0.0:
        return 1.0
    if time_dpd < ramp:
        return 0.5 * (1.0 - math.cos(math.pi * float(time_dpd) / ramp))
    if time_dpd > duration_dpd - ramp:
        return 0.5 * (1.0 - math.cos(math.pi * float(duration_dpd - time_dpd) / ramp))
    return 1.0


def _tone_burst_force_scale(
    *,
    time_dpd: float,
    frequency_dpd: float,
    force_per_vertex: float,
    duration_dpd: float,
    ramp_dpd: float,
) -> float:
    envelope = _tone_burst_envelope(time_dpd, duration_dpd, ramp_dpd)
    return float(force_per_vertex) * envelope * math.sin(2.0 * math.pi * float(frequency_dpd) * float(time_dpd))


def _radial_unit_directions(vertices: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    directions = vertices - center[None, :]
    norms = np.linalg.norm(directions, axis=1)
    nonzero = norms > 0.0
    directions[nonzero] /= norms[nonzero, None]
    return directions


def _current_radial_forces(
    emb: Any,
    *,
    force_per_vertex: float,
    comm: Any,
) -> dict[str, Any]:
    from mpi4py import MPI

    rank = comm.Get_rank()
    can_access = hasattr(emb, "getCoordinates")
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError("Current-radial force requires membrane coordinate access.")
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    payload: dict[str, Any] | None = None
    if rank == access_rank:
        coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise RuntimeError(f"Unexpected membrane coordinate shape: {coordinates.shape}.")
        center = coordinates.mean(axis=0)
        directions = coordinates - center[None, :]
        radii = np.linalg.norm(directions, axis=1)
        nonzero = radii > 0.0
        unit_radial = np.zeros_like(directions)
        unit_radial[nonzero] = directions[nonzero] / radii[nonzero, None]
        forces = float(force_per_vertex) * unit_radial
        force_norms = np.linalg.norm(forces, axis=1)
        payload = {
            "status": "computed",
            "coordinate_rank": int(access_rank),
            "force_per_vertex": float(force_per_vertex),
            "forces": forces.tolist(),
            "particle_count": int(coordinates.shape[0]),
            "center_dpd": [float(item) for item in center],
            "mean_radius_dpd": float(np.mean(radii)),
            "rms_radius_dpd": float(np.sqrt(np.mean(radii * radii))),
            "mean_abs_force_per_vertex": float(np.mean(force_norms)) if len(force_norms) else 0.0,
            "max_abs_force_per_vertex": float(np.max(force_norms)) if len(force_norms) else 0.0,
            "zero_radial_count": int(np.count_nonzero(~nonzero)),
        }
    payload = comm.bcast(payload, root=access_rank)
    comm.Barrier()
    if payload is None:
        raise RuntimeError("Current-radial force computation failed to broadcast payload.")
    return payload


def _current_area_weighted_normal_forces(
    emb: Any,
    faces: np.ndarray,
    *,
    force_per_vertex: float,
    comm: Any,
) -> dict[str, Any]:
    from mpi4py import MPI

    rank = comm.Get_rank()
    can_access = hasattr(emb, "getCoordinates")
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError("Current-normal force requires membrane coordinate access.")
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    payload: dict[str, Any] | None = None
    if rank == access_rank:
        coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise RuntimeError(f"Unexpected membrane coordinate shape: {coordinates.shape}.")
        tri = coordinates[np.asarray(faces, dtype=np.int64)]
        face_area_vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_area_vector_norms = np.linalg.norm(face_area_vectors, axis=1)
        valid = face_area_vector_norms > 0.0
        face_normals = np.zeros_like(face_area_vectors)
        face_normals[valid] = face_area_vectors[valid] / face_area_vector_norms[valid, None]
        center = coordinates.mean(axis=0)
        face_centers = tri.mean(axis=1)
        outward = np.einsum("ij,ij->i", face_normals, face_centers - center[None, :])
        face_normals[outward < 0.0] *= -1.0
        face_areas = 0.5 * face_area_vector_norms

        vertex_normals = np.zeros_like(coordinates)
        vertex_areas = np.zeros(coordinates.shape[0], dtype=np.float64)
        for local_index in range(3):
            vertex_ids = faces[:, local_index]
            area_share = face_areas / 3.0
            np.add.at(vertex_normals, vertex_ids, face_normals * area_share[:, None])
            np.add.at(vertex_areas, vertex_ids, area_share)

        normal_norms = np.linalg.norm(vertex_normals, axis=1)
        nonzero_normals = normal_norms > 0.0
        unit_normals = np.zeros_like(vertex_normals)
        unit_normals[nonzero_normals] = (
            vertex_normals[nonzero_normals] / normal_norms[nonzero_normals, None]
        )
        positive_areas = vertex_areas[vertex_areas > 0.0]
        mean_vertex_area = float(np.mean(positive_areas)) if len(positive_areas) else 1.0
        area_weights = np.ones_like(vertex_areas)
        if mean_vertex_area > 0.0:
            area_weights = vertex_areas / mean_vertex_area
        forces = float(force_per_vertex) * area_weights[:, None] * unit_normals
        force_norms = np.linalg.norm(forces, axis=1)
        radii = np.linalg.norm(coordinates - center[None, :], axis=1)
        payload = {
            "status": "computed",
            "coordinate_rank": int(access_rank),
            "force_per_vertex": float(force_per_vertex),
            "forces": forces.tolist(),
            "particle_count": int(coordinates.shape[0]),
            "center_dpd": [float(item) for item in center],
            "mean_radius_dpd": float(np.mean(radii)),
            "rms_radius_dpd": float(np.sqrt(np.mean(radii * radii))),
            "mean_vertex_area_dpd2": float(mean_vertex_area),
            "min_vertex_area_dpd2": float(np.min(vertex_areas)) if len(vertex_areas) else 0.0,
            "max_vertex_area_dpd2": float(np.max(vertex_areas)) if len(vertex_areas) else 0.0,
            "mean_abs_force_per_vertex": float(np.mean(force_norms)) if len(force_norms) else 0.0,
            "max_abs_force_per_vertex": float(np.max(force_norms)) if len(force_norms) else 0.0,
            "zero_normal_count": int(np.count_nonzero(~nonzero_normals)),
        }
    payload = comm.bcast(payload, root=access_rank)
    comm.Barrier()
    if payload is None:
        raise RuntimeError("Current-normal force computation failed to broadcast payload.")
    return payload


def _normal_force_payload_for_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(payload)
    sidecar.pop("forces", None)
    return sidecar


def _vertex_normal_geometry(coordinates: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise RuntimeError(f"Unexpected membrane coordinate shape: {coordinates.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Unexpected membrane face shape: {faces.shape}.")

    tri = coordinates[faces]
    face_area_vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_area_vector_norms = np.linalg.norm(face_area_vectors, axis=1)
    valid = face_area_vector_norms > 0.0
    face_normals = np.zeros_like(face_area_vectors)
    face_normals[valid] = face_area_vectors[valid] / face_area_vector_norms[valid, None]
    center = coordinates.mean(axis=0)
    face_centers = tri.mean(axis=1)
    outward = np.einsum("ij,ij->i", face_normals, face_centers - center[None, :])
    face_normals[outward < 0.0] *= -1.0
    face_areas = 0.5 * face_area_vector_norms

    vertex_normals = np.zeros_like(coordinates)
    vertex_areas = np.zeros(coordinates.shape[0], dtype=np.float64)
    area_share = face_areas / 3.0
    for local_index in range(3):
        vertex_ids = faces[:, local_index]
        np.add.at(vertex_normals, vertex_ids, face_normals * area_share[:, None])
        np.add.at(vertex_areas, vertex_ids, area_share)

    normal_norms = np.linalg.norm(vertex_normals, axis=1)
    nonzero_normals = normal_norms > 0.0
    unit_normals = np.zeros_like(vertex_normals)
    unit_normals[nonzero_normals] = (
        vertex_normals[nonzero_normals] / normal_norms[nonzero_normals, None]
    )
    positive_areas = vertex_areas[vertex_areas > 0.0]
    mean_vertex_area = float(np.mean(positive_areas)) if len(positive_areas) else 1.0
    area_weights = np.ones_like(vertex_areas)
    if mean_vertex_area > 0.0:
        area_weights = vertex_areas / mean_vertex_area
    radii = np.linalg.norm(coordinates - center[None, :], axis=1)
    return {
        "center": center,
        "unit_normals": unit_normals,
        "vertex_areas": vertex_areas,
        "area_weights": area_weights,
        "mean_vertex_area": mean_vertex_area,
        "radii": radii,
        "zero_normal_count": int(np.count_nonzero(~nonzero_normals)),
    }


def _normal_velocity_kick_shape(
    coordinates: np.ndarray,
    faces: np.ndarray,
    *,
    profile: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    geometry = _vertex_normal_geometry(coordinates, faces)
    if profile == "surface-normal":
        shape = np.asarray(geometry["unit_normals"], dtype=np.float64)
    elif profile == "volume-gradient":
        shape = (
            np.asarray(geometry["area_weights"], dtype=np.float64)[:, None]
            * np.asarray(geometry["unit_normals"], dtype=np.float64)
        )
    else:
        raise ValueError(f"Unsupported normal velocity kick profile: {profile}")

    raw_mean = shape.mean(axis=0)
    shape = shape - raw_mean[None, :]
    rms = float(np.sqrt(np.mean(np.sum(shape * shape, axis=1))))
    if rms <= 0.0:
        raise RuntimeError(f"Degenerate {profile} velocity shape has zero RMS norm.")
    normalized = shape / rms
    normalized_norm = np.sqrt(np.sum(normalized * normalized, axis=1))
    return normalized, {
        "profile": profile,
        "mean_removed_dpd_per_time": [float(item) for item in raw_mean],
        "raw_rms": rms,
        "normalized_rms": float(np.sqrt(np.mean(normalized_norm * normalized_norm))),
        "min_normalized_speed": float(np.min(normalized_norm)) if len(normalized_norm) else 0.0,
        "max_normalized_speed": float(np.max(normalized_norm)) if len(normalized_norm) else 0.0,
        "mean_vertex_area_dpd2": float(geometry["mean_vertex_area"]),
        "min_vertex_area_dpd2": (
            float(np.min(geometry["vertex_areas"])) if len(geometry["vertex_areas"]) else 0.0
        ),
        "max_vertex_area_dpd2": (
            float(np.max(geometry["vertex_areas"])) if len(geometry["vertex_areas"]) else 0.0
        ),
        "zero_normal_count": int(geometry["zero_normal_count"]),
    }


def _lim_mu_from_ka(ka: float, nu: float) -> float:
    """Follow the EMB generator's shared-Yt relationship between Lim ka and mu."""
    ka_value = float(ka)
    nu_value = float(nu)
    if not math.isfinite(ka_value) or ka_value < 0.0:
        raise ValueError(f"ka must be finite and non-negative for Lim mu coupling, got {ka!r}.")
    if not math.isfinite(nu_value) or not (-1.0 < nu_value < 1.0):
        raise ValueError(f"nu must lie strictly between -1 and 1 for Lim mu coupling, got {nu!r}.")
    return ka_value * (1.0 - nu_value) / (1.0 + nu_value)


def prepare_simulation(
    bubble: Bubble,
    symbol_root: Path,
    *,
    dt: float,
    equil_steps: int,
    pulse_steps: int,
    relax_steps: int,
    sample_every: int,
    box_padding_dpd: float,
    membrane_mass_scale: float = 1.0,
    ka_override: float | None = None,
    kb_override: float | None = None,
    lim_mu_policy: str = "legacy-generated",
    force: bool = False,
) -> Path:
    root = repo_root()
    source_dir = root / "emb" / bubble.modality / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing EMB source directory: {source_dir}")

    sim_dir = symbol_root / "simulation"
    if force and sim_dir.exists():
        shutil.rmtree(sim_dir)
    (sim_dir / "parameter").mkdir(parents=True, exist_ok=True)
    (sim_dir / "mesh").mkdir(parents=True, exist_ok=True)
    (sim_dir / "microbubble").mkdir(parents=True, exist_ok=True)
    (sim_dir / "logs").mkdir(parents=True, exist_ok=True)
    (sim_dir / "restart").mkdir(parents=True, exist_ok=True)
    (sim_dir / "particles").mkdir(parents=True, exist_ok=True)
    (sim_dir / "stats").mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_dir / "microbubble" / "sphere_icosphere.py", sim_dir / "microbubble")

    defaults_path = source_dir / "parameters-default.emb.yaml"
    defaults = _load_yaml(defaults_path)
    if lim_mu_policy not in {"legacy-generated", "derive-from-ka"}:
        raise ValueError("lim_mu_policy must be 'legacy-generated' or 'derive-from-ka'.")
    ka_value = float(bubble.ka if ka_override is None else ka_override)
    kb_value = float(bubble.kb if kb_override is None else kb_override)
    nu_value = float(defaults["nu"])
    derived_mu = _lim_mu_from_ka(ka_value, nu_value)
    box_extent = float(math.ceil(2.0 * bubble.radius_dpd + box_padding_dpd))
    defaults.update(
        {
            "radp": float(bubble.radius_dpd),
            "Lx": box_extent,
            "Ly": box_extent,
            "Lz": box_extent,
            "dt": float(dt),
            "dt_eq": float(dt),
            "numsteps": int(pulse_steps + relax_steps),
            "numsteps_eq": int(equil_steps),
            "stslik": max(1, int(math.ceil((pulse_steps + relax_steps) / sample_every))),
            "stslik_eq": max(1, int(math.ceil(max(1, equil_steps) / max(1, sample_every)))),
            "Yt": float(bubble.legacy_yt),
            "Yl": float(bubble.legacy_yt),
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
        }
    )
    default_file = sim_dir / "parameter" / "parameters-default00001.yaml"
    default_eq_file = sim_dir / "parameter" / "parameters-default00001eq.yaml"
    _write_yaml(default_file, defaults)
    _write_yaml(default_eq_file, defaults)

    sys.path.insert(0, str(source_dir))
    try:
        if bubble.modality == "compression":
            from emb.compression.src.parameters import write_parameters
        elif bubble.modality == "indentation":
            from emb.indentation.src.parameters import write_parameters
        else:  # pragma: no cover - guarded by CSV inputs
            raise ValueError(f"Unsupported modality: {bubble.modality}")
        write_parameters(source_path=str(source_dir) + "/", simu_path=str(sim_dir) + "/", simnum="00001")
    finally:
        try:
            sys.path.remove(str(source_dir))
        except ValueError:
            pass

    if float(membrane_mass_scale) <= 0.0:
        raise ValueError("--membrane-mass-scale must be positive.")
    physical_mvert = _physical_membrane_vertex_mass(
        defaults=defaults,
        mesh_path=sim_dir / "mesh" / "emb00001.off",
    )
    for params_name in ("parameters00001.yaml", "parameters00001eq.yaml"):
        params_path = sim_dir / "parameter" / params_name
        if not params_path.exists():
            continue
        params = _load_yaml(params_path)
        raw_generated_mvert = float(params["mvert"])
        if bubble.modality == "indentation":
            raw_over_physical = raw_generated_mvert / physical_mvert
            if math.isclose(raw_over_physical, 5.0, rel_tol=0.05, abs_tol=0.05):
                mass_policy = "physical_membrane_mass_hidden_indentation_x5_removed"
            elif math.isclose(raw_over_physical, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-15):
                mass_policy = "physical_membrane_mass_from_current_indentation_generator"
            else:
                raise RuntimeError(
                    "Indentation generated membrane mass is neither the physical mass nor the known hidden x5 "
                    f"form for {bubble.symbol}: raw/physical={raw_over_physical:.6g}."
                )
            original_mvert = physical_mvert
        else:
            if not math.isclose(raw_generated_mvert, physical_mvert, rel_tol=1.0e-9, abs_tol=1.0e-15):
                raise RuntimeError(
                    "Compression generated membrane mass does not match the physical shell mass "
                    f"for {bubble.symbol}: raw={raw_generated_mvert:.16g}, physical={physical_mvert:.16g}."
                )
            original_mvert = raw_generated_mvert
            mass_policy = "original_generated_physical_membrane_mass"
        params["mvert_raw_generated"] = raw_generated_mvert
        params["mvert_unscaled"] = original_mvert
        params["membrane_mass_scale"] = float(membrane_mass_scale)
        params["membrane_mass_policy"] = mass_policy
        params["mvert"] = original_mvert * float(membrane_mass_scale)
        params["ka"] = ka_value
        params["kb"] = kb_value
        params["mu"] = derived_mu if lim_mu_policy == "derive-from-ka" else float(params["mu"])
        params["lim_mu_policy"] = lim_mu_policy
        params["lim_mu_nu"] = nu_value
        params["lim_mu_to_ka_ratio"] = (1.0 - nu_value) / (1.0 + nu_value)
        _write_yaml(params_path, params)

    for suffix in ("00001", "00001eq"):
        default_path = sim_dir / "parameter" / f"parameters-default{suffix}.yaml"
        if default_path.exists():
            default_payload = _load_yaml(default_path)
            default_payload.update(defaults)
            _write_yaml(default_path, default_payload)

    for prms_name in ("parameters.prms00001.yaml", "parameters.prms00001eq.yaml"):
        prms_path = sim_dir / "parameter" / prms_name
        if not prms_path.exists():
            continue
        prms = _load_yaml(prms_path)
        mu_value = derived_mu if lim_mu_policy == "derive-from-ka" else float(prms["mu"])
        prms.update(
            {
                "ka": ka_value,
                "kb": kb_value,
                "mu": mu_value,
                "a3": 0.0,
                "a4": 0.0,
                "b1": 0.0,
                "b2": 0.0,
            }
        )
        _write_yaml(prms_path, prms)

    if ka_override is not None or kb_override is not None or lim_mu_policy != "legacy-generated":
        override_payload = {
            "schema": "mesouq.emb_breathing_membrane_elastic_override.v1",
            "symbol": bubble.symbol,
            "policy": "diagnostic_non_map_override",
            "map_ka": float(bubble.ka),
            "map_kb": float(bubble.kb),
            "ka": float(bubble.ka if ka_override is None else ka_override),
            "kb": float(bubble.kb if kb_override is None else kb_override),
            "mu": derived_mu if lim_mu_policy == "derive-from-ka" else None,
            "lim_mu_policy": lim_mu_policy,
            "lim_mu_nu": nu_value,
            "lim_mu_to_ka_ratio": (1.0 - nu_value) / (1.0 + nu_value),
            "ka_overridden": ka_override is not None,
            "kb_overridden": kb_override is not None,
            "mu_overridden_from_ka": lim_mu_policy == "derive-from-ka",
        }
        (symbol_root / "membrane_elastic_override.json").write_text(
            json.dumps(override_payload, indent=2), encoding="utf-8"
        )

    return sim_dir


def _prepare_mirheo_log_root(symbol_root: Path) -> None:
    log_root = symbol_root / "mirheo_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    os.environ["MIRHEO_LOG_ROOT"] = str(log_root)


def _apply_post_equilibration_prestrain(
    emb: Any,
    *,
    initial_radius_scale: float,
    comm: Any,
) -> dict[str, Any]:
    from mpi4py import MPI

    scale = float(initial_radius_scale)
    if abs(scale - 1.0) <= 1.0e-14:
        return {"status": "not_requested", "initial_radius_scale": scale}
    rank = comm.Get_rank()
    can_access = hasattr(emb, "getCoordinates") and hasattr(emb, "setCoordinates")
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError("Post-equilibration prestrain requires membrane coordinate access.")
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    payload: dict[str, Any] | None = None
    if rank == access_rank:
        coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
        center = coordinates.mean(axis=0)
        radii_before = np.linalg.norm(coordinates - center[None, :], axis=1)
        scaled = center[None, :] + (coordinates - center[None, :]) * scale
        emb.setCoordinates(scaled.tolist())
        radii_after = np.linalg.norm(scaled - center[None, :], axis=1)
        payload = {
            "status": "applied",
            "coordinate_rank": int(access_rank),
            "initial_radius_scale": scale,
            "center_dpd": [float(item) for item in center],
            "mean_radius_before_dpd": float(np.mean(radii_before)),
            "mean_radius_after_dpd": float(np.mean(radii_after)),
            "rms_radius_before_dpd": float(np.sqrt(np.mean(radii_before * radii_before))),
            "rms_radius_after_dpd": float(np.sqrt(np.mean(radii_after * radii_after))),
        }
    payload = comm.bcast(payload, root=access_rank)
    comm.Barrier()
    if payload is None:
        raise RuntimeError("Post-equilibration prestrain failed to broadcast payload.")
    return payload


def _run_post_equilibration_prestrain_ramp(
    u: Any,
    emb: Any,
    *,
    dt: float,
    initial_radius_scale: float,
    ramp_steps: int,
    reset_velocities: bool,
    comm: Any,
) -> dict[str, Any]:
    """Impose a smooth geometric prestrain before the fixed-shape hold.

    The membrane follows a half-cosine scale ramp about its instantaneous
    centroid. Water and gas continue integrating while the membrane geometry
    is kinematically prescribed. The final membrane state is restored with
    zero velocity before the existing hold begins. No force or constraint
    remains registered for the free response.
    """

    from mpi4py import MPI

    steps = int(ramp_steps)
    target_scale = float(initial_radius_scale)
    if steps <= 0:
        return _apply_post_equilibration_prestrain(
            emb,
            initial_radius_scale=target_scale,
            comm=comm,
        )
    if target_scale <= 0.0:
        raise ValueError("initial_radius_scale must be positive.")

    rank = comm.Get_rank()
    can_access_coordinates = hasattr(emb, "getCoordinates") and hasattr(emb, "setCoordinates")
    can_access_velocities = hasattr(emb, "getVelocities") and hasattr(emb, "setVelocities")
    can_access = can_access_coordinates and (not reset_velocities or can_access_velocities)
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError(
            "Post-equilibration prestrain ramp requires membrane coordinate access and, when requested, velocity access."
        )
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    start_coordinates: np.ndarray | None = None
    target_coordinates: np.ndarray | None = None
    zero_velocities: np.ndarray | None = None
    payload: dict[str, Any] | None = None
    if rank == access_rank:
        start_coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
        if start_coordinates.ndim != 2 or start_coordinates.shape[1] != 3 or start_coordinates.size == 0:
            raise RuntimeError("Post-equilibration prestrain ramp received invalid membrane coordinates.")
        center = start_coordinates.mean(axis=0)
        offsets = start_coordinates - center[None, :]
        target_coordinates = center[None, :] + target_scale * offsets
        zero_velocities = np.zeros_like(start_coordinates) if reset_velocities else None
        start_radii = np.linalg.norm(offsets, axis=1)
        target_radii = np.linalg.norm(target_coordinates - center[None, :], axis=1)
        payload = {
            "status": "smooth_ramp_applied",
            "constraint_type": "half_cosine_coordinate_ramp",
            "coordinate_rank": int(access_rank),
            "rank_count_with_access": int(access_count),
            "initial_radius_scale": target_scale,
            "ramp_steps": steps,
            "ramp_duration_dpd": float(steps * float(dt)),
            "reset_velocities": bool(reset_velocities),
            "center_dpd": [float(item) for item in center],
            "mean_radius_before_dpd": float(np.mean(start_radii)),
            "mean_radius_after_dpd": float(np.mean(target_radii)),
            "rms_radius_before_dpd": float(np.sqrt(np.mean(start_radii * start_radii))),
            "rms_radius_after_dpd": float(np.sqrt(np.mean(target_radii * target_radii))),
        }
    payload = comm.bcast(payload, root=access_rank)
    if payload is None:
        raise RuntimeError("Post-equilibration prestrain ramp failed to initialize.")

    for step in range(1, steps + 1):
        fraction = 0.5 * (1.0 - math.cos(math.pi * float(step) / float(steps)))
        scale = 1.0 + (target_scale - 1.0) * fraction
        if rank == access_rank:
            assert start_coordinates is not None
            center = start_coordinates.mean(axis=0)
            coordinates = center[None, :] + scale * (start_coordinates - center[None, :])
            emb.setCoordinates(coordinates.tolist())
            if zero_velocities is not None:
                emb.setVelocities(zero_velocities.tolist())
        comm.Barrier()
        u.run(1, dt=float(dt))

    if rank == access_rank:
        assert target_coordinates is not None
        emb.setCoordinates(target_coordinates.tolist())
        if zero_velocities is not None:
            emb.setVelocities(zero_velocities.tolist())
        payload["release_coordinates_restored"] = True
        payload["release_velocity_reset"] = bool(zero_velocities is not None)
    payload = comm.bcast(payload, root=access_rank)
    comm.Barrier()
    if payload is None:
        raise RuntimeError("Post-equilibration prestrain ramp failed to finalize.")
    return payload


def _run_post_deflation_coordinate_hold(
    u: Any,
    emb: Any,
    *,
    dt: float,
    hold_steps: int,
    update_every_steps: int,
    reset_velocities: bool,
    comm: Any,
) -> dict[str, Any]:
    """Hold post-deflation membrane coordinates while the fluids settle.

    This is a preparation constraint, not a measurement force.  The exact
    post-deflation coordinates are restored before each integration chunk and
    once more immediately before the free release.  With an update cadence of
    one step and velocity resets enabled, the membrane remains fixed while
    water and gas evolve thermally around it.
    """
    from mpi4py import MPI

    if int(hold_steps) <= 0:
        return {
            "status": "not_requested",
            "hold_steps": int(hold_steps),
            "update_every_steps": int(update_every_steps),
            "reset_velocities": bool(reset_velocities),
        }
    rank = comm.Get_rank()
    can_access_coordinates = hasattr(emb, "getCoordinates") and hasattr(emb, "setCoordinates")
    can_access_velocities = hasattr(emb, "getVelocities") and hasattr(emb, "setVelocities")
    can_access = can_access_coordinates and (not reset_velocities or can_access_velocities)
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError(
            "Post-deflation coordinate hold requires membrane coordinate access and, when requested, velocity access."
        )
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    target_coordinates: np.ndarray | None = None
    target_velocities: np.ndarray | None = None
    payload: dict[str, Any] | None = None
    if rank == access_rank:
        target_coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
        if target_coordinates.ndim != 2 or target_coordinates.shape[1] != 3 or target_coordinates.size == 0:
            raise RuntimeError("Post-deflation coordinate hold received invalid membrane coordinates.")
        target_center = target_coordinates.mean(axis=0)
        target_radii = np.linalg.norm(target_coordinates - target_center[None, :], axis=1)
        target_velocities = np.zeros_like(target_coordinates) if reset_velocities else None
        payload = {
            "status": "applied_then_released",
            "constraint_type": "coordinate_reset",
            "coordinate_rank": int(access_rank),
            "rank_count_with_access": int(access_count),
            "hold_steps": int(hold_steps),
            "hold_duration_dpd": float(int(hold_steps) * float(dt)),
            "update_every_steps": max(1, int(update_every_steps)),
            "reset_velocities": bool(reset_velocities),
            "target_center_dpd": [float(item) for item in target_center],
            "target_mean_radius_dpd": float(np.mean(target_radii)),
            "target_rms_radius_dpd": float(np.sqrt(np.mean(target_radii * target_radii))),
            "max_rms_radius_error_before_reset_dpd": 0.0,
            "max_coordinate_rms_error_before_reset_dpd": 0.0,
            "chunk_count": 0,
        }
    payload = comm.bcast(payload, root=access_rank)
    if payload is None:
        raise RuntimeError("Post-deflation coordinate hold failed to initialize.")

    update_every = max(1, int(update_every_steps))
    remaining = int(hold_steps)
    chunk_count = 0
    while remaining > 0:
        chunk_steps = min(update_every, remaining)
        if rank == access_rank:
            assert target_coordinates is not None
            emb.setCoordinates(target_coordinates.tolist())
            if target_velocities is not None:
                emb.setVelocities(target_velocities.tolist())
        comm.Barrier()
        u.run(int(chunk_steps), dt=float(dt))
        if rank == access_rank:
            assert target_coordinates is not None
            current_coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
            current_center = current_coordinates.mean(axis=0)
            current_radii = np.linalg.norm(current_coordinates - current_center[None, :], axis=1)
            coordinate_error = current_coordinates - target_coordinates
            payload["max_rms_radius_error_before_reset_dpd"] = max(
                float(payload["max_rms_radius_error_before_reset_dpd"]),
                abs(float(np.sqrt(np.mean(current_radii * current_radii))) - float(payload["target_rms_radius_dpd"])),
            )
            payload["max_coordinate_rms_error_before_reset_dpd"] = max(
                float(payload["max_coordinate_rms_error_before_reset_dpd"]),
                float(np.sqrt(np.mean(coordinate_error * coordinate_error))),
            )
        remaining -= chunk_steps
        chunk_count += 1

    if rank == access_rank:
        assert target_coordinates is not None
        emb.setCoordinates(target_coordinates.tolist())
        if target_velocities is not None:
            emb.setVelocities(target_velocities.tolist())
        payload["chunk_count"] = int(chunk_count)
        payload["release_coordinates_restored"] = True
        payload["release_velocity_reset"] = bool(target_velocities is not None)
    payload = comm.bcast(payload, root=access_rank)
    comm.Barrier()
    if payload is None:
        raise RuntimeError("Post-deflation coordinate hold failed to finalize.")
    return payload


def _radial_velocity_kick_profile(
    coordinates: np.ndarray,
    kick: float,
    *,
    profile: str,
) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    center = coordinates.mean(axis=0)
    radial_vectors = coordinates - center[None, :]
    radii = np.linalg.norm(radial_vectors, axis=1)
    directions = np.zeros_like(radial_vectors)
    nonzero = radii > 0.0
    directions[nonzero] = radial_vectors[nonzero] / radii[nonzero, None]
    if profile == "unit-radial":
        delta = float(kick) * directions
        radial_delta = np.einsum("ij,ij->i", delta, directions)
        mean_radius = float(np.mean(radii[nonzero])) if np.any(nonzero) else 0.0
    elif profile == "self-similar-scale":
        mean_radius = float(np.mean(radii[nonzero])) if np.any(nonzero) else 0.0
        if mean_radius <= 0.0:
            raise RuntimeError("Cannot apply self-similar radial kick to degenerate coordinates.")
        radial_delta = float(kick) * radii / mean_radius
        delta = radial_delta[:, None] * directions
    else:
        raise ValueError(f"Unsupported radial velocity kick profile: {profile}")
    return {
        "center": center,
        "directions": directions,
        "radii": radii,
        "mean_radius": mean_radius,
        "delta": delta,
        "radial_delta": radial_delta,
        "profile": profile,
    }


def _radial_velocity_kick_mode_parts(radial_velocity_kick_mode: str) -> tuple[str, str]:
    if radial_velocity_kick_mode in {"add", "replace"}:
        return radial_velocity_kick_mode, "unit-radial"
    if radial_velocity_kick_mode == "scale-add":
        return "add", "self-similar-scale"
    if radial_velocity_kick_mode == "scale-replace":
        return "replace", "self-similar-scale"
    if radial_velocity_kick_mode == "normal-add":
        return "add", "surface-normal"
    if radial_velocity_kick_mode == "normal-replace":
        return "replace", "surface-normal"
    if radial_velocity_kick_mode == "volume-gradient-add":
        return "add", "volume-gradient"
    if radial_velocity_kick_mode == "volume-gradient-replace":
        return "replace", "volume-gradient"
    if radial_velocity_kick_mode == "shape-add":
        return "add", "external-shape"
    if radial_velocity_kick_mode == "shape-replace":
        return "replace", "external-shape"
    raise ValueError(
        "radial_velocity_kick_mode must be 'add', 'replace', 'scale-add', "
        "'scale-replace', 'normal-add', 'normal-replace', 'volume-gradient-add', "
        "'volume-gradient-replace', 'shape-add', or 'shape-replace'."
    )


def _load_velocity_shape(path: Path, *, expected_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_shape = payload.get("shape_vectors")
    if raw_shape is None:
        raw_shape = payload.get("velocity_shape")
    if raw_shape is None:
        raise RuntimeError(f"Velocity shape file has no shape_vectors field: {path}")
    shape = np.asarray(raw_shape, dtype=np.float64)
    if shape.shape != expected_shape:
        raise RuntimeError(
            f"Velocity shape {shape.shape} does not match membrane coordinates {expected_shape}."
        )
    if not np.all(np.isfinite(shape)):
        raise RuntimeError(f"Velocity shape contains non-finite entries: {path}")
    shape = shape - shape.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(shape * shape, axis=1))))
    if rms <= 0.0:
        raise RuntimeError(f"Velocity shape has zero RMS norm: {path}")
    return shape / rms, payload


def _apply_radial_velocity_kick(
    emb: Any,
    *,
    radial_velocity_kick: float,
    radial_velocity_kick_mode: str,
    velocity_shape_json: Path | None,
    faces: np.ndarray | None,
    comm: Any,
) -> dict[str, Any]:
    from mpi4py import MPI

    kick = float(radial_velocity_kick)
    if abs(kick) <= 1.0e-14:
        return {
            "status": "not_requested",
            "radial_velocity_kick_dpd": kick,
            "radial_velocity_kick_mode": radial_velocity_kick_mode,
        }
    kick_action, kick_profile = _radial_velocity_kick_mode_parts(radial_velocity_kick_mode)
    rank = comm.Get_rank()
    can_access = (
        hasattr(emb, "getCoordinates")
        and hasattr(emb, "getVelocities")
        and hasattr(emb, "setVelocities")
    )
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError("Radial velocity kick requires membrane coordinate and velocity access.")
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    payload: dict[str, Any] | None = None
    if rank == access_rank:
        coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
        velocities = np.asarray(emb.getVelocities(), dtype=np.float64)
        if velocities.shape != coordinates.shape:
            raise RuntimeError(
                f"Velocity shape {velocities.shape} does not match coordinate shape {coordinates.shape}."
            )
        uses_shape_profile = kick_profile in {"external-shape", "surface-normal", "volume-gradient"}
        profile_payload = _radial_velocity_kick_profile(
            coordinates,
            0.0 if uses_shape_profile else kick,
            profile="unit-radial" if uses_shape_profile else kick_profile,
        )
        directions = profile_payload["directions"]
        radial_before = np.einsum("ij,ij->i", velocities, directions)
        shape_payload: dict[str, Any] | None = None
        if kick_profile == "external-shape":
            if velocity_shape_json is None:
                raise RuntimeError(
                    "shape-add/shape-replace radial velocity modes require --velocity-shape-json."
                )
            normalized_shape, shape_payload = _load_velocity_shape(
                velocity_shape_json,
                expected_shape=coordinates.shape,
            )
            delta = kick * normalized_shape
            radial_delta = np.einsum("ij,ij->i", delta, directions)
        elif kick_profile in {"surface-normal", "volume-gradient"}:
            if faces is None:
                raise RuntimeError(f"{kick_profile} velocity mode requires membrane mesh faces.")
            normalized_shape, shape_payload = _normal_velocity_kick_shape(
                coordinates,
                faces,
                profile=kick_profile,
            )
            delta = kick * normalized_shape
            radial_delta = np.einsum("ij,ij->i", delta, directions)
        else:
            delta = profile_payload["delta"]
            radial_delta = profile_payload["radial_delta"]
        if kick_action == "replace":
            kicked = delta
        else:
            kicked = velocities + delta
        emb.setVelocities(kicked.tolist())
        radial_after = np.einsum("ij,ij->i", kicked, directions)
        tangential_after = kicked - radial_after[:, None] * directions
        delta_speed = np.sqrt(np.sum(delta * delta, axis=1))
        payload = {
            "status": "applied",
            "velocity_rank": int(access_rank),
            "radial_velocity_kick_dpd": kick,
            "radial_velocity_kick_mode": radial_velocity_kick_mode,
            "radial_velocity_kick_action": kick_action,
            "radial_velocity_kick_profile": kick_profile,
            "mean_radius_dpd": float(profile_payload["mean_radius"]),
            "min_radial_delta_velocity_dpd": float(np.min(radial_delta)),
            "max_radial_delta_velocity_dpd": float(np.max(radial_delta)),
            "rms_delta_velocity_dpd": float(np.sqrt(np.mean(delta_speed * delta_speed))),
            "mean_radial_velocity_before_dpd": float(np.mean(radial_before)),
            "mean_radial_velocity_after_dpd": float(np.mean(radial_after)),
            "rms_radial_velocity_before_dpd": float(np.sqrt(np.mean(radial_before * radial_before))),
            "rms_radial_velocity_after_dpd": float(np.sqrt(np.mean(radial_after * radial_after))),
            "rms_tangential_velocity_after_dpd": float(np.sqrt(np.mean(np.sum(tangential_after * tangential_after, axis=1)))),
        }
        if shape_payload is not None:
            payload["velocity_shape_json"] = None if velocity_shape_json is None else str(velocity_shape_json)
            payload["velocity_shape_schema"] = shape_payload.get("schema")
            payload["velocity_shape_source"] = (
                shape_payload.get("source_result_json")
                if velocity_shape_json is not None
                else "generated_from_current_post_equilibration_mesh"
            )
            payload["velocity_shape_profile"] = shape_payload.get("profile")
            payload["velocity_shape_normalization"] = shape_payload
    payload = comm.bcast(payload, root=access_rank)
    comm.Barrier()
    if payload is None:
        raise RuntimeError("Radial velocity kick failed to broadcast payload.")
    return payload


def _apply_shape_displacement_release(
    emb: Any,
    *,
    shape_displacement_amplitude_dpd: float,
    shape_displacement_projection: str,
    velocity_shape_json: Path | None,
    comm: Any,
) -> dict[str, Any]:
    from mpi4py import MPI

    amplitude = float(shape_displacement_amplitude_dpd)
    if abs(amplitude) <= 1.0e-14:
        return {
            "status": "not_requested",
            "shape_displacement_amplitude_dpd": amplitude,
        }
    if velocity_shape_json is None:
        raise RuntimeError("shape-displacement-release requires --velocity-shape-json.")
    if shape_displacement_projection not in {"full", "radial"}:
        raise ValueError("shape_displacement_projection must be 'full' or 'radial'.")

    rank = comm.Get_rank()
    can_access = hasattr(emb, "getCoordinates") and hasattr(emb, "setCoordinates")
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError("Shape displacement release requires membrane coordinate access.")
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    payload: dict[str, Any] | None = None
    if rank == access_rank:
        coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.size == 0:
            raise RuntimeError(
                f"Membrane coordinate array has invalid shape for shape displacement: {coordinates.shape}."
            )
        center_before = coordinates.mean(axis=0)
        offsets = coordinates - center_before[None, :]
        radii_before = np.linalg.norm(offsets, axis=1)
        directions = np.zeros_like(offsets)
        nonzero = radii_before > 1.0e-12
        directions[nonzero] = offsets[nonzero] / radii_before[nonzero, None]

        normalized_shape, shape_payload = _load_velocity_shape(
            velocity_shape_json,
            expected_shape=coordinates.shape,
        )
        if shape_displacement_projection == "radial":
            radial_shape = np.einsum("ij,ij->i", normalized_shape, directions)
            normalized_shape = radial_shape[:, None] * directions
            projected_rms = float(np.sqrt(np.mean(np.sum(normalized_shape * normalized_shape, axis=1))))
            if projected_rms <= 0.0:
                raise RuntimeError(f"Radial-projected shape has zero RMS norm: {velocity_shape_json}")
            normalized_shape = normalized_shape / projected_rms
        displacement = amplitude * normalized_shape
        radial_delta = np.einsum("ij,ij->i", displacement, directions)
        tangential_delta = displacement - radial_delta[:, None] * directions
        displaced = coordinates + displacement
        emb.setCoordinates(displaced.tolist())

        center_after = displaced.mean(axis=0)
        radii_after = np.linalg.norm(displaced - center_after[None, :], axis=1)
        displacement_norm = np.sqrt(np.sum(displacement * displacement, axis=1))
        tangential_norm = np.sqrt(np.sum(tangential_delta * tangential_delta, axis=1))
        payload = {
            "status": "applied",
            "coordinate_rank": int(access_rank),
            "shape_displacement_amplitude_dpd": amplitude,
            "shape_displacement_projection": shape_displacement_projection,
            "shape_json": str(velocity_shape_json),
            "shape_schema": shape_payload.get("schema"),
            "shape_source": shape_payload.get("source_result_json"),
            "shape_profile": shape_payload.get("profile"),
            "shape_normalization": shape_payload,
            "center_before_dpd": [float(item) for item in center_before],
            "center_after_dpd": [float(item) for item in center_after],
            "rms_radius_before_dpd": float(np.sqrt(np.mean(radii_before * radii_before))),
            "rms_radius_after_dpd": float(np.sqrt(np.mean(radii_after * radii_after))),
            "mean_radius_before_dpd": float(np.mean(radii_before)),
            "mean_radius_after_dpd": float(np.mean(radii_after)),
            "min_radial_displacement_dpd": float(np.min(radial_delta)),
            "max_radial_displacement_dpd": float(np.max(radial_delta)),
            "mean_radial_displacement_dpd": float(np.mean(radial_delta)),
            "rms_radial_displacement_dpd": float(np.sqrt(np.mean(radial_delta * radial_delta))),
            "rms_tangential_displacement_dpd": float(np.sqrt(np.mean(tangential_norm * tangential_norm))),
            "rms_total_displacement_dpd": float(np.sqrt(np.mean(displacement_norm * displacement_norm))),
        }
    payload = comm.bcast(payload, root=access_rank)
    comm.Barrier()
    if payload is None:
        raise RuntimeError("Shape displacement release failed to broadcast payload.")
    return payload


def _particle_vector_center_radius(pv: Any, *, vector_name: str, comm: Any) -> dict[str, Any]:
    from mpi4py import MPI

    rank = comm.Get_rank()
    can_access = hasattr(pv, "getCoordinates")
    local_count = 0
    local_sum = np.zeros(3, dtype=np.float64)
    if can_access:
        coordinates = np.asarray(pv.getCoordinates(), dtype=np.float64)
        if coordinates.ndim == 2 and coordinates.shape[1] == 3 and coordinates.size > 0:
            local_count = int(coordinates.shape[0])
            local_sum = coordinates.sum(axis=0)
    total_count = int(comm.allreduce(local_count, op=MPI.SUM))
    total_sum = np.asarray(comm.allreduce(local_sum, op=MPI.SUM), dtype=np.float64)
    if total_count <= 0:
        raise RuntimeError(f"{vector_name} coordinate access returned no particles.")
    center = total_sum / float(total_count)

    local_r2_sum = 0.0
    local_r_sum = 0.0
    if can_access:
        coordinates = np.asarray(pv.getCoordinates(), dtype=np.float64)
        if coordinates.ndim == 2 and coordinates.shape[1] == 3 and coordinates.size > 0:
            radii = np.linalg.norm(coordinates - center[None, :], axis=1)
            local_r2_sum = float(np.sum(radii * radii))
            local_r_sum = float(np.sum(radii))
    total_r2_sum = float(comm.allreduce(local_r2_sum, op=MPI.SUM))
    total_r_sum = float(comm.allreduce(local_r_sum, op=MPI.SUM))
    return {
        "status": "ok",
        "vector_name": vector_name,
        "coordinate_rank_count": int(comm.allreduce(1 if can_access else 0, op=MPI.SUM)),
        "particle_count": total_count,
        "center_dpd": [float(item) for item in center],
        "mean_radius_dpd": total_r_sum / float(total_count),
        "rms_radius_dpd": math.sqrt(total_r2_sum / float(total_count)),
        "rank": int(rank),
    }


def _membrane_radius_restraint_forces(
    emb: Any,
    *,
    target_radius_dpd: float,
    force_constant: float,
    max_force_per_vertex: float,
    comm: Any,
) -> tuple[list[list[float]], dict[str, Any]]:
    from mpi4py import MPI

    rank = comm.Get_rank()
    can_access = hasattr(emb, "getCoordinates")
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError("Radius restraint requires membrane coordinate access.")
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    forces: list[list[float]] | None = None
    payload: dict[str, Any] | None = None
    if rank == access_rank:
        coordinates = np.asarray(emb.getCoordinates(), dtype=np.float64)
        center = coordinates.mean(axis=0)
        delta = coordinates - center[None, :]
        radii = np.linalg.norm(delta, axis=1)
        directions = np.zeros_like(delta)
        nonzero = radii > 1.0e-12
        directions[nonzero] = delta[nonzero] / radii[nonzero, None]
        force_magnitude = float(force_constant) * (float(target_radius_dpd) - radii)
        max_force = float(max_force_per_vertex)
        if max_force > 0.0:
            force_magnitude = np.clip(force_magnitude, -max_force, max_force)
        force_array = directions * force_magnitude[:, None]
        forces = force_array.tolist()
        payload = {
            "status": "computed",
            "coordinate_rank": int(access_rank),
            "target_radius_dpd": float(target_radius_dpd),
            "mean_radius_dpd": float(np.mean(radii)),
            "rms_radius_dpd": float(np.sqrt(np.mean(radii * radii))),
            "mean_radius_error_dpd": float(float(target_radius_dpd) - np.mean(radii)),
            "rms_radius_error_dpd": float(float(target_radius_dpd) - np.sqrt(np.mean(radii * radii))),
            "max_abs_force_per_vertex": float(np.max(np.abs(force_magnitude))) if force_magnitude.size else 0.0,
            "mean_abs_force_per_vertex": float(np.mean(np.abs(force_magnitude))) if force_magnitude.size else 0.0,
        }
    forces = comm.bcast(forces, root=access_rank)
    payload = comm.bcast(payload, root=access_rank)
    comm.Barrier()
    if forces is None or payload is None:
        raise RuntimeError("Radius restraint force broadcast failed.")
    return forces, payload


def _run_radius_restraint(
    u: Any,
    mir: Any,
    emb: Any,
    *,
    dt: float,
    hold_steps: int,
    update_every_steps: int,
    target_scale: float,
    force_constant: float,
    max_force_per_vertex: float,
    comm: Any,
) -> dict[str, Any]:
    if int(hold_steps) <= 0:
        return {
            "status": "not_requested",
            "hold_steps": int(hold_steps),
            "target_scale": float(target_scale),
        }
    if float(force_constant) <= 0.0:
        raise ValueError("--radius-restraint-force-constant must be positive.")
    start_geometry = _particle_vector_center_radius(emb, vector_name="emb", comm=comm)
    target_radius = float(start_geometry["rms_radius_dpd"]) * float(target_scale)
    update_every = max(1, int(update_every_steps))
    remaining = int(hold_steps)
    elapsed_steps = 0
    chunk_index = 0
    chunks: list[dict[str, Any]] = []
    while remaining > 0:
        chunk_steps = min(update_every, remaining)
        forces, force_payload = _membrane_radius_restraint_forces(
            emb,
            target_radius_dpd=target_radius,
            force_constant=float(force_constant),
            max_force_per_vertex=float(max_force_per_vertex),
            comm=comm,
        )
        plugin = mir.Plugins.createMembraneExtraForce(f"radiusRestraint{chunk_index}", emb, forces)
        u.registerPlugins(plugin)
        u.run(int(chunk_steps), dt=float(dt))
        u.deregisterPlugins(plugin)
        force_payload.update(
            {
                "chunk_index": int(chunk_index),
                "start_step_after_equil": int(elapsed_steps),
                "end_step_after_equil": int(elapsed_steps + chunk_steps),
                "start_time_after_equil_dpd": float(elapsed_steps * float(dt)),
                "end_time_after_equil_dpd": float((elapsed_steps + chunk_steps) * float(dt)),
            }
        )
        chunks.append(force_payload)
        elapsed_steps += chunk_steps
        remaining -= chunk_steps
        chunk_index += 1
    final_geometry = _particle_vector_center_radius(emb, vector_name="emb", comm=comm)
    max_abs_force = max((float(item["max_abs_force_per_vertex"]) for item in chunks), default=0.0)
    return {
        "status": "applied_then_released",
        "target_scale": float(target_scale),
        "target_radius_dpd": float(target_radius),
        "force_constant": float(force_constant),
        "max_force_per_vertex": float(max_force_per_vertex),
        "hold_steps": int(hold_steps),
        "hold_duration_dpd": float(int(hold_steps) * float(dt)),
        "update_every_steps": int(update_every),
        "chunk_count": int(chunk_index),
        "max_abs_force_per_vertex": float(max_abs_force),
        "start_geometry": start_geometry,
        "final_geometry_before_release": final_geometry,
        "fit_start_recommendation_dpd": float(int(hold_steps) * float(dt)),
        "chunks": chunks,
    }


def _minimum_image_delta(delta: np.ndarray, lbox: tuple[float, float, float]) -> np.ndarray:
    wrapped = np.asarray(delta, dtype=np.float64).copy()
    box = np.asarray(lbox, dtype=np.float64)
    valid = np.isfinite(box) & (box > 0.0)
    wrapped[:, valid] -= box[valid] * np.round(wrapped[:, valid] / box[valid])
    return wrapped


def _radial_velocity_profile_weights(
    distances: np.ndarray,
    *,
    radius: float,
    profile: str,
    outer_radius_factor: float,
) -> np.ndarray:
    r = np.asarray(distances, dtype=np.float64)
    weights = np.zeros_like(r)
    if radius <= 0.0:
        return weights
    positive = r > 1.0e-12
    if profile == "outside-potential":
        safe_r = np.maximum(r, float(radius))
        weights[positive] = (float(radius) / safe_r[positive]) ** 2
        if outer_radius_factor > 1.0:
            cutoff = float(outer_radius_factor) * float(radius)
            taper_start = 0.8 * cutoff
            beyond = r >= cutoff
            taper = (r > taper_start) & (r < cutoff)
            weights[beyond] = 0.0
            if cutoff > taper_start:
                x = (r[taper] - taper_start) / (cutoff - taper_start)
                weights[taper] *= 0.5 * (1.0 + np.cos(math.pi * x))
    elif profile == "inside-linear":
        weights[positive] = np.minimum(r[positive] / float(radius), 1.0)
    else:
        raise ValueError(f"Unknown radial velocity profile: {profile}")
    return weights


def _make_radial_force_field(
    *,
    center_dpd: Iterable[float],
    radius_dpd: float,
    lbox: tuple[float, float, float],
    force_per_particle: float,
    profile: str,
    outer_radius_factor: float,
) -> Any:
    cx, cy, cz = [float(item) for item in center_dpd]
    lx, ly, lz = [float(item) for item in lbox]
    radius = float(radius_dpd)
    force = float(force_per_particle)
    cutoff = float(outer_radius_factor) * radius if outer_radius_factor > 1.0 else math.inf
    taper_start = 0.8 * cutoff if math.isfinite(cutoff) else math.inf

    def wrapped_delta(value: float, center: float, box: float) -> float:
        delta = float(value) - center
        if math.isfinite(box) and box > 0.0:
            delta -= box * round(delta / box)
        return delta

    def weight(distance: float) -> float:
        if radius <= 0.0 or distance <= 1.0e-12:
            return 0.0
        if profile == "outside-potential":
            safe_r = max(distance, radius)
            out = (radius / safe_r) ** 2
            if math.isfinite(cutoff):
                if distance >= cutoff:
                    return 0.0
                if cutoff > taper_start and distance > taper_start:
                    x = (distance - taper_start) / (cutoff - taper_start)
                    out *= 0.5 * (1.0 + math.cos(math.pi * x))
            return out
        if profile == "inside-linear":
            return min(distance / radius, 1.0)
        raise ValueError(f"Unknown radial force profile: {profile}")

    def force_field(r: Any) -> tuple[float, float, float]:
        dx = wrapped_delta(r.x, cx, lx)
        dy = wrapped_delta(r.y, cy, ly)
        dz = wrapped_delta(r.z, cz, lz)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        w = weight(distance)
        if w <= 0.0:
            return (0.0, 0.0, 0.0)
        scale = force * w / distance
        return (scale * dx, scale * dy, scale * dz)

    return force_field


def _apply_radial_velocity_profile_kick(
    pv: Any,
    *,
    vector_name: str,
    center_dpd: Iterable[float],
    radius_dpd: float,
    lbox: tuple[float, float, float],
    radial_velocity_kick: float,
    scale: float,
    profile: str,
    radial_velocity_kick_mode: str,
    outer_radius_factor: float,
    comm: Any,
) -> dict[str, Any]:
    from mpi4py import MPI

    kick = float(radial_velocity_kick) * float(scale)
    if pv is None:
        return {"status": "not_available", "vector_name": vector_name}
    if abs(kick) <= 1.0e-14:
        return {
            "status": "not_requested",
            "vector_name": vector_name,
            "radial_velocity_kick_dpd": kick,
            "scale": float(scale),
            "profile": profile,
        }
    kick_action, _ = _radial_velocity_kick_mode_parts(radial_velocity_kick_mode)

    can_access = (
        hasattr(pv, "getCoordinates")
        and hasattr(pv, "getVelocities")
        and hasattr(pv, "setVelocities")
    )
    local = np.zeros(8, dtype=np.float64)
    if can_access:
        coordinates = np.asarray(pv.getCoordinates(), dtype=np.float64)
        velocities = np.asarray(pv.getVelocities(), dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise RuntimeError(f"{vector_name} coordinates must be an Nx3 array.")
        if velocities.shape != coordinates.shape:
            raise RuntimeError(
                f"{vector_name} velocity shape {velocities.shape} does not match coordinate shape {coordinates.shape}."
            )
        if coordinates.size > 0:
            center = np.asarray(list(center_dpd), dtype=np.float64)
            delta = _minimum_image_delta(coordinates - center[None, :], lbox)
            distances = np.linalg.norm(delta, axis=1)
            nonzero = distances > 1.0e-12
            directions = np.zeros_like(delta)
            directions[nonzero] = delta[nonzero] / distances[nonzero, None]
            weights = _radial_velocity_profile_weights(
                distances,
                radius=float(radius_dpd),
                profile=profile,
                outer_radius_factor=float(outer_radius_factor),
            )
            active = nonzero & (weights > 0.0)
            kick_vectors = kick * weights[:, None] * directions
            radial_before = np.einsum("ij,ij->i", velocities, directions)
            if kick_action == "replace":
                kicked = velocities - radial_before[:, None] * directions + kick_vectors
            else:
                kicked = velocities + kick_vectors
            pv.setVelocities(kicked.tolist())
            radial_after = np.einsum("ij,ij->i", kicked, directions)
            local[0] = float(coordinates.shape[0])
            local[1] = float(np.count_nonzero(active))
            local[2] = float(np.sum(radial_before[active])) if np.any(active) else 0.0
            local[3] = float(np.sum(radial_after[active])) if np.any(active) else 0.0
            local[4] = float(np.sum(radial_before[active] ** 2)) if np.any(active) else 0.0
            local[5] = float(np.sum(radial_after[active] ** 2)) if np.any(active) else 0.0
            local[6] = float(np.max(np.abs(kick * weights[active]))) if np.any(active) else 0.0
            local[7] = 1.0
        else:
            local[7] = 1.0
    total = np.asarray(comm.allreduce(local, op=MPI.SUM), dtype=np.float64)
    active_count = int(total[1])
    max_kick = float(comm.allreduce(float(local[6]), op=MPI.MAX))
    payload = {
        "status": "applied" if active_count > 0 else "no_active_particles",
        "vector_name": vector_name,
        "profile": profile,
        "scale": float(scale),
        "radial_velocity_kick_dpd": float(kick),
        "base_radial_velocity_kick_dpd": float(radial_velocity_kick),
        "radial_velocity_kick_mode": radial_velocity_kick_mode,
        "outer_radius_factor": float(outer_radius_factor),
        "particle_count": int(total[0]),
        "active_particle_count": active_count,
        "access_rank_count": int(total[7]),
        "max_abs_applied_radial_velocity_dpd": max_kick,
    }
    if active_count > 0:
        payload.update(
            {
                "mean_radial_velocity_before_dpd": float(total[2] / total[1]),
                "mean_radial_velocity_after_dpd": float(total[3] / total[1]),
                "rms_radial_velocity_before_dpd": float(math.sqrt(total[4] / total[1])),
                "rms_radial_velocity_after_dpd": float(math.sqrt(total[5] / total[1])),
            }
        )
    comm.Barrier()
    return payload


def _fluid_contact_vector_stats(
    pv: Any,
    *,
    vector_name: str,
    center_dpd: Iterable[float],
    radius_dpd: float,
    lbox: tuple[float, float, float],
    margin_dpd: float,
    particle_mass: float,
    comm: Any,
) -> dict[str, Any]:
    from mpi4py import MPI

    if pv is None:
        return {"status": "not_available", "vector_name": vector_name}

    radius = float(radius_dpd)
    margin = max(0.0, float(margin_dpd))
    inner_radius = max(0.0, radius - margin)
    outer_radius = radius + margin
    center = np.asarray(list(center_dpd), dtype=np.float64)
    can_access = hasattr(pv, "getCoordinates")
    has_velocities = can_access and hasattr(pv, "getVelocities")
    local = np.zeros(14, dtype=np.float64)
    local_min_radius = math.inf
    local_max_radius = -math.inf
    if can_access:
        coordinates = np.asarray(pv.getCoordinates(), dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise RuntimeError(f"{vector_name} coordinates must be an Nx3 array.")
        local[13] = 1.0
        count = int(coordinates.shape[0])
        local[0] = float(count)
        if count > 0:
            delta = _minimum_image_delta(coordinates - center[None, :], lbox)
            distances = np.linalg.norm(delta, axis=1)
            local[1] = float(np.count_nonzero(distances < radius))
            local[2] = float(np.count_nonzero(distances < inner_radius))
            local[3] = float(np.count_nonzero(distances > radius))
            local[4] = float(np.count_nonzero(distances > outer_radius))
            local[5] = float(np.count_nonzero(np.abs(distances - radius) <= margin))
            local[6] = float(np.sum(distances))
            local[7] = float(np.sum(distances * distances))
            local_min_radius = float(np.min(distances))
            local_max_radius = float(np.max(distances))
            if has_velocities:
                velocities = np.asarray(pv.getVelocities(), dtype=np.float64)
                if velocities.shape != coordinates.shape:
                    raise RuntimeError(
                        f"{vector_name} velocity shape {velocities.shape} does not match "
                        f"coordinate shape {coordinates.shape}."
                    )
                nonzero = distances > 1.0e-12
                directions = np.zeros_like(delta)
                directions[nonzero] = delta[nonzero] / distances[nonzero, None]
                radial_velocity = np.einsum("ij,ij->i", velocities, directions)
                speed2 = np.einsum("ij,ij->i", velocities, velocities)
                local[8] = float(np.sum(radial_velocity))
                local[9] = float(np.sum(radial_velocity * radial_velocity))
                local[10] = float(np.sum(speed2))
                local[11] = float(np.sum(float(particle_mass) * radial_velocity))
                local[12] = 1.0

    total = np.asarray(comm.allreduce(local, op=MPI.SUM), dtype=np.float64)
    min_radius = float(comm.allreduce(local_min_radius, op=MPI.MIN))
    max_radius = float(comm.allreduce(local_max_radius, op=MPI.MAX))
    count = int(total[0])
    payload: dict[str, Any] = {
        "status": "ok" if count > 0 else "empty",
        "vector_name": vector_name,
        "particle_count": count,
        "coordinate_rank_count": int(total[13]),
        "radius_reference_dpd": radius,
        "margin_dpd": margin,
        "inside_radius_count": int(total[1]),
        "inside_radius_fraction": float(total[1] / total[0]) if count > 0 else 0.0,
        "inside_radius_minus_margin_count": int(total[2]),
        "inside_radius_minus_margin_fraction": float(total[2] / total[0]) if count > 0 else 0.0,
        "outside_radius_count": int(total[3]),
        "outside_radius_fraction": float(total[3] / total[0]) if count > 0 else 0.0,
        "outside_radius_plus_margin_count": int(total[4]),
        "outside_radius_plus_margin_fraction": float(total[4] / total[0]) if count > 0 else 0.0,
        "near_shell_band_count": int(total[5]),
        "near_shell_band_fraction": float(total[5] / total[0]) if count > 0 else 0.0,
        "mean_distance_from_center_dpd": float(total[6] / total[0]) if count > 0 else 0.0,
        "rms_distance_from_center_dpd": float(math.sqrt(total[7] / total[0])) if count > 0 else 0.0,
        "min_distance_from_center_dpd": min_radius if count > 0 and math.isfinite(min_radius) else 0.0,
        "max_distance_from_center_dpd": max_radius if count > 0 and math.isfinite(max_radius) else 0.0,
        "velocity_rank_count": int(total[12]),
    }
    if count > 0 and int(total[12]) > 0:
        payload.update(
            {
                "mean_radial_velocity_dpd": float(total[8] / total[0]),
                "rms_radial_velocity_dpd": float(math.sqrt(total[9] / total[0])),
                "rms_speed_dpd": float(math.sqrt(total[10] / total[0])),
                "sum_radial_momentum_dpd": float(total[11]),
            }
        )
    return payload


def _flatten_fluid_contact_audit_sample(sample: dict[str, Any]) -> dict[str, Any]:
    water = sample.get("water") or {}
    gas = sample.get("gas") or {}
    membrane = sample.get("membrane") or {}
    return {
        "sample_index": sample["sample_index"],
        "elapsed_steps_after_release": sample["elapsed_steps_after_release"],
        "time_after_release_dpd": sample["time_after_release_dpd"],
        "membrane_mean_radius_dpd": membrane.get("mean_radius_dpd", ""),
        "membrane_rms_radius_dpd": membrane.get("rms_radius_dpd", ""),
        "water_particle_count": water.get("particle_count", ""),
        "water_inside_radius_count": water.get("inside_radius_count", ""),
        "water_inside_radius_fraction": water.get("inside_radius_fraction", ""),
        "water_inside_radius_minus_margin_count": water.get("inside_radius_minus_margin_count", ""),
        "water_inside_radius_minus_margin_fraction": water.get("inside_radius_minus_margin_fraction", ""),
        "water_near_shell_band_fraction": water.get("near_shell_band_fraction", ""),
        "water_mean_radial_velocity_dpd": water.get("mean_radial_velocity_dpd", ""),
        "water_sum_radial_momentum_dpd": water.get("sum_radial_momentum_dpd", ""),
        "gas_particle_count": gas.get("particle_count", ""),
        "gas_outside_radius_count": gas.get("outside_radius_count", ""),
        "gas_outside_radius_fraction": gas.get("outside_radius_fraction", ""),
        "gas_outside_radius_plus_margin_count": gas.get("outside_radius_plus_margin_count", ""),
        "gas_outside_radius_plus_margin_fraction": gas.get("outside_radius_plus_margin_fraction", ""),
        "gas_near_shell_band_fraction": gas.get("near_shell_band_fraction", ""),
        "gas_mean_radial_velocity_dpd": gas.get("mean_radial_velocity_dpd", ""),
        "gas_sum_radial_momentum_dpd": gas.get("sum_radial_momentum_dpd", ""),
    }


def _fluid_contact_audit_sample(
    *,
    emb: Any,
    water: Any,
    gas: Any,
    lbox: tuple[float, float, float],
    water_mass: float,
    gas_mass: float,
    margin_dpd: float,
    elapsed_steps_after_release: int,
    dt: float,
    sample_index: int,
    comm: Any,
) -> dict[str, Any]:
    membrane = _particle_vector_center_radius(emb, vector_name="emb", comm=comm)
    center = membrane["center_dpd"]
    radius = float(membrane["rms_radius_dpd"])
    return {
        "schema": "mesouq.emb_breathing_fluid_contact_audit_sample.v1",
        "sample_index": int(sample_index),
        "elapsed_steps_after_release": int(elapsed_steps_after_release),
        "time_after_release_dpd": float(int(elapsed_steps_after_release) * float(dt)),
        "margin_dpd": float(margin_dpd),
        "membrane": membrane,
        "water": _fluid_contact_vector_stats(
            water,
            vector_name="water",
            center_dpd=center,
            radius_dpd=radius,
            lbox=lbox,
            margin_dpd=margin_dpd,
            particle_mass=float(water_mass),
            comm=comm,
        ),
        "gas": _fluid_contact_vector_stats(
            gas,
            vector_name="gas",
            center_dpd=center,
            radius_dpd=radius,
            lbox=lbox,
            margin_dpd=margin_dpd,
            particle_mass=float(gas_mass),
            comm=comm,
        ),
    }


def _write_fluid_contact_audit_outputs(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    audit_every_steps: int,
    margin_dpd: float,
) -> None:
    csv_path = output_dir / "fluid_contact_audit_timeseries.csv"
    json_path = output_dir / "fluid_contact_audit_summary.json"
    rows = [_flatten_fluid_contact_audit_sample(sample) for sample in samples]
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _max_nested(path: tuple[str, str]) -> float:
        values = []
        for sample in samples:
            section = sample.get(path[0]) or {}
            value = section.get(path[1])
            if value is not None:
                values.append(float(value))
        return max(values) if values else 0.0

    summary = {
        "schema": "mesouq.emb_breathing_fluid_contact_audit_summary.v1",
        "status": "written" if samples else "empty",
        "audit_every_steps": int(audit_every_steps),
        "margin_dpd": float(margin_dpd),
        "sample_count": len(samples),
        "timeseries_csv": str(csv_path) if rows else "",
        "max_water_inside_radius_fraction": _max_nested(("water", "inside_radius_fraction")),
        "max_water_inside_radius_minus_margin_fraction": _max_nested(
            ("water", "inside_radius_minus_margin_fraction")
        ),
        "max_gas_outside_radius_fraction": _max_nested(("gas", "outside_radius_fraction")),
        "max_gas_outside_radius_plus_margin_fraction": _max_nested(
            ("gas", "outside_radius_plus_margin_fraction")
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run_relaxation_with_optional_fluid_contact_audit(
    u: Any,
    *,
    emb: Any,
    water: Any,
    gas: Any,
    lbox: tuple[float, float, float],
    water_mass: float,
    gas_mass: float,
    dt: float,
    relax_steps: int,
    audit_every_steps: int,
    audit_margin_dpd: float,
    output_dir: Path,
    comm: Any,
) -> None:
    if int(relax_steps) <= 0:
        return
    if int(audit_every_steps) <= 0 or (water is None and gas is None):
        u.run(int(relax_steps), dt=float(dt))
        return

    rank = comm.Get_rank()
    cadence = max(1, int(audit_every_steps))
    remaining = int(relax_steps)
    elapsed = 0
    sample_index = 0
    samples: list[dict[str, Any]] = []

    sample = _fluid_contact_audit_sample(
        emb=emb,
        water=water,
        gas=gas,
        lbox=lbox,
        water_mass=water_mass,
        gas_mass=gas_mass,
        margin_dpd=float(audit_margin_dpd),
        elapsed_steps_after_release=elapsed,
        dt=float(dt),
        sample_index=sample_index,
        comm=comm,
    )
    if rank == 0:
        samples.append(sample)
    sample_index += 1
    while remaining > 0:
        chunk_steps = min(cadence, remaining)
        u.run(int(chunk_steps), dt=float(dt))
        elapsed += chunk_steps
        remaining -= chunk_steps
        sample = _fluid_contact_audit_sample(
            emb=emb,
            water=water,
            gas=gas,
            lbox=lbox,
            water_mass=water_mass,
            gas_mass=gas_mass,
            margin_dpd=float(audit_margin_dpd),
            elapsed_steps_after_release=elapsed,
            dt=float(dt),
            sample_index=sample_index,
            comm=comm,
        )
        if rank == 0:
            samples.append(sample)
        sample_index += 1
    if rank == 0:
        _write_fluid_contact_audit_outputs(
            output_dir=output_dir,
            samples=samples,
            audit_every_steps=cadence,
            margin_dpd=float(audit_margin_dpd),
        )


def _run_relaxation_with_memory_capture(
    u: Any,
    *,
    emb: Any,
    dt: float,
    relax_steps: int,
    sample_every: int,
    comm: Any,
) -> CapturedTrajectory | None:
    """Sample release coordinates without Mirheo's parallel HDF5 particle dumper.

    The campaign needs only membrane geometry.  This avoids a Vega OpenMPI/HDF5
    failure seen after a long constrained pre-release hold while preserving the
    same release sampling cadence used by the particle-dump path.
    """
    from mpi4py import MPI

    rank = comm.Get_rank()
    can_access = hasattr(emb, "getCoordinates")
    access_count = comm.allreduce(1 if can_access else 0, op=MPI.SUM)
    if access_count <= 0:
        raise RuntimeError("Memory trajectory capture requires membrane coordinate access.")
    access_rank = comm.allreduce(rank if can_access else 1_000_000_000, op=MPI.MIN)
    if access_rank != 0:
        raise RuntimeError(
            "Memory trajectory capture requires rank 0 membrane coordinate access so rank 0 can postprocess."
        )

    frames: list[np.ndarray] | None = [] if rank == 0 else None
    steps_after_release: list[int] | None = [] if rank == 0 else None

    def capture(step: int) -> None:
        capture_error: str | None = None
        if rank == 0:
            try:
                assert frames is not None
                assert steps_after_release is not None
                positions = np.asarray(emb.getCoordinates(), dtype=np.float64)
                if positions.ndim != 2 or positions.shape[1] != 3 or positions.size == 0:
                    raise RuntimeError("Memory trajectory capture received invalid membrane coordinates.")
                if not np.all(np.isfinite(positions)):
                    raise RuntimeError("Memory trajectory capture received non-finite membrane coordinates.")
                frames.append(positions.copy())
                steps_after_release.append(int(step))
            except Exception as exc:
                capture_error = f"{type(exc).__name__}: {exc}"
        capture_error = comm.bcast(capture_error, root=0)
        if capture_error is not None:
            raise RuntimeError(capture_error)

    elapsed = 0
    capture(elapsed)
    remaining = int(relax_steps)
    cadence = max(1, int(sample_every))
    while remaining > 0:
        chunk_steps = min(cadence, remaining)
        u.run(int(chunk_steps), dt=float(dt))
        elapsed += chunk_steps
        remaining -= chunk_steps
        capture(elapsed)
    if rank != 0:
        return None
    assert frames is not None
    assert steps_after_release is not None
    return CapturedTrajectory(frames=frames, steps_after_release=steps_after_release)


def run_simulation(
    bubble: Bubble,
    sim_dir: Path,
    *,
    dt: float,
    equil_steps: int,
    pulse_steps: int,
    relax_steps: int,
    sample_every: int,
    trajectory_capture: str,
    particle_dump_dir: Path,
    pulse_force_per_vertex: float,
    excitation_mode: str,
    initial_radius_scale: float,
    radial_velocity_kick: float,
    radial_velocity_kick_mode: str,
    velocity_shape_json: Path | None,
    shape_displacement_amplitude_dpd: float,
    shape_displacement_projection: str,
    drive_frequency_mhz: float,
    drive_force_per_vertex: float,
    drive_update_every_steps: int,
    tone_burst_cycles: float,
    tone_burst_ramp_cycles: float,
    radius_restraint_target_scale: float,
    radius_restraint_force_constant: float,
    radius_restraint_max_force_per_vertex: float,
    radius_restraint_hold_steps: int,
    radius_restraint_update_every_steps: int,
    fluid_kick_water_scale: float,
    fluid_kick_gas_scale: float,
    fluid_kick_water_outer_radius_factor: float,
    fluid_burst_water_force_per_particle: float,
    fluid_burst_gas_force_per_particle: float,
    fluid_burst_force_field_grid_spacing: float,
    fluid_burst_water_outer_radius_factor: float,
    stress_free_radius_scale: float,
    prestrain_placement: str,
    post_deflation_ramp_steps: int,
    post_deflation_hold_steps: int,
    post_deflation_hold_update_every_steps: int,
    post_deflation_hold_reset_velocities: bool,
    solvent_mode: str,
    water_shell_fsi_scale: float,
    water_shell_gamma_scale: float,
    bouncer_mode: str,
    water_belonging_correct_every: int,
    gas_belonging_correct_every: int,
    membrane_mass_scale: float,
    pin_com: bool,
    checkpoint_every: int,
    restart_from_checkpoint: Path | None,
    particle_checker_every: int,
    dump_fluid_particles: bool,
    fluid_dump_every_steps: int,
    fluid_audit_every_steps: int,
    fluid_audit_margin_dpd: float,
    dump_fluid_density_grid: bool,
    fluid_density_sample_every_steps: int,
    fluid_density_dump_every_steps: int,
    fluid_density_bin_size_dpd: float,
    comm: Any,
) -> CapturedTrajectory | None:
    import mirheo as mir
    import trimesh
    from mpi4py import MPI

    rank = comm.Get_rank()
    _prepare_mirheo_log_root(sim_dir.parent)

    default_path = sim_dir / "parameter" / "parameters-default00001.yaml"
    params_path = sim_dir / "parameter" / "parameters00001.yaml"
    prms_path = sim_dir / "parameter" / "parameters.prms00001.yaml"
    parameters_default = _load_yaml(default_path)
    parameters = _load_yaml(params_path)
    prms_emb = _load_yaml(prms_path)

    obj_type = str(parameters_default["objFile"])[:-4]
    mesh_path = sim_dir / "mesh" / f"{obj_type}00001.off"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    original_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    initial_vertices, stress_free_vertices, post_equilibration_scale = _prestrain_geometry(
        original_vertices,
        initial_radius_scale=initial_radius_scale,
        stress_free_radius_scale=stress_free_radius_scale,
        prestrain_placement=prestrain_placement,
    )
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangle = initial_vertices[faces]
    edges = np.linalg.norm(triangle[:, 1] - triangle[:, 0], axis=1)
    lj_fac = 0.8 * float(np.min(edges))

    lbox = (
        float(parameters_default["Lx"]),
        float(parameters_default["Ly"]),
        float(parameters_default["Lz"]),
    )
    pos_q = np.reshape(np.loadtxt(sim_dir / "posq.txt"), (-1, 7))
    kbt = float(parameters["kbt"])
    aii = float(parameters_default["aii"]) * kbt
    if bubble.modality == "compression":
        bouncer_bins = (500, 150)
        set_membrane_lj = False
    else:
        bouncer_bins = (1000, 150)
        set_membrane_lj = True
    afsi = float(water_shell_fsi_scale) * aii
    water_shell_gamma = float(parameters["gamma_fsi"]) * float(water_shell_gamma_scale)
    if bouncer_mode not in {"on", "off"}:
        raise ValueError("bouncer_mode must be 'on' or 'off'.")
    water_belonging_correct_every = max(0, int(water_belonging_correct_every))
    gas_belonging_correct_every = max(0, int(gas_belonging_correct_every))
    original_mvert = float(parameters.get("mvert_unscaled", parameters["mvert"]))
    scaled_mvert = float(parameters["mvert"])
    recorded_mass_scale = float(parameters.get("membrane_mass_scale", membrane_mass_scale))
    if rank == 0:
        fsi_payload = {
            "water_shell_fsi_conservative_scale": float(water_shell_fsi_scale),
            "aii": float(aii),
            "afsi": float(afsi),
            "kbt": float(kbt),
            "parameters_default_aii": float(parameters_default["aii"]),
            "water_shell_gamma_scale": float(water_shell_gamma_scale),
            "water_shell_gamma": float(water_shell_gamma),
            "base_water_shell_gamma": float(parameters["gamma_fsi"]),
            "modality": bubble.modality,
            "symbol": bubble.symbol,
            "bouncer_bins": list(bouncer_bins),
            "bouncer_mode": bouncer_mode,
            "set_membrane_lj": bool(set_membrane_lj),
            "gas_shell_fsi_conservative_a": 0.0,
            "water_belonging_correct_every": int(water_belonging_correct_every),
            "gas_belonging_correct_every": int(gas_belonging_correct_every),
            "belonging_correction_policy": (
                "diagnostic_periodic_object_belonging_correction"
                if water_belonging_correct_every > 0 or gas_belonging_correct_every > 0
                else "initial_split_only_original_protocol"
            ),
        }
        (sim_dir.parent / "fsi_parameters.json").write_text(
            json.dumps(fsi_payload, indent=2), encoding="utf-8"
        )
        mass_payload = {
            "schema": "mesouq.emb_breathing_membrane_mass.v1",
            "symbol": bubble.symbol,
            "agent": bubble.agent,
            "modality": bubble.modality,
            "policy": str(parameters.get("membrane_mass_policy", "unknown_generated_mass_policy")),
            "mvert": scaled_mvert,
            "mvert_unscaled": original_mvert,
            "mvert_raw_generated": float(parameters.get("mvert_raw_generated", parameters["mvert"])),
            "membrane_mass_scale": recorded_mass_scale,
            "mw": float(parameters["mw"]),
            "mg": float(parameters["mg"]),
            "rho_shell": float(parameters_default["rho_shell"]),
            "shell_th": float(parameters_default["th_fac"]) * float(parameters_default["shell_th"]),
            "tot_area": float(parameters["tot_area"]),
            "nverts": int(parameters["nverts"]),
            "indentation_x5_multiplier_removed": bool(bubble.modality == "indentation"),
            "compression_had_no_x5_multiplier": bool(bubble.modality == "compression"),
        }
        (sim_dir.parent / "membrane_mass_parameters.json").write_text(
            json.dumps(mass_payload, indent=2), encoding="utf-8"
        )

    prms_emb["bpress"] = float(parameters_default["bpress"])
    mirheo_kwargs: dict[str, Any] = {
        "nranks": (1, 1, 1),
        "domain": lbox,
        "debug_level": 0,
        "log_filename": str(sim_dir / "logs" / "mirheo"),
        "no_splash": True,
        "comm_ptr": MPI._addressof(comm),
    }
    if int(checkpoint_every) > 0:
        mirheo_kwargs.update(
            {
                "checkpoint_folder": str(sim_dir / "restart") + "/",
                "checkpoint_every": int(checkpoint_every),
            }
        )
    u = mir.Mirheo(**mirheo_kwargs)

    mesh_emb = mir.ParticleVectors.MembraneMesh(
        vertices=initial_vertices.tolist(),
        stress_free_vertices=stress_free_vertices.tolist(),
        faces=faces.tolist(),
    )
    emb = mir.ParticleVectors.MembraneVector("emb", mass=scaled_mvert, mesh=mesh_emb)
    u.registerParticleVector(emb, mir.InitialConditions.Membrane(pos_q))

    water = None
    gas = None
    use_water = solvent_mode in {"full", "water-only"}
    use_gas = solvent_mode in {"full", "gas-only"}
    if use_water:
        water = mir.ParticleVectors.ParticleVector("water", mass=float(parameters["mw"]))
        u.registerParticleVector(
            water, mir.InitialConditions.Uniform(number_density=float(parameters_default["rhow"]))
        )
        water_checker = mir.BelongingCheckers.Mesh("inner_solvent_checker_2")
        u.registerObjectBelongingChecker(water_checker, emb)
        u.applyObjectBelongingChecker(
            water_checker,
            water,
            correct_every=int(water_belonging_correct_every),
            inside="none",
            outside="",
        )
    if use_gas:
        gas_source = mir.ParticleVectors.ParticleVector("sol2", mass=float(parameters["mg"]))
        u.registerParticleVector(
            gas_source, mir.InitialConditions.Uniform(number_density=float(parameters_default["rhog"]))
        )
        gas_checker = mir.BelongingCheckers.Mesh("inner_checker_1")
        u.registerObjectBelongingChecker(gas_checker, emb)
        gas = u.applyObjectBelongingChecker(
            gas_checker,
            gas_source,
            correct_every=int(gas_belonging_correct_every),
            inside="gas",
            outside="",
        )

    vv = mir.Integrators.VelocityVerlet("vv")
    u.registerIntegrator(vv)
    particle_vectors = [emb]
    if water is not None:
        particle_vectors.append(water)
    if gas is not None:
        particle_vectors.append(gas)
    for pv in particle_vectors:
        u.setIntegrator(vv, pv)

    rc = float(parameters_default["rc"])
    int_emb = mir.Interactions.MembraneForces(
        "int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free=True
    )
    dpd = mir.Interactions.Pairwise(
        "dpd",
        rc,
        kind="DPD",
        a=0.0,
        gamma=0.0,
        kBT=kbt,
        power=float(parameters_default["s"]),
    )
    dpd_wat = mir.Interactions.Pairwise(
        "dpd_wat",
        rc,
        kind="DPD",
        a=aii,
        gamma=float(parameters_default["gamma_dpd"]),
        kBT=kbt,
        power=float(parameters_default["s"]),
    )
    dpd_gas = mir.Interactions.Pairwise(
        "dpd_gas",
        rc,
        kind="DPD",
        a=0.0,
        gamma=float(parameters_default["gamma_dpd_gas"]),
        kBT=kbt,
        power=float(parameters_default["s_g"]),
    )
    dpd_fsi = mir.Interactions.Pairwise(
        "dpd_fsi",
        rc,
        kind="DPD",
        a=afsi,
        gamma=water_shell_gamma,
        kBT=kbt,
        power=float(parameters_default["k_fsi"]),
    )
    dpd_fsi_gas = mir.Interactions.Pairwise(
        "dpd_fsi_gas",
        rc,
        kind="DPD",
        a=0.0,
        gamma=float(parameters["gamma_fsi_gas"]),
        kBT=kbt,
        power=float(parameters_default["k_fsi"]),
    )
    lj = mir.Interactions.Pairwise(
        "lj",
        rc,
        kind="RepulsiveLJ",
        epsilon=0.1,
        sigma=rc / (2 ** (1 / 6)),
        max_force=10.0,
        aware_mode="Object",
    )
    lj_int = mir.Interactions.Pairwise(
        "lj_int",
        lj_fac,
        kind="RepulsiveLJ",
        epsilon=10000.0,
        sigma=0.99 * lj_fac / (2 ** (1 / 6)),
        max_force=10000.0,
    )

    for interaction in (int_emb, dpd, dpd_wat, dpd_gas, dpd_fsi, dpd_fsi_gas, lj, lj_int):
        u.registerInteraction(interaction)
    u.setInteraction(int_emb, emb, emb)
    if set_membrane_lj:
        u.setInteraction(lj, emb, emb)
    u.setInteraction(lj_int, emb, emb)
    if water is not None:
        u.setInteraction(dpd_wat, water, water)
        u.setInteraction(dpd_fsi, emb, water)
    if gas is not None:
        u.setInteraction(dpd_gas, gas, gas)
        u.setInteraction(dpd_fsi_gas, emb, gas)
    if water is not None and gas is not None:
        u.setInteraction(dpd, water, gas)
    if (water is not None or gas is not None) and bouncer_mode == "on":
        bouncer = mir.Bouncers.Mesh("membrane_bounce", bouncer_bins[0], bouncer_bins[1], "bounce_maxwell", kBT=kbt)
        u.registerBouncer(bouncer)
        if water is not None:
            u.setBouncer(bouncer, emb, water)
        if gas is not None:
            u.setBouncer(bouncer, emb, gas)

    if rank == 0:
        print(
            f"[emb_breathing] {bubble.symbol}: equil_steps={equil_steps} "
            f"pulse_steps={pulse_steps} relax_steps={relax_steps} dt={dt} "
            f"sample_every={sample_every} pulse_force_per_vertex={pulse_force_per_vertex} "
            f"excitation_mode={excitation_mode} initial_radius_scale={initial_radius_scale} "
            f"prestrain_placement={prestrain_placement} "
            f"radial_velocity_kick={radial_velocity_kick} radial_velocity_kick_mode={radial_velocity_kick_mode} "
            f"velocity_shape_json={velocity_shape_json or ''} "
            f"shape_displacement_amplitude_dpd={shape_displacement_amplitude_dpd} "
            f"shape_displacement_projection={shape_displacement_projection} "
            f"drive_frequency_mhz={drive_frequency_mhz} drive_force_per_vertex={drive_force_per_vertex} "
            f"tone_burst_cycles={tone_burst_cycles} tone_burst_ramp_cycles={tone_burst_ramp_cycles} "
            f"radius_restraint_target_scale={radius_restraint_target_scale} "
            f"radius_restraint_force_constant={radius_restraint_force_constant} "
            f"radius_restraint_max_force_per_vertex={radius_restraint_max_force_per_vertex} "
            f"radius_restraint_hold_steps={radius_restraint_hold_steps} "
            f"radius_restraint_update_every_steps={radius_restraint_update_every_steps} "
            f"fluid_kick_water_scale={fluid_kick_water_scale} fluid_kick_gas_scale={fluid_kick_gas_scale} "
            f"fluid_kick_water_outer_radius_factor={fluid_kick_water_outer_radius_factor} "
            f"fluid_burst_water_force_per_particle={fluid_burst_water_force_per_particle} "
            f"fluid_burst_gas_force_per_particle={fluid_burst_gas_force_per_particle} "
            f"fluid_burst_force_field_grid_spacing={fluid_burst_force_field_grid_spacing} "
            f"fluid_burst_water_outer_radius_factor={fluid_burst_water_outer_radius_factor} "
            f"stress_free_radius_scale={stress_free_radius_scale} solvent_mode={solvent_mode} "
            f"post_deflation_ramp_steps={post_deflation_ramp_steps} "
            f"post_deflation_hold_steps={post_deflation_hold_steps} "
            f"post_deflation_hold_update_every_steps={post_deflation_hold_update_every_steps} "
            f"post_deflation_hold_reset_velocities={post_deflation_hold_reset_velocities} "
            f"water_shell_fsi_scale={water_shell_fsi_scale} water_shell_gamma_scale={water_shell_gamma_scale} "
            f"bouncer_mode={bouncer_mode} "
            f"water_belonging_correct_every={water_belonging_correct_every} "
            f"gas_belonging_correct_every={gas_belonging_correct_every} "
            f"membrane_mass_scale={recorded_mass_scale} "
            f"restart_from_checkpoint={restart_from_checkpoint or ''} "
            f"pin_com={pin_com} particle_checker_every={particle_checker_every} "
            f"dump_fluid_particles={dump_fluid_particles} fluid_dump_every_steps={fluid_dump_every_steps} "
            f"fluid_audit_every_steps={fluid_audit_every_steps} "
            f"fluid_audit_margin_dpd={fluid_audit_margin_dpd} "
            f"dump_fluid_density_grid={dump_fluid_density_grid} "
            f"fluid_density_sample_every_steps={fluid_density_sample_every_steps} "
            f"fluid_density_dump_every_steps={fluid_density_dump_every_steps} "
            f"fluid_density_bin_size_dpd={fluid_density_bin_size_dpd}",
            flush=True,
        )

    if pin_com:
        pin_path = sim_dir / "pin"
        if rank == 0:
            pin_path.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        unrestricted = mir.Plugins.PinObject.Unrestricted
        u.registerPlugins(
            mir.Plugins.createPinObject(
                "pin",
                emb,
                int(sample_every),
                str(pin_path) + "/",
                [0.0, 0.0, 0.0],
                [unrestricted, unrestricted, unrestricted],
            )
        )

    checker = None
    if int(particle_checker_every) > 0 and restart_from_checkpoint is None:
        # Debug-only Mirheo guard: catches NaN/invalid particle positions or velocities.
        checker = mir.Plugins.createParticleChecker(
            name="debugParticleChecker",
            check_every=int(particle_checker_every),
        )
        u.registerPlugins(checker)

    if restart_from_checkpoint is not None:
        restart_path = Path(restart_from_checkpoint).resolve()
        restart_exists = restart_path.exists() if rank == 0 else None
        restart_exists = comm.bcast(restart_exists, root=0)
        if not restart_exists:
            raise FileNotFoundError(f"Restart checkpoint folder does not exist: {restart_path}")
        u.restart(str(restart_path) + "/")
        if rank == 0:
            restart_payload = {
                "schema": "mesouq.emb_breathing_restart_measurement.v1",
                "status": "restarted",
                "restart_from_checkpoint": str(restart_path),
                "measurement_water_shell_fsi_scale": float(water_shell_fsi_scale),
                "measurement_water_shell_gamma_scale": float(water_shell_gamma_scale),
                "equil_steps_after_restart": int(equil_steps),
                "diagnostic_purpose": (
                    "measurement-window interaction switch; the checkpoint is expected "
                    "to contain an already equilibrated state"
                ),
            }
            (sim_dir.parent / "restart_measurement.json").write_text(
                json.dumps(restart_payload, indent=2), encoding="utf-8"
            )
        if int(particle_checker_every) > 0:
            checker = mir.Plugins.createParticleChecker(
                name="debugParticleChecker",
                check_every=int(particle_checker_every),
            )
            u.registerPlugins(checker)

    if equil_steps > 0:
        u.run(int(equil_steps), dt=float(dt))
    if prestrain_placement == "initial-geometry":
        prestrain_payload = _initial_geometry_prestrain_payload(
            original_vertices,
            initial_vertices,
            stress_free_vertices,
            initial_radius_scale=initial_radius_scale,
            stress_free_radius_scale=stress_free_radius_scale,
        )
    else:
        prestrain_payload = _run_post_equilibration_prestrain_ramp(
            u,
            emb,
            dt=float(dt),
            initial_radius_scale=post_equilibration_scale,
            ramp_steps=int(post_deflation_ramp_steps),
            reset_velocities=bool(post_deflation_hold_reset_velocities),
            comm=comm,
        )
    if rank == 0:
        prestrain_path = sim_dir.parent / "post_equilibration_prestrain.json"
        prestrain_path.write_text(json.dumps(prestrain_payload, indent=2), encoding="utf-8")
    post_deflation_hold_payload = _run_post_deflation_coordinate_hold(
        u,
        emb,
        dt=float(dt),
        hold_steps=int(post_deflation_hold_steps),
        update_every_steps=int(post_deflation_hold_update_every_steps),
        reset_velocities=bool(post_deflation_hold_reset_velocities),
        comm=comm,
    )
    if rank == 0:
        hold_path = sim_dir.parent / "post_deflation_coordinate_hold.json"
        hold_path.write_text(json.dumps(post_deflation_hold_payload, indent=2), encoding="utf-8")
    if excitation_mode in {"velocity-kick", "fluid-velocity-kick"} and abs(float(radial_velocity_kick)) <= 1.0e-14:
        raise ValueError(
            "--radial-velocity-kick must be nonzero when --excitation-mode velocity-kick "
            "or fluid-velocity-kick."
        )
    if excitation_mode == "shape-displacement-release":
        if abs(float(shape_displacement_amplitude_dpd)) <= 1.0e-14:
            raise ValueError(
                "--shape-displacement-amplitude-dpd must be nonzero when "
                "--excitation-mode shape-displacement-release."
            )
        if velocity_shape_json is None:
            raise ValueError("--excitation-mode shape-displacement-release requires --velocity-shape-json.")
    shape_displacement_payload = _apply_shape_displacement_release(
        emb,
        shape_displacement_amplitude_dpd=(
            shape_displacement_amplitude_dpd
            if excitation_mode == "shape-displacement-release"
            else 0.0
        ),
        shape_displacement_projection=shape_displacement_projection,
        velocity_shape_json=velocity_shape_json,
        comm=comm,
    )
    if rank == 0:
        shape_displacement_path = sim_dir.parent / "post_equilibration_shape_displacement.json"
        shape_displacement_path.write_text(
            json.dumps(shape_displacement_payload, indent=2),
            encoding="utf-8",
        )
    velocity_kick_payload = _apply_radial_velocity_kick(
        emb,
        radial_velocity_kick=radial_velocity_kick
        if excitation_mode in {"velocity-kick", "fluid-velocity-kick"}
        else 0.0,
        radial_velocity_kick_mode=radial_velocity_kick_mode,
        velocity_shape_json=velocity_shape_json,
        faces=faces,
        comm=comm,
    )
    if rank == 0:
        velocity_kick_path = sim_dir.parent / "post_equilibration_velocity_kick.json"
        velocity_kick_path.write_text(json.dumps(velocity_kick_payload, indent=2), encoding="utf-8")

    if excitation_mode == "fluid-velocity-kick":
        membrane_geometry = _particle_vector_center_radius(emb, vector_name="emb", comm=comm)
        center = membrane_geometry["center_dpd"]
        radius = float(membrane_geometry["rms_radius_dpd"])
        water_kick_payload = _apply_radial_velocity_profile_kick(
            water,
            vector_name="water",
            center_dpd=center,
            radius_dpd=radius,
            lbox=lbox,
            radial_velocity_kick=radial_velocity_kick,
            scale=fluid_kick_water_scale,
            profile="outside-potential",
            radial_velocity_kick_mode=radial_velocity_kick_mode,
            outer_radius_factor=fluid_kick_water_outer_radius_factor,
            comm=comm,
        )
        gas_kick_payload = _apply_radial_velocity_profile_kick(
            gas,
            vector_name="gas",
            center_dpd=center,
            radius_dpd=radius,
            lbox=lbox,
            radial_velocity_kick=radial_velocity_kick,
            scale=fluid_kick_gas_scale,
            profile="inside-linear",
            radial_velocity_kick_mode=radial_velocity_kick_mode,
            outer_radius_factor=0.0,
            comm=comm,
        )
        if rank == 0:
            fluid_kick_path = sim_dir.parent / "post_equilibration_fluid_velocity_kick.json"
            fluid_kick_path.write_text(
                json.dumps(
                    {
                        "status": "applied",
                        "membrane_geometry": membrane_geometry,
                        "water": water_kick_payload,
                        "gas": gas_kick_payload,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    if trajectory_capture not in {"particle-dump", "memory"}:
        raise ValueError(f"Unsupported trajectory capture mode: {trajectory_capture}")
    if trajectory_capture == "memory" and (
        excitation_mode not in {"prestrain", "none"}
        or int(pulse_steps) != 0
        or int(fluid_audit_every_steps) != 0
        or bool(dump_fluid_particles)
        or bool(dump_fluid_density_grid)
    ):
        raise ValueError(
            "Memory trajectory capture supports only free prestrain/none release without fluid dumps or contact audit."
        )
    if trajectory_capture == "particle-dump":
        dump = mir.Plugins.createDumpParticles(
            name="dumpParticles",
            pv=emb,
            dump_every=int(sample_every),
            channel_names=["positions"],
            path=str(particle_dump_dir / "emb"),
        )
        u.registerPlugins(dump)
    if dump_fluid_particles:
        fluid_dump_every = int(fluid_dump_every_steps) if int(fluid_dump_every_steps) > 0 else int(sample_every)
        if water is not None:
            u.registerPlugins(
                mir.Plugins.createDumpParticles(
                    name="dumpWaterParticles",
                    pv=water,
                    dump_every=fluid_dump_every,
                    channel_names=["positions"],
                    path=str(sim_dir / "particles" / "water"),
                )
            )
        if gas is not None:
            u.registerPlugins(
                mir.Plugins.createDumpParticles(
                    name="dumpGasParticles",
                    pv=gas,
                    dump_every=fluid_dump_every,
                    channel_names=["positions"],
                    path=str(sim_dir / "particles" / "gas"),
                )
            )
    if dump_fluid_density_grid and (water is not None or gas is not None):
        density_sample_every = (
            int(fluid_density_sample_every_steps)
            if int(fluid_density_sample_every_steps) > 0
            else int(sample_every)
        )
        density_dump_every = (
            int(fluid_density_dump_every_steps)
            if int(fluid_density_dump_every_steps) > 0
            else int(relax_steps)
        )
        density_bin_size = max(float(fluid_density_bin_size_dpd), 1.0e-6)
        density_pvs = [pv for pv in (water, gas) if pv is not None]
        density_path = sim_dir / "fluid_density" / "fluid-"
        if rank == 0:
            density_path.parent.mkdir(parents=True, exist_ok=True)
            density_payload = {
                "schema": "mesouq.emb_breathing_fluid_density_grid.v1",
                "status": "enabled",
                "path_prefix": str(density_path),
                "sample_every_steps": int(density_sample_every),
                "dump_every_steps": int(density_dump_every),
                "bin_size_dpd": float(density_bin_size),
                "particle_vectors": [
                    name
                    for name, pv in (("water", water), ("gas", gas))
                    if pv is not None
                ],
                "relative_to_object_vector": "emb",
                "relative_to_object_id": 0,
                "channels": [],
                "note": (
                    "Diagnostic density-grid dump. It writes binned number_densities, "
                    "not full water/gas particle trajectories."
                ),
            }
            (sim_dir.parent / "fluid_density_grid_parameters.json").write_text(
                json.dumps(density_payload, indent=2), encoding="utf-8"
            )
        comm.Barrier()
        u.registerPlugins(
            mir.Plugins.createDumpAverageRelative(
                "fluidDensityGrid",
                density_pvs,
                emb,
                0,
                int(density_sample_every),
                int(density_dump_every),
                (density_bin_size, density_bin_size, density_bin_size),
                [],
                str(density_path),
            )
        )

    directions = _radial_unit_directions(initial_vertices)
    forces = (directions * float(pulse_force_per_vertex)).tolist()
    pulse = None
    if excitation_mode in {"sinusoidal-drive", "normal-sinusoidal-drive", "current-radial-sinusoidal-drive"}:
        drive_frequency_dpd = _mhz_to_dpd_frequency(drive_frequency_mhz)
        if drive_frequency_dpd <= 0.0:
            raise ValueError("--drive-frequency-mhz must be positive for sinusoidal-drive modes.")
        if float(drive_force_per_vertex) == 0.0:
            raise ValueError("--drive-force-per-vertex must be nonzero for sinusoidal-drive modes.")
        update_every = max(1, int(drive_update_every_steps))
        remaining = int(relax_steps)
        elapsed_steps = 0
        chunk_index = 0
        normal_force_chunks: list[dict[str, Any]] = []
        radial_force_chunks: list[dict[str, Any]] = []
        max_abs_force = 0.0
        while remaining > 0:
            chunk_steps = min(update_every, remaining)
            mid_time = (elapsed_steps + 0.5 * chunk_steps) * float(dt)
            scale = float(drive_force_per_vertex) * math.sin(2.0 * math.pi * drive_frequency_dpd * mid_time)
            max_abs_force = max(max_abs_force, abs(scale))
            if excitation_mode == "normal-sinusoidal-drive":
                normal_payload = _current_area_weighted_normal_forces(
                    emb,
                    faces,
                    force_per_vertex=scale,
                    comm=comm,
                )
                normal_chunk = _normal_force_payload_for_sidecar(normal_payload)
                normal_chunk.update(
                    {
                        "chunk_index": int(chunk_index),
                        "start_step_after_equil": int(elapsed_steps),
                        "end_step_after_equil": int(elapsed_steps + chunk_steps),
                        "start_time_after_equil_dpd": float(elapsed_steps * float(dt)),
                        "end_time_after_equil_dpd": float((elapsed_steps + chunk_steps) * float(dt)),
                    }
                )
                normal_force_chunks.append(normal_chunk)
                drive_forces = normal_payload["forces"]
            elif excitation_mode == "current-radial-sinusoidal-drive":
                radial_payload = _current_radial_forces(
                    emb,
                    force_per_vertex=scale,
                    comm=comm,
                )
                radial_chunk = _normal_force_payload_for_sidecar(radial_payload)
                radial_chunk.update(
                    {
                        "chunk_index": int(chunk_index),
                        "start_step_after_equil": int(elapsed_steps),
                        "end_step_after_equil": int(elapsed_steps + chunk_steps),
                        "start_time_after_equil_dpd": float(elapsed_steps * float(dt)),
                        "end_time_after_equil_dpd": float((elapsed_steps + chunk_steps) * float(dt)),
                    }
                )
                radial_force_chunks.append(radial_chunk)
                drive_forces = radial_payload["forces"]
            else:
                drive_forces = (directions * scale).tolist()
            drive = mir.Plugins.createMembraneExtraForce(f"radialDrive{chunk_index}", emb, drive_forces)
            u.registerPlugins(drive)
            u.run(int(chunk_steps), dt=float(dt))
            u.deregisterPlugins(drive)
            elapsed_steps += chunk_steps
            remaining -= chunk_steps
            chunk_index += 1
        if rank == 0:
            drive_payload = {
                "status": "applied",
                "excitation_mode": excitation_mode,
                "drive_frequency_mhz": float(drive_frequency_mhz),
                "drive_frequency_dpd": float(drive_frequency_dpd),
                "drive_period_dpd": float(1.0 / drive_frequency_dpd),
                "drive_force_per_vertex": float(drive_force_per_vertex),
                "drive_update_every_steps": update_every,
                "drive_chunk_count": int(chunk_index),
                "relax_steps": int(relax_steps),
                "max_abs_force_per_vertex": float(max_abs_force),
                "current_normal_force": {
                    "status": "applied" if excitation_mode == "normal-sinusoidal-drive" else "not_requested",
                    "chunk_count_with_force": int(len(normal_force_chunks)),
                    "chunks": normal_force_chunks,
                },
                "current_radial_force": {
                    "status": "applied" if excitation_mode == "current-radial-sinusoidal-drive" else "not_requested",
                    "chunk_count_with_force": int(len(radial_force_chunks)),
                    "chunks": radial_force_chunks,
                },
            }
            if excitation_mode == "normal-sinusoidal-drive":
                drive_path = sim_dir.parent / "post_equilibration_normal_sinusoidal_drive.json"
            elif excitation_mode == "current-radial-sinusoidal-drive":
                drive_path = sim_dir.parent / "post_equilibration_current_radial_sinusoidal_drive.json"
            else:
                drive_path = sim_dir.parent / "post_equilibration_sinusoidal_drive.json"
            drive_path.write_text(json.dumps(drive_payload, indent=2), encoding="utf-8")
    elif excitation_mode == "radius-restraint-release":
        restraint_payload = _run_radius_restraint(
            u,
            mir,
            emb,
            dt=float(dt),
            hold_steps=int(radius_restraint_hold_steps),
            update_every_steps=int(radius_restraint_update_every_steps),
            target_scale=float(radius_restraint_target_scale),
            force_constant=float(radius_restraint_force_constant),
            max_force_per_vertex=float(radius_restraint_max_force_per_vertex),
            comm=comm,
        )
        if rank == 0:
            restraint_path = sim_dir.parent / "post_equilibration_radius_restraint_release.json"
            restraint_path.write_text(json.dumps(restraint_payload, indent=2), encoding="utf-8")
    elif excitation_mode in {"tone-burst", "fluid-force-burst", "normal-tone-burst"}:
        drive_frequency_dpd = _mhz_to_dpd_frequency(drive_frequency_mhz)
        if drive_frequency_dpd <= 0.0:
            raise ValueError(
                "--drive-frequency-mhz must be positive for tone-burst/fluid-force-burst/normal-tone-burst."
            )
        if excitation_mode in {"tone-burst", "normal-tone-burst"} and float(drive_force_per_vertex) == 0.0:
            raise ValueError("--drive-force-per-vertex must be nonzero for tone-burst/normal-tone-burst.")
        if excitation_mode == "fluid-force-burst" and (
            abs(float(drive_force_per_vertex)) <= 1.0e-14
            and abs(float(fluid_burst_water_force_per_particle)) <= 1.0e-14
            and abs(float(fluid_burst_gas_force_per_particle)) <= 1.0e-14
        ):
            raise ValueError(
                "fluid-force-burst requires a nonzero membrane, water, or gas burst force."
            )
        if float(tone_burst_cycles) <= 0.0:
            raise ValueError(
                "--tone-burst-cycles must be positive for tone-burst/fluid-force-burst/normal-tone-burst."
            )
        update_every = max(1, int(drive_update_every_steps))
        burst_duration_dpd = float(tone_burst_cycles) / drive_frequency_dpd
        ramp_duration_dpd = max(0.0, float(tone_burst_ramp_cycles)) / drive_frequency_dpd
        burst_steps = max(1, int(math.ceil(burst_duration_dpd / float(dt))))
        burst_grid_spacing = max(float(fluid_burst_force_field_grid_spacing), 1.0e-6)
        field_h = (burst_grid_spacing, burst_grid_spacing, burst_grid_spacing)
        fluid_force_payload: dict[str, Any] = {"status": "not_requested"}
        membrane_geometry: dict[str, Any] | None = None
        if excitation_mode == "fluid-force-burst":
            membrane_geometry = _particle_vector_center_radius(emb, vector_name="emb", comm=comm)
            fluid_force_payload = {
                "status": "configured",
                "membrane_geometry": membrane_geometry,
                "water_force_per_particle": float(fluid_burst_water_force_per_particle),
                "gas_force_per_particle": float(fluid_burst_gas_force_per_particle),
                "force_field_grid_spacing": float(burst_grid_spacing),
                "water_outer_radius_factor": float(fluid_burst_water_outer_radius_factor),
                "water_profile": "outside-potential",
                "gas_profile": "inside-linear",
                "water_active": bool(
                    water is not None and abs(float(fluid_burst_water_force_per_particle)) > 1.0e-14
                ),
                "gas_active": bool(
                    gas is not None and abs(float(fluid_burst_gas_force_per_particle)) > 1.0e-14
                ),
            }
        remaining = burst_steps
        elapsed_steps = 0
        chunk_index = 0
        max_abs_force = 0.0
        max_abs_water_force = 0.0
        max_abs_gas_force = 0.0
        normal_force_chunks: list[dict[str, Any]] = []
        while remaining > 0:
            chunk_steps = min(update_every, remaining)
            mid_time = (elapsed_steps + 0.5 * chunk_steps) * float(dt)
            scale = _tone_burst_force_scale(
                time_dpd=mid_time,
                frequency_dpd=drive_frequency_dpd,
                force_per_vertex=drive_force_per_vertex,
                duration_dpd=burst_duration_dpd,
                ramp_dpd=ramp_duration_dpd,
            )
            max_abs_force = max(max_abs_force, abs(scale))
            plugins: list[Any] = []
            if abs(scale) > 0.0:
                if excitation_mode == "normal-tone-burst":
                    normal_payload = _current_area_weighted_normal_forces(
                        emb,
                        faces,
                        force_per_vertex=scale,
                        comm=comm,
                    )
                    normal_chunk = _normal_force_payload_for_sidecar(normal_payload)
                    normal_chunk.update(
                        {
                            "chunk_index": int(chunk_index),
                            "start_step_after_equil": int(elapsed_steps),
                            "end_step_after_equil": int(elapsed_steps + chunk_steps),
                            "start_time_after_equil_dpd": float(elapsed_steps * float(dt)),
                            "end_time_after_equil_dpd": float((elapsed_steps + chunk_steps) * float(dt)),
                        }
                    )
                    normal_force_chunks.append(normal_chunk)
                    plugins.append(
                        mir.Plugins.createMembraneExtraForce(
                            f"normalToneBurst{chunk_index}", emb, normal_payload["forces"]
                        )
                    )
                else:
                    burst_forces = (directions * scale).tolist()
                    plugins.append(
                        mir.Plugins.createMembraneExtraForce(f"radialToneBurst{chunk_index}", emb, burst_forces)
                    )
            if excitation_mode == "fluid-force-burst" and membrane_geometry is not None:
                center = membrane_geometry["center_dpd"]
                radius = float(membrane_geometry["rms_radius_dpd"])
                water_force = _tone_burst_force_scale(
                    time_dpd=mid_time,
                    frequency_dpd=drive_frequency_dpd,
                    force_per_vertex=float(fluid_burst_water_force_per_particle),
                    duration_dpd=burst_duration_dpd,
                    ramp_dpd=ramp_duration_dpd,
                )
                gas_force = _tone_burst_force_scale(
                    time_dpd=mid_time,
                    frequency_dpd=drive_frequency_dpd,
                    force_per_vertex=float(fluid_burst_gas_force_per_particle),
                    duration_dpd=burst_duration_dpd,
                    ramp_dpd=ramp_duration_dpd,
                )
                max_abs_water_force = max(max_abs_water_force, abs(water_force))
                max_abs_gas_force = max(max_abs_gas_force, abs(gas_force))
                if water is not None and abs(water_force) > 0.0:
                    water_field = _make_radial_force_field(
                        center_dpd=center,
                        radius_dpd=radius,
                        lbox=lbox,
                        force_per_particle=water_force,
                        profile="outside-potential",
                        outer_radius_factor=float(fluid_burst_water_outer_radius_factor),
                    )
                    plugins.append(
                        mir.Plugins.createAddForceField(
                            f"waterRadialToneBurst{chunk_index}", water, water_field, h=field_h
                        )
                    )
                if gas is not None and abs(gas_force) > 0.0:
                    gas_field = _make_radial_force_field(
                        center_dpd=center,
                        radius_dpd=radius,
                        lbox=lbox,
                        force_per_particle=gas_force,
                        profile="inside-linear",
                        outer_radius_factor=0.0,
                    )
                    plugins.append(
                        mir.Plugins.createAddForceField(
                            f"gasRadialToneBurst{chunk_index}", gas, gas_field, h=field_h
                        )
                    )
            for plugin in plugins:
                u.registerPlugins(plugin)
            u.run(int(chunk_steps), dt=float(dt))
            for plugin in reversed(plugins):
                u.deregisterPlugins(plugin)
            elapsed_steps += chunk_steps
            remaining -= chunk_steps
            chunk_index += 1
        if rank == 0:
            drive_payload = {
                "status": "applied_then_released",
                "excitation_mode": excitation_mode,
                "drive_frequency_mhz": float(drive_frequency_mhz),
                "drive_frequency_dpd": float(drive_frequency_dpd),
                "drive_period_dpd": float(1.0 / drive_frequency_dpd),
                "drive_force_per_vertex": float(drive_force_per_vertex),
                "drive_update_every_steps": update_every,
                "tone_burst_cycles": float(tone_burst_cycles),
                "tone_burst_ramp_cycles": float(tone_burst_ramp_cycles),
                "burst_duration_dpd": float(burst_duration_dpd),
                "burst_steps": int(burst_steps),
                "burst_chunk_count": int(chunk_index),
                "post_burst_relax_steps": int(relax_steps),
                "max_abs_force_per_vertex": float(max_abs_force),
                "fluid_force_burst": fluid_force_payload,
                "max_abs_water_force_per_particle": float(max_abs_water_force),
                "max_abs_gas_force_per_particle": float(max_abs_gas_force),
                "current_normal_force": {
                    "status": "applied" if excitation_mode == "normal-tone-burst" else "not_requested",
                    "chunk_count_with_force": int(len(normal_force_chunks)),
                    "chunks": normal_force_chunks,
                },
            }
            if excitation_mode == "fluid-force-burst":
                drive_path = sim_dir.parent / "post_equilibration_fluid_force_burst.json"
            elif excitation_mode == "normal-tone-burst":
                drive_path = sim_dir.parent / "post_equilibration_normal_tone_burst.json"
            else:
                drive_path = sim_dir.parent / "post_equilibration_tone_burst.json"
            drive_path.write_text(json.dumps(drive_payload, indent=2), encoding="utf-8")
    elif excitation_mode == "normal-force-pulse" and pulse_steps > 0:
        normal_payload = _current_area_weighted_normal_forces(
            emb,
            faces,
            force_per_vertex=float(pulse_force_per_vertex),
            comm=comm,
        )
        pulse = mir.Plugins.createMembraneExtraForce("normalPulse", emb, normal_payload["forces"])
        u.registerPlugins(pulse)
        u.run(int(pulse_steps), dt=float(dt))
        u.deregisterPlugins(pulse)
        if rank == 0:
            pulse_payload = _normal_force_payload_for_sidecar(normal_payload)
            pulse_payload.update(
                {
                    "status": "applied_then_released",
                    "excitation_mode": excitation_mode,
                    "pulse_force_per_vertex": float(pulse_force_per_vertex),
                    "pulse_steps": int(pulse_steps),
                    "pulse_duration_dpd": float(int(pulse_steps) * float(dt)),
                    "post_pulse_relax_steps": int(relax_steps),
                }
            )
            pulse_path = sim_dir.parent / "post_equilibration_normal_force_pulse.json"
            pulse_path.write_text(json.dumps(pulse_payload, indent=2), encoding="utf-8")
    elif excitation_mode == "force-pulse" and pulse_steps > 0:
        pulse = mir.Plugins.createMembraneExtraForce("radialPulse", emb, forces)
        u.registerPlugins(pulse)
        u.run(int(pulse_steps), dt=float(dt))
        u.deregisterPlugins(pulse)
    elif pulse_steps > 0:
        u.run(int(pulse_steps), dt=float(dt))
    sinusoidal_drive_modes = {
        "sinusoidal-drive",
        "normal-sinusoidal-drive",
        "current-radial-sinusoidal-drive",
    }
    captured_trajectory: CapturedTrajectory | None = None
    if relax_steps > 0 and excitation_mode not in sinusoidal_drive_modes:
        if trajectory_capture == "memory":
            captured_trajectory = _run_relaxation_with_memory_capture(
                u,
                emb=emb,
                dt=float(dt),
                relax_steps=int(relax_steps),
                sample_every=int(sample_every),
                comm=comm,
            )
        else:
            _run_relaxation_with_optional_fluid_contact_audit(
                u,
                emb=emb,
                water=water,
                gas=gas,
                lbox=lbox,
                water_mass=float(parameters["mw"]),
                gas_mass=float(parameters["mg"]),
                dt=float(dt),
                relax_steps=int(relax_steps),
                audit_every_steps=int(fluid_audit_every_steps),
                audit_margin_dpd=float(fluid_audit_margin_dpd),
                output_dir=sim_dir.parent,
                comm=comm,
            )
    del u
    return captured_trajectory


def _read_positions(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["position"][:], dtype=np.float64)


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    return float(abs(signed))


def _kabsch_align_to_reference(vertices: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if vertices.shape != reference.shape:
        raise ValueError(
            f"frame/reference shape mismatch: frame={vertices.shape}, reference={reference.shape}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("frame contains non-finite vertex coordinates")
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference contains non-finite vertex coordinates")
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    ref_centered = reference - reference.mean(axis=0, keepdims=True)
    covariance = centered.T @ ref_centered
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Kabsch covariance contains non-finite values")
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return centered @ rotation


def _quadrupole_basis(directions: np.ndarray) -> np.ndarray:
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]
    return np.column_stack(
        [
            x * x - 1.0 / 3.0,
            y * y - 1.0 / 3.0,
            x * y,
            x * z,
            y * z,
        ]
    )


def _mode_purity_failure_summary(
    status: str,
    *,
    rows: list[dict[str, Any]],
    requested_frame_count: int,
    failed_dump_index: int | None = None,
    error: BaseException | str | None = None,
) -> dict[str, Any]:
    if rows:
        summary = summarize_mode_purity_rows(rows)
    else:
        summary = {"frame_count": 0}
    summary.update(
        {
            "status": status,
            "usable_frame_count": len(rows),
            "requested_frame_count": int(requested_frame_count),
        }
    )
    if failed_dump_index is not None:
        summary["failed_dump_index"] = int(failed_dump_index)
    if error is not None:
        summary["error"] = str(error)
        summary["error_type"] = type(error).__name__ if isinstance(error, BaseException) else "error"
    return summary


def compute_mode_purity_timeseries(
    frames: list[np.ndarray],
    *,
    time_dpd: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(frames) < 2:
        return [], {"status": "insufficient_frames", "frame_count": len(frames)}
    reference = np.asarray(frames[0], dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 3:
        return [], {
            "status": "invalid_reference_shape",
            "frame_count": len(frames),
            "reference_shape": list(reference.shape),
        }
    if not np.all(np.isfinite(reference)):
        return [], {"status": "nonfinite_reference", "frame_count": len(frames)}
    ref_centered = reference - reference.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(ref_centered, axis=1)
    if np.any(radii <= 0.0):
        return [], {"status": "degenerate_reference", "frame_count": len(frames)}
    directions = ref_centered / radii[:, None]
    scale_denominator = float(np.sum(ref_centered * ref_centered))
    if scale_denominator <= 0.0:
        return [], {"status": "degenerate_reference_scale", "frame_count": len(frames)}
    mean_reference_radius = float(np.mean(radii))
    l1_basis = directions
    l2_basis = _quadrupole_basis(directions)
    rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        try:
            aligned_centered = _kabsch_align_to_reference(
                np.asarray(frame, dtype=np.float64),
                reference,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            return rows, _mode_purity_failure_summary(
                "alignment_failed",
                rows=rows,
                requested_frame_count=len(frames),
                failed_dump_index=idx,
                error=exc,
            )
        displacement = aligned_centered - ref_centered
        if not np.all(np.isfinite(displacement)):
            return rows, _mode_purity_failure_summary(
                "nonfinite_displacement",
                rows=rows,
                requested_frame_count=len(frames),
                failed_dump_index=idx,
                error="aligned displacement contains non-finite values",
            )
        # The l=0 breathing coordinate for an irregular triangulated shell is a
        # self-similar scale of the reference mesh, not a constant radial offset
        # at every vertex. Removing this best-fit scale avoids counting pure
        # breathing of a non-perfect mesh as artificial l2/residual content.
        scale_delta = float(np.sum(displacement * ref_centered) / scale_denominator)
        l0 = scale_delta * mean_reference_radius
        residual_displacement = displacement - scale_delta * ref_centered
        radial = np.einsum("ij,ij->i", residual_displacement, directions)
        radial_vector = radial[:, None] * directions
        tangential = residual_displacement - radial_vector
        l1_coeff, *_ = np.linalg.lstsq(l1_basis, radial, rcond=None)
        l1_component = l1_basis @ l1_coeff
        after_l1 = radial - l1_component
        l2_coeff, *_ = np.linalg.lstsq(l2_basis, after_l1, rcond=None)
        l2_component = l2_basis @ l2_coeff
        residual = after_l1 - l2_component
        l1_rms = float(np.sqrt(np.mean(l1_component * l1_component)))
        l2_rms = float(np.sqrt(np.mean(l2_component * l2_component)))
        residual_rms = float(np.sqrt(np.mean(residual * residual)))
        tangential_rms = float(np.sqrt(np.mean(np.sum(tangential * tangential, axis=1))))
        row = {
            "dump_index": int(idx),
            "time_dpd": float(time_dpd[idx]),
            "l0_mean_radial_dpd": l0,
            "l0_scale_delta": scale_delta,
            "l1_rms_dpd": l1_rms,
            "l2_rms_dpd": l2_rms,
            "radial_residual_rms_dpd": residual_rms,
            "tangential_rms_dpd": tangential_rms,
        }
        rows.append(row)
    return rows, summarize_mode_purity_rows(rows)


def summarize_mode_purity_rows(mode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(mode_rows) < 2:
        return {"status": "insufficient_frames", "frame_count": len(mode_rows)}
    l0_arr = np.asarray([row["l0_mean_radial_dpd"] for row in mode_rows], dtype=np.float64)
    l1_arr = np.asarray([row["l1_rms_dpd"] for row in mode_rows], dtype=np.float64)
    l2_arr = np.asarray([row["l2_rms_dpd"] for row in mode_rows], dtype=np.float64)
    radial_residual_arr = np.asarray([row["radial_residual_rms_dpd"] for row in mode_rows], dtype=np.float64)
    tangential_arr = np.asarray([row["tangential_rms_dpd"] for row in mode_rows], dtype=np.float64)
    l0_amp = 0.5 * float(np.max(l0_arr) - np.min(l0_arr))
    denom = max(abs(l0_amp), 1.0e-30)

    def _normalized_temporal_rms(values: np.ndarray) -> float:
        # Each value is already a spatial RMS for one frame, so this is the
        # RMS contamination over both vertices and the selected time window.
        return float(np.sqrt(np.mean(np.square(values))) / denom)

    def _normalized_temporal_p95(values: np.ndarray) -> float:
        return float(np.quantile(values, 0.95) / denom)

    summary = {
        "status": "ok",
        "frame_count": len(mode_rows),
        "time_start_dpd": float(mode_rows[0]["time_dpd"]),
        "time_end_dpd": float(mode_rows[-1]["time_dpd"]),
        "l0_peak_to_peak_amplitude_dpd": 2.0 * l0_amp,
        "l0_half_amplitude_dpd": l0_amp,
        "l1_rms_max_over_l0_half_amplitude": float(np.max(l1_arr) / denom),
        "l2_rms_max_over_l0_half_amplitude": float(np.max(l2_arr) / denom),
        "radial_residual_rms_max_over_l0_half_amplitude": float(np.max(radial_residual_arr) / denom),
        "tangential_rms_max_over_l0_half_amplitude": float(np.max(tangential_arr) / denom),
        "l1_spatiotemporal_rms_over_l0_half_amplitude": _normalized_temporal_rms(l1_arr),
        "l2_spatiotemporal_rms_over_l0_half_amplitude": _normalized_temporal_rms(l2_arr),
        "radial_residual_spatiotemporal_rms_over_l0_half_amplitude": _normalized_temporal_rms(
            radial_residual_arr
        ),
        "tangential_spatiotemporal_rms_over_l0_half_amplitude": _normalized_temporal_rms(
            tangential_arr
        ),
        "l1_temporal_p95_over_l0_half_amplitude": _normalized_temporal_p95(l1_arr),
        "l2_temporal_p95_over_l0_half_amplitude": _normalized_temporal_p95(l2_arr),
        "radial_residual_temporal_p95_over_l0_half_amplitude": _normalized_temporal_p95(
            radial_residual_arr
        ),
        "tangential_temporal_p95_over_l0_half_amplitude": _normalized_temporal_p95(
            tangential_arr
        ),
        "mode_purity_clean_ratio_threshold": MAX_MODE_RATIO_FOR_CLEAN_L0,
    }
    l1_clean = bool(np.max(l1_arr) / denom <= MAX_MODE_RATIO_FOR_CLEAN_L0)
    l2_clean = bool(np.max(l2_arr) / denom <= MAX_MODE_RATIO_FOR_CLEAN_L0)
    radial_clean = bool(np.max(radial_residual_arr) / denom <= MAX_MODE_RATIO_FOR_CLEAN_L0)
    tangential_clean = bool(np.max(tangential_arr) / denom <= MAX_MODE_RATIO_FOR_CLEAN_L0)
    summary.update(
        {
            "mode_purity_clean_l1_le_threshold": l1_clean,
            "mode_purity_clean_l2_le_threshold": l2_clean,
            "mode_purity_clean_radial_residual_le_threshold": radial_clean,
            "mode_purity_clean_tangential_le_threshold": tangential_clean,
            # Legacy keys retained for older packagers. They now reflect the
            # active production threshold recorded above, not a hard-coded 0.25.
            "mode_purity_clean_l1_lt_0p25": l1_clean,
            "mode_purity_clean_l2_lt_0p25": l2_clean,
            "mode_purity_clean_radial_residual_lt_0p25": radial_clean,
            "mode_purity_clean_tangential_lt_0p25": tangential_clean,
        }
    )
    summary["mode_purity_clean_l0_dominant"] = bool(
        summary["mode_purity_clean_l1_le_threshold"]
        and summary["mode_purity_clean_l2_le_threshold"]
        and summary["mode_purity_clean_radial_residual_le_threshold"]
        and summary["mode_purity_clean_tangential_le_threshold"]
    )
    return summary


def _linear_detrend(time_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values - values.mean()
    coeff = np.polyfit(time_values, values, deg=1)
    return values - np.polyval(coeff, time_values)


def _local_maxima(time_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    mask = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    return time_values[1:-1][mask], values[1:-1][mask]


def _observed_alternating_cycles(
    time_values: np.ndarray,
    values: np.ndarray,
    *,
    fitted_values: np.ndarray | None,
    fitted_amplitude: float | None,
) -> dict[str, Any]:
    def empty_payload(*, prominence: float = math.nan) -> dict[str, Any]:
        return {
            "observed_alternating_cycles": 0.0,
            "observed_qualified_extrema_count": 0,
            "observed_extrema_prominence_threshold": prominence,
            "observed_tail_qualified_cycles": 0.0,
            "observed_alternating_half_period_cv": math.nan,
            "observed_tail_extremum_to_residual_rms": math.nan,
            "observed_tail_gap_half_periods": math.nan,
            "observed_tail_quality_clean": False,
        }

    if values.size < 5:
        return empty_payload()

    signal_std = float(np.std(values))
    residual_rms = math.nan
    if fitted_values is not None and fitted_values.shape == values.shape:
        residual = values - fitted_values
        residual_rms = float(np.sqrt(np.mean(residual * residual)))
    amplitude = float(fitted_amplitude) if fitted_amplitude is not None and math.isfinite(float(fitted_amplitude)) else math.nan
    candidates = [0.25 * signal_std]
    if math.isfinite(residual_rms):
        candidates.append(2.0 * residual_rms)
    if math.isfinite(amplitude):
        candidates.append(0.15 * amplitude)
    prominence = max([value for value in candidates if math.isfinite(value)] or [math.nan])
    if not math.isfinite(prominence) or prominence <= 0.0:
        return empty_payload(prominence=prominence)

    extrema: list[tuple[float, float, str]] = []
    for idx in range(1, values.size - 1):
        value = float(values[idx])
        if value > values[idx - 1] and value >= values[idx + 1] and value >= prominence:
            extrema.append((float(time_values[idx]), value, "max"))
        elif value < values[idx - 1] and value <= values[idx + 1] and value <= -prominence:
            extrema.append((float(time_values[idx]), value, "min"))

    alternating: list[tuple[float, float, str]] = []
    for item in extrema:
        if not alternating:
            alternating.append(item)
            continue
        if item[2] != alternating[-1][2]:
            alternating.append(item)
            continue
        if (item[2] == "max" and item[1] > alternating[-1][1]) or (
            item[2] == "min" and item[1] < alternating[-1][1]
        ):
            alternating[-1] = item

    cycles = max(0.0, 0.5 * float(len(alternating) - 1))
    half_period_cv = math.nan
    tail_qualified_cycles = 0.0
    tail_extremum_to_residual = math.nan
    tail_gap_half_periods = math.nan
    if len(alternating) >= 2:
        alternating_times = np.asarray([item[0] for item in alternating], dtype=np.float64)
        alternating_values = np.asarray([item[1] for item in alternating], dtype=np.float64)
        intervals = np.diff(alternating_times)
        positive_intervals = intervals[intervals > 0.0]
        if positive_intervals.size:
            mean_interval = float(np.mean(positive_intervals))
            if mean_interval > 0.0:
                half_period_cv = float(np.std(positive_intervals) / mean_interval)
            median_interval = float(np.median(positive_intervals))
            if median_interval > 0.0:
                tail_gap_half_periods = float((float(time_values[-1]) - float(alternating_times[-1])) / median_interval)
        if math.isfinite(residual_rms) and residual_rms > 0.0:
            abs_over_residual = np.abs(alternating_values) / residual_rms
            tail_extremum_to_residual = float(np.min(abs_over_residual[-2:]))
            qualified = abs_over_residual >= MIN_TAIL_EXTREMUM_TO_RESIDUAL_RMS
        else:
            tail_extremum_to_residual = math.inf
            qualified = np.abs(alternating_values) > 0.0

        best_run = 0
        current_run = 0
        for ok in qualified:
            if bool(ok):
                current_run += 1
                best_run = max(best_run, current_run)
            else:
                current_run = 0
        tail_qualified_cycles = max(0.0, 0.5 * float(best_run - 1))

    tail_quality_clean = bool(
        tail_qualified_cycles >= MIN_TAIL_QUALIFIED_CYCLES
        and math.isfinite(half_period_cv)
        and half_period_cv <= MAX_ALTERNATING_HALF_PERIOD_CV
        and math.isfinite(tail_extremum_to_residual)
        and tail_extremum_to_residual >= MIN_TAIL_EXTREMUM_TO_RESIDUAL_RMS
        and math.isfinite(tail_gap_half_periods)
        and tail_gap_half_periods <= MAX_TAIL_GAP_HALF_PERIODS
    )
    return {
        "observed_alternating_cycles": cycles,
        "observed_qualified_extrema_count": int(len(alternating)),
        "observed_extrema_prominence_threshold": float(prominence),
        "observed_tail_qualified_cycles": tail_qualified_cycles,
        "observed_alternating_half_period_cv": half_period_cv,
        "observed_tail_extremum_to_residual_rms": tail_extremum_to_residual,
        "observed_tail_gap_half_periods": tail_gap_half_periods,
        "observed_tail_quality_clean": tail_quality_clean,
    }


def _damped_sine_model(
    time_values: np.ndarray,
    offset: float,
    slope: float,
    sin_coeff: float,
    cos_coeff: float,
    damping: float,
    omega: float,
) -> np.ndarray:
    shifted = time_values - float(time_values[0])
    envelope = np.exp(-np.maximum(0.0, float(damping)) * shifted)
    return offset + slope * shifted + envelope * (
        sin_coeff * np.sin(float(omega) * shifted) + cos_coeff * np.cos(float(omega) * shifted)
    )


def _fit_damped_sine(
    time_values: np.ndarray,
    values: np.ndarray,
    *,
    initial_frequency_dpd: float,
) -> dict[str, Any]:
    if time_values.size < 12 or values.size != time_values.size:
        return {"status": "insufficient_samples", "sample_count": int(time_values.size)}
    if not np.all(np.isfinite(values)):
        return {
            "status": "nonfinite_samples",
            "sample_count": int(values.size),
            "nonfinite_sample_count": int(np.count_nonzero(~np.isfinite(values))),
        }
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:  # pragma: no cover - depends on runtime extras
        return {"status": "scipy_unavailable", "error": repr(exc)}

    t = np.asarray(time_values, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    shifted = t - t[0]
    duration = float(max(shifted[-1], np.finfo(float).eps))
    omega0 = max(float(initial_frequency_dpd) * 2.0 * math.pi, 2.0 * math.pi / duration)
    baseline = float(np.mean(y))
    amp0 = 0.5 * float(np.max(y) - np.min(y))
    if not math.isfinite(amp0) or amp0 <= 0.0:
        amp0 = max(float(np.std(y)), 1.0e-12)
    p0 = [baseline, 0.0, amp0, 0.0, 0.1 / duration, omega0]
    lower = [
        baseline - 10.0 * amp0 - abs(baseline),
        -np.inf,
        -20.0 * amp0,
        -20.0 * amp0,
        0.0,
        0.15 * omega0,
    ]
    upper = [
        baseline + 10.0 * amp0 + abs(baseline),
        np.inf,
        20.0 * amp0,
        20.0 * amp0,
        20.0 / duration,
        6.0 * omega0,
    ]
    try:
        coeffs, cov = curve_fit(
            _damped_sine_model,
            t,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=30000,
        )
    except Exception as exc:
        return {"status": "fit_failed", "error": f"{type(exc).__name__}: {exc}"}

    fitted = _damped_sine_model(t, *coeffs)
    residual = y - fitted
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    signal_std = float(np.std(y))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    amplitude = float(math.hypot(float(coeffs[2]), float(coeffs[3])))
    omega = float(coeffs[5])
    frequency_dpd = omega / (2.0 * math.pi)
    frequency_mhz = _dpd_frequency_to_mhz(frequency_dpd)
    return {
        "status": "ok",
        "frequency_dpd": float(frequency_dpd),
        "frequency_mhz": float(frequency_mhz),
        "omega_dpd": omega,
        "damping_dpd_inv": float(coeffs[4]),
        "offset": float(coeffs[0]),
        "slope": float(coeffs[1]),
        "sin_coeff": float(coeffs[2]),
        "cos_coeff": float(coeffs[3]),
        "amplitude": amplitude,
        "phase_rad": float(math.atan2(float(coeffs[3]), float(coeffs[2]))),
        "residual_rms": residual_rms,
        "signal_std": signal_std,
        "r2": r2,
        "amplitude_to_residual_std": None if residual_rms == 0.0 else amplitude / residual_rms,
        "coefficients": [float(item) for item in coeffs],
        "covariance_diag": [float(item) for item in np.diag(cov)] if np.ndim(cov) == 2 else None,
    }


def estimate_frequency(
    time_dpd: np.ndarray,
    observable: np.ndarray,
    *,
    transient_cut_fraction: float,
    fit_start_dpd: float | None = None,
    fit_end_dpd: float | None = None,
) -> dict[str, Any]:
    if time_dpd.size < 8:
        return {"status": "insufficient_samples", "sample_count": int(time_dpd.size)}
    window_mask = np.ones(time_dpd.shape, dtype=bool)
    if fit_start_dpd is not None:
        window_mask &= time_dpd >= float(fit_start_dpd)
    if fit_end_dpd is not None:
        window_mask &= time_dpd <= float(fit_end_dpd)
    window_indices = np.where(window_mask)[0]
    if window_indices.size < 8:
        return {
            "status": "insufficient_window_samples",
            "sample_count": int(time_dpd.size),
            "fit_start_dpd": None if fit_start_dpd is None else float(fit_start_dpd),
            "fit_end_dpd": None if fit_end_dpd is None else float(fit_end_dpd),
            "window_sample_count": int(window_indices.size),
        }
    window_time = time_dpd[window_indices]
    window_values = observable[window_indices]
    local_start = min(
        window_time.size - 4,
        max(0, int(round(window_time.size * transient_cut_fraction))),
    )
    start_index = int(window_indices[local_start])
    end_index = int(window_indices[-1])
    t = window_time[local_start:]
    y = window_values[local_start:]
    nonfinite_sample_count = int(np.count_nonzero(~np.isfinite(y)))
    if nonfinite_sample_count:
        return {
            "status": "nonfinite_samples",
            "sample_count": int(time_dpd.size),
            "fit_sample_count": int(t.size),
            "nonfinite_sample_count": nonfinite_sample_count,
        }
    detrended = _linear_detrend(t, y)
    dt_sample = float(np.median(np.diff(t)))
    window = np.hanning(detrended.size)
    spectrum = np.fft.rfft(detrended * window)
    freqs_dpd = np.fft.rfftfreq(detrended.size, d=dt_sample)
    power = np.abs(spectrum) ** 2
    if freqs_dpd.size <= 1:
        return {"status": "insufficient_frequency_bins", "sample_count": int(time_dpd.size)}

    nonzero = freqs_dpd > 0.0
    nonzero_indices = np.where(nonzero)[0]
    global_idx = int(nonzero_indices[np.argmax(power[nonzero_indices])])

    peak_times, _ = _local_maxima(t, detrended)
    peak_to_peak_mhz = None
    peak_to_peak_period_dpd = None
    if peak_times.size >= 3:
        periods = np.diff(peak_times)
        periods = periods[periods > 0.0]
        if periods.size:
            peak_to_peak_period_dpd = float(np.median(periods))
            peak_to_peak_mhz = 1.0 / peak_to_peak_period_dpd / UNIT_TIME_SECONDS / 1.0e6

    peak_records: list[dict[str, float]] = []
    for idx in range(1, max(1, power.size - 1)):
        if idx + 1 >= power.size:
            continue
        if power[idx] > power[idx - 1] and power[idx] >= power[idx + 1]:
            frequency_mhz = float(freqs_dpd[idx] / UNIT_TIME_SECONDS / 1.0e6)
            peak_records.append(
                {
                    "frequency_mhz": frequency_mhz,
                    "frequency_dpd": float(freqs_dpd[idx]),
                    "power": float(power[idx]),
                }
            )
    top_power_peaks = sorted(peak_records, key=lambda item: item["power"], reverse=True)[:12]
    if top_power_peaks:
        dominant_peak = top_power_peaks[0]
    else:
        dominant_frequency_mhz = float(freqs_dpd[global_idx] / UNIT_TIME_SECONDS / 1.0e6)
        dominant_peak = {
            "frequency_mhz": dominant_frequency_mhz,
            "frequency_dpd": float(freqs_dpd[global_idx]),
            "power": float(power[global_idx]),
        }
    second_peak = top_power_peaks[1] if len(top_power_peaks) > 1 else None
    dominant_to_second_power_ratio = None
    if second_peak is None:
        dominant_to_second_power_ratio = math.inf
    elif float(second_peak["power"]) > 0.0:
        dominant_to_second_power_ratio = float(dominant_peak["power"]) / float(second_peak["power"])
    damped = _fit_damped_sine(t, y, initial_frequency_dpd=float(dominant_peak["frequency_dpd"]))
    fitted_for_observed = None
    fitted_amplitude = None
    if damped.get("status") == "ok":
        coeffs = damped.get("coefficients") or []
        if len(coeffs) >= 6:
            fitted_for_observed = _linear_detrend(t, _damped_sine_model(t, *[float(item) for item in coeffs]))
        try:
            fitted_amplitude = float(damped.get("amplitude"))
        except (TypeError, ValueError):
            fitted_amplitude = math.nan
    observed = _observed_alternating_cycles(
        t,
        detrended,
        fitted_values=fitted_for_observed,
        fitted_amplitude=fitted_amplitude,
    )
    fft_frequency_mhz = float(dominant_peak["frequency_mhz"])
    damped_frequency_mhz = damped.get("frequency_mhz") if damped.get("status") == "ok" else None
    fft_damped_relative_delta = None
    if damped_frequency_mhz is not None and damped_frequency_mhz != 0.0:
        fft_damped_relative_delta = abs(fft_frequency_mhz - float(damped_frequency_mhz)) / abs(float(damped_frequency_mhz))
    peak_to_peak_damped_relative_delta = None
    if (
        peak_to_peak_mhz is not None
        and damped_frequency_mhz is not None
        and float(damped_frequency_mhz) != 0.0
    ):
        peak_to_peak_damped_relative_delta = abs(float(peak_to_peak_mhz) - float(damped_frequency_mhz)) / abs(
            float(damped_frequency_mhz)
        )
    clean_frequency = (
        damped.get("status") == "ok"
        and damped.get("r2") is not None
        and float(damped["r2"]) >= MIN_DAMPED_FIT_R2
        and damped.get("amplitude_to_residual_std") is not None
        and float(damped["amplitude_to_residual_std"]) >= MIN_AMPLITUDE_TO_RESIDUAL_STD
        and fft_damped_relative_delta is not None
        and fft_damped_relative_delta <= CLEAN_FFT_DAMPED_REL_TOL
        and peak_to_peak_damped_relative_delta is not None
        and peak_to_peak_damped_relative_delta <= CLEAN_PEAK_TO_PEAK_DAMPED_REL_TOL
        and dominant_to_second_power_ratio is not None
        and dominant_to_second_power_ratio >= MIN_DOMINANT_TO_SECOND_FFT_POWER_RATIO
        and observed["observed_alternating_cycles"] >= MIN_OBSERVED_ALTERNATING_CYCLES
        and bool(observed["observed_tail_quality_clean"])
    )
    if clean_frequency:
        frequency_status = "clean_damped_fft_agreement"
        measured_frequency_mhz = float(damped_frequency_mhz)
    elif damped.get("status") == "ok":
        frequency_status = "measured_but_quality_gate_failed"
        measured_frequency_mhz = float(damped_frequency_mhz)
    else:
        frequency_status = "fft_only_damped_fit_failed"
        measured_frequency_mhz = fft_frequency_mhz

    return {
        "status": "ok",
        "estimator": "dominant_nonzero_fft_peak_plus_damped_sine_fit",
        "sample_count": int(time_dpd.size),
        "window_sample_count": int(window_time.size),
        "fit_sample_count": int(t.size),
        "transient_cut_fraction": float(transient_cut_fraction),
        "fit_start_dpd": None if fit_start_dpd is None else float(fit_start_dpd),
        "fit_end_dpd": None if fit_end_dpd is None else float(fit_end_dpd),
        "fit_start_index": start_index,
        "fit_end_index": end_index,
        "fit_time_start_dpd": float(t[0]),
        "fit_time_end_dpd": float(t[-1]),
        "window_time_start_dpd": float(window_time[0]),
        "window_time_end_dpd": float(window_time[-1]),
        "sample_dt_dpd": dt_sample,
        "frequency_bin_resolution_dpd": float(1.0 / (t[-1] - t[0])) if t[-1] > t[0] else None,
        "fft_global_peak_frequency_mhz": float(freqs_dpd[global_idx] / UNIT_TIME_SECONDS / 1.0e6),
        "fft_dominant_peak": dominant_peak,
        "fft_top_power_peaks": top_power_peaks,
        "second_fft_peak": second_peak,
        "dominant_to_second_power_ratio": dominant_to_second_power_ratio,
        "damped_sine": damped,
        "fft_damped_relative_delta": fft_damped_relative_delta,
        "measured_frequency_mhz": measured_frequency_mhz,
        "frequency_status": frequency_status,
        "clean_frequency": bool(clean_frequency),
        "quality_gates": {
            "min_damped_fit_r2": MIN_DAMPED_FIT_R2,
            "min_amplitude_to_residual_std": MIN_AMPLITUDE_TO_RESIDUAL_STD,
            "max_fft_damped_relative_delta": CLEAN_FFT_DAMPED_REL_TOL,
            "max_peak_to_peak_damped_relative_delta": CLEAN_PEAK_TO_PEAK_DAMPED_REL_TOL,
            "min_dominant_to_second_fft_power_ratio": MIN_DOMINANT_TO_SECOND_FFT_POWER_RATIO,
            "min_observed_alternating_cycles": MIN_OBSERVED_ALTERNATING_CYCLES,
            "min_tail_qualified_cycles": MIN_TAIL_QUALIFIED_CYCLES,
            "max_alternating_half_period_cv": MAX_ALTERNATING_HALF_PERIOD_CV,
            "min_tail_extremum_to_residual_rms": MIN_TAIL_EXTREMUM_TO_RESIDUAL_RMS,
            "max_tail_gap_half_periods": MAX_TAIL_GAP_HALF_PERIODS,
        },
        "peak_to_peak_frequency_mhz": peak_to_peak_mhz,
        "peak_to_peak_period_dpd": peak_to_peak_period_dpd,
        "peak_to_peak_damped_relative_delta": peak_to_peak_damped_relative_delta,
        **observed,
        "detrended_min": float(np.min(detrended)),
        "detrended_max": float(np.max(detrended)),
        "detrended_std": float(np.std(detrended)),
        "observable_mean": float(np.mean(y)),
        "observable_std": float(np.std(y)),
    }


def estimate_driven_response(
    time_dpd: np.ndarray,
    observable: np.ndarray,
    *,
    drive_frequency_mhz: float,
    fit_start_dpd: float | None = None,
    fit_end_dpd: float | None = None,
) -> dict[str, Any]:
    drive_frequency_dpd = _mhz_to_dpd_frequency(drive_frequency_mhz)
    if drive_frequency_dpd <= 0.0:
        return {
            "status": "not_requested",
            "drive_frequency_mhz": float(drive_frequency_mhz),
            "drive_frequency_dpd": float(drive_frequency_dpd),
        }
    window_mask = np.ones(time_dpd.shape, dtype=bool)
    if fit_start_dpd is not None:
        window_mask &= time_dpd >= float(fit_start_dpd)
    if fit_end_dpd is not None:
        window_mask &= time_dpd <= float(fit_end_dpd)
    if int(np.count_nonzero(window_mask)) < 8:
        return {
            "status": "insufficient_window_samples",
            "drive_frequency_mhz": float(drive_frequency_mhz),
            "drive_frequency_dpd": float(drive_frequency_dpd),
            "window_sample_count": int(np.count_nonzero(window_mask)),
        }
    t = np.asarray(time_dpd[window_mask], dtype=np.float64)
    y = np.asarray(observable[window_mask], dtype=np.float64)
    if not np.all(np.isfinite(y)):
        return {
            "status": "nonfinite_samples",
            "drive_frequency_mhz": float(drive_frequency_mhz),
            "drive_frequency_dpd": float(drive_frequency_dpd),
            "nonfinite_sample_count": int(np.count_nonzero(~np.isfinite(y))),
        }
    shifted = t - t[0]
    omega = 2.0 * math.pi * drive_frequency_dpd
    design = np.column_stack(
        [
            np.ones_like(shifted),
            shifted,
            np.sin(omega * shifted),
            np.cos(omega * shifted),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coeffs
    residual = y - fitted
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    sin_coeff = float(coeffs[2])
    cos_coeff = float(coeffs[3])
    amplitude = float(math.hypot(sin_coeff, cos_coeff))
    phase_rad = float(math.atan2(cos_coeff, sin_coeff))

    detrended = _linear_detrend(t, y)
    dt_sample = float(np.median(np.diff(t)))
    freqs_dpd = np.fft.rfftfreq(detrended.size, d=dt_sample)
    power = np.abs(np.fft.rfft(detrended * np.hanning(detrended.size))) ** 2
    nearest_idx = int(np.argmin(np.abs(freqs_dpd - drive_frequency_dpd)))
    nearest_frequency_mhz = _dpd_frequency_to_mhz(float(freqs_dpd[nearest_idx]))
    peak_idx = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
    peak_frequency_mhz = _dpd_frequency_to_mhz(float(freqs_dpd[peak_idx])) if peak_idx > 0 else None
    response_to_residual = None if residual_rms == 0.0 else amplitude / residual_rms
    clean_response = bool(
        r2 is not None
        and r2 >= 0.20
        and response_to_residual is not None
        and response_to_residual >= MIN_AMPLITUDE_TO_RESIDUAL_STD
    )
    return {
        "status": "ok",
        "estimator": "fixed_frequency_sinusoidal_response",
        "drive_frequency_mhz": float(drive_frequency_mhz),
        "drive_frequency_dpd": float(drive_frequency_dpd),
        "drive_period_dpd": float(1.0 / drive_frequency_dpd),
        "fit_start_dpd": float(t[0]),
        "fit_end_dpd": float(t[-1]),
        "sample_count": int(time_dpd.size),
        "fit_sample_count": int(t.size),
        "amplitude_dpd": amplitude,
        "phase_rad": phase_rad,
        "offset": float(coeffs[0]),
        "slope": float(coeffs[1]),
        "sin_coeff": sin_coeff,
        "cos_coeff": cos_coeff,
        "residual_rms": residual_rms,
        "signal_std": float(np.std(y)),
        "response_to_residual_std": response_to_residual,
        "r2": None if r2 is None else float(r2),
        "coefficients": [float(item) for item in coeffs],
        "nearest_fft_frequency_mhz": float(nearest_frequency_mhz),
        "nearest_fft_power": float(power[nearest_idx]),
        "dominant_fft_frequency_mhz": peak_frequency_mhz,
        "dominant_fft_power": None if peak_idx <= 0 else float(power[peak_idx]),
        "clean_driven_response": clean_response,
    }


def postprocess_symbol(
    bubble: Bubble,
    symbol_root: Path,
    *,
    seed_index: int | None = None,
    dt: float,
    sample_every: int,
    pulse_steps: int,
    relax_steps: int,
    transient_cut_fraction: float,
    fit_start_dpd: float | None,
    fit_end_dpd: float | None,
    primary_observable: str,
    captured_frames: list[np.ndarray] | None = None,
    captured_frame_steps: list[int] | None = None,
    trajectory_capture: str = "particle-dump",
    particle_dump_dir: Path | None = None,
) -> dict[str, Any]:
    import trimesh

    sim_dir = symbol_root / "simulation"
    setup_manifest_path = symbol_root / "setup_manifest.json"
    setup_protocol: dict[str, Any] = {}
    if setup_manifest_path.exists():
        setup_manifest = json.loads(setup_manifest_path.read_text(encoding="utf-8"))
        setup_protocol = dict(setup_manifest.get("protocol") or {})
    generated_params_path = sim_dir / "parameter" / "parameters00001.yaml"
    generated_material: dict[str, Any] = {
        "parameters_yaml": str(generated_params_path),
        "ka": float(bubble.ka),
        "kb": float(bubble.kb),
        "mu": None,
        "membrane_mass_scale": float(setup_protocol.get("membrane_mass_scale", 1.0)),
    }
    if generated_params_path.is_file():
        generated_params = _load_yaml(generated_params_path)
        for key in ("ka", "kb", "mu", "mvert", "mvert_unscaled", "membrane_mass_scale", "lim_mu_policy"):
            if key in generated_params:
                generated_material[key] = generated_params[key]
    mesh_path = sim_dir / "mesh" / "emb00001.off"
    if not mesh_path.exists():
        mesh_candidates = sorted((sim_dir / "mesh").glob("emb*.off"))
        if not mesh_candidates:
            raise FileNotFoundError(f"No EMB mesh found under {sim_dir / 'mesh'}")
        mesh_path = mesh_candidates[0]
    mesh = trimesh.load_mesh(mesh_path, process=False)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    files: list[Path] = []
    if captured_frames is None:
        resolved_particle_dump_dir = (
            sim_dir / "particles"
            if particle_dump_dir is None
            else Path(particle_dump_dir).resolve()
        )
        files = sorted(resolved_particle_dump_dir.glob("emb*.h5"))
        if not files:
            raise FileNotFoundError(
                f"No membrane particle dumps found under {resolved_particle_dump_dir}"
            )
        frames = [_read_positions(path) for path in files]
    else:
        frames = [np.asarray(frame, dtype=np.float64) for frame in captured_frames]
        if not frames:
            raise RuntimeError("Memory trajectory capture returned no membrane frames.")
        if captured_frame_steps is None or len(captured_frame_steps) != len(frames):
            raise RuntimeError(
                "Memory trajectory capture requires one exact step index per membrane frame."
            )

    rows: list[dict[str, Any]] = []
    for idx, positions in enumerate(frames):
        centroid = positions.mean(axis=0)
        radii = np.linalg.norm(positions - centroid[None, :], axis=1)
        volume = _mesh_volume(positions, faces)
        equivalent_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        step = (
            idx * int(sample_every)
            if captured_frame_steps is None
            else int(captured_frame_steps[idx])
        )
        time_dpd = step * float(dt)
        rows.append(
            {
                "dump_index": idx,
                "step_after_equil": step,
                "time_dpd": time_dpd,
                "time_seconds": time_dpd * UNIT_TIME_SECONDS,
                "phase": "pulse" if step < pulse_steps else "free_relaxation",
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "mean_radius_dpd": float(np.mean(radii)),
                "rms_radius_dpd": float(np.sqrt(np.mean(radii * radii))),
                "equivalent_radius_dpd": float(equivalent_radius),
                "volume_dpd3": volume,
                "hdf5_path": "" if not files else str(files[idx]),
            }
        )

    csv_path = symbol_root / "time_series.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    time_dpd = np.asarray([row["time_dpd"] for row in rows], dtype=np.float64)
    series = {
        "rms_radius_dpd": np.asarray([row["rms_radius_dpd"] for row in rows], dtype=np.float64),
        "mean_radius_dpd": np.asarray([row["mean_radius_dpd"] for row in rows], dtype=np.float64),
        "equivalent_radius_dpd": np.asarray([row["equivalent_radius_dpd"] for row in rows], dtype=np.float64),
        "volume_dpd3": np.asarray([row["volume_dpd3"] for row in rows], dtype=np.float64),
    }
    if primary_observable not in series:
        raise ValueError(f"Unsupported primary observable: {primary_observable}")
    observable = series[primary_observable]
    mean_radius = series["mean_radius_dpd"]
    volume = series["volume_dpd3"]
    fit = estimate_frequency(
        time_dpd,
        observable,
        transient_cut_fraction=transient_cut_fraction,
        fit_start_dpd=fit_start_dpd,
        fit_end_dpd=fit_end_dpd,
    )
    observable_fits = {
        name: estimate_frequency(
            time_dpd,
            values,
            transient_cut_fraction=transient_cut_fraction,
            fit_start_dpd=fit_start_dpd,
            fit_end_dpd=fit_end_dpd,
        )
        for name, values in series.items()
    }
    sensitivity = {
        f"cut_{cut:.2f}": estimate_frequency(
            time_dpd,
            observable,
            transient_cut_fraction=cut,
            fit_start_dpd=fit_start_dpd,
            fit_end_dpd=fit_end_dpd,
        )
        for cut in (0.15, 0.25, 0.35)
    }
    driven_response = None
    if setup_protocol.get("excitation_mode") in {
        "sinusoidal-drive",
        "normal-sinusoidal-drive",
        "current-radial-sinusoidal-drive",
    }:
        drive_frequency_mhz = float(setup_protocol.get("drive_frequency_mhz") or 0.0)
        driven_response = estimate_driven_response(
            time_dpd,
            observable,
            drive_frequency_mhz=drive_frequency_mhz,
            fit_start_dpd=fit_start_dpd,
            fit_end_dpd=fit_end_dpd,
        )

    plots = write_plots(
        bubble,
        symbol_root,
        time_dpd=time_dpd,
        observable=observable,
        observable_name=primary_observable,
        fit=fit,
        transient_cut_fraction=transient_cut_fraction,
        fit_start_dpd=fit_start_dpd,
        fit_end_dpd=fit_end_dpd,
    )
    if driven_response is not None:
        plots.update(
            write_driven_response_plot(
                bubble,
                symbol_root,
                time_dpd=time_dpd,
                observable=observable,
                observable_name=primary_observable,
                response=driven_response,
            )
        )
    mode_rows, mode_summary = compute_mode_purity_timeseries(frames, time_dpd=time_dpd)
    mode_fit_window_summary = {"status": "not_available", "frame_count": 0}
    if mode_rows and fit.get("fit_time_start_dpd") is not None and fit.get("fit_time_end_dpd") is not None:
        fit_start_time = float(fit["fit_time_start_dpd"])
        fit_end_time = float(fit["fit_time_end_dpd"])
        mode_fit_rows = [
            row
            for row in mode_rows
            if fit_start_time <= float(row["time_dpd"]) <= fit_end_time
        ]
        mode_fit_window_summary = summarize_mode_purity_rows(mode_fit_rows)
    mode_gate_summary = (
        mode_fit_window_summary
        if mode_fit_window_summary.get("status") == "ok"
        else mode_summary
    )
    mode_csv_path = symbol_root / "mode_purity_timeseries.csv"
    if mode_rows:
        with mode_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mode_rows)
        plots.update(write_mode_purity_plot(bubble, symbol_root, mode_rows=mode_rows))
    frequency_mhz = fit.get("measured_frequency_mhz")
    result = {
        "protocol_type": "driven-response" if driven_response is not None else "free-run",
        "symbol": bubble.symbol,
        "agent": bubble.agent,
        "modality": bubble.modality,
        "dataset": bubble.dataset,
        "diameter_um": bubble.diameter_um,
        "seed_index": None if seed_index is None else int(seed_index),
        "seed_label": None if seed_index is None else _seed_label(seed_index),
        "symbol_root": str(symbol_root),
        "map": bubble.__dict__,
        "generated_material": generated_material,
        "observable": primary_observable,
        "setup_protocol": setup_protocol,
        "fit_window": {
            "transient_cut_fraction": float(transient_cut_fraction),
            "fit_start_dpd": None if fit_start_dpd is None else float(fit_start_dpd),
            "fit_end_dpd": None if fit_end_dpd is None else float(fit_end_dpd),
        },
        "observable_fits": observable_fits,
        "observable_stats": {
            "primary_min": float(np.min(observable)),
            "primary_max": float(np.max(observable)),
            "primary_mean": float(np.mean(observable)),
            "primary_std": float(np.std(observable)),
            "rms_radius_min_dpd": float(np.min(series["rms_radius_dpd"])),
            "rms_radius_max_dpd": float(np.max(series["rms_radius_dpd"])),
            "rms_radius_mean_dpd": float(np.mean(series["rms_radius_dpd"])),
            "rms_radius_std_dpd": float(np.std(series["rms_radius_dpd"])),
            "mean_radius_min_dpd": float(np.min(mean_radius)),
            "mean_radius_max_dpd": float(np.max(mean_radius)),
            "equivalent_radius_min_dpd": float(np.min(series["equivalent_radius_dpd"])),
            "equivalent_radius_max_dpd": float(np.max(series["equivalent_radius_dpd"])),
            "volume_min_dpd3": float(np.min(volume)),
            "volume_max_dpd3": float(np.max(volume)),
            "volume_mean_dpd3": float(np.mean(volume)),
            "volume_std_dpd3": float(np.std(volume)),
        },
        "time_series_csv": str(csv_path),
        "mode_purity_csv": str(mode_csv_path) if mode_rows else None,
        "mode_purity": mode_gate_summary,
        "mode_purity_gate_basis": "fit_window" if mode_fit_window_summary.get("status") == "ok" else "full_trace",
        "mode_purity_full_trace": mode_summary,
        "mode_purity_fit_window": mode_fit_window_summary,
        "trajectory_capture": trajectory_capture,
        "particle_dump_directory": (
            None
            if trajectory_capture != "particle-dump"
            else str(
                sim_dir / "particles"
                if particle_dump_dir is None
                else Path(particle_dump_dir).resolve()
            )
        ),
        "captured_frame_count": len(frames),
        "captured_final_step_after_release": int(rows[-1]["step_after_equil"]),
        "particle_dump_count": len(files),
        "particle_dump_paths": [str(path) for path in files],
        "fit": fit,
        "driven_response": driven_response,
        "sensitivity": sensitivity,
        "fitted_frequency_mhz": frequency_mhz,
        "frequency_status": fit.get("frequency_status"),
        "fft_dominant_frequency_mhz": (
            None
            if fit.get("fft_dominant_peak") is None
            else fit["fft_dominant_peak"].get("frequency_mhz")
        ),
        "damped_sine_frequency_mhz": (
            None
            if fit.get("damped_sine", {}).get("status") != "ok"
            else fit["damped_sine"].get("frequency_mhz")
        ),
        "fft_damped_relative_delta": fit.get("fft_damped_relative_delta"),
        "plots": plots,
        "checks": {
            "finite_observable": bool(np.all(np.isfinite(observable))),
            "nonzero_observable_std": bool(np.std(observable) > 0.0),
            "finite_volume": bool(np.all(np.isfinite(volume))),
            "nonzero_volume_std": bool(np.std(volume) > 0.0),
            "finite_frequency": frequency_mhz is not None and math.isfinite(float(frequency_mhz)),
            "clean_frequency": bool(fit.get("clean_frequency")),
            "clean_driven_response": (
                None
                if driven_response is None
                else bool(driven_response.get("clean_driven_response"))
            ),
            "clean_mode_purity": bool(mode_gate_summary.get("mode_purity_clean_l0_dominant")),
        },
    }
    result_path = symbol_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_plots(
    bubble: Bubble,
    symbol_root: Path,
    *,
    time_dpd: np.ndarray,
    observable: np.ndarray,
    observable_name: str,
    fit: dict[str, Any],
    transient_cut_fraction: float,
    fit_start_dpd: float | None,
    fit_end_dpd: float | None,
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime extras
        return {"plot_error": repr(exc)}

    plots_dir = symbol_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fit_start_time = fit.get("fit_time_start_dpd")
    fit_end_time = fit.get("fit_time_end_dpd")
    if fit_start_time is None:
        fit_start_time = float(time_dpd[min(time_dpd.size - 1, max(0, int(round(time_dpd.size * transient_cut_fraction))))])
    if fit_end_time is None:
        fit_end_time = float(time_dpd[-1])
    fit_mask = (time_dpd >= float(fit_start_time)) & (time_dpd <= float(fit_end_time))
    if int(np.count_nonzero(fit_mask)) < 4:
        fit_mask = np.ones(time_dpd.shape, dtype=bool)
    stem, ylabel = OBSERVABLE_PLOT_METADATA.get(observable_name, (observable_name, observable_name))

    time_path = plots_dir / f"{bubble.symbol}_{stem}_timeseries.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_dpd, observable, lw=1.0)
    ax.axvline(float(fit_start_time), color="tab:red", ls="--", lw=1.0, label="fit start")
    ax.axvline(float(fit_end_time), color="tab:red", ls=":", lw=1.0, label="fit end")
    damped = fit.get("damped_sine", {})
    if damped.get("status") == "ok" and damped.get("coefficients"):
        t_fit = time_dpd[fit_mask]
        y_fit = _damped_sine_model(t_fit, *[float(item) for item in damped["coefficients"]])
        ax.plot(t_fit, y_fit, color="tab:orange", lw=1.2, label="damped-sine fit")
    ax.set_xlabel("time after equilibration [DPD]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{bubble.symbol} {bubble.dataset}: free breathing observable")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(time_path, dpi=180)
    plt.close(fig)

    spectrum_path = plots_dir / f"{bubble.symbol}_{stem}_spectrum.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    t = time_dpd[fit_mask]
    if t.size >= 4:
        y = _linear_detrend(t, observable[fit_mask])
        dt_sample = float(np.median(np.diff(t)))
        freqs_dpd = np.fft.rfftfreq(y.size, d=dt_sample)
        freqs_mhz = freqs_dpd / UNIT_TIME_SECONDS / 1.0e6
        power = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
        ax.plot(freqs_mhz, power, lw=1.0)
    dominant_peak = fit.get("fft_dominant_peak")
    if dominant_peak is not None:
        ax.axvline(float(dominant_peak["frequency_mhz"]), color="tab:orange", ls="--", label="dominant FFT peak")
    if damped.get("status") == "ok":
        ax.axvline(
            float(damped["frequency_mhz"]),
            color="tab:green",
            ls="-.",
            label="damped-sine fit",
        )
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("periodogram power [a.u.]")
    ax.set_title(f"{bubble.symbol} {bubble.dataset}: {observable_name} spectrum")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=180)
    plt.close(fig)

    return {"time_series_png": str(time_path), "spectrum_png": str(spectrum_path)}


def write_driven_response_plot(
    bubble: Bubble,
    symbol_root: Path,
    *,
    time_dpd: np.ndarray,
    observable: np.ndarray,
    observable_name: str,
    response: dict[str, Any],
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime extras
        return {"driven_response_plot_error": repr(exc)}
    if response.get("status") != "ok":
        return {}

    plots_dir = symbol_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem, ylabel = OBSERVABLE_PLOT_METADATA.get(observable_name, (observable_name, observable_name))
    fit_start = float(response["fit_start_dpd"])
    fit_end = float(response["fit_end_dpd"])
    fit_mask = (time_dpd >= fit_start) & (time_dpd <= fit_end)
    t = time_dpd[fit_mask]
    shifted = t - t[0]
    coeffs = [float(item) for item in response["coefficients"]]
    omega = 2.0 * math.pi * float(response["drive_frequency_dpd"])
    y_fit = coeffs[0] + coeffs[1] * shifted + coeffs[2] * np.sin(omega * shifted)
    y_fit += coeffs[3] * np.cos(omega * shifted)

    time_path = plots_dir / f"{bubble.symbol}_{stem}_driven_response_fit.png"
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.plot(time_dpd, observable, lw=1.0, label="observable")
    ax.plot(t, y_fit, lw=1.2, color="tab:orange", label="fixed-drive fit")
    ax.axvline(fit_start, color="tab:red", ls="--", lw=0.9)
    ax.axvline(fit_end, color="tab:red", ls=":", lw=0.9)
    ax.set_xlabel("time after equilibration [DPD]")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{bubble.symbol} {bubble.dataset}: driven response "
        f"at {float(response['drive_frequency_mhz']):.4g} MHz"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(time_path, dpi=180)
    plt.close(fig)

    spectrum_path = plots_dir / f"{bubble.symbol}_{stem}_driven_response_spectrum.png"
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    if t.size >= 4:
        y = _linear_detrend(t, observable[fit_mask])
        dt_sample = float(np.median(np.diff(t)))
        freqs_dpd = np.fft.rfftfreq(y.size, d=dt_sample)
        freqs_mhz = freqs_dpd / UNIT_TIME_SECONDS / 1.0e6
        power = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
        ax.plot(freqs_mhz, power, lw=1.0)
    ax.axvline(float(response["drive_frequency_mhz"]), color="tab:orange", ls="--", label="drive")
    if response.get("dominant_fft_frequency_mhz") is not None:
        ax.axvline(
            float(response["dominant_fft_frequency_mhz"]),
            color="tab:green",
            ls="-.",
            label="dominant FFT",
        )
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("periodogram power [a.u.]")
    ax.set_title(f"{bubble.symbol} {bubble.dataset}: driven-response spectrum")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=180)
    plt.close(fig)

    return {
        "driven_response_fit_png": str(time_path),
        "driven_response_spectrum_png": str(spectrum_path),
    }


def write_mode_purity_plot(
    bubble: Bubble,
    symbol_root: Path,
    *,
    mode_rows: list[dict[str, Any]],
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime extras
        return {"mode_purity_plot_error": repr(exc)}
    if not mode_rows:
        return {}
    plots_dir = symbol_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    time = np.asarray([row["time_dpd"] for row in mode_rows], dtype=np.float64)
    l0 = np.asarray([row["l0_mean_radial_dpd"] for row in mode_rows], dtype=np.float64)
    l1 = np.asarray([row["l1_rms_dpd"] for row in mode_rows], dtype=np.float64)
    l2 = np.asarray([row["l2_rms_dpd"] for row in mode_rows], dtype=np.float64)
    residual = np.asarray([row["radial_residual_rms_dpd"] for row in mode_rows], dtype=np.float64)
    tangential = np.asarray([row["tangential_rms_dpd"] for row in mode_rows], dtype=np.float64)
    path = plots_dir / f"{bubble.symbol}_mode_purity.png"
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    axes[0].plot(time, l0, lw=1.0, color="black", label="l0 mean radial")
    axes[0].set_ylabel("l0 displacement [DPD]")
    axes[0].legend(loc="best")
    denom = max(0.5 * float(np.max(l0) - np.min(l0)), 1.0e-30)
    axes[1].plot(time, l1 / denom, lw=1.0, label="l1/l0 amp")
    axes[1].plot(time, l2 / denom, lw=1.0, label="l2/l0 amp")
    axes[1].plot(time, residual / denom, lw=1.0, label="radial residual/l0 amp")
    axes[1].plot(time, tangential / denom, lw=1.0, label="tangential/l0 amp")
    axes[1].axhline(
        MAX_MODE_RATIO_FOR_CLEAN_L0,
        color="tab:red",
        ls="--",
        lw=0.8,
        label=f"{MAX_MODE_RATIO_FOR_CLEAN_L0:.2f} guide",
    )
    axes[1].set_xlabel("time after equilibration [DPD]")
    axes[1].set_ylabel("ratio to l0 half-amplitude")
    axes[1].legend(loc="best", ncol=2, fontsize=8)
    fig.suptitle(f"{bubble.symbol} {bubble.dataset}: membrane breathing purity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"mode_purity_png": str(path)}


def aggregate(run_root: Path, bubbles: list[Bubble]) -> dict[str, Any]:
    results = []
    for bubble in bubbles:
        for root in _result_roots_for_bubble(run_root, bubble):
            path = root / "result.json"
            results.append(json.loads(path.read_text(encoding="utf-8")))

    summary_path = run_root / "summary.csv"
    fieldnames = [
        "protocol_type",
        "symbol",
        "seed_index",
        "seed_label",
        "agent",
        "dataset",
        "diameter_um",
        "source_label",
        "sample_index",
        "map_ka",
        "map_kb",
        "ka",
        "kb",
        "mu",
        "membrane_mass_scale",
        "radius_dpd",
        "fitted_frequency_mhz",
        "frequency_status",
        "fft_dominant_frequency_mhz",
        "damped_sine_frequency_mhz",
        "fft_damped_relative_delta",
        "drive_frequency_mhz",
        "driven_amplitude_dpd",
        "driven_phase_rad",
        "driven_response_r2",
        "driven_response_to_residual_std",
        "clean_driven_response",
        "prestrain_placement",
        "initial_radius_scale",
        "stress_free_radius_scale",
        "fit_start_dpd",
        "fit_end_dpd",
        "fit_time_start_dpd",
        "fit_time_end_dpd",
        "clean_frequency",
        "clean_mode_purity",
        "mode_purity_gate_basis",
        "l2_over_l0",
        "tangential_over_l0",
        "full_trace_l2_over_l0",
        "full_trace_tangential_over_l0",
        "particle_dump_count",
        "time_series_csv",
        "mode_purity_csv",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            checks = result.get("checks") or {}
            driven = result.get("driven_response") or {}
            generated_material = result.get("generated_material") or {}
            row = {
                "protocol_type": result.get("protocol_type", "free-run"),
                "symbol": result["symbol"],
                "seed_index": result.get("seed_index"),
                "seed_label": result.get("seed_label"),
                "agent": result["agent"],
                "dataset": result["dataset"],
                "diameter_um": result["diameter_um"],
                "source_label": result["map"]["source_label"],
                "sample_index": result["map"]["sample_index"],
                "map_ka": result["map"]["ka"],
                "map_kb": result["map"]["kb"],
                "ka": generated_material.get("ka", result["map"]["ka"]),
                "kb": generated_material.get("kb", result["map"]["kb"]),
                "mu": generated_material.get("mu"),
                "membrane_mass_scale": generated_material.get("membrane_mass_scale"),
                "radius_dpd": result["map"]["radius_dpd"],
                "fitted_frequency_mhz": result.get("fitted_frequency_mhz"),
                "frequency_status": result.get("frequency_status"),
                "fft_dominant_frequency_mhz": result.get("fft_dominant_frequency_mhz"),
                "damped_sine_frequency_mhz": result.get("damped_sine_frequency_mhz"),
                "fft_damped_relative_delta": result.get("fft_damped_relative_delta"),
                "drive_frequency_mhz": driven.get("drive_frequency_mhz"),
                "driven_amplitude_dpd": driven.get("amplitude_dpd"),
                "driven_phase_rad": driven.get("phase_rad"),
                "driven_response_r2": driven.get("r2"),
                "driven_response_to_residual_std": driven.get("response_to_residual_std"),
                "clean_driven_response": checks.get("clean_driven_response"),
                "prestrain_placement": (result.get("setup_protocol") or {}).get("prestrain_placement"),
                "initial_radius_scale": (result.get("setup_protocol") or {}).get("initial_radius_scale"),
                "stress_free_radius_scale": (result.get("setup_protocol") or {}).get("stress_free_radius_scale"),
                "fit_start_dpd": (result.get("fit_window") or {}).get("fit_start_dpd"),
                "fit_end_dpd": (result.get("fit_window") or {}).get("fit_end_dpd"),
                "fit_time_start_dpd": (result.get("fit") or {}).get("fit_time_start_dpd"),
                "fit_time_end_dpd": (result.get("fit") or {}).get("fit_time_end_dpd"),
                "clean_frequency": checks.get("clean_frequency"),
                "clean_mode_purity": checks.get("clean_mode_purity"),
                "mode_purity_gate_basis": result.get("mode_purity_gate_basis"),
                "l2_over_l0": (result.get("mode_purity") or {}).get("l2_rms_max_over_l0_half_amplitude"),
                "tangential_over_l0": (result.get("mode_purity") or {}).get(
                    "tangential_rms_max_over_l0_half_amplitude"
                ),
                "full_trace_l2_over_l0": (result.get("mode_purity_full_trace") or {}).get(
                    "l2_rms_max_over_l0_half_amplitude"
                ),
                "full_trace_tangential_over_l0": (result.get("mode_purity_full_trace") or {}).get(
                    "tangential_rms_max_over_l0_half_amplitude"
                ),
                "particle_dump_count": result.get("particle_dump_count"),
                "time_series_csv": result.get("time_series_csv"),
                "mode_purity_csv": result.get("mode_purity_csv"),
            }
            writer.writerow(row)

    manifest = {
        "schema": "mesouq.emb_free_shell_breathing_protocol.v1",
        "created_unix": time.time(),
        "run_root": str(run_root),
        "repo_root": str(repo_root()),
        "git_commit": git_commit(),
        "map_sources": {
            "Definity": "attempt081 final 50k seed00",
            "SonoVue": "attempt066 seed00 from 10-seed 50k run",
        },
        "results": results,
        "seed_indices": sorted(
            {
                int(result["seed_index"])
                for result in results
                if result.get("seed_index") is not None
            }
        ),
        "summary_csv": str(summary_path),
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = run_root / "report.md"
    write_report(report_path, manifest)
    manifest["report_md"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# EMB breathing-mode Mirheo DPD report",
        "",
        f"Run root: `{manifest['run_root']}`",
        f"Repo root: `{manifest['repo_root']}`",
        "",
        "## Summary",
        "",
        "| symbol | seed | dataset | measured MHz | FFT MHz | damped MHz | clean freq | clean mode | status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in manifest["results"]:
        freq = result.get("fitted_frequency_mhz")
        fft = result.get("fft_dominant_frequency_mhz")
        damped = result.get("damped_sine_frequency_mhz")
        checks = result.get("checks") or {}
        lines.append(
            "| {symbol} | {seed} | {dataset} | {freq} | {fft} | {damped} | {clean_frequency} | {clean_mode} | {status} |".format(
                symbol=result["symbol"],
                seed="" if result.get("seed_label") is None else result["seed_label"],
                dataset=result["dataset"],
                freq="" if freq is None else f"{float(freq):.9g}",
                fft="" if fft is None else f"{float(fft):.9g}",
                damped="" if damped is None else f"{float(damped):.9g}",
                clean_frequency=checks.get("clean_frequency", ""),
                clean_mode=checks.get("clean_mode_purity", ""),
                status=result.get("frequency_status", ""),
            )
        )
    protocol_types = sorted({str(result.get("protocol_type", "free-run")) for result in manifest["results"]})
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Simulations use the tracked EMB Mirheo mesh and parameter writers.",
            "- The quasi-static compression plates/pinning and indentation anchors/loads are not used.",
            "- The original generated `bpress`, solvent/gas DPD settings, bouncer settings, membrane mass behavior, and thermal `kBT` are preserved.",
            "- The uniform radius perturbation placement is recorded per run: default post-equilibration coordinate release or initial displaced geometry.",
            "- Excitation protocol, primary observable, and prestrain sidecar are recorded in each `setup_manifest.json` and result folder.",
            f"- Protocol types in this manifest: `{', '.join(protocol_types)}`.",
            "- Frequency is reported from damped-sine and FFT measurements without an analytical target gate.",
            "- Membrane trajectory cleanliness is checked with Kabsch-aligned l0/l1/l2/radial-residual/tangential mode metrics.",
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{manifest['summary_csv']}`",
            f"- Manifest: `{Path(path).with_name('manifest.json')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_map = repo_root() / "papers" / "huq_emb" / "resonance_breathing_mode_handoff" / "latest_50k_map_values.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-values", type=Path, default=default_map)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", default=["d2"])
    parser.add_argument("--all", action="store_true", help="Run or aggregate all six symbols.")
    parser.add_argument("--seed-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument(
        "--equil-steps",
        type=int,
        default=-1,
        help="Thermal equilibration steps; negative uses the modality's original EMB default.",
    )
    parser.add_argument("--pulse-steps", type=int, default=0)
    parser.add_argument(
        "--relax-steps",
        type=int,
        default=-1,
        help="Free-release measurement steps; negative uses the modality's original EMB default.",
    )
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument(
        "--trajectory-capture",
        choices=["particle-dump", "memory"],
        default="particle-dump",
        help=(
            "particle-dump preserves the Mirheo HDF5 workflow. memory samples EMB coordinates on rank 0 "
            "during a simple free release and avoids parallel HDF5 output."
        ),
    )
    parser.add_argument(
        "--particle-dump-root",
        type=Path,
        default=None,
        help=(
            "Optional external root for transient membrane particle dumps. "
            "Each symbol/seed is isolated below this root; CSV/JSON/PNG evidence remains under --run-root."
        ),
    )
    parser.add_argument("--pulse-force-per-vertex", type=float, default=1.0e-3)
    parser.add_argument(
        "--excitation-mode",
        choices=[
            "force-pulse",
            "prestrain",
            "velocity-kick",
            "fluid-velocity-kick",
            "sinusoidal-drive",
            "normal-sinusoidal-drive",
            "current-radial-sinusoidal-drive",
            "tone-burst",
            "fluid-force-burst",
            "normal-tone-burst",
            "normal-force-pulse",
            "radius-restraint-release",
            "shape-displacement-release",
            "none",
        ],
        default="prestrain",
    )
    parser.add_argument("--initial-radius-scale", type=float, default=0.96)
    parser.add_argument("--radial-velocity-kick", type=float, default=0.0)
    parser.add_argument(
        "--radial-velocity-kick-mode",
        choices=[
            "add",
            "replace",
            "scale-add",
            "scale-replace",
            "normal-add",
            "normal-replace",
            "volume-gradient-add",
            "volume-gradient-replace",
            "shape-add",
            "shape-replace",
        ],
        default="add",
        help=(
            "`add`/`replace` use a constant radial velocity per vertex. "
            "`scale-add`/`scale-replace` use a self-similar scale-rate profile "
            "whose mean radial velocity equals --radial-velocity-kick. "
            "`normal-add`/`normal-replace` use current post-equilibration "
            "surface normals. `volume-gradient-add`/`volume-gradient-replace` "
            "use area-weighted current normals proportional to dV/dx. "
            "`shape-add`/`shape-replace` load a diagnostic per-vertex velocity "
            "shape from --velocity-shape-json and scale it to the requested RMS speed."
        ),
    )
    parser.add_argument(
        "--velocity-shape-json",
        type=Path,
        default=None,
        help=(
            "Diagnostic per-vertex 3D velocity shape for shape-add/shape-replace modes. "
            "The JSON must contain shape_vectors or velocity_shape with one vector per membrane vertex."
        ),
    )
    parser.add_argument(
        "--shape-displacement-amplitude-dpd",
        type=float,
        default=0.0,
        help=(
            "Diagnostic coordinate-release amplitude in DPD units for "
            "shape-displacement-release. The normalized shape is loaded from "
            "--velocity-shape-json."
        ),
    )
    parser.add_argument(
        "--shape-displacement-projection",
        choices=["full", "radial"],
        default="full",
        help=(
            "Diagnostic projection for shape-displacement-release. full applies "
            "the loaded 3D shape; radial keeps only each vertex's radial component "
            "relative to the post-equilibration center and renormalizes it."
        ),
    )
    parser.add_argument("--drive-frequency-mhz", type=float, default=0.0)
    parser.add_argument("--drive-force-per-vertex", type=float, default=1.0e-4)
    parser.add_argument("--drive-update-every-steps", type=int, default=10)
    parser.add_argument("--tone-burst-cycles", type=float, default=2.0)
    parser.add_argument("--tone-burst-ramp-cycles", type=float, default=0.5)
    parser.add_argument(
        "--radius-restraint-target-scale",
        type=float,
        default=1.01,
        help=(
            "Diagnostic l0 restraint target radius relative to the post-equilibration "
            "RMS membrane radius. Used only for radius-restraint-release."
        ),
    )
    parser.add_argument(
        "--radius-restraint-force-constant",
        type=float,
        default=0.02,
        help="Diagnostic radial harmonic force constant per vertex in DPD units.",
    )
    parser.add_argument(
        "--radius-restraint-max-force-per-vertex",
        type=float,
        default=0.01,
        help="Force cap per membrane vertex for radius-restraint-release; nonpositive disables clipping.",
    )
    parser.add_argument(
        "--radius-restraint-hold-steps",
        type=int,
        default=5000,
        help="Number of post-equilibration steps to hold the l0 radius restraint before release.",
    )
    parser.add_argument(
        "--radius-restraint-update-every-steps",
        type=int,
        default=20,
        help="How often to recompute the radius-restraint force from current membrane coordinates.",
    )
    parser.add_argument("--fluid-kick-water-scale", type=float, default=1.0)
    parser.add_argument("--fluid-kick-gas-scale", type=float, default=1.0)
    parser.add_argument("--fluid-kick-water-outer-radius-factor", type=float, default=2.5)
    parser.add_argument("--fluid-burst-water-force-per-particle", type=float, default=0.0)
    parser.add_argument("--fluid-burst-gas-force-per-particle", type=float, default=0.0)
    parser.add_argument("--fluid-burst-force-field-grid-spacing", type=float, default=1.0)
    parser.add_argument("--fluid-burst-water-outer-radius-factor", type=float, default=2.5)
    parser.add_argument("--stress-free-radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--prestrain-placement",
        choices=["post-equilibration", "initial-geometry"],
        default="post-equilibration",
        help=(
            "Where to place the isotropic shell displacement. post-equilibration "
            "preserves the current protocol by moving membrane coordinates after "
            "equilibration. initial-geometry scales the MembraneMesh before fluid/gas "
            "initialization and does not apply a post-equilibration coordinate jump."
        ),
    )
    parser.add_argument(
        "--post-deflation-ramp-steps",
        type=int,
        default=0,
        help=(
            "Half-cosine coordinate-ramp duration before the fixed-coordinate hold. "
            "The fluids continue integrating during the ramp; 0 preserves the legacy "
            "instantaneous post-equilibration displacement."
        ),
    )
    parser.add_argument(
        "--post-deflation-hold-steps",
        type=int,
        default=0,
        help=(
            "Temporary coordinate-hold duration after post-equilibration deflation and before free release. "
            "0 preserves the legacy immediate-release behavior."
        ),
    )
    parser.add_argument(
        "--post-deflation-hold-update-every-steps",
        type=int,
        default=1,
        help="Coordinate-reset cadence during --post-deflation-hold-steps; 1 holds every integration step.",
    )
    parser.add_argument(
        "--post-deflation-hold-reset-velocities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset membrane vertex velocities during the temporary coordinate hold.",
    )
    parser.add_argument("--solvent-mode", choices=["full", "vacuum", "gas-only", "water-only"], default="full")
    parser.add_argument(
        "--water-shell-fsi-scale",
        type=float,
        default=WATER_SHELL_FSI_CONSERVATIVE_SCALE,
        help=(
            "Diagnostic water-shell conservative FSI scale multiplying aii. "
            "Default preserves the current PDF value."
        ),
    )
    parser.add_argument(
        "--water-shell-gamma-scale",
        type=float,
        default=1.0,
        help="Diagnostic multiplier on generated water-shell gamma_fsi. Default preserves generated parameters.",
    )
    parser.add_argument(
        "--bouncer-mode",
        choices=["on", "off"],
        default="on",
        help="Diagnostic membrane bouncer toggle. Default preserves the current model.",
    )
    parser.add_argument(
        "--water-belonging-correct-every",
        type=int,
        default=0,
        help=(
            "Diagnostic Mirheo object-belonging correction cadence for the exterior water PV. "
            "0 preserves the original initial split only. Positive values periodically remove "
            "water particles that appear inside the membrane."
        ),
    )
    parser.add_argument(
        "--gas-belonging-correct-every",
        type=int,
        default=0,
        help=(
            "Diagnostic Mirheo object-belonging correction cadence for the gas split. "
            "0 preserves the original initial split only. Positive values periodically move "
            "particles between the inside gas PV and the outside source PV."
        ),
    )
    parser.add_argument(
        "--membrane-mass-scale",
        type=float,
        default=1.0,
        help=(
            "Explicit diagnostic multiplier on generated membrane vertex mass. "
            "Default 1.0 preserves the no-x5 breathing mass. Non-1 values are "
            "recorded in parameters00001.yaml, setup_manifest.json, and "
            "membrane_mass_parameters.json."
        ),
    )
    parser.add_argument(
        "--primary-observable",
        choices=["rms_radius_dpd", "mean_radius_dpd", "equivalent_radius_dpd", "volume_dpd3"],
        default="rms_radius_dpd",
    )
    parser.add_argument("--box-padding-dpd", type=float, default=8.0)
    parser.add_argument(
        "--ka-override",
        type=float,
        default=None,
        help="Diagnostic non-MAP membrane ka override; omit to use MAP ka.",
    )
    parser.add_argument(
        "--kb-override",
        type=float,
        default=None,
        help="Diagnostic non-MAP membrane kb override; omit to use MAP kb.",
    )
    parser.add_argument(
        "--lim-mu-policy",
        choices=["legacy-generated", "derive-from-ka"],
        default="legacy-generated",
        help=(
            "How Lim mu is supplied when ka changes. legacy-generated preserves the generated YAML; "
            "derive-from-ka uses mu=ka*(1-nu)/(1+nu), matching the EMB generators' shared-Yt relation."
        ),
    )
    parser.add_argument("--transient-cut-fraction", type=float, default=0.0)
    parser.add_argument(
        "--fit-start-dpd",
        type=float,
        default=0.0,
        help="Start time, in DPD units after release, for the frequency fit window.",
    )
    parser.add_argument(
        "--fit-end-dpd",
        type=float,
        default=0.25,
        help="End time, in DPD units after release, for the frequency fit window; use a negative value for no upper bound.",
    )
    parser.add_argument(
        "--pin-com",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Register Mirheo PinObject with zero translational velocity and unrestricted rotation.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Mirheo checkpoint cadence in steps; 0 disables checkpoints for measurement-only runs.",
    )
    parser.add_argument(
        "--restart-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Diagnostic restart folder for measurement-window protocol switches. "
            "The simulation is rebuilt with the requested measurement interactions, "
            "then restarted from this checkpoint before excitation and dumping."
        ),
    )
    parser.add_argument(
        "--particle-checker-every",
        type=int,
        default=0,
        help=(
            "Debug-only Mirheo particle checker cadence. 0 disables it. "
            "Use only for diagnostics, not production breathing runs."
        ),
    )
    parser.add_argument(
        "--dump-fluid-particles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic-only dump of water/gas particle vectors for crossing/contact audits. "
            "Disabled by default to protect storage quota."
        ),
    )
    parser.add_argument(
        "--fluid-dump-every-steps",
        type=int,
        default=0,
        help=(
            "Diagnostic water/gas dump cadence in Mirheo steps. "
            "Nonpositive values reuse --sample-every when --dump-fluid-particles is enabled."
        ),
    )
    parser.add_argument(
        "--fluid-audit-every-steps",
        type=int,
        default=0,
        help=(
            "Storage-light diagnostic cadence for aggregate water/gas contact counts. "
            "0 disables it. Unlike --dump-fluid-particles, this writes only CSV/JSON summaries."
        ),
    )
    parser.add_argument(
        "--fluid-audit-margin-dpd",
        type=float,
        default=0.1,
        help="Distance margin around the membrane RMS radius for fluid contact/crossing counts.",
    )
    parser.add_argument(
        "--dump-fluid-density-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic-only coarse density-grid dump using Mirheo createDumpAverageRelative. "
            "This writes binned number_densities, not full fluid particle HDF5 trajectories."
        ),
    )
    parser.add_argument("--fluid-density-sample-every-steps", type=int, default=0)
    parser.add_argument("--fluid-density-dump-every-steps", type=int, default=0)
    parser.add_argument("--fluid-density-bin-size-dpd", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bubbles = load_bubbles(args.map_values)
    selected = bubbles if args.all else select_bubbles(bubbles, args.symbols)
    seed_indices = _coerce_seed_indices(args.seed_indices)
    fit_start_dpd = None if args.fit_start_dpd is None else float(args.fit_start_dpd)
    fit_end_dpd = None if args.fit_end_dpd is None or float(args.fit_end_dpd) < 0.0 else float(args.fit_end_dpd)
    if fit_start_dpd is not None and fit_end_dpd is not None and fit_end_dpd <= fit_start_dpd:
        raise ValueError("--fit-end-dpd must be greater than --fit-start-dpd, or negative for no upper bound.")
    if float(args.membrane_mass_scale) <= 0.0:
        raise ValueError("--membrane-mass-scale must be positive.")
    if int(args.post_deflation_ramp_steps) < 0:
        raise ValueError("--post-deflation-ramp-steps must be nonnegative.")
    if int(args.post_deflation_ramp_steps) > 0 and args.prestrain_placement != "post-equilibration":
        raise ValueError(
            "--post-deflation-ramp-steps requires --prestrain-placement post-equilibration."
        )
    args.run_root.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        aggregate(args.run_root, selected)
        return 0

    if args.postprocess_only:
        if args.trajectory_capture == "memory":
            raise ValueError("--postprocess-only is unavailable for memory trajectory capture; rerun the simulation.")
        for bubble in selected:
            resolved_equil_steps, resolved_relax_steps, _ = _resolve_protocol_steps(
                bubble,
                equil_steps=args.equil_steps,
                relax_steps=args.relax_steps,
            )
            roots = []
            for symbol_root in (args.run_root / bubble.symbol, args.run_root / bubble.symbol / bubble.symbol):
                roots.extend(symbol_root / _seed_label(seed_index) for seed_index in seed_indices)
            roots = [root for root in roots if (root / "simulation").is_dir()] or [args.run_root / bubble.symbol]
            for root in roots:
                seed_index = None
                if root.name.startswith("seed-"):
                    seed_index = int(root.name.split("-", maxsplit=1)[1])
                particle_dump_dir = (
                    root / "simulation" / "particles"
                    if args.particle_dump_root is None
                    else args.particle_dump_root.resolve()
                    / bubble.symbol
                    / _seed_label(0 if seed_index is None else seed_index)
                )
                postprocess_symbol(
                    bubble,
                    root,
                    seed_index=seed_index,
                    dt=args.dt,
                    sample_every=args.sample_every,
                    pulse_steps=args.pulse_steps,
                    relax_steps=resolved_relax_steps,
                    transient_cut_fraction=args.transient_cut_fraction,
                    fit_start_dpd=fit_start_dpd,
                    fit_end_dpd=fit_end_dpd,
                    primary_observable=args.primary_observable,
                    particle_dump_dir=particle_dump_dir,
                )
        aggregate(args.run_root, selected)
        return 0

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise RuntimeError(
            "This protocol requires at least 2 MPI ranks so Mirheo postprocess dump "
            "plugins write membrane HDF5 trajectories. Use MPI_RANKS=2 or larger."
        )

    for bubble in selected:
        resolved_equil_steps, resolved_relax_steps, step_resolution = _resolve_protocol_steps(
            bubble,
            equil_steps=args.equil_steps,
            relax_steps=args.relax_steps,
        )
        for seed_index in seed_indices:
            symbol_root = args.run_root / bubble.symbol / _seed_label(seed_index)
            result_path = symbol_root / "result.json"
            sim_dir = symbol_root / "simulation"
            particle_dump_dir = (
                sim_dir / "particles"
                if args.particle_dump_root is None
                else args.particle_dump_root.resolve()
                / bubble.symbol
                / _seed_label(seed_index)
            )
            if rank == 0:
                if args.force_prepare and symbol_root.exists():
                    shutil.rmtree(symbol_root)
                symbol_root.mkdir(parents=True, exist_ok=True)
            comm.Barrier()
            skip = bool(args.resume and result_path.exists())
            skip = comm.bcast(skip, root=0)
            if skip:
                if rank == 0:
                    print(f"[emb_breathing] {bubble.symbol} seed={seed_index}: reusing {result_path}", flush=True)
                continue
            if rank == 0:
                if sim_dir.exists():
                    shutil.rmtree(sim_dir)
                if args.particle_dump_root is not None:
                    if particle_dump_dir.exists():
                        shutil.rmtree(particle_dump_dir)
                    particle_dump_dir.mkdir(parents=True, exist_ok=True)
                sim_dir = prepare_simulation(
                    bubble,
                    symbol_root,
                    dt=args.dt,
                    equil_steps=resolved_equil_steps,
                    pulse_steps=args.pulse_steps,
                    relax_steps=resolved_relax_steps,
                    sample_every=args.sample_every,
                    box_padding_dpd=args.box_padding_dpd,
                    membrane_mass_scale=float(args.membrane_mass_scale),
                    ka_override=args.ka_override,
                    kb_override=args.kb_override,
                    lim_mu_policy=args.lim_mu_policy,
                    force=False,
                )
                setup_manifest = {
                    "bubble": bubble.__dict__,
                    "simulation_dir": str(sim_dir),
                    "seed_index": int(seed_index),
                    "seed_label": _seed_label(seed_index),
                    "rng_seed_control": "replicate_index_recorded_no_explicit_mirheo_seed_api_found",
                    "step_resolution": step_resolution,
                    "protocol": {
                        "dt": args.dt,
                        "equil_steps": resolved_equil_steps,
                        "requested_equil_steps": args.equil_steps,
                        "pulse_steps": args.pulse_steps,
                        "relax_steps": resolved_relax_steps,
                        "requested_relax_steps": args.relax_steps,
                        "sample_every": args.sample_every,
                        "trajectory_capture": args.trajectory_capture,
                        "particle_dump_root": (
                            None
                            if args.particle_dump_root is None
                            else str(args.particle_dump_root.resolve())
                        ),
                        "particle_dump_directory": str(particle_dump_dir),
                        "pulse_force_per_vertex": args.pulse_force_per_vertex,
                        "excitation_mode": args.excitation_mode,
                        "initial_radius_scale": args.initial_radius_scale,
                        "radial_velocity_kick": args.radial_velocity_kick,
                        "radial_velocity_kick_mode": args.radial_velocity_kick_mode,
                        "velocity_shape_json": (
                            None if args.velocity_shape_json is None else str(args.velocity_shape_json)
                        ),
                        "shape_displacement_amplitude_dpd": args.shape_displacement_amplitude_dpd,
                        "shape_displacement_projection": args.shape_displacement_projection,
                        "drive_frequency_mhz": args.drive_frequency_mhz,
                        "drive_force_per_vertex": args.drive_force_per_vertex,
                        "drive_update_every_steps": args.drive_update_every_steps,
                        "tone_burst_cycles": args.tone_burst_cycles,
                        "tone_burst_ramp_cycles": args.tone_burst_ramp_cycles,
                        "radius_restraint_target_scale": args.radius_restraint_target_scale,
                        "radius_restraint_force_constant": args.radius_restraint_force_constant,
                        "radius_restraint_max_force_per_vertex": args.radius_restraint_max_force_per_vertex,
                        "radius_restraint_hold_steps": args.radius_restraint_hold_steps,
                        "radius_restraint_update_every_steps": args.radius_restraint_update_every_steps,
                        "fluid_kick_water_scale": args.fluid_kick_water_scale,
                        "fluid_kick_gas_scale": args.fluid_kick_gas_scale,
                        "fluid_kick_water_outer_radius_factor": args.fluid_kick_water_outer_radius_factor,
                        "fluid_burst_water_force_per_particle": args.fluid_burst_water_force_per_particle,
                        "fluid_burst_gas_force_per_particle": args.fluid_burst_gas_force_per_particle,
                        "fluid_burst_force_field_grid_spacing": args.fluid_burst_force_field_grid_spacing,
                        "fluid_burst_water_outer_radius_factor": args.fluid_burst_water_outer_radius_factor,
                        "stress_free_radius_scale": args.stress_free_radius_scale,
                        "prestrain_placement": args.prestrain_placement,
                        "post_deflation_ramp_steps": int(args.post_deflation_ramp_steps),
                        "post_deflation_ramp_profile": (
                            "half_cosine_coordinate_ramp"
                            if int(args.post_deflation_ramp_steps) > 0
                            else "disabled_legacy_instantaneous_displacement"
                        ),
                        "post_deflation_hold_steps": int(args.post_deflation_hold_steps),
                        "post_deflation_hold_update_every_steps": int(
                            args.post_deflation_hold_update_every_steps
                        ),
                        "post_deflation_hold_reset_velocities": bool(
                            args.post_deflation_hold_reset_velocities
                        ),
                        "solvent_mode": args.solvent_mode,
                        "water_shell_fsi_scale": args.water_shell_fsi_scale,
                        "water_shell_gamma_scale": args.water_shell_gamma_scale,
                        "bouncer_mode": args.bouncer_mode,
                        "water_belonging_correct_every": max(0, int(args.water_belonging_correct_every)),
                        "gas_belonging_correct_every": max(0, int(args.gas_belonging_correct_every)),
                        "belonging_correction_policy": (
                            "diagnostic_periodic_object_belonging_correction"
                            if max(0, int(args.water_belonging_correct_every)) > 0
                            or max(0, int(args.gas_belonging_correct_every)) > 0
                            else "initial_split_only_original_protocol"
                        ),
                        "membrane_mass_scale": float(args.membrane_mass_scale),
                        "membrane_mass_scale_policy": (
                            "generated_no_x5_mass"
                            if math.isclose(float(args.membrane_mass_scale), 1.0)
                            else "explicit_diagnostic_inertia_scale"
                        ),
                        "primary_observable": args.primary_observable,
                        "box_padding_dpd": args.box_padding_dpd,
                        "transient_cut_fraction": args.transient_cut_fraction,
                        "fit_start_dpd": fit_start_dpd,
                        "fit_end_dpd": fit_end_dpd,
                        "pin_com": bool(args.pin_com),
                        "checkpoint_every": int(args.checkpoint_every),
                        "restart_from_checkpoint": (
                            None if args.restart_from_checkpoint is None else str(args.restart_from_checkpoint)
                        ),
                        "particle_checker_every": int(args.particle_checker_every),
                        "particle_checker_policy": (
                            "debug_only_disabled_for_production"
                            if int(args.particle_checker_every) <= 0
                            else "debug_only_mirheo_createParticleChecker"
                        ),
                        "dump_fluid_particles": bool(args.dump_fluid_particles),
                        "fluid_dump_every_steps": int(args.fluid_dump_every_steps),
                        "fluid_dump_policy": (
                            "diagnostic_fluid_particle_dump_enabled"
                            if bool(args.dump_fluid_particles)
                            else "disabled_for_quota"
                        ),
                        "fluid_audit_every_steps": int(args.fluid_audit_every_steps),
                        "fluid_audit_margin_dpd": float(args.fluid_audit_margin_dpd),
                        "fluid_audit_policy": (
                            "storage_light_aggregate_contact_counts_enabled"
                            if int(args.fluid_audit_every_steps) > 0
                            else "disabled"
                        ),
                        "dump_fluid_density_grid": bool(args.dump_fluid_density_grid),
                        "fluid_density_sample_every_steps": int(args.fluid_density_sample_every_steps),
                        "fluid_density_dump_every_steps": int(args.fluid_density_dump_every_steps),
                        "fluid_density_bin_size_dpd": float(args.fluid_density_bin_size_dpd),
                        "fluid_density_grid_policy": (
                            "diagnostic_coarse_density_grid_enabled"
                            if bool(args.dump_fluid_density_grid)
                            else "disabled"
                        ),
                        "water_shell_fsi_conservative_scale": float(args.water_shell_fsi_scale),
                        "water_shell_fsi_conservative_policy": (
                            "default_pdf_value_afsi_0p4_aii"
                            if math.isclose(float(args.water_shell_fsi_scale), WATER_SHELL_FSI_CONSERVATIVE_SCALE)
                            else "diagnostic_override"
                        ),
                        "ka_override": args.ka_override,
                        "kb_override": args.kb_override,
                        "lim_mu_policy": args.lim_mu_policy,
                        "elastic_override_policy": (
                            "diagnostic_non_map_override"
                            if args.ka_override is not None or args.kb_override is not None
                            else "map_values"
                        ),
                    },
                    "environment": {
                        "hostname": os.uname().nodename,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
                        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
                        "mesouq_site": os.environ.get("MESOUQ_SITE"),
                        "mesouq_site_runtime_root": os.environ.get("MESOUQ_SITE_RUNTIME_ROOT"),
                        "python": sys.executable,
                    },
                }
                (symbol_root / "setup_manifest.json").write_text(
                    json.dumps(setup_manifest, indent=2), encoding="utf-8"
                )
            comm.Barrier()
            os.environ["MESOUQ_EMB_PROTOCOL_SEED_INDEX"] = str(seed_index)
            captured_trajectory = run_simulation(
                bubble,
                sim_dir,
                dt=args.dt,
                equil_steps=resolved_equil_steps,
                pulse_steps=args.pulse_steps,
                relax_steps=resolved_relax_steps,
                sample_every=args.sample_every,
                trajectory_capture=args.trajectory_capture,
                particle_dump_dir=particle_dump_dir,
                pulse_force_per_vertex=args.pulse_force_per_vertex,
                excitation_mode=args.excitation_mode,
                initial_radius_scale=args.initial_radius_scale,
                radial_velocity_kick=args.radial_velocity_kick,
                radial_velocity_kick_mode=args.radial_velocity_kick_mode,
                velocity_shape_json=args.velocity_shape_json,
                shape_displacement_amplitude_dpd=args.shape_displacement_amplitude_dpd,
                shape_displacement_projection=args.shape_displacement_projection,
                drive_frequency_mhz=args.drive_frequency_mhz,
                drive_force_per_vertex=args.drive_force_per_vertex,
                drive_update_every_steps=args.drive_update_every_steps,
                tone_burst_cycles=args.tone_burst_cycles,
                tone_burst_ramp_cycles=args.tone_burst_ramp_cycles,
                radius_restraint_target_scale=args.radius_restraint_target_scale,
                radius_restraint_force_constant=args.radius_restraint_force_constant,
                radius_restraint_max_force_per_vertex=args.radius_restraint_max_force_per_vertex,
                radius_restraint_hold_steps=args.radius_restraint_hold_steps,
                radius_restraint_update_every_steps=args.radius_restraint_update_every_steps,
                fluid_kick_water_scale=args.fluid_kick_water_scale,
                fluid_kick_gas_scale=args.fluid_kick_gas_scale,
                fluid_kick_water_outer_radius_factor=args.fluid_kick_water_outer_radius_factor,
                fluid_burst_water_force_per_particle=args.fluid_burst_water_force_per_particle,
                fluid_burst_gas_force_per_particle=args.fluid_burst_gas_force_per_particle,
                fluid_burst_force_field_grid_spacing=args.fluid_burst_force_field_grid_spacing,
                fluid_burst_water_outer_radius_factor=args.fluid_burst_water_outer_radius_factor,
                stress_free_radius_scale=args.stress_free_radius_scale,
                prestrain_placement=args.prestrain_placement,
                post_deflation_ramp_steps=int(args.post_deflation_ramp_steps),
                post_deflation_hold_steps=int(args.post_deflation_hold_steps),
                post_deflation_hold_update_every_steps=int(args.post_deflation_hold_update_every_steps),
                post_deflation_hold_reset_velocities=bool(args.post_deflation_hold_reset_velocities),
                solvent_mode=args.solvent_mode,
                water_shell_fsi_scale=float(args.water_shell_fsi_scale),
                water_shell_gamma_scale=float(args.water_shell_gamma_scale),
                bouncer_mode=args.bouncer_mode,
                water_belonging_correct_every=int(args.water_belonging_correct_every),
                gas_belonging_correct_every=int(args.gas_belonging_correct_every),
                membrane_mass_scale=float(args.membrane_mass_scale),
                pin_com=bool(args.pin_com),
                checkpoint_every=int(args.checkpoint_every),
                restart_from_checkpoint=args.restart_from_checkpoint,
                particle_checker_every=int(args.particle_checker_every),
                dump_fluid_particles=bool(args.dump_fluid_particles),
                fluid_dump_every_steps=int(args.fluid_dump_every_steps),
                fluid_audit_every_steps=int(args.fluid_audit_every_steps),
                fluid_audit_margin_dpd=float(args.fluid_audit_margin_dpd),
                dump_fluid_density_grid=bool(args.dump_fluid_density_grid),
                fluid_density_sample_every_steps=int(args.fluid_density_sample_every_steps),
                fluid_density_dump_every_steps=int(args.fluid_density_dump_every_steps),
                fluid_density_bin_size_dpd=float(args.fluid_density_bin_size_dpd),
                comm=comm,
            )
            comm.Barrier()
            if rank == 0:
                postprocess_symbol(
                    bubble,
                    symbol_root,
                    seed_index=seed_index,
                    dt=args.dt,
                    sample_every=args.sample_every,
                    pulse_steps=args.pulse_steps,
                    relax_steps=resolved_relax_steps,
                    transient_cut_fraction=args.transient_cut_fraction,
                    fit_start_dpd=fit_start_dpd,
                    fit_end_dpd=fit_end_dpd,
                    primary_observable=args.primary_observable,
                    captured_frames=(
                        None if captured_trajectory is None else captured_trajectory.frames
                    ),
                    captured_frame_steps=(
                        None
                        if captured_trajectory is None
                        else captured_trajectory.steps_after_release
                    ),
                    trajectory_capture=args.trajectory_capture,
                    particle_dump_dir=particle_dump_dir,
                )
    if rank == 0:
        aggregate(args.run_root, selected)
    comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
