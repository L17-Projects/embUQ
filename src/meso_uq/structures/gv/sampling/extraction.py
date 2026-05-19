from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .failures import GVSamplingExtractionError
from .types import GVMaterialGeometry, GVSweep


def extract_sampling_channels(
    *,
    experiment: str,
    work_dir: str | Path,
    controls: Mapping[str, float],
    sweep: GVSweep,
    geometry: GVMaterialGeometry,
) -> dict[str, np.ndarray]:
    """Extract finite in-memory channels from one completed GV Mirheo run."""

    root = Path(work_dir)
    if experiment == "stretching":
        channels = _extract_stretching(root, controls=controls, geometry=geometry)
    elif experiment == "buckling":
        channels = _extract_buckling(root, controls=controls)
    elif experiment == "torsion":
        channels = _extract_torsion(root, controls=controls, geometry=geometry)
    elif experiment == "eigenmodes":
        channels = _extract_eigenmodes(root)
    else:
        raise GVSamplingExtractionError(f"Unsupported GV sampling extraction experiment: {experiment!r}.")
    channels.setdefault(sweep.axis, _control_channel(controls, sweep.axis, _channel_length(channels)))
    return _validate_finite_channels(channels)


def merge_sampling_channels(
    channel_sets: tuple[Mapping[str, np.ndarray], ...],
) -> dict[str, np.ndarray]:
    """Merge per-control-point channel payloads into one aggregate sweep payload."""

    if not channel_sets:
        raise GVSamplingExtractionError("At least one channel payload is required.")
    expected_names = {str(name) for name in channel_sets[0]}
    for index, channels in enumerate(channel_sets[1:], start=1):
        names = {str(name) for name in channels}
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing channels {missing}")
            if extra:
                details.append(f"extra channels {extra}")
            raise GVSamplingExtractionError(
                f"Channel payload {index} is inconsistent across sweep values: "
                + "; ".join(details)
                + "."
            )
    names = sorted(expected_names)
    merged: dict[str, np.ndarray] = {}
    for name in names:
        arrays = [np.asarray(channels[name]) for channels in channel_sets]
        merged[name] = _concat_compatible(name, arrays)
    return _validate_finite_channels(merged)


def _extract_stretching(
    root: Path,
    *,
    controls: Mapping[str, float],
    geometry: GVMaterialGeometry,
) -> dict[str, np.ndarray]:
    forward_channels = _extract_stretching_forward_sweep(root, geometry=geometry)
    if forward_channels is not None:
        return forward_channels

    force_path = root / "force" / "emb.csv"
    columns = _read_csv_columns(force_path)
    for required in ("time", "fx", "fy", "fz"):
        if required not in columns:
            raise GVSamplingExtractionError(f"Stretching force CSV is missing column {required!r}: {force_path}")
    force = np.sqrt(columns["fx"] ** 2 + columns["fy"] ** 2 + columns["fz"] ** 2)
    metrics = _stretching_paper_metrics(root, geometry=geometry)
    nominal_force = float(controls["tot_force"])
    channels = {
        "force": np.asarray([nominal_force], dtype=float),
        "force_mean": np.asarray([float(np.mean(force))], dtype=float),
        "force_std": np.asarray([float(np.std(force))], dtype=float),
        "tot_force": _control_channel(controls, "tot_force", 1),
        "displacement": np.asarray([metrics["mean_length"] - metrics["reference_length"]], dtype=float),
        "mean_length": np.asarray([metrics["mean_length"]], dtype=float),
        "std_length": np.asarray([metrics["std_length"]], dtype=float),
        "mean_radius": np.asarray([metrics["mean_radius"]], dtype=float),
        "std_radius": np.asarray([metrics["std_radius"]], dtype=float),
        "reference_length": np.asarray([metrics["reference_length"]], dtype=float),
        "reference_radius": np.asarray([metrics["reference_radius"]], dtype=float),
    }
    if "bpress" in controls:
        channels["bpress"] = _control_channel(controls, "bpress", 1)
    return channels


def _extract_stretching_forward_sweep(
    root: Path,
    *,
    geometry: GVMaterialGeometry,
) -> dict[str, np.ndarray] | None:
    production_folders = _stretching_production_folders(root)
    if len(production_folders) <= 1:
        return None

    reference_path = _first_existing(
        root / "trj_eq" / "sim00001eq" / "emb_0000000.xyz",
        production_folders[0] / "emb_0000000.xyz",
    )
    reference = _read_xyz_positions(reference_path)
    top_indices, bottom_indices = _stretching_anchor_indices(
        reference,
        height=geometry.height,
        fraction=0.3,
    )
    radius_indices = _stretching_center_indices(
        reference,
        height=geometry.height,
        fraction=0.7,
    )
    reference_length = _stretching_anchor_distance(reference, top_indices, bottom_indices)
    reference_radius = _stretching_mean_radius(reference, radius_indices)

    tot_force: list[float] = []
    bpress: list[float] = []
    mean_length: list[float] = []
    std_length: list[float] = []
    mean_radius: list[float] = []
    std_radius: list[float] = []

    for folder in production_folders:
        sim = folder.name.removeprefix("sim")
        defaults_path = root / "parameter" / f"parameters-default{sim}.yaml"
        tot_force.append(_read_yaml_scalar(defaults_path, "tot_force", default=np.nan))
        bpress.append(_read_yaml_scalar(defaults_path, "bpress", default=np.nan))
        metrics = _stretching_folder_metrics(
            folder,
            top_indices=top_indices,
            bottom_indices=bottom_indices,
            radius_indices=radius_indices,
        )
        mean_length.append(metrics["mean_length"])
        std_length.append(metrics["std_length"])
        mean_radius.append(metrics["mean_radius"])
        std_radius.append(metrics["std_radius"])

    length_array = np.asarray(mean_length, dtype=float)
    return {
        "tot_force": np.asarray(tot_force, dtype=float),
        "force": np.asarray(tot_force, dtype=float),
        "bpress": np.asarray(bpress, dtype=float),
        "displacement": length_array - float(length_array[0]),
        "mean_length": length_array,
        "std_length": np.asarray(std_length, dtype=float),
        "mean_radius": np.asarray(mean_radius, dtype=float),
        "std_radius": np.asarray(std_radius, dtype=float),
        "reference_length": np.full(length_array.shape, reference_length, dtype=float),
        "reference_radius": np.full(length_array.shape, reference_radius, dtype=float),
    }


def _extract_buckling(root: Path, *, controls: Mapping[str, float]) -> dict[str, np.ndarray]:
    forward_channels = _extract_buckling_forward_sweep(root)
    if forward_channels is not None:
        return forward_channels

    initial, faces = _read_initial_buckling_mesh(root)
    final = _read_final_buckling_vertices(root)
    if initial.shape != final.shape:
        raise GVSamplingExtractionError(
            f"Buckling initial/final vertex shapes differ: {initial.shape} != {final.shape}."
        )
    initial_centered = initial - np.mean(initial, axis=0)
    final_centered = final - np.mean(final, axis=0)
    delta = final_centered - initial_centered
    deformation = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    radial = np.linalg.norm(final_centered[:, :2], axis=1)
    channels = {
        "buck": _control_channel(controls, "buck", 1),
        "deformation_amplitude": np.asarray([deformation], dtype=float),
        "shape_amplitude": np.asarray([float(np.std(radial))], dtype=float),
    }
    if faces.size:
        initial_volume = _mesh_volume(initial, faces)
        if initial_volume <= 0.0:
            raise GVSamplingExtractionError("Buckling initial mesh volume must be positive.")
        volume_stats = _buckling_trajectory_volume_stats(root, faces, expected_shape=initial.shape)
        if volume_stats is None:
            final_volume = _mesh_volume(final, faces)
            mean_volume = final_volume
            std_volume = 0.0
            analyzed_frame_count = 1
        else:
            mean_volume = volume_stats["mean_volume"]
            std_volume = volume_stats["std_volume"]
            analyzed_frame_count = int(volume_stats["analyzed_frame_count"])
        channels["initial_volume"] = np.asarray([initial_volume], dtype=float)
        channels["mean_volume"] = np.asarray([mean_volume], dtype=float)
        channels["std_volume"] = np.asarray([std_volume], dtype=float)
        channels["analyzed_volume_frame_count"] = np.asarray([analyzed_frame_count], dtype=float)
        channels["relative_volume"] = np.asarray([mean_volume / initial_volume], dtype=float)
        channels["relative_volume_std"] = np.asarray([std_volume / initial_volume], dtype=float)
    if "bpress" in controls:
        channels["bpress"] = _control_channel(controls, "bpress", 1)
    return channels


def _extract_buckling_forward_sweep(root: Path) -> dict[str, np.ndarray] | None:
    production_folders = _stretching_production_folders(root)
    if len(production_folders) <= 1:
        return None

    initial, faces = _read_initial_buckling_mesh(root)
    if faces.size == 0:
        raise GVSamplingExtractionError("Buckling forward sweep requires mesh faces to compute volume.")
    initial_centered = initial - np.mean(initial, axis=0)
    initial_volume = _mesh_volume(initial, faces)
    if initial_volume <= 0.0:
        raise GVSamplingExtractionError("Buckling initial mesh volume must be positive.")

    buck: list[float] = []
    bpress: list[float] = []
    mean_volume: list[float] = []
    std_volume: list[float] = []
    analyzed_count: list[float] = []
    deformation: list[float] = []
    shape: list[float] = []

    for folder in production_folders:
        sim = folder.name.removeprefix("sim")
        defaults_path = root / "parameter" / f"parameters-default{sim}.yaml"
        buck.append(_read_yaml_scalar(defaults_path, "buck", default=np.nan))
        bpress.append(_read_yaml_scalar(defaults_path, "bpress", default=np.nan))
        stats = _buckling_folder_volume_stats(folder, faces, expected_shape=initial.shape)
        mean_volume.append(stats["mean_volume"])
        std_volume.append(stats["std_volume"])
        analyzed_count.append(stats["analyzed_volume_frame_count"])
        final = _read_xyz_positions(stats["final_frame"])
        final_centered = final - np.mean(final, axis=0)
        delta = final_centered - initial_centered
        deformation.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
        radial = np.linalg.norm(final_centered[:, :2], axis=1)
        shape.append(float(np.std(radial)))

    buck_array = np.asarray(buck, dtype=float)
    mean_array = np.asarray(mean_volume, dtype=float)
    std_array = np.asarray(std_volume, dtype=float)
    reference_volume = _buckling_measured_reference_volume(mean_array, buck_values=buck_array)
    return {
        "buck": buck_array,
        "bpress": np.asarray(bpress, dtype=float),
        "deformation_amplitude": np.asarray(deformation, dtype=float),
        "shape_amplitude": np.asarray(shape, dtype=float),
        "initial_volume": np.full(mean_array.shape, initial_volume, dtype=float),
        "reference_volume": np.full(mean_array.shape, reference_volume, dtype=float),
        "mean_volume": mean_array,
        "std_volume": std_array,
        "analyzed_volume_frame_count": np.asarray(analyzed_count, dtype=float),
        "relative_volume": mean_array / reference_volume,
        "relative_volume_std": std_array / reference_volume,
    }


def _buckling_measured_reference_volume(
    mean_volume: np.ndarray,
    *,
    buck_values: np.ndarray | None = None,
) -> float:
    if mean_volume.ndim != 1 or mean_volume.size == 0:
        raise GVSamplingExtractionError("Buckling measured volume reference requires a non-empty 1D mean_volume array.")
    if buck_values is not None:
        if buck_values.shape != mean_volume.shape:
            raise GVSamplingExtractionError("Buckling buck and mean_volume arrays must share a shape.")
        zero_indices = np.flatnonzero(np.isclose(buck_values, 0.0, rtol=0.0, atol=1.0e-12))
        reference_index = int(zero_indices[0]) if zero_indices.size else 0
    else:
        reference_index = 0
    reference_volume = float(mean_volume[reference_index])
    if reference_volume <= 0.0:
        raise GVSamplingExtractionError("Buckling measured reference volume must be positive.")
    return reference_volume


def _extract_torsion(
    root: Path,
    *,
    controls: Mapping[str, float],
    geometry: GVMaterialGeometry,
) -> dict[str, np.ndarray]:
    forward_channels = _extract_torsion_forward_sweep(root, geometry=geometry)
    if forward_channels is not None:
        return forward_channels

    anchor_channels = _extract_torsion_anchor_channels(root, controls=controls, geometry=geometry)
    if anchor_channels is not None:
        return anchor_channels

    force_path = root / "force" / "emb.csv"
    columns = _read_csv_columns(force_path)
    for required in ("time", "fx", "fy", "fz"):
        if required not in columns:
            raise GVSamplingExtractionError(f"Torsion force CSV is missing column {required!r}: {force_path}")
    force = np.sqrt(columns["fx"] ** 2 + columns["fy"] ** 2 + columns["fz"] ** 2)
    area = 2.0 * np.pi * geometry.radGV * geometry.height
    theta = float(controls["theta"])
    gamma = geometry.radGV * theta / geometry.height
    count = int(force.size)
    return {
        "time": columns["time"],
        "theta": np.full(count, theta, dtype=float),
        "gamma": np.full(count, gamma, dtype=float),
        "force": force,
        "sigma_phi_r": force / area,
    }


def _extract_torsion_forward_sweep(
    root: Path,
    *,
    geometry: GVMaterialGeometry,
) -> dict[str, np.ndarray] | None:
    mesh_path = _optional_existing(root / "mesh" / "gv00001.off", root / "gas_vesicle" / "gv.off")
    if mesh_path is None:
        return None
    anchor_pairs = _torsion_anchor_pairs(root)
    if len(anchor_pairs) <= 1:
        return None

    from .torsion import reconstruct_torsion_paper_channels

    vertices, _ = _read_off_mesh(mesh_path)
    theta_values: list[float] = []
    gamma: list[float] = []
    sigma_phi_r: list[float] = []
    sigma_std: list[float] = []
    for sim, anchor_min_path, anchor_max_path in anchor_pairs:
        defaults_path = root / "parameter" / f"parameters-default{sim}.yaml"
        theta = _read_yaml_scalar(defaults_path, "theta", default=np.nan)
        channels = reconstruct_torsion_paper_channels(
            {
                "controls": {"theta": theta},
                "geometry": {"radius": geometry.radGV, "height": geometry.height},
                "channels": {
                    "mesh_vertices": vertices,
                    "constrained_vertex_forces_min": _read_anchor_force_csv(anchor_min_path),
                    "constrained_vertex_forces_max": _read_anchor_force_csv(anchor_max_path),
                },
            },
            geometry={"radius": geometry.radGV, "height": geometry.height},
            controls={"theta": theta},
        )
        theta_values.append(theta)
        gamma.append(float(np.asarray(channels["gamma"], dtype=float).reshape(-1)[0]))
        sigma_phi_r.append(float(np.asarray(channels["sigma_phi_r"], dtype=float).reshape(-1)[0]))
        sigma_std.append(float(np.asarray(channels["sigma_std"], dtype=float).reshape(-1)[0]))
    return {
        "theta": np.asarray(theta_values, dtype=float),
        "gamma": np.asarray(gamma, dtype=float),
        "sigma_phi_r": np.asarray(sigma_phi_r, dtype=float),
        "sigma_std": np.asarray(sigma_std, dtype=float),
    }


def _extract_eigenmodes(root: Path, *, mode_count: int = 30) -> dict[str, np.ndarray]:
    output_dir = root / "analysis" / "output"
    raw_eigenvalue_path = _optional_existing(output_dir / "eigvalues.txt", root / "eigvalues.txt")
    mode_window_manifest_path = _optional_existing(output_dir / "mode_window_manifest.json", root / "mode_window_manifest.json")
    mode_window_manifest = _read_json_mapping(mode_window_manifest_path) if mode_window_manifest_path is not None else {}
    eigenvalue_path = _first_existing(
        output_dir / "eigvalues_new.txt",
        output_dir / "eigvalues.txt",
        root / "eigvalues_new.txt",
        root / "eigvalues.txt",
    )
    if not eigenvalue_path.read_text(encoding="utf-8").strip():
        raise GVSamplingExtractionError(f"Eigenmodes file contains no eigenvalues: {eigenvalue_path}")
    eigenvalues = np.atleast_1d(np.loadtxt(eigenvalue_path, dtype=float))
    if eigenvalues.size == 0:
        raise GVSamplingExtractionError(f"Eigenmodes file contains no eigenvalues: {eigenvalue_path}")
    raw_eigenvalues = (
        np.atleast_1d(np.loadtxt(raw_eigenvalue_path, dtype=float))
        if raw_eigenvalue_path is not None
        else eigenvalues
    )
    count = min(int(mode_count), int(eigenvalues.size))
    parameter_path = _optional_existing(root / "parameter" / "parameters00001.yaml", root / "parameters00001.yaml")
    kbt = _read_yaml_scalar(parameter_path, "kbt", default=1.0) if parameter_path is not None else 1.0
    selected_eigenvalues = np.asarray(eigenvalues[:count], dtype=float)
    if np.any(selected_eigenvalues <= 0.0):
        raise GVSamplingExtractionError(f"Eigenmodes eigenvalues must be positive: {eigenvalue_path}")
    channels: dict[str, np.ndarray] = {
        "mode_index": np.arange(count, dtype=float),
        "eigenvalues": selected_eigenvalues,
        "kBT": np.asarray([kbt], dtype=float),
        "eigenfrequencies": np.sqrt(np.sort(kbt / selected_eigenvalues)[:count]),
        "raw_eigenpair_count": np.asarray([float(raw_eigenvalues.size)], dtype=float),
        "final_mode_count": np.asarray([float(count)], dtype=float),
        "final_mode_indices": _mode_window_indices(
            mode_window_manifest,
            key="final_mode_indices",
            count=count,
            default=np.arange(count, dtype=int),
        ).astype(float),
        "selected_paper_mode_indices": _mode_window_indices(
            mode_window_manifest,
            key="selected_paper_mode_indices",
            count=count,
            default=np.arange(count, dtype=int),
        ).astype(float),
    }
    selected_raw_indices = _mode_window_indices(
        mode_window_manifest,
        key="selected_raw_mode_indices",
        count=count,
        default=None,
    )
    if selected_raw_indices is not None:
        channels["selected_raw_mode_indices"] = selected_raw_indices.astype(float)
    mode_min_frequency = mode_window_manifest.get("min_frequency_tau_inv")
    if mode_min_frequency is not None:
        channels["mode_window_min_frequency_tau_inv"] = np.asarray([float(mode_min_frequency)], dtype=float)
    vector_path = _optional_existing(
        output_dir / "eigvectors_new.txt",
        output_dir / "eigvectors.txt",
        root / "eigvectors_new.txt",
        root / "eigvectors.txt",
    )
    if vector_path is not None:
        vectors = np.atleast_1d(np.loadtxt(vector_path, dtype=float))
        if vectors.ndim == 1:
            channels["eigenvectors"] = vectors[:count]
        else:
            channels["eigenvectors"] = vectors[:count, ...]
    reference_path = _optional_existing(
        root / "analysis" / "emb_0000000.xyz",
        output_dir / "ref1.xyz",
        root / "emb_0000000.xyz",
        root / "ref1.xyz",
    )
    if reference_path is not None:
        channels["reference_positions"] = _read_xyz_positions(reference_path)
    mesh_path = _optional_existing(
        root / "mesh" / "gv00001.off",
        root / "gas_vesicle" / "gv.off",
        root / "gv00001.off",
        root / "gv.off",
    )
    if mesh_path is not None:
        _, faces = _read_off_mesh(mesh_path)
        if faces.size:
            channels["mesh_faces"] = faces.astype(float)
    return channels


def _read_json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GVSamplingExtractionError(f"Invalid JSON mode-window manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise GVSamplingExtractionError(f"Mode-window manifest must be a JSON object: {path}")
    return payload


def _mode_window_indices(
    manifest: Mapping[str, object],
    *,
    key: str,
    count: int,
    default: np.ndarray | None,
) -> np.ndarray | None:
    raw_indices = manifest.get(key)
    if raw_indices is None:
        return None if default is None else np.asarray(default, dtype=int)
    indices = np.asarray(raw_indices, dtype=float)
    if indices.ndim != 1 or indices.size < count:
        raise GVSamplingExtractionError(f"Mode-window manifest {key} has invalid shape.")
    if not np.all(np.isfinite(indices)) or not np.all(np.equal(indices, np.floor(indices))):
        raise GVSamplingExtractionError(f"Mode-window manifest {key} must contain finite integer indices.")
    if np.any(indices < 0):
        raise GVSamplingExtractionError(f"Mode-window manifest {key} must contain non-negative indices.")
    return indices[:count].astype(int)


def _stretching_paper_metrics(root: Path, *, geometry: GVMaterialGeometry) -> dict[str, float]:
    reference_path = _first_existing(
        root / "trj_eq" / "sim00001eq" / "emb_0000000.xyz",
        root / "trj_eq" / "sim00001" / "emb_0000000.xyz",
    )
    production_frames = tuple(
        sorted(
            path
            for folder in (root / "trj_eq").glob("sim*")
            if folder.is_dir() and not folder.name.endswith("eq")
            for path in folder.glob("emb_*.xyz")
        )
    )
    if not production_frames:
        raise GVSamplingExtractionError(
            f"Stretching paper replay requires production XYZ frames under {root / 'trj_eq'}."
        )
    if len(production_frames) < 2:
        raise GVSamplingExtractionError(
            f"Stretching displacement requires at least two XYZ frames under {root / 'trj_eq'}."
        )

    reference = _read_xyz_positions(reference_path)
    top_indices, bottom_indices = _stretching_anchor_indices(
        reference,
        height=geometry.height,
        fraction=0.3,
    )
    radius_indices = _stretching_center_indices(
        reference,
        height=geometry.height,
        fraction=0.7,
    )

    lengths: list[float] = []
    radii: list[float] = []
    for frame in production_frames:
        positions = _read_xyz_positions(frame)
        lengths.append(_stretching_anchor_distance(positions, top_indices, bottom_indices))
        radii.append(_stretching_mean_radius(positions, radius_indices))

    start = min(int(0.25 * len(lengths)), max(len(lengths) - 1, 0))
    steady_lengths = np.asarray(lengths[start:], dtype=float)
    steady_radii = np.asarray(radii[start:], dtype=float)
    if steady_lengths.size == 0 or steady_radii.size == 0:
        raise GVSamplingExtractionError("Stretching paper replay found no steady-state frames.")

    return {
        "mean_length": float(np.mean(steady_lengths)),
        "std_length": float(np.std(steady_lengths)),
        "mean_radius": float(np.mean(steady_radii)),
        "std_radius": float(np.std(steady_radii)),
        "reference_length": _stretching_anchor_distance(reference, top_indices, bottom_indices),
        "reference_radius": _stretching_mean_radius(reference, radius_indices),
    }


def _stretching_production_folders(root: Path) -> tuple[Path, ...]:
    trj_root = root / "trj_eq"
    return tuple(
        sorted(
            (
                folder
                for folder in trj_root.glob("sim*")
                if folder.is_dir() and not folder.name.endswith("eq")
            ),
            key=_stretching_sim_sort_key,
        )
    )


def _stretching_sim_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("sim")
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (10**9, path.name)


def _stretching_folder_metrics(
    folder: Path,
    *,
    top_indices: np.ndarray,
    bottom_indices: np.ndarray,
    radius_indices: np.ndarray,
) -> dict[str, float]:
    frames = tuple(sorted(folder.glob("emb_*.xyz")))
    if len(frames) < 2:
        raise GVSamplingExtractionError(
            f"Stretching forward sweep requires at least two XYZ frames in {folder}."
        )
    lengths: list[float] = []
    radii: list[float] = []
    for frame in frames:
        positions = _read_xyz_positions(frame)
        lengths.append(_stretching_anchor_distance(positions, top_indices, bottom_indices))
        radii.append(_stretching_mean_radius(positions, radius_indices))
    start = min(int(0.25 * len(lengths)), max(len(lengths) - 1, 0))
    steady_lengths = np.asarray(lengths[start:], dtype=float)
    steady_radii = np.asarray(radii[start:], dtype=float)
    if steady_lengths.size == 0 or steady_radii.size == 0:
        raise GVSamplingExtractionError(f"Stretching forward sweep found no steady-state frames in {folder}.")
    return {
        "mean_length": float(np.mean(steady_lengths)),
        "std_length": float(np.std(steady_lengths)),
        "mean_radius": float(np.mean(steady_radii)),
        "std_radius": float(np.std(steady_radii)),
    }


def _stretching_rotate_positions(positions: np.ndarray) -> np.ndarray:
    centered = np.asarray(positions, dtype=float) - np.mean(positions, axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    return np.asarray(centered @ eigenvectors, dtype=float)


def _stretching_anchor_indices(
    positions: np.ndarray,
    *,
    height: float,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotated = _stretching_rotate_positions(positions)
    z0 = 0.5 * float(fraction) * float(height)
    tolerance = 0.1
    top_distance = np.abs(rotated[:, 2] - z0)
    bottom_distance = np.abs(rotated[:, 2] + z0)
    top_count = int(np.count_nonzero(top_distance < tolerance))
    bottom_count = int(np.count_nonzero(bottom_distance < tolerance))
    if top_count == 0 or bottom_count == 0:
        raise GVSamplingExtractionError(
            "Stretching paper replay could not reconstruct axial particle bands."
        )
    return (
        np.argsort(top_distance)[:top_count],
        np.argsort(bottom_distance)[:bottom_count],
    )


def _stretching_center_indices(
    positions: np.ndarray,
    *,
    height: float,
    fraction: float,
) -> np.ndarray:
    rotated = _stretching_rotate_positions(positions)
    z_distance = np.abs(rotated[:, 2])
    mask_count = int(np.count_nonzero(z_distance < 0.5 * float(fraction) * float(height)))
    if mask_count == 0:
        raise GVSamplingExtractionError(
            "Stretching paper replay could not reconstruct central radius particles."
        )
    return np.argsort(z_distance)[:mask_count]


def _stretching_anchor_distance(
    positions: np.ndarray,
    top_indices: np.ndarray,
    bottom_indices: np.ndarray,
) -> float:
    rotated = _stretching_rotate_positions(positions)
    top = np.mean(rotated[np.asarray(top_indices, dtype=int)], axis=0)
    bottom = np.mean(rotated[np.asarray(bottom_indices, dtype=int)], axis=0)
    return float(np.linalg.norm(top - bottom))


def _stretching_mean_radius(positions: np.ndarray, center_indices: np.ndarray) -> float:
    rotated = _stretching_rotate_positions(positions)
    indices = np.asarray(center_indices, dtype=int)
    return float(np.mean(np.sqrt(rotated[indices, 0] ** 2 + rotated[indices, 1] ** 2)))


def _extract_torsion_anchor_channels(
    root: Path,
    *,
    controls: Mapping[str, float],
    geometry: GVMaterialGeometry,
) -> dict[str, np.ndarray] | None:
    mesh_path = _optional_existing(root / "mesh" / "gv00001.off", root / "gas_vesicle" / "gv.off")
    anchor_min_path = _optional_existing(root / "anchor_min" / "sim00001" / "emb.csv")
    anchor_max_path = _optional_existing(root / "anchor_max" / "sim00001" / "emb.csv")
    if mesh_path is None and anchor_min_path is None and anchor_max_path is None:
        return None
    if mesh_path is None or anchor_min_path is None or anchor_max_path is None:
        raise GVSamplingExtractionError(
            "Torsion paper replay requires mesh/gv00001.off plus anchor_min and anchor_max force CSVs."
        )
    from .torsion import reconstruct_torsion_paper_channels

    vertices, _ = _read_off_mesh(mesh_path)
    theta = float(controls["theta"])
    channels = reconstruct_torsion_paper_channels(
        {
            "controls": {"theta": theta},
            "geometry": {"radius": geometry.radGV, "height": geometry.height},
            "channels": {
                "mesh_vertices": vertices,
                "constrained_vertex_forces_min": _read_anchor_force_csv(anchor_min_path),
                "constrained_vertex_forces_max": _read_anchor_force_csv(anchor_max_path),
            },
        },
        geometry={"radius": geometry.radGV, "height": geometry.height},
        controls={"theta": theta},
    )
    return {
        "theta": np.asarray([theta], dtype=float),
        **channels,
    }


def _read_anchor_force_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise GVSamplingExtractionError(f"Required anchor force CSV output is missing: {path}")
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    except ValueError as exc:
        raise GVSamplingExtractionError(f"Anchor force CSV {path} has non-numeric fields.") from exc
    data = np.atleast_2d(data)
    if data.size == 0 or data.shape[1] < 4:
        raise GVSamplingExtractionError(f"Anchor force CSV contains no force rows: {path}")
    force_columns = data[:, 1:]
    if force_columns.shape[1] % 3 != 0:
        raise GVSamplingExtractionError(f"Anchor force CSV has an invalid force column count: {path}")
    return force_columns.reshape(force_columns.shape[0], force_columns.shape[1] // 3, 3)


def _read_yaml_scalar(path: Path, key: str, *, default: float) -> float:
    if not path.is_file():
        return float(default)
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GVSamplingExtractionError(f"Could not read YAML scalar {key!r} from {path}.") from exc
    if not isinstance(payload, Mapping) or key not in payload:
        return float(default)
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise GVSamplingExtractionError(f"YAML scalar {key!r} in {path} is not numeric.") from exc
    if not np.isfinite(value):
        raise GVSamplingExtractionError(f"YAML scalar {key!r} in {path} is not finite.")
    return value


def _read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise GVSamplingExtractionError(f"Required Mirheo CSV output is missing: {path}")
    rows: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise GVSamplingExtractionError(f"CSV output has no header: {path}")
        for field in reader.fieldnames:
            rows[str(field)] = []
        for row in reader:
            for field in reader.fieldnames:
                try:
                    rows[str(field)].append(float(row[field]))  # type: ignore[index]
                except (TypeError, ValueError) as exc:
                    raise GVSamplingExtractionError(f"CSV output {path} has non-numeric field {field!r}.") from exc
    if not rows or not any(rows.values()):
        raise GVSamplingExtractionError(f"CSV output contains no rows: {path}")
    return {name: np.asarray(values, dtype=float) for name, values in rows.items()}


def _stretching_displacement(root: Path) -> dict[str, float]:
    candidates = sorted(root.glob("trj_eq/sim*eq/emb_*.xyz")) or sorted(root.glob("trj_eq/sim*/emb_*.xyz"))
    if len(candidates) < 2:
        raise GVSamplingExtractionError(
            f"Stretching displacement requires at least two XYZ frames under {root / 'trj_eq'}."
        )
    first = _read_xyz_positions(candidates[0])
    last = _read_xyz_positions(candidates[-1])
    initial_extent = float(np.max(first[:, 2]) - np.min(first[:, 2]))
    final_extent = float(np.max(last[:, 2]) - np.min(last[:, 2]))
    return {
        "initial_extent": initial_extent,
        "displacement": final_extent - initial_extent,
    }


def _read_initial_buckling_vertices(root: Path) -> np.ndarray:
    return _read_initial_buckling_mesh(root)[0]


def _read_initial_buckling_mesh(root: Path) -> tuple[np.ndarray, np.ndarray]:
    return _read_off_mesh(_first_existing(root / "mesh" / "gv00001.off", root / "gas_vesicle" / "gv.off"))


def _read_final_buckling_vertices(root: Path) -> np.ndarray:
    hdf5_candidates = sorted((root / "restart").glob("emb.PV-*.h5"))
    if hdf5_candidates:
        try:
            return _read_hdf5_dataset(hdf5_candidates[-1], "position")
        except GVSamplingExtractionError:
            raise
        except Exception as exc:
            raise GVSamplingExtractionError(f"Could not read buckling HDF5 positions from {hdf5_candidates[-1]}.") from exc
    xyz_candidates = sorted(root.glob("trj_eq/sim*/emb_*.xyz"))
    if xyz_candidates:
        return _read_xyz_positions(xyz_candidates[-1])
    return _read_xyz_positions(_first_existing(root / "gas_vesicle" / "test.xyz"))


def _buckling_trajectory_volume_stats(
    root: Path,
    faces: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> dict[str, float] | None:
    candidates = sorted(root.glob("trj_eq/sim*/emb_*.xyz"))
    production_frames = [path for path in candidates if not path.parent.name.endswith("eq")]
    if not production_frames:
        return None

    volumes: list[float] = []
    for frame_path in production_frames:
        vertices = _read_xyz_positions(frame_path)
        if vertices.shape != expected_shape:
            raise GVSamplingExtractionError(
                f"Buckling trajectory vertex shape differs from mesh: {vertices.shape} != {expected_shape}."
            )
        volumes.append(_mesh_volume(vertices, faces))

    if not volumes:
        return None
    volume_array = np.asarray(volumes, dtype=float)
    start = int(0.5 * volume_array.size)
    steady = volume_array[start:-1]
    if steady.size == 0:
        steady = volume_array[start:]
    if steady.size == 0:
        steady = volume_array
    return {
        "mean_volume": float(np.mean(steady)),
        "std_volume": float(np.std(steady)),
        "analyzed_frame_count": float(steady.size),
    }


def _buckling_folder_volume_stats(
    folder: Path,
    faces: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> dict[str, float | Path]:
    frames = tuple(sorted(folder.glob("emb_*.xyz")))
    if len(frames) < 2:
        raise GVSamplingExtractionError(
            f"Buckling forward sweep requires at least two XYZ frames in {folder}."
        )
    volumes: list[float] = []
    for frame in frames:
        vertices = _read_xyz_positions(frame)
        if vertices.shape != expected_shape:
            raise GVSamplingExtractionError(
                f"Buckling trajectory vertex shape differs from mesh: {vertices.shape} != {expected_shape}."
            )
        volumes.append(_mesh_volume(vertices, faces))
    volume_array = np.asarray(volumes, dtype=float)
    start = int(0.5 * volume_array.size)
    steady = volume_array[start:-1]
    if steady.size == 0:
        steady = volume_array[start:]
    if steady.size == 0:
        steady = volume_array
    return {
        "mean_volume": float(np.mean(steady)),
        "std_volume": float(np.std(steady)),
        "analyzed_volume_frame_count": float(steady.size),
        "final_frame": frames[-1],
    }


def _torsion_anchor_pairs(root: Path) -> tuple[tuple[str, Path, Path], ...]:
    min_root = root / "anchor_min"
    max_root = root / "anchor_max"
    if not min_root.is_dir() or not max_root.is_dir():
        return ()
    pairs: list[tuple[str, Path, Path]] = []
    for min_folder in sorted(min_root.glob("sim*"), key=_stretching_sim_sort_key):
        if not min_folder.is_dir():
            continue
        sim = min_folder.name.removeprefix("sim")
        max_folder = max_root / min_folder.name
        min_path = min_folder / "emb.csv"
        max_path = max_folder / "emb.csv"
        if min_path.is_file() and max_path.is_file():
            pairs.append((sim, min_path, max_path))
    return tuple(pairs)


def _read_hdf5_dataset(path: Path, dataset_name: str) -> np.ndarray:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise GVSamplingExtractionError(
            f"h5py is required to read Mirheo HDF5 output {path}; install it in the GV runtime environment."
        ) from exc
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise GVSamplingExtractionError(f"HDF5 output {path} is missing dataset {dataset_name!r}.")
        return np.asarray(handle[dataset_name][()], dtype=float)


def _read_xyz_positions(path: Path) -> np.ndarray:
    if not path.is_file():
        raise GVSamplingExtractionError(f"XYZ output is missing: {path}")
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not rows:
        raise GVSamplingExtractionError(f"XYZ output contains no particle positions: {path}")
    return np.asarray(rows, dtype=float)


def _read_off_vertices(path: Path) -> np.ndarray:
    return _read_off_mesh(path)[0]


def _read_off_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise GVSamplingExtractionError(f"OFF mesh output is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header != "OFF":
            raise GVSamplingExtractionError(f"OFF mesh has invalid header: {path}")
        counts = handle.readline().split()
        if not counts:
            raise GVSamplingExtractionError(f"OFF mesh is missing counts: {path}")
        vertex_count = int(counts[0])
        face_count = int(counts[1]) if len(counts) > 1 else 0
        vertices = [[float(item) for item in handle.readline().split()[:3]] for _ in range(vertex_count)]
        faces: list[list[int]] = []
        for _ in range(face_count):
            parts = handle.readline().split()
            if not parts:
                continue
            count = int(parts[0])
            indices = [int(item) for item in parts[1 : 1 + count]]
            if len(indices) < 3:
                continue
            for offset in range(1, len(indices) - 1):
                faces.append([indices[0], indices[offset], indices[offset + 1]])
    if not vertices:
        raise GVSamplingExtractionError(f"OFF mesh contains no vertices: {path}")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if faces.size == 0:
        raise GVSamplingExtractionError("Mesh volume requires triangular faces.")
    vertex_count = int(vertices.shape[0])
    if np.any(faces < 0) or np.any(faces >= vertex_count):
        raise GVSamplingExtractionError("OFF mesh face index is outside the vertex range.")
    triangles = vertices[faces]
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    return float(abs(np.sum(signed)) / 6.0)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise GVSamplingExtractionError(f"None of the expected Mirheo output files exists: {joined}")


def _optional_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _control_channel(controls: Mapping[str, float], name: str, count: int) -> np.ndarray:
    if name not in controls:
        raise GVSamplingExtractionError(f"Control {name!r} is required to build sampling channels.")
    return np.full(int(max(count, 1)), float(controls[name]), dtype=float)


def _channel_length(channels: Mapping[str, np.ndarray]) -> int:
    for values in channels.values():
        array = np.asarray(values)
        if array.shape == ():
            return 1
        if array.shape:
            return int(array.shape[0])
    return 1


def _concat_compatible(name: str, arrays: list[np.ndarray]) -> np.ndarray:
    normalized = [np.asarray(array, dtype=float) for array in arrays]
    if len(normalized) == 1:
        return normalized[0]
    ranks = {array.ndim for array in normalized}
    if ranks == {0}:
        return np.asarray([float(array) for array in normalized], dtype=float)
    normalized = [array.reshape(1) if array.ndim == 0 else array for array in normalized]
    tail_shapes = {array.shape[1:] for array in normalized}
    if len(tail_shapes) != 1:
        raise GVSamplingExtractionError(f"Channel {name!r} has incompatible shapes across sweep values.")
    return np.concatenate(normalized, axis=0)


def _validate_finite_channels(channels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for name, values in channels.items():
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise GVSamplingExtractionError(f"Channel {name!r} is empty.")
        if not np.all(np.isfinite(array)):
            raise GVSamplingExtractionError(f"Channel {name!r} contains NaN/Inf values.")
        normalized[str(name)] = array
    return normalized


__all__ = [
    "extract_sampling_channels",
    "merge_sampling_channels",
]
