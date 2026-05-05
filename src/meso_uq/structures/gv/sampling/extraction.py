from __future__ import annotations

import csv
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
        channels = _extract_stretching(root, controls=controls)
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


def _extract_stretching(root: Path, *, controls: Mapping[str, float]) -> dict[str, np.ndarray]:
    force_path = root / "force" / "emb.csv"
    columns = _read_csv_columns(force_path)
    for required in ("time", "fx", "fy", "fz"):
        if required not in columns:
            raise GVSamplingExtractionError(f"Stretching force CSV is missing column {required!r}: {force_path}")
    force = np.sqrt(columns["fx"] ** 2 + columns["fy"] ** 2 + columns["fz"] ** 2)
    count = int(force.size)
    displacement = _stretching_displacement(root)
    initial_extent = displacement["initial_extent"]
    delta = displacement["displacement"]
    channels = {
        "time": columns["time"],
        "fx": columns["fx"],
        "fy": columns["fy"],
        "fz": columns["fz"],
        "force": force,
        "tot_force": _control_channel(controls, "tot_force", count),
        "displacement": np.full(count, delta, dtype=float),
    }
    if initial_extent > 0.0:
        channels["longitudinal_strain"] = np.full(count, delta / initial_extent, dtype=float)
    if "bpress" in controls:
        channels["bpress"] = _control_channel(controls, "bpress", count)
    return channels


def _extract_buckling(root: Path, *, controls: Mapping[str, float]) -> dict[str, np.ndarray]:
    initial = _read_initial_buckling_vertices(root)
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
    if "bpress" in controls:
        channels["bpress"] = _control_channel(controls, "bpress", 1)
    return channels


def _extract_torsion(
    root: Path,
    *,
    controls: Mapping[str, float],
    geometry: GVMaterialGeometry,
) -> dict[str, np.ndarray]:
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


def _extract_eigenmodes(root: Path, *, mode_count: int = 30) -> dict[str, np.ndarray]:
    output_dir = root / "analysis" / "output"
    eigenvalue_path = _first_existing(
        output_dir / "eigvalues.txt",
        output_dir / "eigvalues_new.txt",
    )
    eigenvalues = np.atleast_1d(np.loadtxt(eigenvalue_path, dtype=float))
    if eigenvalues.size == 0:
        raise GVSamplingExtractionError(f"Eigenmodes file contains no eigenvalues: {eigenvalue_path}")
    count = min(int(mode_count), int(eigenvalues.size))
    channels: dict[str, np.ndarray] = {
        "mode_index": np.arange(count, dtype=float),
        "eigenvalues": eigenvalues[:count],
        "eigenfrequencies": np.sqrt(np.abs(eigenvalues[:count])),
    }
    vector_path = _optional_existing(output_dir / "eigvectors.txt", output_dir / "eigvectors_new.txt")
    if vector_path is not None:
        vectors = np.atleast_1d(np.loadtxt(vector_path, dtype=float))
        if vectors.ndim == 1:
            channels["eigenvectors"] = vectors[:count]
        else:
            channels["eigenvectors"] = vectors[:count, ...]
    return channels


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
    return _read_off_vertices(_first_existing(root / "mesh" / "gv00001.off", root / "gas_vesicle" / "gv.off"))


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
        rows = [[float(item) for item in handle.readline().split()[:3]] for _ in range(vertex_count)]
    if not rows:
        raise GVSamplingExtractionError(f"OFF mesh contains no vertices: {path}")
    return np.asarray(rows, dtype=float)


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
