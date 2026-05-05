from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from meso_uq.structures.gv.sampling import (
    GVMaterialGeometry,
    GVSweep,
    GVSamplingExtractionError,
    extract_sampling_channels,
    merge_sampling_channels,
)
from meso_uq.structures.gv.sampling.extraction import (
    _channel_length,
    _control_channel,
    _read_csv_columns,
    _read_hdf5_dataset,
    _read_off_vertices,
    _read_xyz_positions,
    _first_existing,
)


def _write_xyz(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(points)), "# test frame"]
    lines.extend(f"0 {x} {y} {z}" for x, y, z in points)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_off(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["OFF", f"{len(points)} 0 0"]
    lines.extend(f"{x} {y} {z}" for x, y, z in points)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _geometry() -> GVMaterialGeometry:
    return GVMaterialGeometry(radGV=2.0, height=14.28)


def test_extract_stretching_channels_from_force_and_xyz_outputs(tmp_path: Path) -> None:
    work = tmp_path / "stretching"
    force_dir = work / "force"
    force_dir.mkdir(parents=True)
    (force_dir / "emb.csv").write_text(
        "objId,time,fx,fy,fz\n0,0.1,3,4,0\n0,0.2,0,0,6\n",
        encoding="utf-8",
    )
    _write_xyz(work / "trj_eq" / "sim00001eq" / "emb_0000000.xyz", [(0, 0, 0), (0, 0, 2)])
    _write_xyz(work / "trj_eq" / "sim00001eq" / "emb_0000001.xyz", [(0, 0, 0), (0, 0, 3)])

    channels = extract_sampling_channels(
        experiment="stretching",
        work_dir=work,
        controls={"tot_force": 750.0, "bpress": -91.0},
        sweep=GVSweep("tot_force", (750.0,)),
        geometry=_geometry(),
    )

    assert channels["force"].tolist() == [5.0, 6.0]
    assert channels["displacement"].tolist() == [1.0, 1.0]
    assert channels["longitudinal_strain"].tolist() == [0.5, 0.5]


def test_extract_buckling_channels_from_mesh_and_xyz_outputs(tmp_path: Path) -> None:
    work = tmp_path / "buckling"
    initial = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    final = [(10, 10, 0), (11.2, 10, 0), (10, 11, 0)]
    _write_off(work / "mesh" / "gv00001.off", initial)
    _write_xyz(work / "gas_vesicle" / "test.xyz", final)

    channels = extract_sampling_channels(
        experiment="buckling",
        work_dir=work,
        controls={"buck": 0.25, "bpress": -91.0},
        sweep=GVSweep("buck", (0.25,)),
        geometry=_geometry(),
    )

    assert channels["buck"].tolist() == [0.25]
    assert channels["deformation_amplitude"][0] > 0.0
    assert channels["shape_amplitude"][0] > 0.0


def test_extract_torsion_channels_rejects_non_finite_force_output(tmp_path: Path) -> None:
    work = tmp_path / "torsion"
    force_dir = work / "force"
    force_dir.mkdir(parents=True)
    (force_dir / "emb.csv").write_text("objId,time,fx,fy,fz\n0,0.1,nan,0,0\n", encoding="utf-8")

    with pytest.raises(GVSamplingExtractionError, match="NaN|Inf|non-finite"):
        extract_sampling_channels(
            experiment="torsion",
            work_dir=work,
            controls={"theta": 0.03},
            sweep=GVSweep("theta", (0.03,)),
            geometry=_geometry(),
        )


def test_extract_eigenmodes_and_merge_sweep_channels(tmp_path: Path) -> None:
    work = tmp_path / "eigenmodes" / "analysis" / "output"
    work.mkdir(parents=True)
    (work / "eigvalues.txt").write_text("4\n9\n16\n", encoding="utf-8")

    channels = extract_sampling_channels(
        experiment="eigenmodes",
        work_dir=tmp_path / "eigenmodes",
        controls={"bpress": -91.0},
        sweep=GVSweep("bpress", (-91.0,)),
        geometry=_geometry(),
    )
    merged = merge_sampling_channels((channels, {"eigenvalues": np.asarray([25.0]), "bpress": np.asarray([-92.0])}))

    assert channels["eigenfrequencies"].tolist() == [2.0, 3.0, 4.0]
    assert merged["eigenvalues"].tolist() == [4.0, 9.0, 16.0, 25.0]


def test_extract_sampling_channels_rejects_unknown_experiment() -> None:
    with pytest.raises(
        GVSamplingExtractionError,
        match="Unsupported GV sampling extraction experiment",
    ):
        extract_sampling_channels(
            experiment="unsupported",
            work_dir=".",
            controls={"tot_force": 500.0},
            sweep=GVSweep("tot_force", (500.0,)),
            geometry=_geometry(),
        )


def test_extract_sampling_channels_rejects_buckling_without_required_mesh_or_vesicle(tmp_path: Path) -> None:
    with pytest.raises(GVSamplingExtractionError, match="None of the expected Mirheo output files exists"):
        extract_sampling_channels(
            experiment="buckling",
            work_dir=tmp_path / "missing-buckling",
            controls={"buck": 0.5, "bpress": -91.0},
            sweep=GVSweep("buck", (0.5,)),
            geometry=_geometry(),
        )


def test_extract_sampling_channels_rejects_stretching_missing_force_header(tmp_path: Path) -> None:
    work = tmp_path / "stretching"
    force_dir = work / "force"
    force_dir.mkdir(parents=True)
    (force_dir / "emb.csv").write_text("objId,time,fx,fz\n0,0.1,3,0\n", encoding="utf-8")

    with pytest.raises(GVSamplingExtractionError, match="missing column"):
        extract_sampling_channels(
            experiment="stretching",
            work_dir=work,
            controls={"tot_force": 750.0, "bpress": -91.0},
            sweep=GVSweep("tot_force", (750.0,)),
            geometry=_geometry(),
        )


def test_extract_sampling_channels_rejects_non_numeric_csv_fields(tmp_path: Path) -> None:
    work = tmp_path / "stretching-bad"
    force_dir = work / "force"
    force_dir.mkdir(parents=True)
    (force_dir / "emb.csv").write_text(
        "objId,time,fx,fy,fz\n0,0.1,bad,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GVSamplingExtractionError,
        match="CSV output .* has non-numeric field",
    ):
        extract_sampling_channels(
            experiment="stretching",
            work_dir=work,
            controls={"tot_force": 750.0, "bpress": -91.0},
            sweep=GVSweep("tot_force", (750.0,)),
            geometry=_geometry(),
        )


def test_extract_eigenmodes_without_vector_file_uses_eigenvalues_only(tmp_path: Path) -> None:
    work = tmp_path / "eigenmodes" / "analysis" / "output"
    work.mkdir(parents=True)
    (work / "eigvalues.txt").write_text("9\n16\n", encoding="utf-8")

    channels = extract_sampling_channels(
        experiment="eigenmodes",
        work_dir=tmp_path / "eigenmodes",
        controls={"bpress": -91.0},
        sweep=GVSweep("bpress", (-91.0,)),
        geometry=_geometry(),
    )

    assert "eigenvectors" not in channels
    assert channels["mode_index"].shape == (2,)


def test_merge_sampling_channels_rejects_incompatible_shapes() -> None:
    with pytest.raises(GVSamplingExtractionError, match="incompatible shapes"):
        merge_sampling_channels(({"amplitude": np.asarray([[1.0, 2.0], [3.0, 4.0]])}, {"amplitude": np.asarray([1.0, 2.0])}))


def test_merge_sampling_channels_rejects_empty_channels() -> None:
    with pytest.raises(GVSamplingExtractionError, match="Channel 'empty' is empty"):
        merge_sampling_channels(({"empty": np.asarray([])},))


def test_merge_sampling_channels_rejects_empty_payload() -> None:
    with pytest.raises(GVSamplingExtractionError, match="At least one channel payload is required."):
        merge_sampling_channels(())


def test_merge_sampling_channels_merges_scalar_channels() -> None:
    channels = merge_sampling_channels(({"axis": np.asarray(1.0)}, {"axis": np.asarray(2.0)}))
    assert channels["axis"].tolist() == [1.0, 2.0]


def test_control_channel_requires_control_value() -> None:
    with pytest.raises(GVSamplingExtractionError, match="Control 'tot_force' is required"):
        _control_channel({}, "tot_force", 4)


def test_read_csv_columns_rejects_file_without_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="CSV output has no header"):
        _read_csv_columns(csv_path)


def test_read_csv_columns_rejects_file_with_no_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "header_only.csv"
    csv_path.write_text("objId,time,fx,fy,fz\n", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="CSV output contains no rows"):
        _read_csv_columns(csv_path)


def test_read_xyz_positions_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GVSamplingExtractionError, match="XYZ output is missing"):
        _read_xyz_positions(tmp_path / "missing.xyz")


def test_read_xyz_positions_rejects_no_particle_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.xyz"
    path.write_text("2\n0\n", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="XYZ output contains no particle positions"):
        _read_xyz_positions(path)


def test_first_existing_requires_candidates() -> None:
    with pytest.raises(GVSamplingExtractionError, match="None of the expected Mirheo output files exists"):
        _first_existing(Path("missing1"), Path("missing2"))


def test_read_off_vertices_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GVSamplingExtractionError, match="OFF mesh output is missing"):
        _read_off_vertices(tmp_path / "missing.off")


def test_read_off_vertices_rejects_invalid_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.off"
    path.write_text("NOT_OFF\n1 0 0\n", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="OFF mesh has invalid header"):
        _read_off_vertices(path)


def test_read_off_vertices_rejects_missing_counts(tmp_path: Path) -> None:
    path = tmp_path / "counts.off"
    path.write_text("OFF\n\n1 0 0\n", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="OFF mesh is missing counts"):
        _read_off_vertices(path)


def test_read_off_vertices_rejects_no_vertices(tmp_path: Path) -> None:
    path = tmp_path / "novertices.off"
    path.write_text("OFF\n0 0 0\n", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="OFF mesh contains no vertices"):
        _read_off_vertices(path)


def test_extract_buckling_rejects_shape_mismatch_between_initial_and_final_vertices(tmp_path: Path) -> None:
    work = tmp_path / "buckling-shape-mismatch"
    _write_off(work / "mesh" / "gv00001.off", [(0, 0, 0)])
    _write_xyz(work / "gas_vesicle" / "test.xyz", [(0, 0, 0), (1, 0, 0)])

    with pytest.raises(GVSamplingExtractionError, match="initial/final vertex shapes differ"):
        extract_sampling_channels(
            experiment="buckling",
            work_dir=work,
            controls={"buck": 0.25, "bpress": -91.0},
            sweep=GVSweep("buck", (0.25,)),
            geometry=_geometry(),
        )


def test_extract_torsion_rejects_missing_force_column(tmp_path: Path) -> None:
    work = tmp_path / "torsion-missing"
    force_dir = work / "force"
    force_dir.mkdir(parents=True)
    (force_dir / "emb.csv").write_text("objId,time,fx,fy\n0,0.1,1,2\n", encoding="utf-8")

    with pytest.raises(GVSamplingExtractionError, match="missing column"):
        extract_sampling_channels(
            experiment="torsion",
            work_dir=work,
            controls={"theta": 0.03},
            sweep=GVSweep("theta", (0.03,)),
            geometry=_geometry(),
        )


def test_extract_eigenmodes_rejects_empty_values(tmp_path: Path) -> None:
    output = tmp_path / "eigenmodes-empty" / "analysis" / "output"
    output.mkdir(parents=True)
    (output / "eigvalues.txt").write_text("", encoding="utf-8")

    with pytest.raises(GVSamplingExtractionError, match="contains no eigenvalues"):
        extract_sampling_channels(
            experiment="eigenmodes",
            work_dir=tmp_path / "eigenmodes-empty",
            controls={"bpress": -91.0},
            sweep=GVSweep("bpress", (-91.0,)),
            geometry=_geometry(),
        )


def test_extract_eigenmodes_uses_vector_1d_path(tmp_path: Path) -> None:
    output = tmp_path / "eigenmodes-eigvec" / "analysis" / "output"
    output.mkdir(parents=True)
    (output / "eigvalues.txt").write_text("4\n9\n", encoding="utf-8")
    (output / "eigvectors.txt").write_text("1.0\n2.0\n", encoding="utf-8")

    channels = extract_sampling_channels(
        experiment="eigenmodes",
        work_dir=tmp_path / "eigenmodes-eigvec",
        controls={"bpress": -91.0},
        sweep=GVSweep("bpress", (-91.0,)),
        geometry=_geometry(),
    )

    assert channels["eigenvectors"].shape == (2,)
    assert np.allclose(channels["eigenvectors"], [1.0, 2.0])


def test_read_hdf5_dataset_requires_optional_dependency(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.h5"
    path.write_text("", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="h5py is required"):
        _read_hdf5_dataset(path, "position")


def test_extract_eigenmodes_uses_vector_2d_path(tmp_path: Path) -> None:
    output = tmp_path / "eigenmodes-eigvec-2d" / "analysis" / "output"
    output.mkdir(parents=True)
    (output / "eigvalues.txt").write_text("4\n9\n", encoding="utf-8")
    (output / "eigvectors.txt").write_text("1.0 2.0\n3.0 4.0\n", encoding="utf-8")

    channels = extract_sampling_channels(
        experiment="eigenmodes",
        work_dir=tmp_path / "eigenmodes-eigvec-2d",
        controls={"bpress": -91.0},
        sweep=GVSweep("bpress", (-91.0,)),
        geometry=_geometry(),
    )

    assert channels["eigenvectors"].shape == (2, 2)


def test_extract_stretching_rejects_insufficient_xyz_frames(tmp_path: Path) -> None:
    work = tmp_path / "stretching-one-frame"
    force_dir = work / "force"
    force_dir.mkdir(parents=True)
    (force_dir / "emb.csv").write_text("objId,time,fx,fy,fz\n0,0.1,1,0,0\n", encoding="utf-8")
    _write_xyz(work / "trj_eq" / "sim00001" / "emb_0000000.xyz", [(0, 0, 0), (0, 0, 1)])

    with pytest.raises(GVSamplingExtractionError, match="requires at least two XYZ frames"):
        extract_sampling_channels(
            experiment="stretching",
            work_dir=work,
            controls={"tot_force": 750.0, "bpress": -91.0},
            sweep=GVSweep("tot_force", (750.0,)),
            geometry=_geometry(),
        )


def test_read_csv_columns_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GVSamplingExtractionError, match="Required Mirheo CSV output is missing"):
        _read_csv_columns(tmp_path / "missing.csv")


def test_read_xyz_positions_skips_short_rows_and_keeps_valid_positions(tmp_path: Path) -> None:
    path = tmp_path / "mixed.xyz"
    path.write_text("2\n# frame\nbad\n0 1 2 3\n", encoding="utf-8")

    assert _read_xyz_positions(path).tolist() == [[1.0, 2.0, 3.0]]


def test_channel_length_handles_scalar_and_empty_payloads() -> None:
    assert _channel_length({"scalar": np.asarray(1.0)}) == 1
    assert _channel_length({}) == 1


def test_extract_buckling_uses_xyz_fallback_before_legacy_test_xyz(tmp_path: Path) -> None:
    work = tmp_path / "buckling-xyz"
    _write_off(work / "mesh" / "gv00001.off", [(0, 0, 0), (1, 0, 0)])
    _write_xyz(work / "trj_eq" / "sim00001" / "emb_0000000.xyz", [(0, 0, 0), (1.5, 0, 0)])

    channels = extract_sampling_channels(
        experiment="buckling",
        work_dir=work,
        controls={"buck": 0.25, "bpress": -91.0},
        sweep=GVSweep("buck", (0.25,)),
        geometry=_geometry(),
    )

    assert channels["deformation_amplitude"][0] > 0.0


def test_read_hdf5_dataset_rejects_missing_dataset(monkeypatch, tmp_path: Path) -> None:
    class _FakeH5File:
        def __init__(self, path, mode):
            self.path = path
            self.mode = mode

        def __enter__(self):
            return {"velocity": np.asarray([[1.0, 2.0, 3.0]])}

        def __exit__(self, exc_type, exc, traceback):
            return False

    class _FakeH5Py:
        File = _FakeH5File

    monkeypatch.setitem(sys.modules, "h5py", _FakeH5Py)
    path = tmp_path / "positions.h5"
    path.write_text("", encoding="utf-8")

    with pytest.raises(GVSamplingExtractionError, match="missing dataset"):
        _read_hdf5_dataset(path, "position")


def test_extract_buckling_wraps_unreadable_hdf5_positions(monkeypatch, tmp_path: Path) -> None:
    class _FakeH5File:
        def __init__(self, path, mode):
            raise RuntimeError("bad hdf5")

    class _FakeH5Py:
        File = _FakeH5File

    monkeypatch.setitem(sys.modules, "h5py", _FakeH5Py)
    work = tmp_path / "buckling-bad-hdf5"
    _write_off(work / "mesh" / "gv00001.off", [(0, 0, 0), (1, 0, 0)])
    restart = work / "restart"
    restart.mkdir(parents=True)
    (restart / "emb.PV-0000000.h5").write_text("not hdf5", encoding="utf-8")

    with pytest.raises(GVSamplingExtractionError, match="Could not read buckling HDF5 positions"):
        extract_sampling_channels(
            experiment="buckling",
            work_dir=work,
            controls={"buck": 0.25, "bpress": -91.0},
            sweep=GVSweep("buck", (0.25,)),
            geometry=_geometry(),
        )
