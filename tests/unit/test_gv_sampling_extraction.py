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
    _read_initial_buckling_vertices,
    _read_hdf5_dataset,
    _mesh_volume,
    _read_anchor_force_csv,
    _read_off_vertices,
    _read_xyz_positions,
    _first_existing,
)


def _write_xyz(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(points)), "# test frame"]
    lines.extend(f"0 {x} {y} {z}" for x, y, z in points)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_off(
    path: Path,
    points: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    face_rows = list(faces or [])
    lines = ["OFF", f"{len(points)} {len(face_rows)} 0"]
    lines.extend(f"{x} {y} {z}" for x, y, z in points)
    lines.extend(" ".join(str(item) for item in (len(face), *face)) for face in face_rows)
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
    reference = [
        (0.10, 0.00, 2.142),
        (0.20, 0.00, 2.142),
        (-0.10, 0.00, -2.142),
        (-0.20, 0.00, -2.142),
        (1.00, 0.00, 0.0),
        (0.00, 1.00, 0.0),
        (-1.00, 0.00, 0.0),
        (0.00, -1.00, 0.0),
    ]
    stretched_1 = [
        (0.10, 0.00, 2.40),
        (0.20, 0.00, 2.40),
        (-0.10, 0.00, -2.40),
        (-0.20, 0.00, -2.40),
        (0.90, 0.00, 0.0),
        (0.00, 0.90, 0.0),
        (-0.90, 0.00, 0.0),
        (0.00, -0.90, 0.0),
    ]
    stretched_2 = [
        (0.10, 0.00, 2.50),
        (0.20, 0.00, 2.50),
        (-0.10, 0.00, -2.50),
        (-0.20, 0.00, -2.50),
        (0.85, 0.00, 0.0),
        (0.00, 0.85, 0.0),
        (-0.85, 0.00, 0.0),
        (0.00, -0.85, 0.0),
    ]
    _write_xyz(work / "trj_eq" / "sim00001eq" / "emb_0000000.xyz", reference)
    _write_xyz(work / "trj_eq" / "sim00001" / "emb_0000000.xyz", stretched_1)
    _write_xyz(work / "trj_eq" / "sim00001" / "emb_0000001.xyz", stretched_2)

    channels = extract_sampling_channels(
        experiment="stretching",
        work_dir=work,
        controls={"tot_force": 750.0, "bpress": -91.0},
        sweep=GVSweep("tot_force", (750.0,)),
        geometry=_geometry(),
    )

    assert channels["force"].tolist() == [750.0]
    assert channels["force_mean"].tolist() == [5.5]
    assert channels["displacement"][0] > 0.0
    assert channels["mean_length"][0] > channels["reference_length"][0]
    assert channels["mean_radius"][0] < channels["reference_radius"][0]


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


def test_extract_stretching_forward_sweep_channels_from_sim_folders(tmp_path: Path) -> None:
    work = tmp_path / "stretching-forward"
    reference = [
        (0.10, 0.00, 2.142),
        (0.20, 0.00, 2.142),
        (-0.10, 0.00, -2.142),
        (-0.20, 0.00, -2.142),
        (1.00, 0.00, 0.0),
        (0.00, 1.00, 0.0),
        (-1.00, 0.00, 0.0),
        (0.00, -1.00, 0.0),
    ]
    _write_xyz(work / "trj_eq" / "sim00001eq" / "emb_0000000.xyz", reference)
    for sim, force, scale_z, radius in (
        ("00001", 500.0, 2.20, 0.95),
        ("00002", 1000.0, 2.40, 0.90),
    ):
        (work / "parameter").mkdir(parents=True, exist_ok=True)
        (work / "parameter" / f"parameters-default{sim}.yaml").write_text(
            f"tot_force: {force}\nbpress: -91.0\n",
            encoding="utf-8",
        )
        for frame in range(2):
            points = [
                (0.10, 0.00, scale_z),
                (0.20, 0.00, scale_z),
                (-0.10, 0.00, -scale_z),
                (-0.20, 0.00, -scale_z),
                (radius, 0.00, 0.0),
                (0.00, radius, 0.0),
                (-radius, 0.00, 0.0),
                (0.00, -radius, 0.0),
            ]
            _write_xyz(work / "trj_eq" / f"sim{sim}" / f"emb_{frame:07d}.xyz", points)

    channels = extract_sampling_channels(
        experiment="stretching",
        work_dir=work,
        controls={"tot_force": 500.0, "bpress": -91.0},
        sweep=GVSweep("tot_force", (500.0, 1000.0)),
        geometry=_geometry(),
    )

    assert channels["tot_force"].tolist() == [500.0, 1000.0]
    assert channels["force"].tolist() == [500.0, 1000.0]
    assert channels["bpress"].tolist() == [-91.0, -91.0]
    assert channels["mean_length"][1] > channels["mean_length"][0]
    assert channels["mean_radius"][1] < channels["mean_radius"][0]
    assert channels["displacement"].tolist() == pytest.approx([0.0, channels["mean_length"][1] - channels["mean_length"][0]])


def test_extract_buckling_channels_include_relative_volume_when_mesh_faces_exist(tmp_path: Path) -> None:
    work = tmp_path / "buckling-volume"
    initial = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    final = [(10, 10, 10), (12, 10, 10), (10, 12, 10), (10, 10, 12)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    _write_off(work / "mesh" / "gv00001.off", initial, faces=faces)
    _write_xyz(work / "gas_vesicle" / "test.xyz", final)

    channels = extract_sampling_channels(
        experiment="buckling",
        work_dir=work,
        controls={"buck": 0.25, "bpress": -91.0},
        sweep=GVSweep("bpress", (-91.0,)),
        geometry=_geometry(),
    )

    assert channels["relative_volume"].tolist() == [8.0]


def test_extract_torsion_channels_from_force_csv_fallback(tmp_path: Path) -> None:
    work = tmp_path / "torsion-force"
    (work / "force").mkdir(parents=True)
    (work / "force" / "emb.csv").write_text(
        "time,fx,fy,fz\n0.0,3.0,4.0,0.0\n1.0,0.0,0.0,6.0\n",
        encoding="utf-8",
    )

    channels = extract_sampling_channels(
        experiment="torsion",
        work_dir=work,
        controls={"theta": 0.2},
        sweep=GVSweep("theta", (0.2,)),
        geometry=_geometry(),
    )

    assert channels["theta"].tolist() == [0.2, 0.2]
    assert channels["gamma"].tolist() == pytest.approx([2.0 * 0.2 / 14.28] * 2)
    assert channels["force"].tolist() == [5.0, 6.0]
    assert channels["sigma_phi_r"].tolist() == pytest.approx(
        (np.array([5.0, 6.0]) / (2.0 * np.pi * 2.0 * 14.28)).tolist()
    )


def test_extract_torsion_channels_from_paper_anchor_csvs(tmp_path: Path) -> None:
    work = tmp_path / "torsion-anchors"
    mesh_vertices = [
        (0.0, 0.0, -6.0),
        (1.0, 0.0, -5.5),
        (0.0, 0.0, 5.5),
        (1.0, 0.0, 6.0),
        (0.0, 0.0, 0.0),
    ]
    _write_off(work / "mesh" / "gv00001.off", mesh_vertices)
    for root_name, scale in (("anchor_min", 1.0), ("anchor_max", 1.5)):
        path = work / root_name / "sim00001" / "emb.csv"
        path.parent.mkdir(parents=True)
        rows = [
            "time,p0x,p0y,p0z,p1x,p1y,p1z",
            f"0.0,0.0,{scale},0.0,0.0,{2.0 * scale},0.0",
            f"1.0,0.0,{2.0 * scale},0.0,0.0,{4.0 * scale},0.0",
            f"2.0,0.0,{3.0 * scale},0.0,0.0,{6.0 * scale},0.0",
            f"3.0,0.0,{4.0 * scale},0.0,0.0,{8.0 * scale},0.0",
        ]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    channels = extract_sampling_channels(
        experiment="torsion",
        work_dir=work,
        controls={"theta": 0.1},
        sweep=GVSweep("theta", (0.1,)),
        geometry=_geometry(),
    )

    assert channels["theta"].tolist() == [0.1]
    assert channels["gamma"][0] > 0.0
    assert channels["sigma_phi_r"][0] > 0.0
    assert channels["sigma_std"].shape == (1,)


def test_read_anchor_force_csv_rejects_invalid_outputs(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("time,p0x,p0y,p0z\n0.0,not,numeric,row\n", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="non-numeric"):
        _read_anchor_force_csv(bad)

    too_short = tmp_path / "too-short.csv"
    too_short.write_text("time,p0x\n0.0,1.0\n", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="contains no force rows"):
        _read_anchor_force_csv(too_short)

    bad_width = tmp_path / "bad-width.csv"
    bad_width.write_text("time,p0x,p0y,p0z,extra\n0.0,1.0,2.0,3.0,4.0\n", encoding="utf-8")
    with pytest.raises(GVSamplingExtractionError, match="invalid force column count"):
        _read_anchor_force_csv(bad_width)


def test_extract_buckling_uses_last_half_trajectory_volume_stats_when_available(tmp_path: Path) -> None:
    work = tmp_path / "buckling-volume-trajectory"
    initial = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    _write_off(work / "mesh" / "gv00001.off", initial, faces=faces)
    for index, scale in enumerate((1.0, 1.0, 2.0, 2.0, 3.0, 3.0)):
        frame = [(10 + scale * x, 10 + scale * y, 10 + scale * z) for x, y, z in initial]
        _write_xyz(work / "trj_eq" / "sim00001" / f"emb_{index:07d}.xyz", frame)

    channels = extract_sampling_channels(
        experiment="buckling",
        work_dir=work,
        controls={"buck": 0.25, "bpress": -91.0},
        sweep=GVSweep("bpress", (-91.0,)),
        geometry=_geometry(),
    )

    assert channels["analyzed_volume_frame_count"].tolist() == [2.0]
    assert channels["relative_volume"].tolist() == pytest.approx([17.5])
    assert channels["relative_volume_std"].tolist() == pytest.approx([9.5])


def test_extract_buckling_forward_sweep_channels_from_sim_folders(tmp_path: Path) -> None:
    work = tmp_path / "buckling-forward"
    initial = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    _write_off(work / "mesh" / "gv00001.off", initial, faces=faces)
    (work / "parameter").mkdir(parents=True)
    for sim, buck, scale in (("00001", 0.0, 1.0), ("00002", 0.75, 2.0)):
        (work / "parameter" / f"parameters-default{sim}.yaml").write_text(
            f"buck: {buck}\nbpress: -91.0\n",
            encoding="utf-8",
        )
        for frame_index, frame_scale in enumerate((scale, scale, scale * 1.1)):
            frame = [
                (10.0 + frame_scale * x, 10.0 + frame_scale * y, 10.0 + frame_scale * z)
                for x, y, z in initial
            ]
            _write_xyz(work / "trj_eq" / f"sim{sim}" / f"emb_{frame_index:07d}.xyz", frame)

    channels = extract_sampling_channels(
        experiment="buckling",
        work_dir=work,
        controls={"buck": 0.0, "bpress": -91.0},
        sweep=GVSweep("buck", (0.0, 0.75)),
        geometry=_geometry(),
    )

    assert channels["buck"].tolist() == [0.0, 0.75]
    assert channels["bpress"].tolist() == [-91.0, -91.0]
    assert channels["analyzed_volume_frame_count"].tolist() == [1.0, 1.0]
    assert channels["relative_volume"].tolist() == pytest.approx([1.0, 8.0])
    assert channels["deformation_amplitude"][1] > channels["deformation_amplitude"][0]


def test_extract_buckling_forward_sweep_uses_measured_zero_pressure_volume_reference(tmp_path: Path) -> None:
    work = tmp_path / "buckling-forward-measured-reference"
    initial = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    _write_off(work / "mesh" / "gv00001.off", initial, faces=faces)
    (work / "parameter").mkdir(parents=True)
    for sim, buck, scale in (("00001", 0.0, 1.2), ("00002", 0.75, 1.5)):
        (work / "parameter" / f"parameters-default{sim}.yaml").write_text(
            f"buck: {buck}\nbpress: -91.0\n",
            encoding="utf-8",
        )
        for frame_index in range(3):
            frame = [
                (10.0 + scale * x, 10.0 + scale * y, 10.0 + scale * z)
                for x, y, z in initial
            ]
            _write_xyz(work / "trj_eq" / f"sim{sim}" / f"emb_{frame_index:07d}.xyz", frame)

    channels = extract_sampling_channels(
        experiment="buckling",
        work_dir=work,
        controls={"buck": 0.0, "bpress": -91.0},
        sweep=GVSweep("buck", (0.0, 0.75)),
        geometry=_geometry(),
    )

    assert channels["relative_volume"].tolist() == pytest.approx([1.0, (1.5 / 1.2) ** 3])
    assert channels["reference_volume"].tolist() == pytest.approx([channels["mean_volume"][0]] * 2)
    assert channels["initial_volume"][0] != pytest.approx(channels["reference_volume"][0])


def test_extract_torsion_forward_sweep_channels_from_anchor_folders(tmp_path: Path) -> None:
    work = tmp_path / "torsion-forward"
    vertices = [
        (1.0, 0.0, 6.0),
        (0.0, 1.0, 6.0),
        (1.0, 0.0, -6.0),
        (0.0, 1.0, -6.0),
    ]
    _write_off(work / "mesh" / "gv00001.off", vertices)
    (work / "parameter").mkdir(parents=True)
    header = "time,fx0,fy0,fz0,fx1,fy1,fz1\n"
    for sim, theta, force in (("00001", 0.01, 10.0), ("00002", 0.02, 20.0)):
        (work / "parameter" / f"parameters-default{sim}.yaml").write_text(
            f"theta: {theta}\n",
            encoding="utf-8",
        )
        for anchor_name in ("anchor_min", "anchor_max"):
            path = work / anchor_name / f"sim{sim}" / "emb.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                header
                + f"0,{0.0},{force},{0.0},{0.0},{0.0},{0.0}\n"
                + f"1,{0.0},{force},{0.0},{0.0},{0.0},{0.0}\n",
                encoding="utf-8",
            )

    channels = extract_sampling_channels(
        experiment="torsion",
        work_dir=work,
        controls={"theta": 0.01},
        sweep=GVSweep("theta", (0.01, 0.02)),
        geometry=_geometry(),
    )

    assert channels["theta"].tolist() == [0.01, 0.02]
    assert channels["gamma"][1] > channels["gamma"][0]
    assert channels["sigma_phi_r"][1] > channels["sigma_phi_r"][0]
    assert channels["sigma_std"].tolist() == pytest.approx([0.0, 0.0])


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
    merged = merge_sampling_channels(
        (
            channels,
            {
                "bpress": np.asarray([-92.0]),
                "eigenfrequencies": np.asarray([5.0]),
                "eigenvalues": np.asarray([25.0]),
                "kBT": np.asarray([1.0]),
                "mode_index": np.asarray([0.0]),
            },
        )
    )

    assert np.allclose(channels["eigenfrequencies"], [0.25, 1.0 / 3.0, 0.5])
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


def test_merge_sampling_channels_rejects_inconsistent_channel_sets() -> None:
    with pytest.raises(GVSamplingExtractionError, match="inconsistent across sweep values.*missing channels"):
        merge_sampling_channels(
            (
                {"force": np.asarray([1.0]), "time": np.asarray([0.0])},
                {"force": np.asarray([2.0])},
            )
        )

    with pytest.raises(GVSamplingExtractionError, match="inconsistent across sweep values.*extra channels"):
        merge_sampling_channels(
            (
                {"force": np.asarray([1.0])},
                {"force": np.asarray([2.0]), "time": np.asarray([0.0])},
            )
        )


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


def test_extract_buckling_rejects_non_positive_initial_volume(tmp_path: Path) -> None:
    work = tmp_path / "buckling-flat"
    flat_points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    faces = [(0, 1, 2), (1, 3, 2)]
    _write_off(work / "mesh" / "gv00001.off", flat_points, faces=faces)
    _write_xyz(work / "gas_vesicle" / "test.xyz", [(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0)])

    with pytest.raises(GVSamplingExtractionError, match="volume must be positive"):
        extract_sampling_channels(
            experiment="buckling",
            work_dir=work,
            controls={"buck": 0.25, "bpress": -91.0},
            sweep=GVSweep("bpress", (-91.0,)),
            geometry=_geometry(),
        )


def test_buckling_mesh_helpers_cover_face_edge_cases(tmp_path: Path) -> None:
    mesh_dir = tmp_path / "mesh-helper" / "mesh"
    mesh_path = mesh_dir / "gv00001.off"
    mesh_dir.mkdir(parents=True)
    mesh_path.write_text(
        "OFF\n4 3 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n\n2 0 1\n3 0 1 3\n",
        encoding="utf-8",
    )

    vertices = _read_off_vertices(mesh_path)
    assert vertices.shape == (4, 3)
    assert _read_initial_buckling_vertices(mesh_dir.parent).shape == (4, 3)

    with pytest.raises(GVSamplingExtractionError, match="triangular faces"):
        _mesh_volume(np.asarray(vertices, dtype=float), np.asarray([], dtype=int))
    with pytest.raises(GVSamplingExtractionError, match="outside the vertex range"):
        _mesh_volume(np.asarray(vertices, dtype=float), np.asarray([[0, 1, 10]], dtype=int))
