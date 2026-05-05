from __future__ import annotations

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
