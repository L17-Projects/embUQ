from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from meso_uq.structures import get_structure
from meso_uq.structures.gv import build_geometry, geometry_id
from meso_uq.structures.gv.geometries import DEFAULT_GV_GEOMETRY
from meso_uq.structures.gv.geometry_sources import (
    DEFAULT_GV_HEIGHT,
    DEFAULT_GV_RADIUS,
    gv_canonical_geometry_default_source,
    gv_default_geometry_defaults,
    gv_generated_artifacts,
)
import meso_uq.structures.gv.geometry_sources as geometry_sources
import meso_uq.structures.gv.geometries as geometries_module


def test_gv_geometry_ids_are_stable_and_filesystem_safe() -> None:
    assert geometry_id(radius=2.0, height=14.28) == "gv_rad2_height14_28"
    geometry = build_geometry(radius=2.0, height=14.28, source="test")
    assert geometry.id == "gv_rad2_height14_28"
    assert "." not in geometry.id
    assert "/" not in geometry.id


def test_gv_default_geometry_is_registered() -> None:
    gv = get_structure("gv")
    geometry = gv.get_geometry("gv_rad2_height14_28")
    assert geometry.parameters == {"radius": 2.0, "height": 14.28}


def test_default_geometry_derives_from_canonical_staged_source() -> None:
    assert gv_default_geometry_defaults() == (2.0, 14.28)
    assert DEFAULT_GV_GEOMETRY.source == gv_canonical_geometry_default_source()
    assert not Path(DEFAULT_GV_GEOMETRY.source).is_absolute()
    assert DEFAULT_GV_GEOMETRY.source == "gv/stretching/src/parameters-default.gv.yaml"


def test_default_geometry_import_does_not_read_canonical_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_if_called() -> tuple[float, float]:
        raise AssertionError("default geometry import must not read canonical YAML")

    monkeypatch.setattr(geometry_sources, "gv_default_geometry_defaults", _fail_if_called)

    reloaded = importlib.reload(geometries_module)

    assert reloaded.DEFAULT_GV_GEOMETRY.parameters == {
        "radius": DEFAULT_GV_RADIUS,
        "height": DEFAULT_GV_HEIGHT,
    }


def test_default_geometry_defaults_fall_back_without_repo_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(geometry_sources, "gv_canonical_geometry_default_path", lambda: tmp_path / "missing.yaml")

    assert geometry_sources.gv_default_geometry_defaults() == (DEFAULT_GV_RADIUS, DEFAULT_GV_HEIGHT)


@pytest.mark.parametrize("pattern", ["/temp/tilen/", "/net/o364/"])
def test_gv_geometry_generators_do_not_embed_cluster_absolutes(pattern: str) -> None:
    scripts = [
        Path("gv/stretching/src/gas_vesicle/create_gv.py"),
        Path("gv/buckling/src/gas_vesicle/create_gv.py"),
        Path("gv/torsion/src/gas_vesicle/create_gv.py"),
        Path("gv/eigenmodes/src/gas_vesicle/create_gv.py"),
    ]
    for script_path in scripts:
        text = script_path.read_text(encoding="utf-8")
        assert pattern not in text, (script_path, pattern)


@pytest.mark.parametrize("script", [
    Path("gv/stretching/src/gas_vesicle/create_gv.py"),
    Path("gv/buckling/src/gas_vesicle/create_gv.py"),
    Path("gv/torsion/src/gas_vesicle/create_gv.py"),
    Path("gv/eigenmodes/src/gas_vesicle/create_gv.py"),
])
def test_gv_geometry_generators_document_scale_space_lookup_paths(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert "GV_SCALE_SPACE_BINARY" in text
    assert "GV_CGAL_TOOLS_ROOT" in text
    assert "shutil.which(\"scale_space\")" in text


def test_gv_off_is_treated_as_generated_artifact() -> None:
    assert "gv.off" in gv_generated_artifacts()


def test_gv_default_geometry_rejects_unknown_id() -> None:
    gv = get_structure("gv")
    with pytest.raises(KeyError, match="Unknown geometry"):
        gv.get_geometry("missing_geometry")


def test_buckling_geometry_faces_are_oriented_like_paper_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path("gv/buckling/src/gas_vesicle/add_to_off.py")
    monkeypatch.syspath_prepend(str(script.parent.resolve()))
    spec = importlib.util.spec_from_file_location("buckling_add_to_off_under_test", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    positive_faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
    )

    oriented = module.orient_faces_to_paper_winding(positive_faces, points)

    assert module.signed_mesh_volume(points, positive_faces) > 0.0
    assert module.signed_mesh_volume(points, oriented) < 0.0


def test_buckling_create_gv_enforces_paper_mesh_winding() -> None:
    text = Path("gv/buckling/src/gas_vesicle/create_gv.py").read_text(encoding="utf-8")

    assert "_orient_off_to_paper_winding(\"out.off\")" in text
    assert "archived paper meshes use negative signed volume" in text
