from __future__ import annotations

from pathlib import Path

import pytest

from meso_uq.structures import get_structure
from meso_uq.structures.gv import build_geometry, geometry_id
from meso_uq.structures.gv.geometries import DEFAULT_GV_GEOMETRY
from meso_uq.structures.gv.geometry_sources import (
    gv_canonical_geometry_default_source,
    gv_default_geometry_defaults,
    gv_generated_artifacts,
)


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
