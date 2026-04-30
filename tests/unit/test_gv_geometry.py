from __future__ import annotations

from meso_uq.structures import get_structure
from meso_uq.structures.gv import build_geometry, geometry_id


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
