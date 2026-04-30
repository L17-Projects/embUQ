from __future__ import annotations

from decimal import Decimal

from ..registry import GeometrySpec


def _normalize_decimal(value: float) -> str:
    decimal = Decimal(str(value)).normalize()
    text = format(decimal, "f").rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "_")


def geometry_id(*, radius: float, height: float) -> str:
    return f"gv_rad{_normalize_decimal(radius)}_height{_normalize_decimal(height)}"


def build_geometry(*, radius: float, height: float, source: str) -> GeometrySpec:
    return GeometrySpec(
        id=geometry_id(radius=radius, height=height),
        label=f"GV radius={radius:g}, height={height:g}",
        shape="gas_vesicle",
        parameters={"radius": radius, "height": height},
        source=source,
    )


DEFAULT_GV_GEOMETRY = build_geometry(
    radius=2.0,
    height=14.28,
    source="gv_simulation_files/*/gv/parameters-default.gv.yaml",
)
