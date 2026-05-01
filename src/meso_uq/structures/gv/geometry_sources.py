from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
GV_SOURCE_ROOT: Final[Path] = REPO_ROOT / "gv"

_CANONICAL_DEFAULT_GEOMETRY_EXPERIMENT: Final[str] = "stretching"
DEFAULT_GV_RADIUS: Final[float] = 2.0
DEFAULT_GV_HEIGHT: Final[float] = 14.28


def gv_canonical_geometry_default_path() -> Path:
    return GV_SOURCE_ROOT / _CANONICAL_DEFAULT_GEOMETRY_EXPERIMENT / "src" / "parameters-default.gv.yaml"


def gv_canonical_geometry_default_source() -> str:
    return gv_canonical_geometry_default_path().relative_to(REPO_ROOT).as_posix()


def gv_default_geometry_defaults() -> tuple[float, float]:
    parameters_path = gv_canonical_geometry_default_path()
    if not parameters_path.is_file():
        return DEFAULT_GV_RADIUS, DEFAULT_GV_HEIGHT
    with parameters_path.open("r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)
    return float(parameters["radGV"]), float(parameters["height"])


def gv_generated_artifacts() -> tuple[str, ...]:
    return ("gv.off",)
