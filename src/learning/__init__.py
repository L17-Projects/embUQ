"""Deprecated compatibility package for legacy surrogate imports."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

__all__ = ["model"]


def _warn() -> None:
    warnings.warn(
        "`learning` is a deprecated MesoUQ surrogate compatibility import; "
        "use `meso_uq.surrogate` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name: str) -> Any:
    if name == "model":
        _warn()
        return importlib.import_module("learning.model")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
