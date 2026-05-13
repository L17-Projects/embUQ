"""Shared surrogate-model utilities for MesoUQ."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MLP",
    "SURROGATE_SERIALIZATION_ALIASES",
    "SerializedSurrogateAlias",
    "init_weights",
    "list_serialized_surrogate_aliases",
    "load_model_states",
    "save_model_states",
    "train_model",
]


def __getattr__(name: str) -> Any:
    if name in {"MLP", "init_weights", "load_model_states", "save_model_states"}:
        module = import_module(".model", __name__)
        return getattr(module, name)
    if name in {
        "SURROGATE_SERIALIZATION_ALIASES",
        "SerializedSurrogateAlias",
        "list_serialized_surrogate_aliases",
    }:
        module = import_module(".compat", __name__)
        return getattr(module, name)
    if name == "train_model":
        module = import_module(".training", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
