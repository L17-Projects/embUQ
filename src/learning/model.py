"""Deprecated shim for surrogate artifacts that reference learning.model."""

from __future__ import annotations

import warnings

warnings.warn(
    "`learning.model` is a deprecated MesoUQ surrogate compatibility import; "
    "use `meso_uq.surrogate.model` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from meso_uq.surrogate.model import MLP, init_weights, load_model_states, save_model_states

__all__ = ["MLP", "init_weights", "load_model_states", "save_model_states"]
