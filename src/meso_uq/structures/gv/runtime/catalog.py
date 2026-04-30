from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Mapping, cast

from .base import GV_RUNTIME_DEFAULT_ROOT, RuntimeDescriptor, RuntimeDryRun


RUNTIME_EXPERIMENTS = ("stretching", "buckling", "torsion", "eigenmodes", "shear_flow")


def runtime_module_name(experiment: str) -> str:
    _validate_runtime_experiment(experiment)
    return f"meso_uq.structures.gv.runtime.{experiment}"


def load_runtime_descriptor(experiment: str) -> RuntimeDescriptor:
    module = import_module(runtime_module_name(experiment))
    descriptor = getattr(module, "DESCRIPTOR", None)
    if isinstance(descriptor, RuntimeDescriptor) or callable(getattr(descriptor, "plan", None)):
        return cast(RuntimeDescriptor, descriptor)
    raise TypeError(
        f"GV runtime module '{module.__name__}' must expose DESCRIPTOR as a RuntimeDescriptor-compatible object."
    )


def plan_runtime(
    experiment: str,
    *,
    output_root: str | Path = GV_RUNTIME_DEFAULT_ROOT,
    geometry: str | None = None,
    controls: Mapping[str, float] | None = None,
    include_experimental: bool = False,
) -> RuntimeDryRun:
    descriptor = load_runtime_descriptor(experiment)
    kwargs = {
        "output_root": output_root,
        "controls": controls,
        "include_experimental": include_experimental,
    }
    if geometry is not None:
        kwargs["geometry"] = geometry
    return descriptor.plan(**kwargs)


def _validate_runtime_experiment(experiment: str) -> None:
    if experiment not in RUNTIME_EXPERIMENTS:
        supported = ", ".join(RUNTIME_EXPERIMENTS)
        raise ValueError(f"Unsupported GV runtime experiment '{experiment}'. Supported: {supported}")
