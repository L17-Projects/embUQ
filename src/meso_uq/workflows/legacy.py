from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meso_uq.core import Modality, coerce_modality


PathExists = Callable[[str], bool]


@dataclass(frozen=True)
class LegacyWorkflowSurface:
    family: str
    legacy_path: str
    replacement_api: str
    purpose: str
    status: str = "compatibility_wrapper"

    @property
    def warning_message(self) -> str:
        return (
            f"{self.legacy_path} is a MesoUQ compatibility entry point for {self.purpose}; "
            f"use {self.replacement_api} for new code."
        )


@dataclass(frozen=True)
class LegacySurrogateRuntime:
    backend: str = "dnn"
    predictive_mc_samples: int = 32
    predictive_mc_chunk_size: int = 8

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.backend, self.predictive_mc_samples, self.predictive_mc_chunk_size)


LEGACY_WORKFLOW_SURFACES: tuple[LegacyWorkflowSurface, ...] = (
    LegacyWorkflowSurface(
        family="compression",
        legacy_path="emb/compression/src/generate.py",
        replacement_api="meso_uq.simulation.generate_emb_simulation",
        purpose="EMB compression simulation generation",
    ),
    LegacyWorkflowSurface(
        family="indentation",
        legacy_path="emb/indentation/src/generate.py",
        replacement_api="meso_uq.simulation.generate_emb_simulation",
        purpose="EMB indentation simulation generation",
    ),
    LegacyWorkflowSurface(
        family="compression",
        legacy_path="emb/compression/evalkit/posterior_compression.py",
        replacement_api="meso_uq.workflows.legacy + emb.compression.evalkit compatibility functions",
        purpose="compression posterior evaluation and surrogate runtime setup",
    ),
    LegacyWorkflowSurface(
        family="indentation",
        legacy_path="emb/indentation/evalkit/posterior_indentation.py",
        replacement_api="meso_uq.workflows.legacy + emb.indentation.evalkit compatibility functions",
        purpose="indentation posterior evaluation and surrogate runtime setup",
    ),
    LegacyWorkflowSurface(
        family="inference",
        legacy_path="inference/scripts/run_phase_1.py",
        replacement_api="meso_uq.workflow_acceleration and meso_uq.experiments",
        purpose="hierarchical inference phase 1 launcher",
    ),
    LegacyWorkflowSurface(
        family="inference",
        legacy_path="inference/scripts/run_phase_2.py",
        replacement_api="meso_uq.workflow_acceleration and meso_uq.experiments",
        purpose="hierarchical inference phase 2 launcher",
    ),
    LegacyWorkflowSurface(
        family="inference",
        legacy_path="inference/scripts/run_phase_3b.py",
        replacement_api="meso_uq.workflow_acceleration and meso_uq.experiments",
        purpose="hierarchical inference phase 3b launcher",
    ),
    LegacyWorkflowSurface(
        family="reduced",
        legacy_path="reduced/scripts/run_phase_1.py",
        replacement_api="inference/scripts/run_phase_1.py with reduced config",
        purpose="reduced-model phase 1 launcher",
    ),
    LegacyWorkflowSurface(
        family="reduced",
        legacy_path="reduced/scripts/run_phase_2.py",
        replacement_api="inference/scripts/run_phase_2.py with reduced config",
        purpose="reduced-model phase 2 launcher",
    ),
    LegacyWorkflowSurface(
        family="reduced",
        legacy_path="reduced/scripts/run_phase_3b.py",
        replacement_api="inference/scripts/run_phase_3b.py with reduced config",
        purpose="reduced-model phase 3b launcher",
    ),
    LegacyWorkflowSurface(
        family="propagation",
        legacy_path="propagation/scripts/run_phase1_propagation.py",
        replacement_api="meso_uq.postprocess.propagation",
        purpose="phase 1 posterior propagation launcher",
    ),
    LegacyWorkflowSurface(
        family="propagation",
        legacy_path="propagation/scripts/run_phase3b_propagation.py",
        replacement_api="meso_uq.postprocess.propagation",
        purpose="phase 3b posterior propagation launcher",
    ),
)

_WARNED_SURFACES: set[str] = set()


def list_legacy_workflow_surfaces() -> tuple[LegacyWorkflowSurface, ...]:
    return LEGACY_WORKFLOW_SURFACES


def _path_exists(path: Path, exists: PathExists | None) -> bool:
    if exists is None:
        return path.exists()
    return exists(str(path))


def _candidate_roots(start_path: str | Path, *, max_parent_depth: int) -> tuple[Path, ...]:
    start = Path(start_path).expanduser()
    if start.is_file():
        start = start.parent
    candidates = [start]
    current = start
    for _ in range(max_parent_depth):
        parent = current.parent
        if parent == current:
            break
        candidates.append(parent)
        current = parent
    return tuple(candidates)


def _anchor_roots(anchor_file: str | Path) -> tuple[Path, ...]:
    anchor = Path(anchor_file).expanduser().resolve()
    base = anchor.parent if anchor.is_file() or anchor.suffix else anchor
    return (base, *base.parents)


def resolve_legacy_project_root(
    *,
    marker_parts: Sequence[str],
    start_path: str | Path | None = None,
    anchor_file: str | Path | None = None,
    max_parent_depth: int = 2,
    exists: PathExists | None = None,
) -> Path:
    if not marker_parts:
        raise ValueError("marker_parts must not be empty.")

    start = Path.cwd() if start_path is None else Path(start_path)
    roots: list[Path] = list(_candidate_roots(start, max_parent_depth=max_parent_depth))
    if anchor_file is not None:
        roots.extend(_anchor_roots(anchor_file))

    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if _path_exists(root.joinpath(*marker_parts), exists):
            return root

    marker = "/".join(marker_parts)
    raise RuntimeError(f"Could not find project root ({marker}) from {Path.cwd()}")


def resolve_legacy_relative_path(project_root: str | Path, path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = Path(project_root) / selected
    return selected.resolve()


_INFERENCE_CONFIG_FILENAMES: dict[Modality, str] = {
    Modality.COMPRESSION: "inference_config_compression.yaml",
    Modality.INDENTATION: "inference_config_indentation.yaml",
}


def _legacy_config_candidates(project_root: Path, modality: Modality) -> tuple[Path, ...]:
    filename = _INFERENCE_CONFIG_FILENAMES[modality]
    return (
        project_root / "inference" / "configs" / "production" / filename,
        Path("../../inference/configs/production") / filename,
        Path("inference/configs/production") / filename,
    )


def resolve_legacy_inference_config_path(
    project_root: str | Path,
    modality: Modality | str,
    *,
    exists: PathExists | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    selected = coerce_modality(modality)
    if selected not in _INFERENCE_CONFIG_FILENAMES:
        raise ValueError(
            f"Legacy EMB inference configs are registered only for compression and indentation, got '{selected.value}'."
        )

    project_root_path = Path(project_root)
    env_map = os.environ if env is None else env
    override = env_map.get("HUQ_INFERENCE_CONFIG") or env_map.get("CONFIG_PATH")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute() and not _path_exists(candidate, exists):
            candidate = project_root_path / candidate
        if _path_exists(candidate, exists):
            return candidate

    for candidate in _legacy_config_candidates(project_root_path, selected):
        if _path_exists(candidate, exists):
            return candidate

    raise FileNotFoundError(f"Could not find {_INFERENCE_CONFIG_FILENAMES[selected]}")


def _surrogate_config_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    surrogate_cfg = config.get("surrogate", {})
    if surrogate_cfg is None:
        surrogate_cfg = {}
    if not isinstance(surrogate_cfg, Mapping):
        raise ValueError("Expected 'surrogate' config section to be a mapping.")
    return surrogate_cfg


def resolve_legacy_surrogate_backend(config: Mapping[str, Any]) -> str:
    surrogate_cfg = _surrogate_config_section(config)
    backend = str(surrogate_cfg.get("backend", "dnn")).strip().lower()
    if backend not in {"dnn", "bnn"}:
        raise ValueError(f"Unsupported surrogate backend '{backend}'. Expected 'dnn' or 'bnn'.")
    return backend


def resolve_legacy_surrogate_runtime(config: Mapping[str, Any]) -> LegacySurrogateRuntime:
    surrogate_cfg = _surrogate_config_section(config)
    backend = resolve_legacy_surrogate_backend(config)
    predictive_mc_samples = int(surrogate_cfg.get("predictive_mc_samples", 32))
    predictive_mc_chunk_size = int(surrogate_cfg.get("predictive_mc_chunk_size", 8))
    if predictive_mc_samples < 1:
        raise ValueError("surrogate.predictive_mc_samples must be >= 1.")
    if predictive_mc_chunk_size < 1:
        raise ValueError("surrogate.predictive_mc_chunk_size must be >= 1.")
    return LegacySurrogateRuntime(
        backend=backend,
        predictive_mc_samples=predictive_mc_samples,
        predictive_mc_chunk_size=predictive_mc_chunk_size,
    )


def resolve_legacy_surrogate_trained_dir(
    project_root: str | Path, modality: Modality | str, diameter_um: float
) -> Path:
    selected = coerce_modality(modality)
    if selected not in {Modality.COMPRESSION, Modality.INDENTATION}:
        raise ValueError(
            f"Legacy EMB surrogate directories are registered only for compression and indentation, got '{selected.value}'."
        )
    return Path(project_root) / "emb" / selected.value / "surrogate" / "diameters" / f"{diameter_um}um" / "trained"


def legacy_evalkit_import_paths(project_root: str | Path) -> tuple[Path, ...]:
    root = Path(project_root)
    return (
        root / "emb" / "compression",
        root / "emb" / "compression" / "evalkit",
        root / "emb" / "indentation",
        root / "emb" / "indentation" / "evalkit",
    )


def prepend_legacy_evalkit_paths(project_root: str | Path) -> tuple[str, ...]:
    inserted: list[str] = []
    for path in legacy_evalkit_import_paths(project_root):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
            inserted.append(rendered)
    return tuple(inserted)


def warn_legacy_workflow_surface(legacy_path: str, *, stacklevel: int = 2) -> None:
    if os.environ.get("MESOUQ_SUPPRESS_LEGACY_WARNINGS", "").lower() in {"1", "true", "yes"}:
        return
    if legacy_path in _WARNED_SURFACES:
        return
    surface = next(
        (item for item in LEGACY_WORKFLOW_SURFACES if item.legacy_path == legacy_path),
        None,
    )
    message = (
        surface.warning_message
        if surface is not None
        else f"{legacy_path} is a MesoUQ compatibility entry point; prefer package APIs for new code."
    )
    _WARNED_SURFACES.add(legacy_path)
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


__all__ = [
    "LEGACY_WORKFLOW_SURFACES",
    "LegacySurrogateRuntime",
    "LegacyWorkflowSurface",
    "legacy_evalkit_import_paths",
    "list_legacy_workflow_surfaces",
    "prepend_legacy_evalkit_paths",
    "resolve_legacy_inference_config_path",
    "resolve_legacy_project_root",
    "resolve_legacy_relative_path",
    "resolve_legacy_surrogate_backend",
    "resolve_legacy_surrogate_runtime",
    "resolve_legacy_surrogate_trained_dir",
    "warn_legacy_workflow_surface",
]
