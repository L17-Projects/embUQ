from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

from meso_uq.vega_workflows import (
    VALID_EXPERIMENTS,
    VALID_MODEL_FAMILIES,
    VegaWorkflowSelection,
    expand_selection_matrix,
    parse_selection,
    resolve_workflow_config_path,
    selection_key,
    selection_slug,
)

DEFAULT_PRODUCTION_SANITY_SELECTION = VegaWorkflowSelection("compression", "full-model", "production")
PRODUCTION_SMOKE_OVERRIDES = {
    "pop_size": 10,
    "max_gen": 1,
    "hbi_pop_size": 10,
    "phase3a_pop_size": 10,
    "phase3a_max_gen": 1,
    "phase3b_pop_size": 10,
    "phase3b_max_gen": 1,
    "map_n_displacements": 1,
}


def _deduplicate(selections: Iterable[VegaWorkflowSelection]) -> list[VegaWorkflowSelection]:
    ordered: list[VegaWorkflowSelection] = []
    seen: set[str] = set()
    for selection in selections:
        key = selection_key(selection)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(selection)
    return ordered


def resolve_production_sanity_selections(
    values: Iterable[str],
    *,
    all_lanes: bool = False,
) -> list[VegaWorkflowSelection]:
    explicit = [parse_selection(value) for value in values]
    if explicit:
        selections = explicit
    elif all_lanes:
        selections = expand_selection_matrix(VALID_EXPERIMENTS, VALID_MODEL_FAMILIES, ("production",))
    else:
        selections = [DEFAULT_PRODUCTION_SANITY_SELECTION]

    for selection in selections:
        if selection.profile != "production":
            raise ValueError(
                "Production sanity only supports production selections. "
                f"Got: {selection_key(selection)}"
            )
    return _deduplicate(selections)


def build_production_smoke_config(
    repo_root: Path | str,
    selection: VegaWorkflowSelection,
) -> tuple[Path, dict[str, object]]:
    repo_root = Path(repo_root).resolve()
    base_config = resolve_workflow_config_path(repo_root, selection)
    with base_config.open("rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {base_config}, got {type(config).__name__}")

    smoke_config = dict(config)
    for key, value in PRODUCTION_SMOKE_OVERRIDES.items():
        if key in smoke_config:
            smoke_config[key] = value
    smoke_config["description"] = (
        f"Production sanity smoke derived from shipped production config {base_config.name}"
    )
    return base_config, smoke_config


def write_production_smoke_config(
    repo_root: Path | str,
    selection: VegaWorkflowSelection,
    output_root: Path | str,
) -> tuple[Path, Path]:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    base_config, smoke_config = build_production_smoke_config(repo_root, selection)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{selection_slug(selection)}.yaml"
    config_path.write_text(yaml.safe_dump(smoke_config, sort_keys=False), encoding="utf-8")
    return base_config, config_path


def load_korali_build_state(repo_root: Path | str) -> dict[str, object]:
    repo_root = Path(repo_root).resolve()
    build_options_path = (
        repo_root / "_vega" / "korali" / "build" / "meson-info" / "intro-buildoptions.json"
    )
    if not build_options_path.exists():
        return {
            "status": "unknown",
            "build_options_path": str(build_options_path),
            "reason": "repo-local Korali Meson build metadata not found",
        }

    data = json.loads(build_options_path.read_text(encoding="utf-8"))
    selected = {}
    for item in data:
        name = item.get("name")
        if name in {"buildtype", "mpi", "mpi4py", "openmp", "native_cuda_batch"}:
            selected[name] = item.get("value")

    return {
        "status": "detected",
        "build_options_path": str(build_options_path),
        "build_options": selected,
    }
