from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from meso_uq.experiments import load_experiments
from meso_uq.hpc_paths import default_runs_root, detect_hpc_site

VALID_EXPERIMENTS = ("compression", "indentation")
VALID_MODEL_FAMILIES = ("full-model", "reduced-model")
VALID_PROFILES = ("production", "validation")
VALID_INFERENCE_STAGES = ("phase1", "phase2", "phase3b")
VALID_PROPAGATION_STAGES = ("phase1", "phase3b")
VALID_MAP_STAGES = ("phase1", "phase3b")


@dataclass(frozen=True)
class VegaWorkflowSelection:
    experiment: str
    model_family: str
    profile: str

    def __post_init__(self) -> None:
        if self.experiment not in VALID_EXPERIMENTS:
            raise ValueError(f"Unsupported experiment: {self.experiment}")
        if self.model_family not in VALID_MODEL_FAMILIES:
            raise ValueError(f"Unsupported model family: {self.model_family}")
        if self.profile not in VALID_PROFILES:
            raise ValueError(f"Unsupported profile: {self.profile}")


def _resolve_repo_path(repo_root: Path | str, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = Path(repo_root, candidate)
    return candidate.resolve()


def selection_key(selection: VegaWorkflowSelection) -> str:
    return f"{selection.experiment}:{selection.model_family}:{selection.profile}"


def selection_slug(selection: VegaWorkflowSelection) -> str:
    return "__".join((selection.experiment, selection.model_family, selection.profile))


def parse_selection(value: str) -> VegaWorkflowSelection:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(
            "Workflow selection must use experiment:model-family:profile. " f"Got: {value}"
        )
    return VegaWorkflowSelection(parts[0], parts[1], parts[2])


def expand_selection_matrix(
    experiments: Iterable[str],
    model_families: Iterable[str],
    profiles: Iterable[str],
) -> list[VegaWorkflowSelection]:
    selections: list[VegaWorkflowSelection] = []
    for experiment in experiments:
        for model_family in model_families:
            for profile in profiles:
                selections.append(VegaWorkflowSelection(experiment, model_family, profile))
    return selections


def resolve_workflow_config_path(
    repo_root: Path | str,
    selection: VegaWorkflowSelection,
    override: str | Path | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve()
    override_path = _resolve_repo_path(repo_root, override)
    if override_path is not None:
        return override_path

    if selection.model_family == "full-model":
        filename = (
            f"inference_config_{selection.experiment}.yaml"
            if selection.profile == "production"
            else f"validation_config_{selection.experiment}.yaml"
        )
        return (repo_root / "inference" / "configs" / selection.profile / filename).resolve()

    filename = (
        f"reduced_config_{selection.experiment}.yaml"
        if selection.profile == "production"
        else f"validation_config_{selection.experiment}.yaml"
    )
    return (repo_root / "reduced" / "configs" / selection.profile / filename).resolve()


def resolve_workflow_output_root(
    repo_root: Path | str,
    selection: VegaWorkflowSelection,
    output_dir: str | Path | None = None,
    *,
    run_tag: str | None = None,
    site: str | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve()
    resolved = _resolve_repo_path(repo_root, output_dir)
    if resolved is not None:
        return resolved
    # Keep a stable way to reuse one run tag across multiple commands.
    effective_tag = run_tag
    if effective_tag is None:
        env_tag = os.environ.get("MESOUQ_RUN_TAG", "").strip()
        effective_tag = env_tag or None
    effective_site = site if site is not None else detect_hpc_site()
    return (
        default_runs_root(repo_root, "runs", site=effective_site, run_tag=effective_tag)
        / selection.experiment
        / selection.model_family
        / selection.profile
    ).resolve()


def resolve_inference_stage_driver(
    repo_root: Path | str,
    stage: str,
    model_family: str,
) -> Path:
    repo_root = Path(repo_root).resolve()
    if stage not in VALID_INFERENCE_STAGES:
        raise ValueError(f"Unsupported inference stage: {stage}")

    if model_family == "reduced-model":
        if stage == "phase1":
            return repo_root / "reduced" / "scripts" / "run_phase_1.py"
        if stage == "phase2":
            return repo_root / "reduced" / "scripts" / "run_phase_2.py"
        return repo_root / "reduced" / "scripts" / "run_phase_3b.py"
    if stage == "phase1":
        return repo_root / "inference" / "scripts" / "run_phase_1.py"
    if stage == "phase2":
        return repo_root / "inference" / "scripts" / "run_phase_2.py"
    return repo_root / "inference" / "scripts" / "run_phase_3b.py"


def resolve_propagation_driver(repo_root: Path | str, stage: str) -> Path:
    repo_root = Path(repo_root).resolve()
    if stage not in VALID_PROPAGATION_STAGES:
        raise ValueError(f"Unsupported propagation stage: {stage}")
    if stage == "phase1":
        return repo_root / "propagation" / "scripts" / "run_phase1_propagation.py"
    return repo_root / "propagation" / "scripts" / "run_phase3b_propagation.py"


def resolve_map_stage_input_root(output_root: Path | str, stage: str) -> Path:
    output_root = Path(output_root).resolve()
    if stage not in VALID_MAP_STAGES:
        raise ValueError(f"Unsupported MAP stage: {stage}")
    return output_root / ("results_phase_1" if stage == "phase1" else "results_phase_3b")


def resolve_map_output_root(
    output_root: Path | str, stage: str, maps_dir: str | Path | None = None
) -> Path:
    output_root = Path(output_root).resolve()
    resolved = _resolve_repo_path(output_root, maps_dir)
    if resolved is not None:
        return resolved
    suffix = "phase1" if stage == "phase1" else "phase3b"
    return output_root / f"map_{suffix}"


def load_workflow_datasets(
    repo_root: Path | str,
    config_path: Path | str,
    experiment: str,
) -> list[tuple[float, str]]:
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)

    entries: list[tuple[float, str]] = []
    for spec in load_experiments(config, repo_root):
        if not spec.enabled or spec.name != experiment:
            continue
        for diameter in spec.diameters:
            entries.append((float(diameter), spec.dataset_name(diameter)))

    if not entries:
        raise ValueError(
            f"No enabled datasets found for experiment '{experiment}' in {config_path}"
        )
    return entries


def select_workflow_datasets(
    datasets: Iterable[tuple[float, str]],
    dataset_name: str | None = None,
    diameter: float | None = None,
) -> list[tuple[float, str]]:
    dataset_list = list(datasets)
    if dataset_name is not None and diameter is not None:
        raise ValueError("Use either dataset_name or diameter, not both.")
    if dataset_name is not None:
        selected = [entry for entry in dataset_list if entry[1] == dataset_name]
        if not selected:
            raise ValueError(f"Dataset '{dataset_name}' not found in resolved workflow datasets.")
        return selected
    if diameter is not None:
        selected = [entry for entry in dataset_list if abs(entry[0] - float(diameter)) < 1e-9]
        if not selected:
            raise ValueError(f"Diameter '{diameter}' not found in resolved workflow datasets.")
        return selected
    return dataset_list


def build_inference_command(
    repo_root: Path | str,
    selection: VegaWorkflowSelection,
    stage: str,
    python_bin: str,
    config_path: Path | str,
    output_root: Path | str,
    cpu_ranks: int = 1,
    profiling: bool = False,
    restart: bool = False,
    dry_run: bool = False,
    device: str = "cpu",
) -> list[str]:
    driver = resolve_inference_stage_driver(repo_root, stage, selection.model_family)
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()

    if stage != "phase2" and cpu_ranks != 1:
        raise ValueError(
            f"cpu_ranks is only supported for phase2, got stage={stage} cpu_ranks={cpu_ranks}"
        )
    if stage != "phase1" and restart:
        raise ValueError("restart is only supported for phase1")
    if stage != "phase1" and dry_run:
        raise ValueError("dry_run is only supported for phase1")

    base_command = [
        python_bin,
        str(driver),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_root),
    ]
    if profiling:
        base_command.append("--profiling")
    if stage == "phase1" and restart:
        base_command.append("--restart")
    if stage == "phase1" and dry_run:
        base_command.append("--dry_run")
    if stage == "phase2":
        if cpu_ranks > 1:
            return [
                "mpirun",
                "--bind-to",
                "none",
                "--oversubscribe",
                "-np",
                str(cpu_ranks),
                *base_command,
            ]
        return base_command
    # phase1 and phase3b: device-aware
    base_command.extend(["--device", device])
    if device == "gpu":
        return base_command
    # cpu: Distributed MPI
    return ["mpirun", "--bind-to", "none", "-np", str(cpu_ranks), *base_command]


def build_propagation_command(
    repo_root: Path | str,
    stage: str,
    python_bin: str,
    config_path: Path | str,
    output_root: Path | str,
    device: str = "cpu",
) -> list[str]:
    driver = resolve_propagation_driver(repo_root, stage)
    cmd = [
        python_bin,
        str(driver),
        "--config",
        str(Path(config_path).resolve()),
        "--output-dir",
        str(Path(output_root).resolve()),
    ]
    if stage in {"phase1", "phase3b"}:
        cmd.extend(["--device", device])
    return cmd


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
