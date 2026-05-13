from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from meso_uq.core import AgentFamily, ArtifactClass, Modality, RuntimeCapability, coerce_modality


ConfigPurpose = Literal["generation", "parameters"]


@dataclass(frozen=True)
class EmbParameterFileContract:
    default_template: str = "parameters-default.emb.yaml"
    generated_default_pattern: str = "parameters-default{simnum}.yaml"
    generated_equil_pattern: str = "parameters-default{simnum}eq.yaml"
    runtime_parameter_pattern: str = "parameters{simnum}.yaml"
    runtime_prms_pattern: str = "parameters.prms{simnum}.yaml"
    parameter_dir: str = "parameter"
    mesh_dir: str = "mesh"
    microbubble_dir: str = "microbubble"
    position_quaternion_file: str = "posq.txt"

    def generated_default_file(self, simnum: str) -> str:
        return self.generated_default_pattern.format(simnum=simnum)

    def generated_equil_file(self, simnum: str) -> str:
        return self.generated_equil_pattern.format(simnum=simnum)

    def runtime_parameter_file(self, simnum: str) -> str:
        return self.runtime_parameter_pattern.format(simnum=simnum)

    def runtime_prms_file(self, simnum: str) -> str:
        return self.runtime_prms_pattern.format(simnum=simnum)


@dataclass(frozen=True)
class EmbGenerationWorkflow:
    modality: Modality
    legacy_root: str
    generation_script: str
    parameters_script: str
    default_object: str = "emb"
    parameter_files: EmbParameterFileContract = EmbParameterFileContract()
    generation_config_candidates: tuple[str, ...] = ()
    parameter_config_candidates: tuple[str, ...] = ()
    indentation_mass_multiplier: float = 1.0
    artifact_classes: tuple[ArtifactClass, ...] = (
        ArtifactClass.CONFIG,
        ArtifactClass.SIMULATION_OUTPUT,
        ArtifactClass.RUNTIME_MANIFEST,
    )
    capabilities: tuple[RuntimeCapability, ...] = (
        RuntimeCapability.SIMULATION,
        RuntimeCapability.INFERENCE,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", coerce_modality(self.modality))

    @property
    def family(self) -> AgentFamily:
        return AgentFamily.EMB

    @property
    def config_schema(self) -> str:
        return f"meso_uq.emb.{self.modality.value}.generation.v1"

    def script_path(self, repo_root: Path, script: str) -> Path:
        return repo_root / script


_COMPRESSION_GENERATION_CONFIGS = (
    "../../inference/configs/production/inference_config_compression.yaml",
    "../../inference/configs/production/inference_config.yaml",
    "inference/configs/production/inference_config_compression.yaml",
    "inference/configs/production/inference_config.yaml",
    "../inference/configs/production/inference_config_compression.yaml",
    "../inference/configs/production/inference_config.yaml",
)

_COMPRESSION_PARAMETER_CONFIGS = (
    *_COMPRESSION_GENERATION_CONFIGS,
    "configs/production/baseline_config.yaml",
    "configs/test/baseline_config_test.yaml",
    "../baseline/configs/production/baseline_config.yaml",
    "../baseline/configs/test/baseline_config_test.yaml",
)

_INDENTATION_GENERATION_CONFIGS = (
    "../../inference/configs/production/inference_config_indentation.yaml",
    "inference/configs/production/inference_config_indentation.yaml",
    "../inference/configs/production/inference_config_indentation.yaml",
)

_INDENTATION_PARAMETER_CONFIGS = (
    "inference/configs/production/inference_config_indentation.yaml",
)


EMB_GENERATION_WORKFLOWS: dict[Modality, EmbGenerationWorkflow] = {
    Modality.COMPRESSION: EmbGenerationWorkflow(
        modality=Modality.COMPRESSION,
        legacy_root="compression",
        generation_script="compression/src/generate.py",
        parameters_script="compression/src/parameters.py",
        generation_config_candidates=_COMPRESSION_GENERATION_CONFIGS,
        parameter_config_candidates=_COMPRESSION_PARAMETER_CONFIGS,
        indentation_mass_multiplier=1.0,
    ),
    Modality.INDENTATION: EmbGenerationWorkflow(
        modality=Modality.INDENTATION,
        legacy_root="indentation",
        generation_script="indentation/src/generate.py",
        parameters_script="indentation/src/parameters.py",
        generation_config_candidates=_INDENTATION_GENERATION_CONFIGS,
        parameter_config_candidates=_INDENTATION_PARAMETER_CONFIGS,
        indentation_mass_multiplier=5.0,
    ),
}


def list_emb_generation_workflows() -> tuple[EmbGenerationWorkflow, ...]:
    return tuple(EMB_GENERATION_WORKFLOWS[modality] for modality in (Modality.COMPRESSION, Modality.INDENTATION))


def get_emb_generation_workflow(modality: Modality | str) -> EmbGenerationWorkflow:
    selected = coerce_modality(modality)
    try:
        return EMB_GENERATION_WORKFLOWS[selected]
    except KeyError as exc:
        supported = ", ".join(workflow.modality.value for workflow in list_emb_generation_workflows())
        raise ValueError(f"EMB generation workflow for modality '{selected.value}' is not registered. Expected one of: {supported}.") from exc


def _legacy_repo_root(anchor_file: str | Path) -> Path:
    return Path(anchor_file).resolve().parents[2]


def emb_config_candidate_paths(
    modality: Modality | str,
    *,
    purpose: ConfigPurpose,
    anchor_file: str | Path | None = None,
    cwd: str | Path | None = None,
) -> tuple[Path, ...]:
    workflow = get_emb_generation_workflow(modality)
    if purpose == "generation":
        candidates = workflow.generation_config_candidates
    elif purpose == "parameters":
        candidates = workflow.parameter_config_candidates
    else:
        raise ValueError(f"Unsupported EMB config purpose '{purpose}'. Expected 'generation' or 'parameters'.")

    base = Path.cwd() if cwd is None else Path(cwd)
    paths: list[Path] = []
    seen: set[str] = set()

    def append(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)

    for candidate in candidates:
        path = Path(candidate)
        append(path if path.is_absolute() else base / path)

    if anchor_file is not None:
        repo_root = _legacy_repo_root(anchor_file)
        if workflow.modality is Modality.COMPRESSION:
            append(repo_root / "inference/configs/production/inference_config_compression.yaml")
            if purpose == "generation":
                append(repo_root / "inference/configs/production/inference_config.yaml")
        elif workflow.modality is Modality.INDENTATION:
            append(repo_root / "inference/configs/production/inference_config_indentation.yaml")

    return tuple(paths)


def resolve_emb_config_file(
    modality: Modality | str,
    *,
    purpose: ConfigPurpose,
    anchor_file: str | Path | None = None,
    cwd: str | Path | None = None,
) -> str:
    candidates = emb_config_candidate_paths(modality, purpose=purpose, anchor_file=anchor_file, cwd=cwd)
    for path in candidates:
        if path.exists():
            return str(path)
    rendered = [str(path) for path in candidates]
    raise FileNotFoundError(f"Could not find {purpose} config for EMB {coerce_modality(modality).value} in any of: {rendered}")


__all__ = [
    "ConfigPurpose",
    "EMB_GENERATION_WORKFLOWS",
    "EmbGenerationWorkflow",
    "EmbParameterFileContract",
    "emb_config_candidate_paths",
    "get_emb_generation_workflow",
    "list_emb_generation_workflows",
    "resolve_emb_config_file",
]
