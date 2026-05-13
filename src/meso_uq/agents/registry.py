from __future__ import annotations

from meso_uq.core import (
    AgentDefinition,
    AgentFamily,
    ArtifactClass,
    Modality,
    ModelBackend,
    Platform,
    coerce_agent_family,
    coerce_modality,
)
from meso_uq.modalities import get_modality_descriptor, list_modality_descriptors


AGENT_REGISTRY: dict[AgentFamily, AgentDefinition] = {
    AgentFamily.EMB: AgentDefinition(
        family=AgentFamily.EMB,
        label="Elastic microbubble",
        supported_modalities=(Modality.COMPRESSION, Modality.INDENTATION),
        supported_backends=(ModelBackend.DNN, ModelBackend.BNN, ModelBackend.DPD),
        default_backend=ModelBackend.DNN,
        platforms=(Platform.WORKSTATION, Platform.VEGA, Platform.KAROLINA),
        artifact_classes=(ArtifactClass.REFERENCE, ArtifactClass.SURROGATE, ArtifactClass.TRAINING_MANIFEST),
        metadata={"legacy_roots": ("compression", "indentation")},
    ),
    AgentFamily.GV: AgentDefinition(
        family=AgentFamily.GV,
        label="Gas vesicle",
        supported_modalities=(Modality.BUCKLING, Modality.TORSION, Modality.EIGENMODES, Modality.SHEAR_FLOW),
        supported_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD),
        default_backend=ModelBackend.DNN,
        platforms=(Platform.WORKSTATION, Platform.VEGA, Platform.KAROLINA),
        artifact_classes=(ArtifactClass.REFERENCE, ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.METADATA),
        metadata={"legacy_roots": ("gv/buckling", "gv/torsion", "gv/eigenmodes", "gv/shear_flow")},
    ),
}


def list_agent_definitions() -> tuple[AgentDefinition, ...]:
    return tuple(AGENT_REGISTRY.values())


def get_agent_definition(family: AgentFamily | str) -> AgentDefinition:
    selected_family = coerce_agent_family(family)
    try:
        return AGENT_REGISTRY[selected_family]
    except KeyError as exc:
        supported = ", ".join(definition.family.value for definition in AGENT_REGISTRY.values())
        raise ValueError(f"Agent family '{selected_family.value}' is not registered. Expected one of: {supported}.") from exc


def supported_modalities_for_agent(family: AgentFamily | str) -> tuple[Modality, ...]:
    return get_agent_definition(family).supported_modalities


def resolve_agent_modality(family: AgentFamily | str, modality: Modality | str):
    definition = get_agent_definition(family)
    selected_modality = coerce_modality(modality)
    if selected_modality not in definition.supported_modalities:
        supported = ", ".join(item.value for item in definition.supported_modalities)
        raise ValueError(
            f"Agent family '{definition.family.value}' does not support modality '{selected_modality.value}'. "
            f"Supported modalities: {supported}."
        )
    descriptor = get_modality_descriptor(selected_modality)
    if descriptor.family != definition.family:
        raise ValueError(
            f"Registry mismatch: modality '{selected_modality.value}' belongs to '{descriptor.family.value}', "
            f"not '{definition.family.value}'."
        )
    return definition, descriptor


def list_agent_modalities(family: AgentFamily | str):
    definition = get_agent_definition(family)
    descriptors = {descriptor.modality: descriptor for descriptor in list_modality_descriptors(definition.family)}
    return tuple(descriptors[modality] for modality in definition.supported_modalities)


__all__ = [
    "AGENT_REGISTRY",
    "get_agent_definition",
    "list_agent_definitions",
    "list_agent_modalities",
    "resolve_agent_modality",
    "supported_modalities_for_agent",
]
