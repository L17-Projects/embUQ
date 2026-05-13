from __future__ import annotations

from meso_uq.core import (
    AgentFamily,
    ArtifactClass,
    Modality,
    ModalityDescriptor,
    ModelBackend,
    Platform,
    coerce_agent_family,
    coerce_modality,
)


_COMMON_PLATFORMS = (Platform.WORKSTATION, Platform.VEGA, Platform.KAROLINA)

_MODALITIES: tuple[ModalityDescriptor, ...] = (
    ModalityDescriptor(
        modality=Modality.COMPRESSION,
        family=AgentFamily.EMB,
        label="EMB compression",
        summary="Elastic microbubble compression calibration and surrogate metadata.",
        artifact_classes=(ArtifactClass.REFERENCE, ArtifactClass.SURROGATE, ArtifactClass.TRAINING_MANIFEST),
        model_backends=(ModelBackend.DNN, ModelBackend.BNN, ModelBackend.DPD),
        platforms=_COMMON_PLATFORMS,
        metadata={"legacy_root": "compression", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.INDENTATION,
        family=AgentFamily.EMB,
        label="EMB indentation",
        summary="Elastic microbubble indentation calibration and surrogate metadata.",
        artifact_classes=(ArtifactClass.REFERENCE, ArtifactClass.SURROGATE, ArtifactClass.TRAINING_MANIFEST),
        model_backends=(ModelBackend.DNN, ModelBackend.BNN, ModelBackend.DPD),
        platforms=_COMMON_PLATFORMS,
        metadata={"legacy_root": "indentation", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.BUCKLING,
        family=AgentFamily.GV,
        label="GV buckling",
        summary="Gas vesicle buckling lane metadata for smoke-level orchestration.",
        artifact_classes=(ArtifactClass.REFERENCE, ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.METADATA),
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD),
        platforms=_COMMON_PLATFORMS,
        metadata={"legacy_root": "gv/buckling", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.TORSION,
        family=AgentFamily.GV,
        label="GV torsion",
        summary="Gas vesicle torsion lane metadata for smoke-level orchestration.",
        artifact_classes=(ArtifactClass.REFERENCE, ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.METADATA),
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD),
        platforms=_COMMON_PLATFORMS,
        metadata={"legacy_root": "gv/torsion", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.EIGENMODES,
        family=AgentFamily.GV,
        label="GV eigenmodes",
        summary="Gas vesicle eigenmodes lane metadata for smoke-level orchestration.",
        artifact_classes=(ArtifactClass.REFERENCE, ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.METADATA),
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD),
        platforms=_COMMON_PLATFORMS,
        metadata={"legacy_root": "gv/eigenmodes", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.SHEAR_FLOW,
        family=AgentFamily.GV,
        label="GV shear flow",
        summary="Experimental gas vesicle shear-flow lane metadata for smoke-level orchestration.",
        artifact_classes=(ArtifactClass.REFERENCE, ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.METADATA),
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD),
        platforms=_COMMON_PLATFORMS,
        metadata={
            "legacy_root": "gv/shear_flow",
            "smoke_scope": "metadata",
            "experimental": True,
            "known_runtime_status": "staging_out_of_scope",
        },
    ),
)

MODALITY_REGISTRY: dict[Modality, ModalityDescriptor] = {descriptor.modality: descriptor for descriptor in _MODALITIES}


def list_modality_descriptors(family: AgentFamily | str | None = None) -> tuple[ModalityDescriptor, ...]:
    if family is None:
        return tuple(MODALITY_REGISTRY.values())
    selected_family = coerce_agent_family(family)
    return tuple(descriptor for descriptor in MODALITY_REGISTRY.values() if descriptor.family == selected_family)


def get_modality_descriptor(modality: Modality | str) -> ModalityDescriptor:
    selected_modality = coerce_modality(modality)
    try:
        return MODALITY_REGISTRY[selected_modality]
    except KeyError as exc:
        supported = ", ".join(descriptor.modality.value for descriptor in MODALITY_REGISTRY.values())
        raise ValueError(f"Modality '{selected_modality.value}' is not registered. Expected one of: {supported}.") from exc


def assert_modality_family(modality: Modality | str, family: AgentFamily | str) -> ModalityDescriptor:
    descriptor = get_modality_descriptor(modality)
    selected_family = coerce_agent_family(family)
    if descriptor.family != selected_family:
        raise ValueError(
            f"Modality '{descriptor.modality.value}' belongs to agent family '{descriptor.family.value}', "
            f"not '{selected_family.value}'."
        )
    return descriptor


__all__ = [
    "MODALITY_REGISTRY",
    "assert_modality_family",
    "get_modality_descriptor",
    "list_modality_descriptors",
]
