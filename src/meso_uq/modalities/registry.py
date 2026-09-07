from __future__ import annotations

from meso_uq.core import (
    AgentFamily,
    ArtifactClass,
    InferenceBackend,
    Modality,
    ModalityDescriptor,
    ModelBackend,
    Platform,
    RequirementState,
    RuntimeCapability,
    RuntimeRequirement,
    RuntimeRequirementKind,
    coerce_agent_family,
    coerce_modality,
)


_COMMON_PLATFORMS = (Platform.WORKSTATION, Platform.VEGA, Platform.KAROLINA)
_EMB_SURROGATE_REQUIREMENTS = (
    RuntimeRequirement(
        name="torch",
        kind=RuntimeRequirementKind.PYTHON_PACKAGE,
        state=RequirementState.REQUIRED,
        package="torch",
        description="Required for EMB surrogate training and prediction when surrogate workflows are invoked.",
    ),
    RuntimeRequirement(
        name="pyro",
        kind=RuntimeRequirementKind.PYTHON_PACKAGE,
        state=RequirementState.OPTIONAL,
        package="pyro",
        description="Required only for BNN/Pyro EMB surrogate workflows.",
    ),
)
_SURROGATE_ARTIFACTS = (
    ArtifactClass.REFERENCE,
    ArtifactClass.SURROGATE,
    ArtifactClass.SURROGATE_CHECKPOINT,
    ArtifactClass.TRAINING_MANIFEST,
)
_EMB_CAPABILITIES = (
    RuntimeCapability.SIMULATION,
    RuntimeCapability.SURROGATE_TRAINING,
    RuntimeCapability.SURROGATE_PREDICTION,
    RuntimeCapability.INFERENCE,
    RuntimeCapability.PLOTTING,
    RuntimeCapability.REPORTING,
)
_MODALITIES: tuple[ModalityDescriptor, ...] = (
    ModalityDescriptor(
        modality=Modality.COMPRESSION,
        family=AgentFamily.EMB,
        label="EMB compression",
        summary="Elastic microbubble compression calibration and surrogate metadata.",
        artifact_classes=_SURROGATE_ARTIFACTS,
        model_backends=(ModelBackend.DNN, ModelBackend.BNN, ModelBackend.PYRO_BNN, ModelBackend.DPD),
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.PYRO, InferenceBackend.DRY_RUN),
        platforms=_COMMON_PLATFORMS,
        capabilities=_EMB_CAPABILITIES,
        runtime_requirements=_EMB_SURROGATE_REQUIREMENTS,
        input_controls=("diameter_um", "displacement"),
        observables={"force": "force", "displacement": "length"},
        surrogate_inputs=("initial_diameter", "displacement"),
        surrogate_outputs=("force",),
        config_schema="meso_uq.emb.compression.v1",
        artifact_manifest_kinds=(ArtifactClass.TRAINING_MANIFEST, ArtifactClass.RUN_MANIFEST),
        metadata={"legacy_root": "emb/compression", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.INDENTATION,
        family=AgentFamily.EMB,
        label="EMB indentation",
        summary="Elastic microbubble indentation calibration and surrogate metadata.",
        artifact_classes=_SURROGATE_ARTIFACTS,
        model_backends=(ModelBackend.DNN, ModelBackend.BNN, ModelBackend.PYRO_BNN, ModelBackend.DPD),
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.PYRO, InferenceBackend.DRY_RUN),
        platforms=_COMMON_PLATFORMS,
        capabilities=_EMB_CAPABILITIES,
        runtime_requirements=_EMB_SURROGATE_REQUIREMENTS,
        input_controls=("diameter_um", "force"),
        observables={"force": "force", "displacement": "length"},
        surrogate_inputs=("initial_diameter", "force"),
        surrogate_outputs=("displacement",),
        config_schema="meso_uq.emb.indentation.v1",
        artifact_manifest_kinds=(ArtifactClass.TRAINING_MANIFEST, ArtifactClass.RUN_MANIFEST),
        metadata={"legacy_root": "emb/indentation", "smoke_scope": "metadata"},
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
