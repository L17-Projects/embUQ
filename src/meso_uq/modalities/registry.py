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
_GV_RUNTIME_REQUIREMENTS = (
    RuntimeRequirement(
        name="mirheo",
        kind=RuntimeRequirementKind.PYTHON_PACKAGE,
        state=RequirementState.EXTERNAL,
        package="mirheo",
        description="Required only when the standard GV runtime lane is executed.",
        platforms=(Platform.VEGA, Platform.KAROLINA),
    ),
)
_GV_SURROGATE_REQUIREMENTS = (
    RuntimeRequirement(
        name="torch",
        kind=RuntimeRequirementKind.PYTHON_PACKAGE,
        state=RequirementState.OPTIONAL,
        package="torch",
        description="Required only when GV surrogate training or prediction is invoked.",
    ),
)
_SURROGATE_ARTIFACTS = (
    ArtifactClass.REFERENCE,
    ArtifactClass.SURROGATE,
    ArtifactClass.SURROGATE_CHECKPOINT,
    ArtifactClass.TRAINING_MANIFEST,
)
_GV_ARTIFACTS = (
    ArtifactClass.REFERENCE,
    ArtifactClass.RUNTIME_MANIFEST,
    ArtifactClass.METADATA,
    ArtifactClass.SIMULATION_OUTPUT,
)
_EMB_CAPABILITIES = (
    RuntimeCapability.SIMULATION,
    RuntimeCapability.SURROGATE_TRAINING,
    RuntimeCapability.SURROGATE_PREDICTION,
    RuntimeCapability.INFERENCE,
    RuntimeCapability.PLOTTING,
    RuntimeCapability.REPORTING,
)
_GV_CAPABILITIES = (
    RuntimeCapability.SIMULATION,
    RuntimeCapability.SURROGATE_PREDICTION,
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
    ModalityDescriptor(
        modality=Modality.STRETCHING,
        family=AgentFamily.GV,
        label="GV stretching",
        summary="Gas vesicle stretching lane metadata for manifest-driven runtime staging.",
        artifact_classes=_GV_ARTIFACTS,
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD, ModelBackend.ANALYTICAL),
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.DRY_RUN),
        platforms=_COMMON_PLATFORMS,
        capabilities=_GV_CAPABILITIES,
        runtime_requirements=(*_GV_RUNTIME_REQUIREMENTS, *_GV_SURROGATE_REQUIREMENTS),
        input_controls=("tot_force", "bpress"),
        observables={"extension_curve": "curve"},
        surrogate_inputs=("calibrated_parameters", "radius", "height", "tot_force", "bpress", "observable_axis"),
        surrogate_outputs=("response",),
        config_schema="meso_uq.gv.stretching.v1",
        artifact_manifest_kinds=(ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.TRAINING_MANIFEST),
        metadata={"legacy_root": "gv/stretching", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.BUCKLING,
        family=AgentFamily.GV,
        label="GV buckling",
        summary="Gas vesicle buckling lane metadata for smoke-level orchestration.",
        artifact_classes=_GV_ARTIFACTS,
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD, ModelBackend.ANALYTICAL),
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.DRY_RUN),
        platforms=_COMMON_PLATFORMS,
        capabilities=_GV_CAPABILITIES,
        runtime_requirements=(*_GV_RUNTIME_REQUIREMENTS, *_GV_SURROGATE_REQUIREMENTS),
        input_controls=("buck", "bpress"),
        observables={"buckling_response": "curve"},
        surrogate_inputs=("calibrated_parameters", "radius", "height", "buck", "bpress", "observable_axis"),
        surrogate_outputs=("response",),
        config_schema="meso_uq.gv.buckling.v1",
        artifact_manifest_kinds=(ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.TRAINING_MANIFEST),
        metadata={"legacy_root": "gv/buckling", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.TORSION,
        family=AgentFamily.GV,
        label="GV torsion",
        summary="Gas vesicle torsion lane metadata for smoke-level orchestration.",
        artifact_classes=_GV_ARTIFACTS,
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD, ModelBackend.ANALYTICAL),
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.DRY_RUN),
        platforms=_COMMON_PLATFORMS,
        capabilities=_GV_CAPABILITIES,
        runtime_requirements=(*_GV_RUNTIME_REQUIREMENTS, *_GV_SURROGATE_REQUIREMENTS),
        input_controls=("theta",),
        observables={"torsion_response": "curve"},
        surrogate_inputs=("calibrated_parameters", "radius", "height", "theta", "observable_axis"),
        surrogate_outputs=("response",),
        config_schema="meso_uq.gv.torsion.v1",
        artifact_manifest_kinds=(ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.TRAINING_MANIFEST),
        metadata={"legacy_root": "gv/torsion", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.EIGENMODES,
        family=AgentFamily.GV,
        label="GV eigenmodes",
        summary="Gas vesicle eigenmodes lane metadata for smoke-level orchestration.",
        artifact_classes=_GV_ARTIFACTS,
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD, ModelBackend.ANALYTICAL),
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.DRY_RUN),
        platforms=_COMMON_PLATFORMS,
        capabilities=_GV_CAPABILITIES,
        runtime_requirements=(*_GV_RUNTIME_REQUIREMENTS, *_GV_SURROGATE_REQUIREMENTS),
        input_controls=("bpress",),
        observables={"eigenmode_spectrum": "spectrum"},
        surrogate_inputs=("calibrated_parameters", "radius", "height", "bpress", "observable_axis"),
        surrogate_outputs=("response",),
        config_schema="meso_uq.gv.eigenmodes.v1",
        artifact_manifest_kinds=(ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.TRAINING_MANIFEST),
        metadata={"legacy_root": "gv/eigenmodes", "smoke_scope": "metadata"},
    ),
    ModalityDescriptor(
        modality=Modality.SHEAR_FLOW,
        family=AgentFamily.GV,
        label="GV shear flow",
        summary="Experimental gas vesicle shear-flow lane metadata for smoke-level orchestration.",
        artifact_classes=_GV_ARTIFACTS,
        model_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD, ModelBackend.ANALYTICAL),
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.DRY_RUN),
        platforms=_COMMON_PLATFORMS,
        capabilities=_GV_CAPABILITIES,
        runtime_requirements=(
            RuntimeRequirement(
                name="mirheoOBMD",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.EXTERNAL,
                package="mirheoOBMD",
                description="Required only when the experimental GV shear-flow runtime is invoked.",
                platforms=(Platform.VEGA, Platform.KAROLINA),
            ),
            *_GV_SURROGATE_REQUIREMENTS,
        ),
        input_controls=("ptan", "afsi", "bpress"),
        observables={"shear_flow_response": "curve"},
        surrogate_inputs=("calibrated_parameters", "radius", "height", "ptan", "afsi", "bpress", "shear_coord"),
        surrogate_outputs=("shear_response",),
        config_schema="meso_uq.gv.shear_flow.v1",
        artifact_manifest_kinds=(ArtifactClass.RUNTIME_MANIFEST, ArtifactClass.TRAINING_MANIFEST),
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
