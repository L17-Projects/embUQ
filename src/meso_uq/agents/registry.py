from __future__ import annotations

from meso_uq.core import (
    AgentDefinition,
    AgentFamily,
    ArtifactClass,
    InferenceBackend,
    Modality,
    ModelBackend,
    Platform,
    RequirementState,
    RuntimeRequirement,
    RuntimeRequirementKind,
    coerce_agent_family,
    coerce_modality,
)
from meso_uq.modalities import get_modality_descriptor, list_modality_descriptors


AGENT_REGISTRY: dict[AgentFamily, AgentDefinition] = {
    AgentFamily.EMB: AgentDefinition(
        family=AgentFamily.EMB,
        label="Elastic microbubble",
        aliases=("elastic_microbubble", "elastic-microbubble", "microbubble", "uqdpd", "emb"),
        supported_modalities=(Modality.COMPRESSION, Modality.INDENTATION),
        supported_backends=(ModelBackend.DNN, ModelBackend.BNN, ModelBackend.PYRO_BNN, ModelBackend.DPD),
        default_backend=ModelBackend.DNN,
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.PYRO, InferenceBackend.DRY_RUN),
        platforms=(Platform.WORKSTATION, Platform.VEGA, Platform.KAROLINA),
        artifact_classes=(
            ArtifactClass.REFERENCE,
            ArtifactClass.SURROGATE,
            ArtifactClass.SURROGATE_CHECKPOINT,
            ArtifactClass.TRAINING_MANIFEST,
        ),
        runtime_requirements=(
            RuntimeRequirement(
                name="torch",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.REQUIRED,
                package="torch",
                description="Required for EMB surrogate training and prediction.",
            ),
            RuntimeRequirement(
                name="pyro",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.OPTIONAL,
                package="pyro",
                description="Required only for BNN/Pyro surrogate workflows.",
            ),
            RuntimeRequirement(
                name="korali",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.EXTERNAL,
                package="korali",
                description="Required only for Korali-backed inference phases.",
                platforms=(Platform.VEGA, Platform.KAROLINA),
            ),
        ),
        default_for_legacy=True,
        metadata={"legacy_roots": ("compression", "indentation")},
    ),
    AgentFamily.GV: AgentDefinition(
        family=AgentFamily.GV,
        label="Gas vesicle",
        aliases=("gas_vesicle", "gas-vesicle", "vesicle", "gv"),
        supported_modalities=(
            Modality.STRETCHING,
            Modality.BUCKLING,
            Modality.TORSION,
            Modality.EIGENMODES,
            Modality.SHEAR_FLOW,
        ),
        supported_backends=(ModelBackend.DNN, ModelBackend.SYNTHETIC, ModelBackend.DPD, ModelBackend.ANALYTICAL),
        default_backend=ModelBackend.DNN,
        inference_backends=(InferenceBackend.KORALI, InferenceBackend.DRY_RUN),
        platforms=(Platform.WORKSTATION, Platform.VEGA, Platform.KAROLINA),
        artifact_classes=(
            ArtifactClass.REFERENCE,
            ArtifactClass.RUNTIME_MANIFEST,
            ArtifactClass.METADATA,
            ArtifactClass.SIMULATION_OUTPUT,
        ),
        runtime_requirements=(
            RuntimeRequirement(
                name="mirheo",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.EXTERNAL,
                package="mirheo",
                description="Standard GV runtime package for non-shear lanes when runtime execution is invoked.",
                platforms=(Platform.VEGA, Platform.KAROLINA),
            ),
            RuntimeRequirement(
                name="mirheoOBMD",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.EXTERNAL,
                package="mirheoOBMD",
                description="Shear-flow runtime package used only by the experimental GV shear-flow lane.",
                platforms=(Platform.VEGA, Platform.KAROLINA),
            ),
        ),
        default_for_legacy=False,
        metadata={"legacy_roots": ("gv/stretching", "gv/buckling", "gv/torsion", "gv/eigenmodes", "gv/shear_flow")},
    ),
}

_AGENT_ALIAS_TO_FAMILY: dict[str, AgentFamily] = {
    alias: definition.family for definition in AGENT_REGISTRY.values() for alias in (definition.family.value, *definition.aliases)
}


def list_agent_definitions() -> tuple[AgentDefinition, ...]:
    return tuple(AGENT_REGISTRY.values())


def resolve_agent_family_identifier(family: AgentFamily | str) -> AgentFamily:
    if isinstance(family, AgentFamily):
        return family
    key = str(family).strip().lower()
    if key in _AGENT_ALIAS_TO_FAMILY:
        return _AGENT_ALIAS_TO_FAMILY[key]
    return coerce_agent_family(key)


def get_agent_definition(family: AgentFamily | str) -> AgentDefinition:
    selected_family = resolve_agent_family_identifier(family)
    try:
        return AGENT_REGISTRY[selected_family]
    except KeyError as exc:
        supported = ", ".join(definition.family.value for definition in AGENT_REGISTRY.values())
        raise ValueError(f"Agent family '{selected_family.value}' is not registered. Expected one of: {supported}.") from exc


def supported_modalities_for_agent(family: AgentFamily | str) -> tuple[Modality, ...]:
    return get_agent_definition(family).supported_modalities


def runtime_requirements_for_agent(family: AgentFamily | str) -> tuple[RuntimeRequirement, ...]:
    return get_agent_definition(family).runtime_requirements


def missing_dependency_requirements_for_agent(
    family: AgentFamily | str,
    availability: dict[str, bool],
) -> tuple[RuntimeRequirement, ...]:
    missing = []
    for requirement in runtime_requirements_for_agent(family):
        key = requirement.package or requirement.name
        if not requirement.is_missing_dependency_state:
            continue
        if availability.get(key, True) is False:
            missing.append(requirement)
    return tuple(missing)


def classify_agent_modality_support(family: AgentFamily | str, modality: Modality | str) -> dict[str, str]:
    definition = get_agent_definition(family)
    selected_modality = coerce_modality(modality)
    if selected_modality not in definition.supported_modalities:
        supported = ", ".join(item.value for item in definition.supported_modalities)
        return {
            "status": "unsupported",
            "agent_family": definition.family.value,
            "modality": selected_modality.value,
            "reason": f"Supported modalities: {supported}.",
        }
    return {
        "status": "supported",
        "agent_family": definition.family.value,
        "modality": selected_modality.value,
        "reason": "Declared in the agent-family registry.",
    }


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
    "classify_agent_modality_support",
    "get_agent_definition",
    "list_agent_definitions",
    "list_agent_modalities",
    "missing_dependency_requirements_for_agent",
    "resolve_agent_modality",
    "resolve_agent_family_identifier",
    "runtime_requirements_for_agent",
    "supported_modalities_for_agent",
]
