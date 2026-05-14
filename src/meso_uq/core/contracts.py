from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeVar


class AgentFamily(str, Enum):
    EMB = "emb"
    GV = "gv"


class Modality(str, Enum):
    COMPRESSION = "compression"
    INDENTATION = "indentation"
    STRETCHING = "stretching"
    BUCKLING = "buckling"
    TORSION = "torsion"
    EIGENMODES = "eigenmodes"
    SHEAR_FLOW = "shear_flow"


class ModelBackend(str, Enum):
    DNN = "dnn"
    BNN = "bnn"
    PYRO_BNN = "pyro_bnn"
    DPD = "dpd"
    ANALYTICAL = "analytical"
    SYNTHETIC = "synthetic"
    NONE = "none"


class InferenceBackend(str, Enum):
    KORALI = "korali"
    PYRO = "pyro"
    DRY_RUN = "dry_run"
    NONE = "none"


class NoiseModelKind(str, Enum):
    NONE = "none"
    MEASUREMENT_ERROR = "measurement_error"
    DISCREPANCY = "discrepancy"
    SURROGATE_ERROR = "surrogate_error"
    HIERARCHICAL = "hierarchical"


class Platform(str, Enum):
    WORKSTATION = "workstation"
    VEGA = "vega"
    KAROLINA = "karolina"
    GENERIC_SLURM = "generic_slurm"
    GENERIC = "generic"


class ArtifactClass(str, Enum):
    RAW = "raw"
    REFERENCE = "reference"
    PROCESSED = "processed"
    SIMULATION_OUTPUT = "simulation_output"
    GENERATED = "generated"
    SURROGATE = "surrogate"
    SURROGATE_CHECKPOINT = "surrogate_checkpoint"
    POSTERIOR = "posterior"
    POSTERIOR_SAMPLE = "posterior_sample"
    CONFIG = "config"
    REPORT = "report"
    TRAINING_MANIFEST = "training_manifest"
    RUNTIME_MANIFEST = "runtime_manifest"
    RUN_MANIFEST = "run_manifest"
    METADATA = "metadata"
    FIGURE = "figure"
    LOG = "log"


class RuntimeCapability(str, Enum):
    SIMULATION = "simulation"
    SURROGATE_TRAINING = "surrogate_training"
    SURROGATE_PREDICTION = "surrogate_prediction"
    INFERENCE = "inference"
    ACTIVE_LEARNING = "active_learning"
    PLOTTING = "plotting"
    REPORTING = "reporting"


class RuntimeRequirementKind(str, Enum):
    PYTHON_PACKAGE = "python_package"
    SIMULATOR_BINARY = "simulator_binary"
    SOURCE_TREE = "source_tree"
    HPC_PLATFORM = "hpc_platform"
    ARTIFACT = "artifact"
    CONFIG = "config"


class RequirementState(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    EXTERNAL = "external"
    UNSUPPORTED = "unsupported"


class UnitSystem(str, Enum):
    SI = "si"
    MICROMETER = "micrometer"
    DIMENSIONLESS = "dimensionless"
    CUSTOM = "custom"


EnumT = TypeVar("EnumT", bound=Enum)


def enum_values(enum_type: type[EnumT]) -> tuple[str, ...]:
    return tuple(item.value for item in enum_type)


def _coerce_enum(enum_type: type[EnumT], value: EnumT | str, field_name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        expected = ", ".join(enum_values(enum_type))
        raise ValueError(f"Unsupported {field_name} '{value}'. Expected one of: {expected}.") from exc


def coerce_agent_family(value: AgentFamily | str) -> AgentFamily:
    return _coerce_enum(AgentFamily, value, "agent family")


def coerce_modality(value: Modality | str) -> Modality:
    return _coerce_enum(Modality, value, "modality")


def coerce_model_backend(value: ModelBackend | str) -> ModelBackend:
    return _coerce_enum(ModelBackend, value, "model backend")


def coerce_inference_backend(value: InferenceBackend | str) -> InferenceBackend:
    return _coerce_enum(InferenceBackend, value, "inference backend")


def coerce_noise_model_kind(value: NoiseModelKind | str) -> NoiseModelKind:
    return _coerce_enum(NoiseModelKind, value, "noise model kind")


def coerce_platform(value: Platform | str) -> Platform:
    return _coerce_enum(Platform, value, "platform")


def coerce_artifact_class(value: ArtifactClass | str) -> ArtifactClass:
    return _coerce_enum(ArtifactClass, value, "artifact class")


def coerce_runtime_capability(value: RuntimeCapability | str) -> RuntimeCapability:
    return _coerce_enum(RuntimeCapability, value, "runtime capability")


def coerce_runtime_requirement_kind(value: RuntimeRequirementKind | str) -> RuntimeRequirementKind:
    return _coerce_enum(RuntimeRequirementKind, value, "runtime requirement kind")


def coerce_requirement_state(value: RequirementState | str) -> RequirementState:
    return _coerce_enum(RequirementState, value, "requirement state")


def coerce_unit_system(value: UnitSystem | str) -> UnitSystem:
    return _coerce_enum(UnitSystem, value, "unit system")


def _metadata_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metadata.items()}


@dataclass(frozen=True)
class RuntimeRequirement:
    name: str
    kind: RuntimeRequirementKind
    state: RequirementState = RequirementState.REQUIRED
    description: str = ""
    package: str | None = None
    environment_variable: str | None = None
    platforms: tuple[Platform, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", coerce_runtime_requirement_kind(self.kind))
        object.__setattr__(self, "state", coerce_requirement_state(self.state))
        object.__setattr__(self, "platforms", tuple(coerce_platform(item) for item in self.platforms))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    @property
    def is_missing_dependency_state(self) -> bool:
        return self.state in {RequirementState.REQUIRED, RequirementState.EXTERNAL}

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "state": self.state.value,
            "description": self.description,
            "package": self.package,
            "environment_variable": self.environment_variable,
            "platforms": [item.value for item in self.platforms],
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeRequirement":
        return cls(
            name=str(payload["name"]),
            kind=payload["kind"],
            state=payload.get("state", RequirementState.REQUIRED.value),
            description=str(payload.get("description", "")),
            package=payload.get("package"),
            environment_variable=payload.get("environment_variable"),
            platforms=tuple(payload.get("platforms", ())),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class DatasetSourceMetadata:
    dataset_id: str
    source: str
    artifact_class: ArtifactClass = ArtifactClass.REFERENCE
    schema_version: str = "meso_uq.dataset.v1"
    path: str | None = None
    checksum: str | None = None
    unit_system: UnitSystem = UnitSystem.CUSTOM
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_class", coerce_artifact_class(self.artifact_class))
        object.__setattr__(self, "unit_system", coerce_unit_system(self.unit_system))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source": self.source,
            "artifact_class": self.artifact_class.value,
            "schema_version": self.schema_version,
            "path": self.path,
            "checksum": self.checksum,
            "unit_system": self.unit_system.value,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetSourceMetadata":
        return cls(
            dataset_id=str(payload["dataset_id"]),
            source=str(payload["source"]),
            artifact_class=payload.get("artifact_class", ArtifactClass.REFERENCE.value),
            schema_version=str(payload.get("schema_version", "meso_uq.dataset.v1")),
            path=payload.get("path"),
            checksum=payload.get("checksum"),
            unit_system=payload.get("unit_system", UnitSystem.CUSTOM.value),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    artifact_class: ArtifactClass
    uri: str
    checksum: str | None = None
    schema_version: str = "meso_uq.artifact_ref.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_class", coerce_artifact_class(self.artifact_class))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_class": self.artifact_class.value,
            "uri": self.uri,
            "checksum": self.checksum,
            "schema_version": self.schema_version,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactReference":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_class=payload["artifact_class"],
            uri=str(payload["uri"]),
            checksum=payload.get("checksum"),
            schema_version=str(payload.get("schema_version", "meso_uq.artifact_ref.v1")),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class SurrogateIdentifier:
    surrogate_id: str
    agent_family: AgentFamily
    modality: Modality
    backend: ModelBackend
    artifact: ArtifactReference | None = None
    schema_version: str = "meso_uq.surrogate.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_family", coerce_agent_family(self.agent_family))
        object.__setattr__(self, "modality", coerce_modality(self.modality))
        object.__setattr__(self, "backend", coerce_model_backend(self.backend))
        if self.artifact is not None and not isinstance(self.artifact, ArtifactReference):
            object.__setattr__(self, "artifact", ArtifactReference.from_dict(self.artifact))  # type: ignore[arg-type]
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "surrogate_id": self.surrogate_id,
            "agent_family": self.agent_family.value,
            "modality": self.modality.value,
            "backend": self.backend.value,
            "artifact": self.artifact.as_dict() if self.artifact else None,
            "schema_version": self.schema_version,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurrogateIdentifier":
        artifact = payload.get("artifact")
        return cls(
            surrogate_id=str(payload["surrogate_id"]),
            agent_family=payload["agent_family"],
            modality=payload["modality"],
            backend=payload["backend"],
            artifact=ArtifactReference.from_dict(artifact) if artifact else None,
            schema_version=str(payload.get("schema_version", "meso_uq.surrogate.v1")),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class ModalityDescriptor:
    modality: Modality
    family: AgentFamily
    label: str
    summary: str
    artifact_classes: tuple[ArtifactClass, ...] = ()
    model_backends: tuple[ModelBackend, ...] = ()
    inference_backends: tuple[InferenceBackend, ...] = (InferenceBackend.NONE,)
    platforms: tuple[Platform, ...] = (Platform.GENERIC,)
    capabilities: tuple[RuntimeCapability, ...] = ()
    runtime_requirements: tuple[RuntimeRequirement, ...] = ()
    input_controls: tuple[str, ...] = ()
    observables: Mapping[str, str] = field(default_factory=dict)
    surrogate_inputs: tuple[str, ...] = ()
    surrogate_outputs: tuple[str, ...] = ()
    config_schema: str | None = None
    artifact_manifest_kinds: tuple[ArtifactClass, ...] = ()
    smoke_level: str = "metadata"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", coerce_modality(self.modality))
        object.__setattr__(self, "family", coerce_agent_family(self.family))
        object.__setattr__(
            self,
            "artifact_classes",
            tuple(coerce_artifact_class(item) for item in self.artifact_classes),
        )
        object.__setattr__(
            self,
            "model_backends",
            tuple(coerce_model_backend(item) for item in self.model_backends),
        )
        object.__setattr__(
            self,
            "inference_backends",
            tuple(coerce_inference_backend(item) for item in self.inference_backends),
        )
        object.__setattr__(self, "platforms", tuple(coerce_platform(item) for item in self.platforms))
        object.__setattr__(
            self,
            "capabilities",
            tuple(coerce_runtime_capability(item) for item in self.capabilities),
        )
        object.__setattr__(
            self,
            "runtime_requirements",
            tuple(
                item if isinstance(item, RuntimeRequirement) else RuntimeRequirement.from_dict(item)
                for item in self.runtime_requirements
            ),
        )
        object.__setattr__(self, "input_controls", tuple(str(item) for item in self.input_controls))
        object.__setattr__(self, "observables", _metadata_dict(self.observables))
        object.__setattr__(self, "surrogate_inputs", tuple(str(item) for item in self.surrogate_inputs))
        object.__setattr__(self, "surrogate_outputs", tuple(str(item) for item in self.surrogate_outputs))
        object.__setattr__(
            self,
            "artifact_manifest_kinds",
            tuple(coerce_artifact_class(item) for item in self.artifact_manifest_kinds),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    @property
    def id(self) -> str:
        return self.modality.value

    @property
    def family_id(self) -> str:
        return self.family.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality.value,
            "family": self.family.value,
            "label": self.label,
            "summary": self.summary,
            "artifact_classes": [item.value for item in self.artifact_classes],
            "model_backends": [item.value for item in self.model_backends],
            "inference_backends": [item.value for item in self.inference_backends],
            "platforms": [item.value for item in self.platforms],
            "capabilities": [item.value for item in self.capabilities],
            "runtime_requirements": [item.as_dict() for item in self.runtime_requirements],
            "input_controls": list(self.input_controls),
            "observables": _metadata_dict(self.observables),
            "surrogate_inputs": list(self.surrogate_inputs),
            "surrogate_outputs": list(self.surrogate_outputs),
            "config_schema": self.config_schema,
            "artifact_manifest_kinds": [item.value for item in self.artifact_manifest_kinds],
            "smoke_level": self.smoke_level,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModalityDescriptor":
        return cls(
            modality=payload["modality"],
            family=payload["family"],
            label=str(payload["label"]),
            summary=str(payload["summary"]),
            artifact_classes=tuple(payload.get("artifact_classes", ())),
            model_backends=tuple(payload.get("model_backends", ())),
            inference_backends=tuple(payload.get("inference_backends", (InferenceBackend.NONE.value,))),
            platforms=tuple(payload.get("platforms", (Platform.GENERIC.value,))),
            capabilities=tuple(payload.get("capabilities", ())),
            runtime_requirements=tuple(payload.get("runtime_requirements", ())),
            input_controls=tuple(payload.get("input_controls", ())),
            observables=payload.get("observables", {}),
            surrogate_inputs=tuple(payload.get("surrogate_inputs", ())),
            surrogate_outputs=tuple(payload.get("surrogate_outputs", ())),
            config_schema=payload.get("config_schema"),
            artifact_manifest_kinds=tuple(payload.get("artifact_manifest_kinds", ())),
            smoke_level=str(payload.get("smoke_level", "metadata")),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class AgentDefinition:
    family: AgentFamily
    label: str
    supported_modalities: tuple[Modality, ...]
    aliases: tuple[str, ...] = ()
    supported_backends: tuple[ModelBackend, ...] = (ModelBackend.NONE,)
    default_backend: ModelBackend = ModelBackend.NONE
    inference_backends: tuple[InferenceBackend, ...] = (InferenceBackend.NONE,)
    platforms: tuple[Platform, ...] = (Platform.GENERIC,)
    artifact_classes: tuple[ArtifactClass, ...] = ()
    runtime_requirements: tuple[RuntimeRequirement, ...] = ()
    default_for_legacy: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", coerce_agent_family(self.family))
        object.__setattr__(
            self,
            "supported_modalities",
            tuple(coerce_modality(item) for item in self.supported_modalities),
        )
        object.__setattr__(self, "aliases", tuple(str(item).strip().lower() for item in self.aliases))
        object.__setattr__(
            self,
            "supported_backends",
            tuple(coerce_model_backend(item) for item in self.supported_backends),
        )
        object.__setattr__(self, "default_backend", coerce_model_backend(self.default_backend))
        object.__setattr__(
            self,
            "inference_backends",
            tuple(coerce_inference_backend(item) for item in self.inference_backends),
        )
        object.__setattr__(self, "platforms", tuple(coerce_platform(item) for item in self.platforms))
        object.__setattr__(
            self,
            "artifact_classes",
            tuple(coerce_artifact_class(item) for item in self.artifact_classes),
        )
        object.__setattr__(
            self,
            "runtime_requirements",
            tuple(
                item if isinstance(item, RuntimeRequirement) else RuntimeRequirement.from_dict(item)
                for item in self.runtime_requirements
            ),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if self.default_backend not in self.supported_backends:
            supported = ", ".join(item.value for item in self.supported_backends)
            raise ValueError(
                f"Default backend '{self.default_backend.value}' is not in supported backends: {supported}."
            )

    @property
    def id(self) -> str:
        return self.family.value

    def supports_modality(self, modality: Modality | str) -> bool:
        return coerce_modality(modality) in self.supported_modalities

    def supports_backend(self, backend: ModelBackend | str) -> bool:
        return coerce_model_backend(backend) in self.supported_backends

    def supports_inference_backend(self, backend: InferenceBackend | str) -> bool:
        return coerce_inference_backend(backend) in self.inference_backends

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "label": self.label,
            "supported_modalities": [item.value for item in self.supported_modalities],
            "aliases": list(self.aliases),
            "supported_backends": [item.value for item in self.supported_backends],
            "default_backend": self.default_backend.value,
            "inference_backends": [item.value for item in self.inference_backends],
            "platforms": [item.value for item in self.platforms],
            "artifact_classes": [item.value for item in self.artifact_classes],
            "runtime_requirements": [item.as_dict() for item in self.runtime_requirements],
            "default_for_legacy": self.default_for_legacy,
            "metadata": _metadata_dict(self.metadata),
        }


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    agent_family: AgentFamily
    modality: Modality
    model_backend: ModelBackend = ModelBackend.NONE
    inference_backend: InferenceBackend = InferenceBackend.NONE
    noise_model: NoiseModelKind = NoiseModelKind.NONE
    platform: Platform = Platform.GENERIC
    experiment_id: str | None = None
    code_version: str | None = None
    config_hash: str | None = None
    data_hash: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)
    artifact_classes: tuple[ArtifactClass, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_family", coerce_agent_family(self.agent_family))
        object.__setattr__(self, "modality", coerce_modality(self.modality))
        object.__setattr__(self, "model_backend", coerce_model_backend(self.model_backend))
        object.__setattr__(self, "inference_backend", coerce_inference_backend(self.inference_backend))
        object.__setattr__(self, "noise_model", coerce_noise_model_kind(self.noise_model))
        object.__setattr__(self, "platform", coerce_platform(self.platform))
        object.__setattr__(self, "environment", _metadata_dict(self.environment))
        object.__setattr__(
            self,
            "artifact_classes",
            tuple(coerce_artifact_class(item) for item in self.artifact_classes),
        )
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_family": self.agent_family.value,
            "modality": self.modality.value,
            "model_backend": self.model_backend.value,
            "inference_backend": self.inference_backend.value,
            "noise_model": self.noise_model.value,
            "platform": self.platform.value,
            "experiment_id": self.experiment_id,
            "code_version": self.code_version,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "environment": _metadata_dict(self.environment),
            "artifact_classes": [item.value for item in self.artifact_classes],
            "tags": list(self.tags),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunMetadata":
        return cls(
            run_id=str(payload["run_id"]),
            agent_family=payload["agent_family"],
            modality=payload["modality"],
            model_backend=payload.get("model_backend", ModelBackend.NONE.value),
            inference_backend=payload.get("inference_backend", InferenceBackend.NONE.value),
            noise_model=payload.get("noise_model", NoiseModelKind.NONE.value),
            platform=payload.get("platform", Platform.GENERIC.value),
            experiment_id=payload.get("experiment_id"),
            code_version=payload.get("code_version"),
            config_hash=payload.get("config_hash"),
            data_hash=payload.get("data_hash"),
            environment=payload.get("environment", {}),
            artifact_classes=tuple(payload.get("artifact_classes", ())),
            tags=tuple(payload.get("tags", ())),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class ManifestMetadata:
    schema_version: str
    manifest_kind: ArtifactClass
    run: RunMetadata
    generated_by: str = "meso_uq"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_kind", coerce_artifact_class(self.manifest_kind))
        if not isinstance(self.run, RunMetadata):
            object.__setattr__(self, "run", RunMetadata.from_dict(self.run))  # type: ignore[arg-type]
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_kind": self.manifest_kind.value,
            "generated_by": self.generated_by,
            "run": self.run.as_dict(),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifestMetadata":
        return cls(
            schema_version=str(payload["schema_version"]),
            manifest_kind=payload["manifest_kind"],
            generated_by=str(payload.get("generated_by", "meso_uq")),
            run=RunMetadata.from_dict(payload["run"]),
            metadata=payload.get("metadata", {}),
        )

    def to_json(self, **json_kwargs: Any) -> str:
        kwargs = {"sort_keys": True}
        kwargs.update(json_kwargs)
        return json.dumps(self.as_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "ManifestMetadata":
        return cls.from_dict(json.loads(payload))


__all__ = [
    "AgentDefinition",
    "AgentFamily",
    "ArtifactReference",
    "ArtifactClass",
    "DatasetSourceMetadata",
    "InferenceBackend",
    "ManifestMetadata",
    "Modality",
    "ModalityDescriptor",
    "ModelBackend",
    "NoiseModelKind",
    "Platform",
    "RequirementState",
    "RuntimeCapability",
    "RuntimeRequirement",
    "RuntimeRequirementKind",
    "RunMetadata",
    "SurrogateIdentifier",
    "UnitSystem",
    "coerce_agent_family",
    "coerce_artifact_class",
    "coerce_inference_backend",
    "coerce_modality",
    "coerce_model_backend",
    "coerce_noise_model_kind",
    "coerce_platform",
    "coerce_requirement_state",
    "coerce_runtime_capability",
    "coerce_runtime_requirement_kind",
    "coerce_unit_system",
    "enum_values",
]
