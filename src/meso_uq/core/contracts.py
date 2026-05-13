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
    BUCKLING = "buckling"
    TORSION = "torsion"
    EIGENMODES = "eigenmodes"
    SHEAR_FLOW = "shear_flow"


class ModelBackend(str, Enum):
    DNN = "dnn"
    BNN = "bnn"
    DPD = "dpd"
    SYNTHETIC = "synthetic"
    NONE = "none"


class Platform(str, Enum):
    WORKSTATION = "workstation"
    VEGA = "vega"
    KAROLINA = "karolina"
    GENERIC = "generic"


class ArtifactClass(str, Enum):
    RAW = "raw"
    REFERENCE = "reference"
    PROCESSED = "processed"
    GENERATED = "generated"
    SURROGATE = "surrogate"
    POSTERIOR = "posterior"
    CONFIG = "config"
    REPORT = "report"
    TRAINING_MANIFEST = "training_manifest"
    RUNTIME_MANIFEST = "runtime_manifest"
    RUN_MANIFEST = "run_manifest"
    METADATA = "metadata"
    FIGURE = "figure"
    LOG = "log"


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


def coerce_platform(value: Platform | str) -> Platform:
    return _coerce_enum(Platform, value, "platform")


def coerce_artifact_class(value: ArtifactClass | str) -> ArtifactClass:
    return _coerce_enum(ArtifactClass, value, "artifact class")


def _metadata_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metadata.items()}


@dataclass(frozen=True)
class ModalityDescriptor:
    modality: Modality
    family: AgentFamily
    label: str
    summary: str
    artifact_classes: tuple[ArtifactClass, ...] = ()
    model_backends: tuple[ModelBackend, ...] = ()
    platforms: tuple[Platform, ...] = (Platform.GENERIC,)
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
        object.__setattr__(self, "platforms", tuple(coerce_platform(item) for item in self.platforms))
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
            "platforms": [item.value for item in self.platforms],
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
            platforms=tuple(payload.get("platforms", (Platform.GENERIC.value,))),
            smoke_level=str(payload.get("smoke_level", "metadata")),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class AgentDefinition:
    family: AgentFamily
    label: str
    supported_modalities: tuple[Modality, ...]
    supported_backends: tuple[ModelBackend, ...] = (ModelBackend.NONE,)
    default_backend: ModelBackend = ModelBackend.NONE
    platforms: tuple[Platform, ...] = (Platform.GENERIC,)
    artifact_classes: tuple[ArtifactClass, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", coerce_agent_family(self.family))
        object.__setattr__(
            self,
            "supported_modalities",
            tuple(coerce_modality(item) for item in self.supported_modalities),
        )
        object.__setattr__(
            self,
            "supported_backends",
            tuple(coerce_model_backend(item) for item in self.supported_backends),
        )
        object.__setattr__(self, "default_backend", coerce_model_backend(self.default_backend))
        object.__setattr__(self, "platforms", tuple(coerce_platform(item) for item in self.platforms))
        object.__setattr__(
            self,
            "artifact_classes",
            tuple(coerce_artifact_class(item) for item in self.artifact_classes),
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "label": self.label,
            "supported_modalities": [item.value for item in self.supported_modalities],
            "supported_backends": [item.value for item in self.supported_backends],
            "default_backend": self.default_backend.value,
            "platforms": [item.value for item in self.platforms],
            "artifact_classes": [item.value for item in self.artifact_classes],
            "metadata": _metadata_dict(self.metadata),
        }


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    agent_family: AgentFamily
    modality: Modality
    model_backend: ModelBackend = ModelBackend.NONE
    platform: Platform = Platform.GENERIC
    artifact_classes: tuple[ArtifactClass, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_family", coerce_agent_family(self.agent_family))
        object.__setattr__(self, "modality", coerce_modality(self.modality))
        object.__setattr__(self, "model_backend", coerce_model_backend(self.model_backend))
        object.__setattr__(self, "platform", coerce_platform(self.platform))
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
            "platform": self.platform.value,
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
            platform=payload.get("platform", Platform.GENERIC.value),
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
    "ArtifactClass",
    "ManifestMetadata",
    "Modality",
    "ModalityDescriptor",
    "ModelBackend",
    "Platform",
    "RunMetadata",
    "coerce_agent_family",
    "coerce_artifact_class",
    "coerce_modality",
    "coerce_model_backend",
    "coerce_platform",
    "enum_values",
]
