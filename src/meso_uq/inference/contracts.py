from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.core import (
    AgentFamily,
    ArtifactClass,
    InferenceBackend,
    Modality,
    ModelBackend,
    NoiseModelKind,
    Platform,
    RequirementState,
    RuntimeRequirement,
    coerce_agent_family,
    coerce_artifact_class,
    coerce_inference_backend,
    coerce_modality,
    coerce_model_backend,
    coerce_noise_model_kind,
    coerce_platform,
)


class InferenceLayer(str, Enum):
    INDIVIDUAL = "individual"
    POPULATION = "population"
    MEASUREMENT = "measurement"
    DISCREPANCY = "discrepancy"
    SURROGATE_ERROR = "surrogate_error"


class InferenceSupportState(str, Enum):
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class PosteriorStorageKind(str, Enum):
    SUMMARY = "summary"
    SAMPLES = "samples"
    TRACE = "trace"
    DIAGNOSTICS = "diagnostics"


def _as_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(str(value) for value in (values or ()))


def _metadata_dict(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): value for key, value in (metadata or {}).items()}


def _coerce_layer(value: InferenceLayer | str) -> InferenceLayer:
    if isinstance(value, InferenceLayer):
        return value
    try:
        return InferenceLayer(str(value))
    except ValueError as exc:
        expected = ", ".join(item.value for item in InferenceLayer)
        raise ValueError(f"Unsupported inference layer '{value}'. Expected one of: {expected}.") from exc


def _coerce_support_state(value: InferenceSupportState | str) -> InferenceSupportState:
    if isinstance(value, InferenceSupportState):
        return value
    try:
        return InferenceSupportState(str(value))
    except ValueError as exc:
        expected = ", ".join(item.value for item in InferenceSupportState)
        raise ValueError(f"Unsupported inference support state '{value}'. Expected one of: {expected}.") from exc


def _coerce_posterior_storage_kind(value: PosteriorStorageKind | str) -> PosteriorStorageKind:
    if isinstance(value, PosteriorStorageKind):
        return value
    try:
        return PosteriorStorageKind(str(value))
    except ValueError as exc:
        expected = ", ".join(item.value for item in PosteriorStorageKind)
        raise ValueError(f"Unsupported posterior storage kind '{value}'. Expected one of: {expected}.") from exc


@dataclass(frozen=True)
class PriorContract:
    name: str
    layer: InferenceLayer
    bounds: tuple[float, float]
    units: str = "dimensionless"
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _coerce_layer(self.layer))
        if len(self.bounds) != 2:
            raise ValueError(f"Prior '{self.name}' must define exactly two bounds.")
        lower, upper = float(self.bounds[0]), float(self.bounds[1])
        if lower >= upper:
            raise ValueError(f"Prior '{self.name}' has invalid bounds: {self.bounds}.")
        object.__setattr__(self, "bounds", (lower, upper))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer.value,
            "bounds": list(self.bounds),
            "units": self.units,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PriorContract":
        return cls(
            name=str(payload["name"]),
            layer=str(payload["layer"]),
            bounds=tuple(payload["bounds"]),
            units=str(payload.get("units", "dimensionless")),
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True)
class LikelihoodComponentContract:
    name: str
    noise_model: NoiseModelKind
    layer: InferenceLayer
    observables: tuple[str, ...]
    required_surrogate_uncertainty: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "noise_model", coerce_noise_model_kind(self.noise_model))
        object.__setattr__(self, "layer", _coerce_layer(self.layer))
        object.__setattr__(self, "observables", _as_tuple(self.observables))
        object.__setattr__(
            self,
            "required_surrogate_uncertainty",
            _as_tuple(self.required_surrogate_uncertainty),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if not self.observables:
            raise ValueError(f"Likelihood component '{self.name}' must declare observables.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "noise_model": self.noise_model.value,
            "layer": self.layer.value,
            "observables": list(self.observables),
            "required_surrogate_uncertainty": list(self.required_surrogate_uncertainty),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LikelihoodComponentContract":
        return cls(
            name=str(payload["name"]),
            noise_model=payload["noise_model"],
            layer=payload["layer"],
            observables=tuple(payload.get("observables", ())),
            required_surrogate_uncertainty=tuple(payload.get("required_surrogate_uncertainty", ())),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class SamplerBackendContract:
    backend: InferenceBackend
    requirements: tuple[RuntimeRequirement, ...] = ()
    settings_schema_version: str = "meso_uq.inference.sampler.v1"
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", coerce_inference_backend(self.backend))
        object.__setattr__(self, "requirements", tuple(self.requirements))
        object.__setattr__(self, "settings", _metadata_dict(self.settings))

    def missing_requirement_messages(self) -> tuple[str, ...]:
        messages: list[str] = []
        for requirement in self.requirements:
            if requirement.state in {RequirementState.REQUIRED, RequirementState.EXTERNAL}:
                label = requirement.package or requirement.environment_variable or requirement.name
                messages.append(
                    f"{self.backend.value} requires {label}: {requirement.description or requirement.kind.value}"
                )
        return tuple(messages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "requirements": [requirement.as_dict() for requirement in self.requirements],
            "settings_schema_version": self.settings_schema_version,
            "settings": _metadata_dict(self.settings),
        }


@dataclass(frozen=True)
class InferenceContract:
    contract_id: str
    agent_family: AgentFamily
    modality: Modality
    dataset_id: str
    surrogate_backend: ModelBackend
    sampler: SamplerBackendContract
    priors: tuple[PriorContract, ...]
    likelihood_components: tuple[LikelihoodComponentContract, ...]
    platform: Platform = Platform.GENERIC
    support_state: InferenceSupportState = InferenceSupportState.SUPPORTED
    artifact_class: ArtifactClass = ArtifactClass.POSTERIOR
    surrogate_uncertainty_fields: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_family", coerce_agent_family(self.agent_family))
        object.__setattr__(self, "modality", coerce_modality(self.modality))
        object.__setattr__(self, "surrogate_backend", coerce_model_backend(self.surrogate_backend))
        object.__setattr__(self, "platform", coerce_platform(self.platform))
        object.__setattr__(self, "support_state", _coerce_support_state(self.support_state))
        object.__setattr__(self, "artifact_class", coerce_artifact_class(self.artifact_class))
        object.__setattr__(self, "priors", tuple(self.priors))
        object.__setattr__(self, "likelihood_components", tuple(self.likelihood_components))
        object.__setattr__(self, "surrogate_uncertainty_fields", _as_tuple(self.surrogate_uncertainty_fields))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if not self.priors:
            raise ValueError(f"Inference contract '{self.contract_id}' must declare priors.")
        if not self.likelihood_components:
            raise ValueError(f"Inference contract '{self.contract_id}' must declare likelihood components.")
        if self.support_state is InferenceSupportState.UNSUPPORTED and not self.unsupported_reason:
            raise ValueError(f"Unsupported inference contract '{self.contract_id}' must explain why.")

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.support_state is InferenceSupportState.UNSUPPORTED:
            errors.append(f"{self.contract_id} is unsupported: {self.unsupported_reason}")
        available_uncertainty = set(self.surrogate_uncertainty_fields)
        for component in self.likelihood_components:
            missing = sorted(set(component.required_surrogate_uncertainty) - available_uncertainty)
            if missing:
                errors.append(
                    f"{component.name} requires surrogate uncertainty fields not provided by "
                    f"{self.surrogate_backend.value}: {missing}"
                )
        errors.extend(self.sampler.missing_requirement_messages())
        return tuple(errors)

    def require_supported(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise ValueError("; ".join(errors))

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "agent_family": self.agent_family.value,
            "modality": self.modality.value,
            "dataset_id": self.dataset_id,
            "surrogate_backend": self.surrogate_backend.value,
            "sampler": self.sampler.as_dict(),
            "priors": [prior.as_dict() for prior in self.priors],
            "likelihood_components": [component.as_dict() for component in self.likelihood_components],
            "platform": self.platform.value,
            "support_state": self.support_state.value,
            "artifact_class": self.artifact_class.value,
            "surrogate_uncertainty_fields": list(self.surrogate_uncertainty_fields),
            "unsupported_reason": self.unsupported_reason,
            "metadata": _metadata_dict(self.metadata),
        }


@dataclass(frozen=True)
class PosteriorArtifactContract:
    artifact_id: str
    path: str
    storage_kind: PosteriorStorageKind
    parameters: tuple[str, ...]
    sample_count: int | None = None
    schema_version: str = "meso_uq.posterior_artifact.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage_kind", _coerce_posterior_storage_kind(self.storage_kind))
        object.__setattr__(self, "parameters", _as_tuple(self.parameters))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if Path(self.path).is_absolute():
            raise ValueError("Posterior artifact paths must be relative or placeholder-based.")
        if self.sample_count is not None and int(self.sample_count) < 0:
            raise ValueError("Posterior sample_count must be non-negative.")

    def as_manifest_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_class": ArtifactClass.POSTERIOR_SAMPLE.value
            if self.storage_kind is PosteriorStorageKind.SAMPLES
            else ArtifactClass.POSTERIOR.value,
            "path": self.path,
            "storage_kind": self.storage_kind.value,
            "parameters": list(self.parameters),
            "sample_count": self.sample_count,
            "metadata": _metadata_dict(self.metadata),
        }


def unsupported_gv_inference_contract(
    *,
    contract_id: str,
    modality: Modality | str,
    dataset_id: str,
    reason: str,
) -> InferenceContract:
    return InferenceContract(
        contract_id=contract_id,
        agent_family=AgentFamily.GV,
        modality=modality,
        dataset_id=dataset_id,
        surrogate_backend=ModelBackend.DNN,
        sampler=SamplerBackendContract(backend=InferenceBackend.NONE),
        priors=(PriorContract("placeholder", InferenceLayer.POPULATION, (0.0, 1.0)),),
        likelihood_components=(
            LikelihoodComponentContract(
                name="placeholder",
                noise_model=NoiseModelKind.MEASUREMENT_ERROR,
                layer=InferenceLayer.MEASUREMENT,
                observables=("placeholder",),
            ),
        ),
        support_state=InferenceSupportState.UNSUPPORTED,
        unsupported_reason=reason,
    )
