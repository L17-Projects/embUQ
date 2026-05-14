from __future__ import annotations

from dataclasses import dataclass

from meso_uq.noise.contracts import (
    DiscrepancyConfig,
    MeasurementErrorConfig,
    NoiseModelConfig,
    NoiseModelSupportMetadata,
    PosteriorUncertaintyConfig,
    SurrogateErrorConfig,
    MeasurementErrorKind,
)


_MODEL_SUPPORT: dict[str, NoiseModelSupportMetadata] = {
    "emb/compression": NoiseModelSupportMetadata(
        model_id="emb/compression",
        family="emb",
        supported_observables=("force",),
        observable_units={"force": "micro_newton"},
        requires_measurement_error=True,
        required_measurement_error_kinds=(
            MeasurementErrorKind.ABSOLUTE_GAUSSIAN,
            MeasurementErrorKind.RELATIVE_GAUSSIAN,
        ),
        supports_discrepancy=False,
        supports_surrogate_error=True,
        supports_posterior_uncertainty=True,
    ),
    "gv/stretching": NoiseModelSupportMetadata(
        model_id="gv/stretching",
        family="gv",
        supported_observables=("extension", "force"),
        observable_units={"extension": "micrometer", "force": "nano_newton"},
        requires_measurement_error=True,
        required_measurement_error_kinds=(MeasurementErrorKind.ABSOLUTE_GAUSSIAN,),
        supports_discrepancy=True,
        supports_surrogate_error=True,
        supports_posterior_uncertainty=False,
    ),
}


def list_model_support_metadata() -> tuple[NoiseModelSupportMetadata, ...]:
    return tuple(_MODEL_SUPPORT.values())


def list_model_ids() -> tuple[str, ...]:
    return tuple(_MODEL_SUPPORT)


def get_model_support(model_id: str) -> NoiseModelSupportMetadata:
    key = str(model_id).strip().lower()
    try:
        return _MODEL_SUPPORT[key]
    except KeyError as exc:
        supported = ", ".join(_MODEL_SUPPORT)
        raise ValueError(f"Unknown noise model '{key}'. Expected one of: {supported}.") from exc


def build_model_config(
    model_id: str,
    *,
    observable: str,
    unit: str,
    measurement_error: MeasurementErrorConfig | None = None,
    discrepancy: DiscrepancyConfig | None = None,
    surrogate_error: SurrogateErrorConfig | None = None,
    posterior_uncertainty: PosteriorUncertaintyConfig | None = None,
) -> NoiseModelConfig:
    support = get_model_support(model_id)
    return NoiseModelConfig(
        support=support,
        observable=observable,
        unit=unit,
        measurement_error=measurement_error,
        discrepancy=discrepancy,
        surrogate_error=surrogate_error,
        posterior_uncertainty=posterior_uncertainty,
    )


def supports_for_observable(model_id: str, observable: str) -> bool:
    support = get_model_support(model_id)
    return support.supports_observable(observable)


@dataclass(frozen=True)
class NoiseModelSpec:
    model_id: str
    support: NoiseModelSupportMetadata


__all__ = [
    "NoiseModelSpec",
    "build_model_config",
    "get_model_support",
    "list_model_ids",
    "list_model_support_metadata",
    "supports_for_observable",
]
