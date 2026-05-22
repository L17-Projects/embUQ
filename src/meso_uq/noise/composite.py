from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class LikelihoodStage(str, Enum):
    M0 = "M0"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    M5 = "M5"
    M6 = "M6"
    M7 = "M7"


class LikelihoodComponent(str, Enum):
    LEGACY = "legacy"
    ADDITIVE_NOISE = "additive_noise"
    RELATIVE_NOISE = "relative_noise"
    CORRELATED_CURVE_NOISE = "correlated_curve_noise"
    HEAVY_TAIL = "heavy_tail"
    CONTACT_ALIGNMENT = "contact_alignment"
    GEOMETRY = "geometry"
    SURROGATE_COVARIANCE = "surrogate_covariance"
    MODEL_DISCREPANCY = "model_discrepancy"
    TOTAL_COVARIANCE = "total_covariance"
    SYNTHETIC_RECOVERY = "synthetic_recovery"
    PREDICTIVE_CHECKS = "predictive_checks"
    EMB_COMPARISON = "emb_comparison"


_STAGE_ORDER: tuple[LikelihoodStage, ...] = (
    LikelihoodStage.M0,
    LikelihoodStage.M1,
    LikelihoodStage.M2,
    LikelihoodStage.M3,
    LikelihoodStage.M4,
    LikelihoodStage.M5,
    LikelihoodStage.M6,
    LikelihoodStage.M7,
)

_STAGE_DEFAULTS: Mapping[LikelihoodStage, tuple[LikelihoodComponent, ...]] = {
    LikelihoodStage.M0: (LikelihoodComponent.LEGACY,),
    LikelihoodStage.M1: (LikelihoodComponent.LEGACY,),
    LikelihoodStage.M2: (
        LikelihoodComponent.ADDITIVE_NOISE,
        LikelihoodComponent.RELATIVE_NOISE,
        LikelihoodComponent.CORRELATED_CURVE_NOISE,
        LikelihoodComponent.HEAVY_TAIL,
    ),
    LikelihoodStage.M3: (
        LikelihoodComponent.ADDITIVE_NOISE,
        LikelihoodComponent.RELATIVE_NOISE,
        LikelihoodComponent.CONTACT_ALIGNMENT,
        LikelihoodComponent.GEOMETRY,
    ),
    LikelihoodStage.M4: (
        LikelihoodComponent.ADDITIVE_NOISE,
        LikelihoodComponent.RELATIVE_NOISE,
        LikelihoodComponent.CONTACT_ALIGNMENT,
        LikelihoodComponent.GEOMETRY,
        LikelihoodComponent.SURROGATE_COVARIANCE,
    ),
    LikelihoodStage.M5: (
        LikelihoodComponent.ADDITIVE_NOISE,
        LikelihoodComponent.RELATIVE_NOISE,
        LikelihoodComponent.CONTACT_ALIGNMENT,
        LikelihoodComponent.GEOMETRY,
        LikelihoodComponent.SURROGATE_COVARIANCE,
        LikelihoodComponent.MODEL_DISCREPANCY,
        LikelihoodComponent.TOTAL_COVARIANCE,
    ),
    LikelihoodStage.M6: (
        LikelihoodComponent.SYNTHETIC_RECOVERY,
        LikelihoodComponent.PREDICTIVE_CHECKS,
    ),
    LikelihoodStage.M7: (
        LikelihoodComponent.EMB_COMPARISON,
        LikelihoodComponent.TOTAL_COVARIANCE,
    ),
}


def _coerce_stage(value: LikelihoodStage | str) -> LikelihoodStage:
    if isinstance(value, LikelihoodStage):
        return value
    normalized = str(value).strip().upper()
    try:
        return LikelihoodStage(normalized)
    except ValueError as exc:
        expected = ", ".join(stage.value for stage in LikelihoodStage)
        raise ValueError(f"Unsupported likelihood stage '{value}'. Expected one of: {expected}.") from exc


def _coerce_component(value: LikelihoodComponent | str) -> LikelihoodComponent:
    if isinstance(value, LikelihoodComponent):
        return value
    normalized = str(value).strip().lower()
    try:
        return LikelihoodComponent(normalized)
    except ValueError as exc:
        expected = ", ".join(component.value for component in LikelihoodComponent)
        raise ValueError(f"Unsupported likelihood component '{value}'. Expected one of: {expected}.") from exc


def _cumulative_allowed_components(stage: LikelihoodStage) -> tuple[LikelihoodComponent, ...]:
    allowed: list[LikelihoodComponent] = []
    selected_index = _STAGE_ORDER.index(stage)
    for candidate in _STAGE_ORDER[: selected_index + 1]:
        for component in _STAGE_DEFAULTS[candidate]:
            if component not in allowed:
                allowed.append(component)
    return tuple(allowed)


@dataclass(frozen=True)
class CompositeLikelihoodSpec:
    stage: LikelihoodStage
    components: tuple[LikelihoodComponent, ...]
    legacy_mode: str | None = None

    def __post_init__(self) -> None:
        stage = _coerce_stage(self.stage)
        components = tuple(_coerce_component(component) for component in self.components)
        if not components:
            raise ValueError("Composite likelihood requires at least one component.")
        allowed = _cumulative_allowed_components(stage)
        for component in components:
            if component not in allowed:
                allowed_values = ", ".join(item.value for item in allowed)
                raise ValueError(
                    f"Likelihood component '{component.value}' is not available for stage {stage.value}. "
                    f"Allowed: {allowed_values}."
                )
        if LikelihoodComponent.LEGACY in components and not self.legacy_mode:
            raise ValueError("Legacy component requires legacy_mode.")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "components", components)
        if self.legacy_mode is not None:
            object.__setattr__(self, "legacy_mode", str(self.legacy_mode).strip())

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CompositeLikelihoodSpec":
        stage = _coerce_stage(config.get("stage", LikelihoodStage.M0.value))
        raw_components = config.get("components")
        if raw_components is None:
            components: Sequence[str | LikelihoodComponent] = _STAGE_DEFAULTS[stage]
        elif isinstance(raw_components, str):
            components = (raw_components,)
        else:
            components = tuple(raw_components)
        legacy_mode = config.get("legacy_mode")
        return cls(stage=stage, components=tuple(components), legacy_mode=None if legacy_mode is None else str(legacy_mode))

    @property
    def is_legacy(self) -> bool:
        return self.components == (LikelihoodComponent.LEGACY,)


LikelihoodEvaluator = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class CompositeLikelihood:
    spec: CompositeLikelihoodSpec
    evaluators: Mapping[LikelihoodComponent, LikelihoodEvaluator]

    def __post_init__(self) -> None:
        missing = [component.value for component in self.spec.components if component not in self.evaluators]
        if missing:
            raise ValueError("Composite likelihood has no evaluator for component(s): " + ", ".join(missing))
        object.__setattr__(self, "evaluators", dict(self.evaluators))

    def evaluate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component in self.spec.components:
            results[component.value] = self.evaluators[component](payload)
        return results


def build_composite_likelihood_spec(config: Mapping[str, Any]) -> CompositeLikelihoodSpec:
    return CompositeLikelihoodSpec.from_mapping(config)


def build_composite_likelihood(
    config: Mapping[str, Any],
    evaluators: Mapping[LikelihoodComponent | str, LikelihoodEvaluator],
) -> CompositeLikelihood:
    spec = build_composite_likelihood_spec(config)
    normalized = {_coerce_component(component): evaluator for component, evaluator in evaluators.items()}
    return CompositeLikelihood(spec=spec, evaluators=normalized)


__all__ = [
    "CompositeLikelihood",
    "CompositeLikelihoodSpec",
    "LikelihoodComponent",
    "LikelihoodEvaluator",
    "LikelihoodStage",
    "build_composite_likelihood",
    "build_composite_likelihood_spec",
]
