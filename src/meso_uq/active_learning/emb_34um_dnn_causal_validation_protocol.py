from __future__ import annotations

"""Protocol contract for the EMB 3.4um DNN causal AL-vs-LHS protocol."""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol.v1"
)
# Backward-compatible alias used by existing callers.
EMB_34UM_DNN_CAUSAL_VALIDATION_SCHEMA_VERSION = EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION

EMB_34UM_DNN_CAUSAL_EXPERIMENT = "indentation"
EMB_34UM_DNN_CAUSAL_FAMILY = "emb"
EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES = ("ka", "kb")
EMB_34UM_DNN_CAUSAL_BOUNDS = {"ka": (1e2, 6e5), "kb": (400.0, 70000.0)}
EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE = "log10"

EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT = 5
EMB_34UM_DNN_CAUSAL_REPLICATES = tuple(range(1, EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT + 1))
EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE = 100
EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE = 100
EMB_34UM_DNN_CAUSAL_STEP_SIZE = 100
EMB_34UM_DNN_CAUSAL_MIN_CYCLES = 5
EMB_34UM_DNN_CAUSAL_MAX_CYCLES = 10

EMB_34UM_DNN_CAUSAL_FORCE_GRID = tuple(5000.0 * index / 7.0 for index in range(8))
EMB_34UM_DNN_CAUSAL_FORCE_GRID_POLICY = "linspace_0_5000_8_including_endpoints"

EMB_34UM_DNN_CAUSAL_LHS_SOURCE = "fresh_dpd"
EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE = "fresh_dpd_shared_unseen"

EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND = "fixed_architecture_dnn"
EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE = "same_architecture_dnn_ensemble"

# Geometry coefficients fixed for this protocol variant.
EMB_34UM_DNN_CAUSAL_B1_VALUE = 0.0
EMB_34UM_DNN_CAUSAL_B2_VALUE = 0.0
EMB_34UM_DNN_CAUSAL_A3_VALUE = 0.0
EMB_34UM_DNN_CAUSAL_A4_VALUE = 0.0

EMB_34UM_DNN_CAUSAL_FORBIDDEN_FINAL_LHS_SOURCES = frozenset(
    {
        "samples_all",
        "samples_all.dat",
        "historical",
        "preexisting",
        "replay",
        "archive",
    }
)

EMB_34UM_DNN_CAUSAL_FORBIDDEN_SELECTOR_TOKENS = frozenset(
    {
        "emb_34um_final_gate_surrogate",
        "lightweight",
        "numpy",
        "poly",
        "random-feature",
        "random_feature",
        "ridge",
    }
)

# Production DNN ensemble contract.
EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE = 10

# Timing canary contract:
#  - required before production by default.
#  - can be downgraded to 5 points only with explicit override reason.
EMB_34UM_DNN_CAUSAL_TIMING_CANARY_REQUIRED = True
EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT = 8
EMB_34UM_DNN_CAUSAL_TIMING_CANARY_DOWNTICK = 5


def _coerce_positive_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return number


def _coerce_nonnegative_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _coerce_float(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _coerce_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if value.strip().lower() in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be a boolean.")


def _coerce_sequence(values: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(values)


def _coerce_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty.")
    return text


def _coerce_float_sequence(values: object, *, label: str) -> tuple[float, ...]:
    values = _coerce_sequence(values, label=label)
    parsed = tuple(_coerce_float(value, label=label) for value in values)
    if not parsed:
        raise ValueError(f"{label} must not be empty.")
    return parsed


def validate_dnn_causal_force_grid(values: Sequence[object]) -> tuple[float, ...]:
    """Validate the production 8-point force grid for the DNN causal gate."""

    normalized = _coerce_float_sequence(values, label="force_grid")
    if len(normalized) != len(EMB_34UM_DNN_CAUSAL_FORCE_GRID):
        raise ValueError(
            f"DNN causal force grid must contain {len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)} points, "
            f"got {len(normalized)}."
        )
    for actual, expected in zip(normalized, EMB_34UM_DNN_CAUSAL_FORCE_GRID):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "DNN causal force grid must be exactly 8 evenly spaced points from 0 to 5000 including endpoints."
            )
    return normalized


def validate_dnn_causal_cycle_count(cycle_count: object) -> int:
    cycles = _coerce_positive_int(cycle_count, label="cycle_count")
    if cycles < EMB_34UM_DNN_CAUSAL_MIN_CYCLES:
        raise ValueError(
            f"cycle_count must be at least {EMB_34UM_DNN_CAUSAL_MIN_CYCLES}."
        )
    if cycles > EMB_34UM_DNN_CAUSAL_MAX_CYCLES:
        raise ValueError(
            f"cycle_count must be at most {EMB_34UM_DNN_CAUSAL_MAX_CYCLES}."
        )
    return cycles


def validate_dnn_causal_replicate_count(replicate_count: object) -> int:
    value = _coerce_positive_int(replicate_count, label="replicate_count")
    if value != EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT:
        raise ValueError(f"replicate_count is fixed at {EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT}.")
    return value


def validate_dnn_causal_ensemble_size(ensemble_size: object) -> int:
    value = _coerce_positive_int(ensemble_size, label="ensemble_size")
    if value != EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE:
        raise ValueError(f"ensemble_size is fixed at {EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE}.")
    return value


def validate_dnn_causal_lhs_source(source: object) -> str:
    normalized = str(source or "").strip().lower()
    if not normalized:
        raise ValueError("lhs_source must be non-empty.")
    if normalized in EMB_34UM_DNN_CAUSAL_FORBIDDEN_FINAL_LHS_SOURCES:
        raise ValueError(
            "Final DNN causal LHS must use fresh DPD curves, not samples_all.dat, replay, archive, or historical data."
        )
    if normalized != EMB_34UM_DNN_CAUSAL_LHS_SOURCE:
        raise ValueError(f"lhs_source must be {EMB_34UM_DNN_CAUSAL_LHS_SOURCE!r}.")
    return normalized


def validate_dnn_causal_selector_backend(backend: object) -> str:
    normalized = str(backend or "").strip().lower()
    if not normalized:
        raise ValueError("selector_backend must be non-empty.")
    for token in EMB_34UM_DNN_CAUSAL_FORBIDDEN_SELECTOR_TOKENS:
        if token in normalized:
            raise ValueError(
                "DNN causal selection must not use the lightweight NumPy/ridge/random-feature selector."
            )
    if normalized != EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND:
        raise ValueError(
            f"selector_backend must be the fixed value {EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND!r}."
        )
    return normalized


def validate_dnn_causal_timing_canary(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        raise ValueError("timing_canary is required.")
    required = _coerce_bool(payload.get("required", None), label="timing_canary.required")
    if not required:
        raise ValueError("timing_canary must be required before production.")

    point_count = _coerce_positive_int(
        payload.get("point_count", EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT),
        label="timing_canary.point_count",
    )
    if point_count == EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT:
        return {
            "required": True,
            "point_count": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
        }
    if point_count == EMB_34UM_DNN_CAUSAL_TIMING_CANARY_DOWNTICK:
        reason = _coerce_text(
            payload.get("override_reason", ""),
            label="timing_canary.override_reason",
        )
        return {
            "required": True,
            "point_count": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_DOWNTICK,
            "override_reason": reason,
        }

    raise ValueError(
        f"timing_canary.point_count must be {EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT} "
        f"or downgraded explicitly to {EMB_34UM_DNN_CAUSAL_TIMING_CANARY_DOWNTICK}."
    )


def dnn_causal_training_curve_count(*, cycle_count: int) -> int:
    cycles = validate_dnn_causal_cycle_count(cycle_count)
    return EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE + cycles * EMB_34UM_DNN_CAUSAL_STEP_SIZE


def dnn_causal_branch_curve_count(*, cycle_count: int) -> int:
    return dnn_causal_training_curve_count(cycle_count=cycle_count)


def dnn_causal_total_dpd_curve_count(*, cycle_count: int) -> int:
    cycles = validate_dnn_causal_cycle_count(cycle_count)
    per_replicate = (
        EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE
        + 2 * cycles * EMB_34UM_DNN_CAUSAL_STEP_SIZE
    )
    return EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE + EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT * per_replicate


def dnn_causal_training_row_count(*, curve_count: int) -> int:
    curves = _coerce_positive_int(curve_count, label="curve_count")
    return curves * len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)


def validate_dnn_causal_protocol_payload(payload: Mapping[str, Any]) -> "Emb34umDnnCausalValidationProtocol":
    if not isinstance(payload, Mapping):
        raise ValueError("protocol payload must be a mapping.")

    if "ensemble_size" not in payload:
        raise ValueError("protocol payload must include ensemble_size.")
    if "timing_canary" not in payload:
        raise ValueError("protocol payload must include timing_canary.")

    return Emb34umDnnCausalValidationProtocol(
        cycle_count=payload.get("cycle_count", EMB_34UM_DNN_CAUSAL_MIN_CYCLES),
        replicate_count=payload.get("replicate_count", EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT),
        force_grid=payload.get("force_grid", EMB_34UM_DNN_CAUSAL_FORCE_GRID),
        lhs_source=payload.get("lhs_source", EMB_34UM_DNN_CAUSAL_LHS_SOURCE),
        selector_backend=payload.get("selector_backend", EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND),
        uncertainty_source=payload.get(
            "uncertainty_source",
            EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE,
        ),
        ensemble_size=payload["ensemble_size"],
        timing_canary=payload["timing_canary"],
    )


@dataclass(frozen=True)
class Emb34umDnnCausalValidationProtocol:
    cycle_count: int = EMB_34UM_DNN_CAUSAL_MIN_CYCLES
    replicate_count: int = EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    force_grid: tuple[float, ...] = EMB_34UM_DNN_CAUSAL_FORCE_GRID
    lhs_source: str = EMB_34UM_DNN_CAUSAL_LHS_SOURCE
    selector_backend: str = EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND
    uncertainty_source: str = EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE
    ensemble_size: int = EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    timing_canary: Mapping[str, Any] = field(
        default_factory=lambda: {
            "required": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_REQUIRED,
            "point_count": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_count", validate_dnn_causal_cycle_count(self.cycle_count))
        object.__setattr__(
            self,
            "replicate_count",
            validate_dnn_causal_replicate_count(self.replicate_count),
        )
        object.__setattr__(
            self,
            "force_grid",
            validate_dnn_causal_force_grid(self.force_grid),
        )
        object.__setattr__(self, "lhs_source", validate_dnn_causal_lhs_source(self.lhs_source))
        object.__setattr__(
            self,
            "selector_backend",
            validate_dnn_causal_selector_backend(self.selector_backend),
        )
        object.__setattr__(
            self,
            "uncertainty_source",
            _coerce_text(self.uncertainty_source, label="uncertainty_source"),
        )
        object.__setattr__(
            self,
            "ensemble_size",
            validate_dnn_causal_ensemble_size(self.ensemble_size),
        )
        object.__setattr__(
            self,
            "timing_canary",
            validate_dnn_causal_timing_canary(dict(self.timing_canary)),
        )

    @property
    def rows_per_curve(self) -> int:
        return len(self.force_grid)

    @property
    def total_dpd_curve_count(self) -> int:
        return dnn_causal_total_dpd_curve_count(cycle_count=self.cycle_count)

    @property
    def fixed_coefficients(self) -> dict[str, float]:
        return {
            "b1": EMB_34UM_DNN_CAUSAL_B1_VALUE,
            "b2": EMB_34UM_DNN_CAUSAL_B2_VALUE,
            "a3": EMB_34UM_DNN_CAUSAL_A3_VALUE,
            "a4": EMB_34UM_DNN_CAUSAL_A4_VALUE,
        }

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION,
            "family": EMB_34UM_DNN_CAUSAL_FAMILY,
            "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
            "active_variables": list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES),
            "bounds": {name: list(bounds) for name, bounds in EMB_34UM_DNN_CAUSAL_BOUNDS.items()},
            "parameter_space": EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE,
            "fixed_coefficients": self.fixed_coefficients,
            "replicates": list(EMB_34UM_DNN_CAUSAL_REPLICATES),
            "replicate_count": self.replicate_count,
            "shared_initial_size": EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
            "unseen_test_size": EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
            "step_size": EMB_34UM_DNN_CAUSAL_STEP_SIZE,
            "min_cycles": EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
            "max_cycles": EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
            "cycle_count": self.cycle_count,
            "force_grid": list(self.force_grid),
            "force_grid_policy": EMB_34UM_DNN_CAUSAL_FORCE_GRID_POLICY,
            "lhs_source": self.lhs_source,
            "test_set_source": EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
            "selector_backend": self.selector_backend,
            "uncertainty_source": self.uncertainty_source,
            "ensemble_size": self.ensemble_size,
            "timing_canary": dict(self.timing_canary),
            "rows_per_curve": self.rows_per_curve,
            "branch_training_curve_count_at_final_cycle": dnn_causal_branch_curve_count(
                cycle_count=self.cycle_count,
            ),
            "total_dpd_curve_count": self.total_dpd_curve_count,
        }


def validate_dnn_causal_protocol_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = validate_dnn_causal_protocol_payload(payload)
    return protocol.as_manifest()
