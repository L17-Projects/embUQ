"""Strict-support, exact-diameter PCHIP DPD breathing-frequency emulators.

Each emulator represents one physical bubble diameter and predicts the
breathing frequency from ``ka`` only.  The bank deliberately does not
interpolate across diameter: every diameter used by inference must have its
own accepted full-fluid DPD campaign and fitted emulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FREQUENCY_EMULATOR_BANK_SCHEMA = "meso_uq.emb_dpd_frequency_emulator_bank.v2"
POLYNOMIAL_FREQUENCY_BANK_SCHEMA = "meso_uq.emb_dpd_frequency_polynomial_bank.v1"
FREQUENCY_UNIT = "MHz"
DIAMETER_UNIT = "um"
FREQUENCY_SQUARED_TARGET = "frequency_squared_mhz2"
FREQUENCY_RESPONSE_ENCODING = "frequency_mhz"
FREQUENCY_SQUARED_RESPONSE_ENCODING = "frequency_mhz_squared"
POLYNOMIAL_METHOD = "affine_low_order_horner"
POLYNOMIAL_BASIS = "ascending_power_of_affine_ka"
POLYNOMIAL_COEFFICIENT_ORDER = "ascending_power"
POLYNOMIAL_ALLOWED_DEGREES = (1, 2)
PCHIP_METHOD = "scipy_pchip"
COEFFICIENT_ORDER = "local_ascending_power"
KNOT_ORDER = "increasing_ka_dpd"
DIAMETER_TOLERANCE_UM = 1.0e-9
FIT_PRIMARY_LABEL_ADMISSION_POLICY = "fit_primary"
NONMONOTONE_FREQUENCY_POLICY = "allow_fit_primary_if_dense_finite_positive"
DENSE_VALIDATION_MIN_POINTS = 4097


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number.")
    return result


def _positive_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return result


def _bounds(payload: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or len(payload) != 2:
        raise ValueError(f"{field_name} must contain exactly [minimum, maximum].")
    lower = _finite_float(payload[0], f"{field_name}[0]")
    upper = _finite_float(payload[1], f"{field_name}[1]")
    if lower < 0.0 or upper <= lower:
        raise ValueError(f"{field_name} must be an increasing non-negative supported range.")
    return lower, upper


def _agent_name(value: Any) -> str:
    result = str(value).strip().lower()
    if result not in {"definity", "sonovue"}:
        raise ValueError("agent must be either 'definity' or 'sonovue'.")
    return result


def _optional_mapping(payload: Any, field_name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return dict(payload)


def _sha256_digest(value: Any, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256 digest.")
    return digest


def _hashed_source_datasets(payload: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or not payload:
        raise ValueError(
            "Frequency emulator bank must record non-empty hashed source datasets "
            "in provenance.source_datasets."
        )
    records: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping) or not str(item.get("path") or "").strip():
            raise ValueError(
                f"Frequency emulator bank source_datasets[{index}] must record a path."
            )
        records.append(
            {
                "path": str(item["path"]),
                "sha256": _sha256_digest(
                    item.get("sha256"),
                    f"provenance.source_datasets[{index}].sha256",
                ),
            }
        )
    return tuple(records)


def _float_diameter_set(payload: Any, field_name: str) -> set[float]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of exact diameters.")
    result = {_positive_float(value, field_name) for value in payload}
    if len(result) != len(payload):
        raise ValueError(f"{field_name} contains duplicate diameters.")
    return result


def _knot_array(payload: Any) -> np.ndarray:
    values = np.asarray(payload, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("pchip.knots_ka_dpd must be a one-dimensional array with at least two knots.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("pchip.knots_ka_dpd must contain finite non-negative values.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("pchip.knots_ka_dpd must be strictly increasing.")
    result = np.array(values, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _local_cubic_array(payload: Any, *, knot_count: int) -> np.ndarray:
    values = np.asarray(payload, dtype=np.float64)
    expected_shape = (knot_count - 1, 4)
    if values.shape != expected_shape:
        raise ValueError(
            f"pchip.interval_coefficients must have shape {expected_shape}, got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("pchip.interval_coefficients must contain only finite values.")
    result = np.array(values, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _legacy_normalized_polynomial_to_local_cubic(
    coefficients: Any,
    *,
    degree: int,
    ka_bounds_dpd: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    expected_shape = (degree + 1,)
    if values.shape != expected_shape:
        raise ValueError(
            f"polynomial.coefficients must have shape {expected_shape}, got {values.shape}."
        )
    if degree > 3:
        raise ValueError("Legacy polynomial fixture conversion supports degree <= 3.")
    if not np.all(np.isfinite(values)):
        raise ValueError("polynomial.coefficients must contain only finite values.")
    span = ka_bounds_dpd[1] - ka_bounds_dpd[0]
    local = np.zeros((1, 4), dtype=np.float64)
    for power, coefficient in enumerate(values):
        local[0, power] = coefficient / (span**power)
    local.flags.writeable = False
    return local


def _broadcast_inputs(ka_dpd: Any, diameter_um: Any) -> tuple[np.ndarray, np.ndarray]:
    ka = np.asarray(ka_dpd, dtype=np.float64)
    diameter = np.asarray(diameter_um, dtype=np.float64)
    try:
        ka, diameter = np.broadcast_arrays(ka, diameter)
    except ValueError as error:
        raise ValueError("ka_dpd and diameter_um must be broadcast-compatible.") from error
    if not np.all(np.isfinite(ka)):
        raise ValueError("ka_dpd must contain only finite values.")
    if not np.all(np.isfinite(diameter)) or np.any(diameter <= 0.0):
        raise ValueError("diameter_um must contain finite positive values.")
    return ka, diameter


@dataclass(frozen=True)
class DpdFrequencyEmulator:
    """One bounded PCHIP emulator for a single physical diameter."""

    diameter_um: float
    ka_bounds_dpd: tuple[float, float]
    knots_ka_dpd: Any | None = None
    interval_coefficients: Any | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    coefficients: Any | None = None
    degree: int | None = None

    def __post_init__(self) -> None:
        diameter = _positive_float(self.diameter_um, "diameter_um")
        bounds = _bounds(self.ka_bounds_dpd, "ka_bounds_dpd")
        if self.knots_ka_dpd is None or self.interval_coefficients is None:
            if self.coefficients is None or self.degree is None:
                raise ValueError(
                    "DpdFrequencyEmulator requires PCHIP knots/coefficients or "
                    "legacy polynomial coefficients/degree."
                )
            degree = int(self.degree)
            if degree < 0:
                raise ValueError("Polynomial degree must be non-negative.")
            knots = _knot_array([bounds[0], bounds[1]])
            interval_coefficients = _legacy_normalized_polynomial_to_local_cubic(
                self.coefficients,
                degree=degree,
                ka_bounds_dpd=bounds,
            )
        else:
            knots = _knot_array(self.knots_ka_dpd)
            interval_coefficients = _local_cubic_array(
                self.interval_coefficients,
                knot_count=knots.size,
            )
            if not np.isclose(knots[0], bounds[0], rtol=0.0, atol=0.0) or not np.isclose(
                knots[-1],
                bounds[1],
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError("PCHIP endpoint knots must exactly match ka_bounds_dpd.")
            degree = 3

        object.__setattr__(self, "diameter_um", diameter)
        object.__setattr__(self, "ka_bounds_dpd", bounds)
        object.__setattr__(self, "knots_ka_dpd", knots)
        object.__setattr__(self, "interval_coefficients", interval_coefficients)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "coefficients", interval_coefficients)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "validation", dict(self.validation))

        endpoint_values = interval_coefficients[[0, -1], 0].copy()
        final_delta = knots[-1] - knots[-2]
        final_coefficients = interval_coefficients[-1]
        endpoint_values[1] = (
            ((final_coefficients[3] * final_delta + final_coefficients[2]) * final_delta
            + final_coefficients[1])
            * final_delta
            + final_coefficients[0]
        )
        if np.any(~np.isfinite(endpoint_values)) or np.any(endpoint_values <= 0.0):
            raise ValueError("PCHIP endpoint f_squared values must be finite and positive.")

    @property
    def normalization(self) -> dict[str, float]:
        return {
            "ka_min_dpd": self.ka_bounds_dpd[0],
            "ka_max_dpd": self.ka_bounds_dpd[1],
        }

    def _checked_ka(self, ka_dpd: Any) -> np.ndarray:
        ka = np.asarray(ka_dpd, dtype=np.float64)
        if not np.all(np.isfinite(ka)):
            raise ValueError("ka_dpd must contain only finite values.")
        lower, upper = self.ka_bounds_dpd
        if np.any(ka < lower) or np.any(ka > upper):
            raise ValueError(
                f"ka_dpd is outside the supported range [{lower:g}, {upper:g}] "
                f"for diameter {self.diameter_um:g} um."
            )
        return ka

    def predict_frequency_squared_mhz2(self, ka_dpd: Any) -> np.ndarray:
        """Evaluate f^2 without allowing ka extrapolation."""

        ka = self._checked_ka(ka_dpd)
        original_shape = ka.shape
        flat_ka = ka.reshape(-1)
        interval_index = np.searchsorted(self.knots_ka_dpd, flat_ka, side="right") - 1
        interval_index = np.clip(interval_index, 0, self.knots_ka_dpd.size - 2)
        delta = flat_ka - self.knots_ka_dpd[interval_index]
        coefficients = self.interval_coefficients[interval_index]
        result = ((coefficients[:, 3] * delta + coefficients[:, 2]) * delta + coefficients[:, 1]) * delta + coefficients[:, 0]
        if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError(
                "Frequency emulator produced non-positive or non-finite f_squared "
                f"for diameter {self.diameter_um:g} um."
            )
        return np.asarray(result, dtype=np.float64).reshape(original_shape)

    def predict_mhz(self, ka_dpd: Any) -> np.ndarray:
        return np.sqrt(self.predict_frequency_squared_mhz2(ka_dpd))

    def knot_frequencies_are_monotone(self) -> bool:
        frequency = self.predict_mhz(self.knots_ka_dpd)
        differences = np.diff(frequency)
        return bool(np.all(differences >= 0.0) or np.all(differences <= 0.0))

    def dense_finite_positive_validation(
        self,
        *,
        point_count: int = DENSE_VALIDATION_MIN_POINTS,
    ) -> dict[str, float | int | bool]:
        count = int(point_count)
        if count < DENSE_VALIDATION_MIN_POINTS:
            raise ValueError(
                f"Dense PCHIP validation requires at least {DENSE_VALIDATION_MIN_POINTS} points."
            )
        ka = np.linspace(self.ka_bounds_dpd[0], self.ka_bounds_dpd[1], count)
        frequency_squared = self.predict_frequency_squared_mhz2(ka)
        passed = bool(
            np.all(np.isfinite(frequency_squared)) and np.all(frequency_squared > 0.0)
        )
        if not passed:
            raise ValueError(
                "Dense PCHIP validation produced non-positive or non-finite f_squared "
                f"for diameter {self.diameter_um:g} um."
            )
        return {
            "point_count": count,
            "finite_positive": True,
            "minimum_frequency_squared_mhz2": float(np.min(frequency_squared)),
            "maximum_frequency_squared_mhz2": float(np.max(frequency_squared)),
        }

    def frequency_roots_ka_dpd(self, frequency_mhz: float) -> tuple[float, ...]:
        """Return every in-support PCHIP root for one positive frequency."""

        target = _positive_float(frequency_mhz, "frequency_mhz") ** 2
        roots: list[float] = []
        for index, coefficients in enumerate(self.interval_coefficients):
            polynomial = np.asarray(coefficients, dtype=np.float64).copy()
            polynomial[0] -= target
            nonzero = np.flatnonzero(np.abs(polynomial) > 1.0e-15)
            if nonzero.size == 0:
                roots.extend(
                    [float(self.knots_ka_dpd[index]), float(self.knots_ka_dpd[index + 1])]
                )
                continue
            degree = int(nonzero[-1])
            local_roots = np.roots(polynomial[: degree + 1][::-1])
            interval_width = self.knots_ka_dpd[index + 1] - self.knots_ka_dpd[index]
            tolerance = 1.0e-9 * max(1.0, interval_width)
            for root in local_roots:
                if abs(float(root.imag)) > tolerance:
                    continue
                delta = float(root.real)
                if -tolerance <= delta <= interval_width + tolerance:
                    roots.append(
                        float(
                            self.knots_ka_dpd[index]
                            + min(max(delta, 0.0), interval_width)
                        )
                    )
        ordered: list[float] = []
        for value in sorted(roots):
            if not ordered or not np.isclose(value, ordered[-1], rtol=0.0, atol=1.0e-7):
                ordered.append(value)
        return tuple(ordered)

    def frequency_compatible_branches_ka_dpd(
        self,
        frequency_bounds_mhz: Sequence[float],
    ) -> tuple[tuple[float, float], ...]:
        """Return all support intervals whose PCHIP frequencies lie in the target band."""

        lower, upper = _bounds(frequency_bounds_mhz, "frequency_bounds_mhz")
        if lower <= 0.0:
            raise ValueError("frequency_bounds_mhz must contain positive frequencies.")
        boundaries = [self.ka_bounds_dpd[0], self.ka_bounds_dpd[1]]
        boundaries.extend(self.frequency_roots_ka_dpd(lower))
        boundaries.extend(self.frequency_roots_ka_dpd(upper))
        ordered: list[float] = []
        for value in sorted(boundaries):
            if not ordered or not np.isclose(value, ordered[-1], rtol=0.0, atol=1.0e-7):
                ordered.append(value)
        branches: list[tuple[float, float]] = []
        for left, right in zip(ordered, ordered[1:]):
            midpoint = 0.5 * (left + right)
            value = float(self.predict_mhz(midpoint))
            if lower <= value <= upper:
                if branches and np.isclose(branches[-1][1], left, rtol=0.0, atol=1.0e-7):
                    branches[-1] = (branches[-1][0], right)
                else:
                    branches.append((left, right))
        return tuple(branches)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "diameter_um": self.diameter_um,
            "ka_bounds_dpd": list(self.ka_bounds_dpd),
            "normalization": self.normalization,
            "pchip": {
                "method": PCHIP_METHOD,
                "target": FREQUENCY_SQUARED_TARGET,
                "knot_order": KNOT_ORDER,
                "coefficient_order": COEFFICIENT_ORDER,
                "knots_ka_dpd": self.knots_ka_dpd.tolist(),
                "interval_coefficients": self.interval_coefficients.tolist(),
            },
            "provenance": dict(self.provenance),
            "validation": dict(self.validation),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DpdFrequencyEmulator":
        if not isinstance(payload, Mapping):
            raise ValueError("Frequency emulator entry must be a mapping.")
        pchip = payload.get("pchip")
        if not isinstance(pchip, Mapping):
            raise ValueError("Frequency emulator entry is missing its pchip mapping.")
        if pchip.get("method") != PCHIP_METHOD:
            raise ValueError(f"Frequency emulator pchip.method must be {PCHIP_METHOD}.")
        if pchip.get("target") != FREQUENCY_SQUARED_TARGET:
            raise ValueError("Frequency emulator PCHIP must target frequency_squared_mhz2.")
        if pchip.get("knot_order") != KNOT_ORDER:
            raise ValueError(f"Frequency emulator knot_order must be {KNOT_ORDER}.")
        if pchip.get("coefficient_order") != COEFFICIENT_ORDER:
            raise ValueError(
                f"Frequency emulator coefficient_order must be {COEFFICIENT_ORDER}."
            )
        return cls(
            diameter_um=payload.get("diameter_um"),
            ka_bounds_dpd=_bounds(payload.get("ka_bounds_dpd"), "ka_bounds_dpd"),
            knots_ka_dpd=pchip.get("knots_ka_dpd"),
            interval_coefficients=pchip.get("interval_coefficients"),
            provenance=_optional_mapping(payload.get("provenance"), "provenance"),
            validation=_optional_mapping(payload.get("validation"), "validation"),
        )


@dataclass(frozen=True)
class DpdFrequencyEmulatorBank:
    """An agent-specific collection of exact-diameter frequency emulators."""

    agent: str
    emulators: Sequence[DpdFrequencyEmulator]
    conditions: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    _diameter_values: np.ndarray = field(init=False, repr=False, compare=False)
    _ka_lower_bounds: np.ndarray = field(init=False, repr=False, compare=False)
    _ka_upper_bounds: np.ndarray = field(init=False, repr=False, compare=False)
    _knot_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    _coefficient_tensor: np.ndarray = field(init=False, repr=False, compare=False)
    _interval_counts: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent", _agent_name(self.agent))
        entries = tuple(self.emulators)
        if not entries:
            raise ValueError("Frequency emulator bank must contain at least one emulator.")
        ordered = tuple(sorted(entries, key=lambda entry: entry.diameter_um))
        for previous, current in zip(ordered, ordered[1:]):
            if abs(current.diameter_um - previous.diameter_um) <= DIAMETER_TOLERANCE_UM:
                raise ValueError(
                    "Frequency emulator bank contains duplicate diameter "
                    f"{current.diameter_um:g} um."
                )
        lower = max(entry.ka_bounds_dpd[0] for entry in ordered)
        upper = min(entry.ka_bounds_dpd[1] for entry in ordered)
        if upper <= lower:
            raise ValueError("Frequency emulator entries have no common ka support.")

        max_knots = max(entry.knots_ka_dpd.size for entry in ordered)
        knot_matrix = np.full((len(ordered), max_knots), np.inf, dtype=np.float64)
        coefficient_tensor = np.zeros((len(ordered), max_knots - 1, 4), dtype=np.float64)
        interval_counts = np.zeros(len(ordered), dtype=np.int64)
        for index, entry in enumerate(ordered):
            knot_count = entry.knots_ka_dpd.size
            interval_count = knot_count - 1
            knot_matrix[index, :knot_count] = entry.knots_ka_dpd
            coefficient_tensor[index, :interval_count, :] = entry.interval_coefficients
            interval_counts[index] = interval_count
        for array in (knot_matrix, coefficient_tensor, interval_counts):
            array.flags.writeable = False

        object.__setattr__(self, "emulators", ordered)
        object.__setattr__(self, "conditions", dict(self.conditions))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "validation", dict(self.validation))
        object.__setattr__(
            self,
            "_diameter_values",
            np.asarray([entry.diameter_um for entry in ordered], dtype=np.float64),
        )
        object.__setattr__(
            self,
            "_ka_lower_bounds",
            np.asarray([entry.ka_bounds_dpd[0] for entry in ordered], dtype=np.float64),
        )
        object.__setattr__(
            self,
            "_ka_upper_bounds",
            np.asarray([entry.ka_bounds_dpd[1] for entry in ordered], dtype=np.float64),
        )
        object.__setattr__(self, "_knot_matrix", knot_matrix)
        object.__setattr__(self, "_coefficient_tensor", coefficient_tensor)
        object.__setattr__(self, "_interval_counts", interval_counts)
        self._diameter_values.flags.writeable = False
        self._ka_lower_bounds.flags.writeable = False
        self._ka_upper_bounds.flags.writeable = False

    @property
    def diameters_um(self) -> tuple[float, ...]:
        return tuple(entry.diameter_um for entry in self.emulators)

    @property
    def ka_bounds_dpd(self) -> tuple[float, float]:
        return (
            max(entry.ka_bounds_dpd[0] for entry in self.emulators),
            min(entry.ka_bounds_dpd[1] for entry in self.emulators),
        )

    def emulator_for_diameter(self, diameter_um: float) -> DpdFrequencyEmulator:
        diameter = _positive_float(diameter_um, "diameter_um")
        matches = [
            entry
            for entry in self.emulators
            if abs(entry.diameter_um - diameter) <= DIAMETER_TOLERANCE_UM
        ]
        if not matches:
            supported = ", ".join(f"{value:g}" for value in self.diameters_um)
            raise ValueError(
                f"No exact DPD frequency emulator is available for diameter {diameter:g} um; "
                f"supported diameters are [{supported}] um."
            )
        return matches[0]

    def predict_for_diameters_mhz(self, ka_dpd: Any, diameter_um: Any) -> np.ndarray:
        ka, diameter = _broadcast_inputs(ka_dpd, diameter_um)
        original_shape = ka.shape
        flat_ka = ka.reshape(-1)
        flat_diameter = diameter.reshape(-1)

        right = np.searchsorted(self._diameter_values, flat_diameter, side="left")
        right = np.clip(right, 0, self._diameter_values.size - 1)
        left = np.clip(right - 1, 0, self._diameter_values.size - 1)
        left_distance = np.abs(flat_diameter - self._diameter_values[left])
        right_distance = np.abs(flat_diameter - self._diameter_values[right])
        emulator_index = np.where(left_distance <= right_distance, left, right)
        diameter_distance = np.abs(flat_diameter - self._diameter_values[emulator_index])
        diameter_supported = diameter_distance <= DIAMETER_TOLERANCE_UM
        if not np.all(diameter_supported):
            missing = np.unique(flat_diameter[~diameter_supported])
            missing_text = ", ".join(f"{value:g}" for value in missing)
            supported = ", ".join(f"{value:g}" for value in self.diameters_um)
            raise ValueError(
                f"No exact DPD frequency emulator is available for diameter(s) "
                f"[{missing_text}] um; supported diameters are [{supported}] um."
            )

        lower = self._ka_lower_bounds[emulator_index]
        upper = self._ka_upper_bounds[emulator_index]
        ka_supported = (flat_ka >= lower) & (flat_ka <= upper)
        if not np.all(ka_supported):
            raise ValueError(
                "ka_dpd is outside the supported range for one or more "
                "exact-diameter DPD frequency emulators."
            )

        selected_knots = self._knot_matrix[emulator_index]
        interval_index = np.sum(flat_ka[:, None] >= selected_knots[:, 1:], axis=1)
        interval_index = np.minimum(interval_index, self._interval_counts[emulator_index] - 1)
        coefficients = self._coefficient_tensor[emulator_index, interval_index, :]
        delta = flat_ka - selected_knots[np.arange(flat_ka.size), interval_index]
        frequency_squared = (
            ((coefficients[:, 3] * delta + coefficients[:, 2]) * delta + coefficients[:, 1])
            * delta
            + coefficients[:, 0]
        )
        if np.any(~np.isfinite(frequency_squared)) or np.any(frequency_squared <= 0.0):
            raise ValueError(
                "Frequency emulator bank produced non-positive or non-finite "
                "f_squared values."
            )
        return np.sqrt(frequency_squared).reshape(original_shape)

    def _mapping_without_integrity(self) -> dict[str, Any]:
        return {
            "schema": FREQUENCY_EMULATOR_BANK_SCHEMA,
            "agent": self.agent,
            "response": {
                "frequency_unit": FREQUENCY_UNIT,
                "target": FREQUENCY_SQUARED_TARGET,
                "api_output_unit": FREQUENCY_UNIT,
            },
            "coordinates": {
                "diameter_mode": "exact",
                "diameter_unit": DIAMETER_UNIT,
                "diameter_tolerance_um": DIAMETER_TOLERANCE_UM,
                "ka_support": "inclusive",
                "common_ka_bounds_dpd": list(self.ka_bounds_dpd),
            },
            "runtime_contract": {
                "diameter_interpolation_allowed": False,
                "ka_clipping_allowed": False,
                "ka_extrapolation_allowed": False,
                "analytical_fallback_allowed": False,
                "runtime_requires_scipy": False,
                "population_evaluation": "numpy_searchsorted_gather_horner",
                "label_admission_policy": FIT_PRIMARY_LABEL_ADMISSION_POLICY,
                "nonmonotone_frequency_policy": NONMONOTONE_FREQUENCY_POLICY,
                "dense_finite_positive_validation_min_points": DENSE_VALIDATION_MIN_POINTS,
            },
            "emulators": [entry.to_mapping() for entry in self.emulators],
            "conditions": dict(self.conditions),
            "provenance": dict(self.provenance),
            "validation": dict(self.validation),
        }

    def to_mapping(self) -> dict[str, Any]:
        payload = self._mapping_without_integrity()
        payload["integrity"] = {
            "algorithm": "sha256",
            "canonical_payload_sha256": _canonical_json_hash(payload),
        }
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DpdFrequencyEmulatorBank":
        if not isinstance(payload, Mapping):
            raise ValueError("Frequency emulator bank artifact must be a mapping.")
        if payload.get("schema") != FREQUENCY_EMULATOR_BANK_SCHEMA:
            raise ValueError(
                f"Unsupported frequency emulator bank schema: {payload.get('schema')!r}."
            )
        integrity = payload.get("integrity")
        if not isinstance(integrity, Mapping) or integrity.get("algorithm") != "sha256":
            raise ValueError("Frequency emulator bank is missing SHA-256 integrity metadata.")
        expected_hash = str(integrity.get("canonical_payload_sha256") or "")
        if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
            raise ValueError("Frequency emulator bank integrity hash is invalid.")
        payload_without_integrity = dict(payload)
        payload_without_integrity.pop("integrity", None)
        actual_hash = _canonical_json_hash(payload_without_integrity)
        if actual_hash != expected_hash:
            raise ValueError("Frequency emulator bank integrity hash does not match payload.")

        response = payload.get("response")
        coordinates = payload.get("coordinates")
        runtime_contract = payload.get("runtime_contract")
        entries = payload.get("emulators")
        if (
            not isinstance(response, Mapping)
            or not isinstance(coordinates, Mapping)
            or not isinstance(runtime_contract, Mapping)
        ):
            raise ValueError(
                "Frequency emulator bank is missing response, coordinates, or runtime_contract mappings."
            )
        if response.get("frequency_unit") != FREQUENCY_UNIT or response.get("api_output_unit") != FREQUENCY_UNIT:
            raise ValueError(f"Frequency emulator bank must use {FREQUENCY_UNIT}.")
        if response.get("target") != FREQUENCY_SQUARED_TARGET:
            raise ValueError("Frequency emulator bank must target frequency_squared_mhz2.")
        if coordinates.get("diameter_mode") != "exact":
            raise ValueError("Frequency emulator bank diameter_mode must be exact.")
        if coordinates.get("diameter_unit") != DIAMETER_UNIT:
            raise ValueError(f"Frequency emulator bank diameter_unit must be {DIAMETER_UNIT}.")
        if coordinates.get("ka_support") != "inclusive":
            raise ValueError("Frequency emulator bank ka_support must be inclusive.")
        required_false = (
            "diameter_interpolation_allowed",
            "ka_clipping_allowed",
            "ka_extrapolation_allowed",
            "analytical_fallback_allowed",
            "runtime_requires_scipy",
        )
        for field_name in required_false:
            if runtime_contract.get(field_name) is not False:
                raise ValueError(f"Frequency emulator bank runtime_contract.{field_name} must be false.")
        if runtime_contract.get("label_admission_policy") != FIT_PRIMARY_LABEL_ADMISSION_POLICY:
            raise ValueError("Frequency emulator bank must declare fit-primary label admission.")
        if runtime_contract.get("nonmonotone_frequency_policy") != NONMONOTONE_FREQUENCY_POLICY:
            raise ValueError(
                "Frequency emulator bank must declare the approved fit-primary, "
                "dense-finite-positive nonmonotone policy."
            )
        if int(runtime_contract.get("dense_finite_positive_validation_min_points", 0)) < DENSE_VALIDATION_MIN_POINTS:
            raise ValueError(
                "Frequency emulator bank dense validation contract is below the runtime minimum."
            )
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("Frequency emulator bank emulators must be a sequence.")
        bank = cls(
            agent=payload.get("agent"),
            emulators=[DpdFrequencyEmulator.from_mapping(entry) for entry in entries],
            conditions=_optional_mapping(payload.get("conditions"), "conditions"),
            provenance=_optional_mapping(payload.get("provenance"), "provenance"),
            validation=_optional_mapping(payload.get("validation"), "validation"),
        )
        _hashed_source_datasets(bank.provenance.get("source_datasets"))
        if bank.validation.get("label_admission_policy") != FIT_PRIMARY_LABEL_ADMISSION_POLICY:
            raise ValueError("Frequency emulator bank validation must declare fit-primary labels.")
        if bank.validation.get("nonmonotone_frequency_policy") != NONMONOTONE_FREQUENCY_POLICY:
            raise ValueError("Frequency emulator bank validation has no approved nonmonotone policy.")
        if bank.validation.get("all_dense_grids_finite_positive") is not True:
            raise ValueError("Frequency emulator bank is missing dense finite-positive validation.")

        declared_nonmonotone = _float_diameter_set(
            bank.validation.get("nonmonotone_frequency_label_diameters_um"),
            "validation.nonmonotone_frequency_label_diameters_um",
        )
        actual_nonmonotone: set[float] = set()
        for entry in bank.emulators:
            entry_validation = entry.validation
            if entry_validation.get("label_admission_policy") != FIT_PRIMARY_LABEL_ADMISSION_POLICY:
                raise ValueError(
                    f"Frequency emulator at {entry.diameter_um:g} um is not declared fit-primary."
                )
            if entry_validation.get("fit_primary_gate_passed") is not True:
                raise ValueError(
                    f"Frequency emulator at {entry.diameter_um:g} um failed fit-primary admission."
                )
            if entry_validation.get("dense_grid_finite_positive") is not True:
                raise ValueError(
                    f"Frequency emulator at {entry.diameter_um:g} um lacks dense positive validation."
                )
            point_count = int(entry_validation.get("dense_grid_point_count", 0))
            if point_count < DENSE_VALIDATION_MIN_POINTS:
                raise ValueError(
                    f"Frequency emulator at {entry.diameter_um:g} um dense validation is too sparse."
                )
            recorded_monotone = entry_validation.get("frequency_labels_monotone")
            if not isinstance(recorded_monotone, bool):
                raise ValueError(
                    f"Frequency emulator at {entry.diameter_um:g} um must declare label monotonicity."
                )
            computed_monotone = entry.knot_frequencies_are_monotone()
            if recorded_monotone != computed_monotone:
                raise ValueError(
                    "Frequency emulator label monotonicity declaration contradicts its PCHIP knots "
                    f"at diameter {entry.diameter_um:g} um."
                )
            if not computed_monotone:
                if entry_validation.get("nonmonotone_frequency_policy") != NONMONOTONE_FREQUENCY_POLICY:
                    raise ValueError(
                        f"Nonmonotone frequency emulator at {entry.diameter_um:g} um lacks explicit policy."
                    )
                actual_nonmonotone.add(entry.diameter_um)
            entry.dense_finite_positive_validation(point_count=DENSE_VALIDATION_MIN_POINTS)

        if declared_nonmonotone != actual_nonmonotone:
            raise ValueError(
                "Frequency emulator bank nonmonotone diameter declaration does not match its entries."
            )
        all_monotone = not actual_nonmonotone
        if bank.validation.get("frequency_labels_monotone_by_diameter") is not all_monotone:
            raise ValueError("Frequency emulator bank aggregate monotonicity declaration is inconsistent.")
        legacy_gate = bank.validation.get("all_diameters_positive_and_monotonic")
        if legacy_gate is not None and legacy_gate is not all_monotone:
            raise ValueError(
                "Frequency emulator bank legacy monotonicity gate contradicts its PCHIP knots."
            )
        recorded_bounds = coordinates.get("common_ka_bounds_dpd")
        if recorded_bounds is not None and _bounds(
            recorded_bounds,
            "coordinates.common_ka_bounds_dpd",
        ) != bank.ka_bounds_dpd:
            raise ValueError(
                "Frequency emulator bank common_ka_bounds_dpd does not match its entries."
            )
        recorded_tolerance = _finite_float(
            coordinates.get("diameter_tolerance_um"),
            "coordinates.diameter_tolerance_um",
        )
        if recorded_tolerance != DIAMETER_TOLERANCE_UM:
            raise ValueError("Frequency emulator bank diameter tolerance does not match runtime.")
        return bank

    @classmethod
    def load(cls, path: str | Path) -> "DpdFrequencyEmulatorBank":
        artifact_path = Path(path)
        with artifact_path.open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def write(self, path: str | Path) -> Path:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_mapping(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return artifact_path


@dataclass(frozen=True)
class DpdPolynomialFrequencyEmulator:
    """One exact-diameter, bounded low-order polynomial frequency emulator."""

    diameter_um: float
    ka_bounds_dpd: tuple[float, float]
    coefficients: Any
    degree: int
    response_encoding: str
    ka_offset_dpd: float
    ka_scale_dpd: float
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        diameter = _positive_float(self.diameter_um, "diameter_um")
        bounds = _bounds(self.ka_bounds_dpd, "ka_bounds_dpd")
        degree = int(self.degree)
        if degree not in POLYNOMIAL_ALLOWED_DEGREES:
            raise ValueError(
                "Polynomial frequency emulator degree must be exactly 1 or 2."
            )
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.shape != (degree + 1,):
            raise ValueError(
                "polynomial.coefficients must contain degree + 1 ascending-power values."
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("polynomial.coefficients must contain only finite values.")
        response_encoding = str(self.response_encoding).strip().lower()
        if response_encoding not in {
            FREQUENCY_RESPONSE_ENCODING,
            FREQUENCY_SQUARED_RESPONSE_ENCODING,
        }:
            raise ValueError(
                "polynomial.response_encoding must be frequency_mhz or "
                "frequency_mhz_squared."
            )
        offset = _finite_float(self.ka_offset_dpd, "polynomial.input_transform.offset_dpd")
        scale = _positive_float(self.ka_scale_dpd, "polynomial.input_transform.scale_dpd")

        coefficients = np.array(coefficients, dtype=np.float64, copy=True)
        coefficients.flags.writeable = False
        object.__setattr__(self, "diameter_um", diameter)
        object.__setattr__(self, "ka_bounds_dpd", bounds)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "response_encoding", response_encoding)
        object.__setattr__(self, "ka_offset_dpd", offset)
        object.__setattr__(self, "ka_scale_dpd", scale)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "validation", dict(self.validation))

        self.dense_finite_positive_validation()

    def _checked_ka(self, ka_dpd: Any) -> np.ndarray:
        ka = np.asarray(ka_dpd, dtype=np.float64)
        if not np.all(np.isfinite(ka)):
            raise ValueError("ka_dpd must contain only finite values.")
        lower, upper = self.ka_bounds_dpd
        if np.any(ka < lower) or np.any(ka > upper):
            raise ValueError(
                f"ka_dpd is outside the supported range [{lower:g}, {upper:g}] "
                f"for diameter {self.diameter_um:g} um."
            )
        return ka

    def _predict_encoded(self, ka_dpd: Any) -> np.ndarray:
        ka = self._checked_ka(ka_dpd)
        x = (ka - self.ka_offset_dpd) / self.ka_scale_dpd
        if self.degree == 1:
            encoded = self.coefficients[1] * x + self.coefficients[0]
        else:
            encoded = (
                (self.coefficients[2] * x + self.coefficients[1]) * x
                + self.coefficients[0]
            )
        if np.any(~np.isfinite(encoded)) or np.any(encoded <= 0.0):
            raise ValueError(
                "Polynomial frequency emulator produced a non-positive or non-finite "
                f"{self.response_encoding} value for diameter {self.diameter_um:g} um."
            )
        return np.asarray(encoded, dtype=np.float64)

    def predict_mhz(self, ka_dpd: Any) -> np.ndarray:
        encoded = self._predict_encoded(ka_dpd)
        if self.response_encoding == FREQUENCY_SQUARED_RESPONSE_ENCODING:
            return np.sqrt(encoded)
        return encoded

    def dense_finite_positive_validation(
        self,
        *,
        point_count: int = DENSE_VALIDATION_MIN_POINTS,
    ) -> dict[str, float | int | bool]:
        count = int(point_count)
        if count < DENSE_VALIDATION_MIN_POINTS:
            raise ValueError(
                "Dense polynomial validation requires at least "
                f"{DENSE_VALIDATION_MIN_POINTS} points."
            )
        ka = np.linspace(self.ka_bounds_dpd[0], self.ka_bounds_dpd[1], count)
        frequency = self.predict_mhz(ka)
        if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
            raise ValueError(
                "Dense polynomial validation produced non-positive or non-finite "
                f"frequency for diameter {self.diameter_um:g} um."
            )
        return {
            "point_count": count,
            "finite_positive": True,
            "minimum_frequency_mhz": float(np.min(frequency)),
            "maximum_frequency_mhz": float(np.max(frequency)),
        }

    def frequency_roots_ka_dpd(self, frequency_mhz: float) -> tuple[float, ...]:
        target = _positive_float(frequency_mhz, "frequency_mhz")
        if self.response_encoding == FREQUENCY_SQUARED_RESPONSE_ENCODING:
            target = target**2
        shifted = np.array(self.coefficients, dtype=np.float64, copy=True)
        shifted[0] -= target
        nonzero = np.flatnonzero(np.abs(shifted) > 1.0e-15)
        if nonzero.size == 0:
            return self.ka_bounds_dpd
        roots = np.roots(shifted[: int(nonzero[-1]) + 1][::-1])
        lower, upper = self.ka_bounds_dpd
        tolerance = 1.0e-9 * max(1.0, upper - lower)
        accepted: list[float] = []
        for root in roots:
            if abs(float(root.imag)) > 1.0e-9:
                continue
            ka = self.ka_offset_dpd + float(root.real) * self.ka_scale_dpd
            if lower - tolerance <= ka <= upper + tolerance:
                accepted.append(float(min(max(ka, lower), upper)))
        ordered: list[float] = []
        for value in sorted(accepted):
            if not ordered or not np.isclose(value, ordered[-1], rtol=0.0, atol=1.0e-7):
                ordered.append(value)
        return tuple(ordered)

    def frequency_compatible_branches_ka_dpd(
        self,
        frequency_bounds_mhz: Sequence[float],
    ) -> tuple[tuple[float, float], ...]:
        lower_frequency, upper_frequency = _bounds(
            frequency_bounds_mhz,
            "frequency_bounds_mhz",
        )
        if lower_frequency <= 0.0:
            raise ValueError("frequency_bounds_mhz must contain positive frequencies.")
        boundaries = [self.ka_bounds_dpd[0], self.ka_bounds_dpd[1]]
        boundaries.extend(self.frequency_roots_ka_dpd(lower_frequency))
        boundaries.extend(self.frequency_roots_ka_dpd(upper_frequency))
        ordered: list[float] = []
        for value in sorted(boundaries):
            if not ordered or not np.isclose(value, ordered[-1], rtol=0.0, atol=1.0e-7):
                ordered.append(value)
        branches: list[tuple[float, float]] = []
        for left, right in zip(ordered, ordered[1:]):
            midpoint_value = float(self.predict_mhz(0.5 * (left + right)))
            if lower_frequency <= midpoint_value <= upper_frequency:
                if branches and np.isclose(branches[-1][1], left, rtol=0.0, atol=1.0e-7):
                    branches[-1] = (branches[-1][0], right)
                else:
                    branches.append((left, right))
        return tuple(branches)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "diameter_um": self.diameter_um,
            "ka_bounds_dpd": list(self.ka_bounds_dpd),
            "polynomial": {
                "method": POLYNOMIAL_METHOD,
                "basis": POLYNOMIAL_BASIS,
                "coefficient_order": POLYNOMIAL_COEFFICIENT_ORDER,
                "degree": self.degree,
                "coefficients": self.coefficients.tolist(),
                "response_encoding": self.response_encoding,
                "input_transform": {
                    "type": "affine",
                    "offset_dpd": self.ka_offset_dpd,
                    "scale_dpd": self.ka_scale_dpd,
                },
            },
            "provenance": dict(self.provenance),
            "validation": dict(self.validation),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DpdPolynomialFrequencyEmulator":
        if not isinstance(payload, Mapping):
            raise ValueError("Polynomial frequency emulator entry must be a mapping.")
        polynomial = payload.get("polynomial")
        if not isinstance(polynomial, Mapping):
            raise ValueError("Polynomial frequency emulator entry is missing polynomial metadata.")
        if polynomial.get("method") != POLYNOMIAL_METHOD:
            raise ValueError(f"polynomial.method must be {POLYNOMIAL_METHOD}.")
        if polynomial.get("basis") != POLYNOMIAL_BASIS:
            raise ValueError(f"polynomial.basis must be {POLYNOMIAL_BASIS}.")
        if polynomial.get("coefficient_order") != POLYNOMIAL_COEFFICIENT_ORDER:
            raise ValueError(
                f"polynomial.coefficient_order must be {POLYNOMIAL_COEFFICIENT_ORDER}."
            )
        transform = polynomial.get("input_transform")
        if not isinstance(transform, Mapping) or transform.get("type") != "affine":
            raise ValueError("polynomial.input_transform must be an explicit affine mapping.")
        return cls(
            diameter_um=payload.get("diameter_um"),
            ka_bounds_dpd=_bounds(payload.get("ka_bounds_dpd"), "ka_bounds_dpd"),
            coefficients=polynomial.get("coefficients"),
            degree=polynomial.get("degree"),
            response_encoding=polynomial.get("response_encoding"),
            ka_offset_dpd=transform.get("offset_dpd"),
            ka_scale_dpd=transform.get("scale_dpd"),
            provenance=_optional_mapping(payload.get("provenance"), "provenance"),
            validation=_optional_mapping(payload.get("validation"), "validation"),
        )


@dataclass(frozen=True)
class DpdPolynomialFrequencyEmulatorBank:
    """Vectorized exact-diameter bank of low-order polynomial emulators."""

    agent: str
    emulators: Sequence[DpdPolynomialFrequencyEmulator]
    conditions: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    _diameter_values: np.ndarray = field(init=False, repr=False, compare=False)
    _ka_lower_bounds: np.ndarray = field(init=False, repr=False, compare=False)
    _ka_upper_bounds: np.ndarray = field(init=False, repr=False, compare=False)
    _coefficient_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    _ka_offsets: np.ndarray = field(init=False, repr=False, compare=False)
    _ka_scales: np.ndarray = field(init=False, repr=False, compare=False)
    _squared_response: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent", _agent_name(self.agent))
        entries = tuple(self.emulators)
        if not entries:
            raise ValueError("Polynomial frequency emulator bank must contain at least one entry.")
        ordered = tuple(sorted(entries, key=lambda entry: entry.diameter_um))
        for previous, current in zip(ordered, ordered[1:]):
            if abs(current.diameter_um - previous.diameter_um) <= DIAMETER_TOLERANCE_UM:
                raise ValueError(
                    "Polynomial frequency emulator bank contains duplicate diameter "
                    f"{current.diameter_um:g} um."
                )
        lower = max(entry.ka_bounds_dpd[0] for entry in ordered)
        upper = min(entry.ka_bounds_dpd[1] for entry in ordered)
        if upper <= lower:
            raise ValueError("Polynomial frequency emulator entries have no common ka support.")

        coefficients = np.zeros((len(ordered), 3), dtype=np.float64)
        for index, entry in enumerate(ordered):
            coefficients[index, : entry.degree + 1] = entry.coefficients
        arrays = {
            "_diameter_values": np.asarray(
                [entry.diameter_um for entry in ordered], dtype=np.float64
            ),
            "_ka_lower_bounds": np.asarray(
                [entry.ka_bounds_dpd[0] for entry in ordered], dtype=np.float64
            ),
            "_ka_upper_bounds": np.asarray(
                [entry.ka_bounds_dpd[1] for entry in ordered], dtype=np.float64
            ),
            "_coefficient_matrix": coefficients,
            "_ka_offsets": np.asarray(
                [entry.ka_offset_dpd for entry in ordered], dtype=np.float64
            ),
            "_ka_scales": np.asarray(
                [entry.ka_scale_dpd for entry in ordered], dtype=np.float64
            ),
            "_squared_response": np.asarray(
                [
                    entry.response_encoding == FREQUENCY_SQUARED_RESPONSE_ENCODING
                    for entry in ordered
                ],
                dtype=bool,
            ),
        }
        for array in arrays.values():
            array.flags.writeable = False

        object.__setattr__(self, "emulators", ordered)
        object.__setattr__(self, "conditions", dict(self.conditions))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "validation", dict(self.validation))
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @property
    def diameters_um(self) -> tuple[float, ...]:
        return tuple(entry.diameter_um for entry in self.emulators)

    @property
    def ka_bounds_dpd(self) -> tuple[float, float]:
        return (
            max(entry.ka_bounds_dpd[0] for entry in self.emulators),
            min(entry.ka_bounds_dpd[1] for entry in self.emulators),
        )

    @property
    def response_encodings(self) -> tuple[str, ...]:
        return tuple(sorted({entry.response_encoding for entry in self.emulators}))

    def emulator_for_diameter(self, diameter_um: float) -> DpdPolynomialFrequencyEmulator:
        diameter = _positive_float(diameter_um, "diameter_um")
        matches = [
            entry
            for entry in self.emulators
            if abs(entry.diameter_um - diameter) <= DIAMETER_TOLERANCE_UM
        ]
        if not matches:
            supported = ", ".join(f"{value:g}" for value in self.diameters_um)
            raise ValueError(
                "No exact polynomial frequency emulator is available for diameter "
                f"{diameter:g} um; supported diameters are [{supported}] um."
            )
        return matches[0]

    def predict_for_diameters_mhz(self, ka_dpd: Any, diameter_um: Any) -> np.ndarray:
        ka, diameter = _broadcast_inputs(ka_dpd, diameter_um)
        original_shape = ka.shape
        flat_ka = ka.reshape(-1)
        flat_diameter = diameter.reshape(-1)

        right = np.searchsorted(self._diameter_values, flat_diameter, side="left")
        right = np.clip(right, 0, self._diameter_values.size - 1)
        left = np.clip(right - 1, 0, self._diameter_values.size - 1)
        left_distance = np.abs(flat_diameter - self._diameter_values[left])
        right_distance = np.abs(flat_diameter - self._diameter_values[right])
        emulator_index = np.where(left_distance <= right_distance, left, right)
        diameter_supported = (
            np.abs(flat_diameter - self._diameter_values[emulator_index])
            <= DIAMETER_TOLERANCE_UM
        )
        if not np.all(diameter_supported):
            missing = ", ".join(
                f"{value:g}" for value in np.unique(flat_diameter[~diameter_supported])
            )
            supported = ", ".join(f"{value:g}" for value in self.diameters_um)
            raise ValueError(
                "No exact polynomial frequency emulator is available for diameter(s) "
                f"[{missing}] um; supported diameters are [{supported}] um."
            )

        lower = self._ka_lower_bounds[emulator_index]
        upper = self._ka_upper_bounds[emulator_index]
        if np.any(flat_ka < lower) or np.any(flat_ka > upper):
            raise ValueError(
                "ka_dpd is outside the supported range for one or more exact-diameter "
                "polynomial frequency emulators."
            )
        x = (
            (flat_ka - self._ka_offsets[emulator_index])
            / self._ka_scales[emulator_index]
        )
        coefficients = self._coefficient_matrix[emulator_index]
        encoded = (coefficients[:, 2] * x + coefficients[:, 1]) * x + coefficients[:, 0]
        if np.any(~np.isfinite(encoded)) or np.any(encoded <= 0.0):
            raise ValueError(
                "Polynomial frequency emulator bank produced non-positive or non-finite "
                "encoded response values."
            )
        frequency = np.where(
            self._squared_response[emulator_index],
            np.sqrt(encoded),
            encoded,
        )
        if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
            raise ValueError(
                "Polynomial frequency emulator bank produced non-positive or non-finite "
                "frequency values."
            )
        return frequency.reshape(original_shape)

    def predict_with_support_mhz(
        self,
        ka_dpd: Any,
        diameter_um: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a population and return a hard support mask without scalar loops."""

        ka = np.asarray(ka_dpd, dtype=np.float64)
        diameter = np.asarray(diameter_um, dtype=np.float64)
        try:
            ka, diameter = np.broadcast_arrays(ka, diameter)
        except ValueError as error:
            raise ValueError("ka_dpd and diameter_um must be broadcast-compatible.") from error
        if not np.all(np.isfinite(diameter)) or np.any(diameter <= 0.0):
            raise ValueError("diameter_um must contain finite positive values.")
        original_shape = ka.shape
        flat_ka = ka.reshape(-1)
        flat_diameter = diameter.reshape(-1)

        right = np.searchsorted(self._diameter_values, flat_diameter, side="left")
        right = np.clip(right, 0, self._diameter_values.size - 1)
        left = np.clip(right - 1, 0, self._diameter_values.size - 1)
        emulator_index = np.where(
            np.abs(flat_diameter - self._diameter_values[left])
            <= np.abs(flat_diameter - self._diameter_values[right]),
            left,
            right,
        )
        diameter_supported = (
            np.abs(flat_diameter - self._diameter_values[emulator_index])
            <= DIAMETER_TOLERANCE_UM
        )
        if not np.all(diameter_supported):
            missing = ", ".join(
                f"{value:g}" for value in np.unique(flat_diameter[~diameter_supported])
            )
            raise ValueError(
                "No exact polynomial frequency emulator is available for diameter(s) "
                f"[{missing}] um."
            )

        lower = self._ka_lower_bounds[emulator_index]
        upper = self._ka_upper_bounds[emulator_index]
        support = np.isfinite(flat_ka) & (flat_ka >= lower) & (flat_ka <= upper)
        safe_ka = np.where(support, flat_ka, self._ka_offsets[emulator_index])
        x = (
            (safe_ka - self._ka_offsets[emulator_index])
            / self._ka_scales[emulator_index]
        )
        coefficients = self._coefficient_matrix[emulator_index]
        encoded = (coefficients[:, 2] * x + coefficients[:, 1]) * x + coefficients[:, 0]
        invalid_supported = support & ((~np.isfinite(encoded)) | (encoded <= 0.0))
        if np.any(invalid_supported):
            raise ValueError(
                "Polynomial frequency emulator bank produced a non-positive or non-finite "
                "encoded response inside declared support."
            )
        safe_encoded = np.where(support, encoded, 1.0)
        frequency = np.where(
            self._squared_response[emulator_index],
            np.sqrt(safe_encoded),
            safe_encoded,
        )
        frequency = np.where(support, frequency, 0.0)
        return frequency.reshape(original_shape), support.reshape(original_shape)

    def _mapping_without_integrity(self) -> dict[str, Any]:
        return {
            "schema": POLYNOMIAL_FREQUENCY_BANK_SCHEMA,
            "agent": self.agent,
            "model_family": "low_order_polynomial",
            "response": {
                "api_output_unit": FREQUENCY_UNIT,
                "coefficient_response_encodings": list(self.response_encodings),
            },
            "coordinates": {
                "diameter_mode": "exact",
                "diameter_unit": DIAMETER_UNIT,
                "diameter_tolerance_um": DIAMETER_TOLERANCE_UM,
                "ka_support": "inclusive",
                "common_ka_bounds_dpd": list(self.ka_bounds_dpd),
            },
            "runtime_contract": {
                "allowed_polynomial_degrees": list(POLYNOMIAL_ALLOWED_DEGREES),
                "diameter_interpolation_allowed": False,
                "ka_clipping_allowed": False,
                "ka_extrapolation_allowed": False,
                "analytical_fallback_allowed": False,
                "pchip_fallback_allowed": False,
                "runtime_requires_scipy": False,
                "population_evaluation": "numpy_exact_dispatch_gather_affine_horner",
            },
            "emulators": [entry.to_mapping() for entry in self.emulators],
            "conditions": dict(self.conditions),
            "provenance": dict(self.provenance),
            "validation": dict(self.validation),
        }

    def to_mapping(self) -> dict[str, Any]:
        payload = self._mapping_without_integrity()
        payload["integrity"] = {
            "algorithm": "sha256",
            "canonical_payload_sha256": _canonical_json_hash(payload),
        }
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DpdPolynomialFrequencyEmulatorBank":
        if not isinstance(payload, Mapping):
            raise ValueError("Polynomial frequency bank artifact must be a mapping.")
        if payload.get("schema") != POLYNOMIAL_FREQUENCY_BANK_SCHEMA:
            raise ValueError(
                f"Unsupported polynomial frequency bank schema: {payload.get('schema')!r}."
            )
        integrity = payload.get("integrity")
        if not isinstance(integrity, Mapping) or integrity.get("algorithm") != "sha256":
            raise ValueError("Polynomial frequency bank is missing SHA-256 integrity metadata.")
        expected_hash = _sha256_digest(
            integrity.get("canonical_payload_sha256"),
            "integrity.canonical_payload_sha256",
        )
        unsigned = dict(payload)
        unsigned.pop("integrity", None)
        if _canonical_json_hash(unsigned) != expected_hash:
            raise ValueError("Polynomial frequency bank integrity hash does not match payload.")
        if payload.get("model_family") != "low_order_polynomial":
            raise ValueError("Polynomial frequency bank model_family must be low_order_polynomial.")
        response = payload.get("response")
        coordinates = payload.get("coordinates")
        runtime = payload.get("runtime_contract")
        entries = payload.get("emulators")
        if not all(isinstance(value, Mapping) for value in (response, coordinates, runtime)):
            raise ValueError(
                "Polynomial frequency bank is missing response, coordinates, or runtime_contract."
            )
        if response.get("api_output_unit") != FREQUENCY_UNIT:
            raise ValueError(f"Polynomial frequency bank API output must use {FREQUENCY_UNIT}.")
        if coordinates.get("diameter_mode") != "exact":
            raise ValueError("Polynomial frequency bank diameter_mode must be exact.")
        if coordinates.get("diameter_unit") != DIAMETER_UNIT:
            raise ValueError(f"Polynomial frequency bank diameter_unit must be {DIAMETER_UNIT}.")
        if coordinates.get("ka_support") != "inclusive":
            raise ValueError("Polynomial frequency bank ka_support must be inclusive.")
        if runtime.get("allowed_polynomial_degrees") != list(POLYNOMIAL_ALLOWED_DEGREES):
            raise ValueError("Polynomial frequency bank must allow exactly degrees 1 and 2.")
        for field_name in (
            "diameter_interpolation_allowed",
            "ka_clipping_allowed",
            "ka_extrapolation_allowed",
            "analytical_fallback_allowed",
            "pchip_fallback_allowed",
            "runtime_requires_scipy",
        ):
            if runtime.get(field_name) is not False:
                raise ValueError(
                    f"Polynomial frequency bank runtime_contract.{field_name} must be false."
                )
        if runtime.get("population_evaluation") != "numpy_exact_dispatch_gather_affine_horner":
            raise ValueError("Polynomial frequency bank has an unsupported population evaluator.")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("Polynomial frequency bank emulators must be a sequence.")
        bank = cls(
            agent=payload.get("agent"),
            emulators=[DpdPolynomialFrequencyEmulator.from_mapping(entry) for entry in entries],
            conditions=_optional_mapping(payload.get("conditions"), "conditions"),
            provenance=_optional_mapping(payload.get("provenance"), "provenance"),
            validation=_optional_mapping(payload.get("validation"), "validation"),
        )
        _hashed_source_datasets(bank.provenance.get("source_datasets"))
        recorded_encodings = response.get("coefficient_response_encodings")
        if not isinstance(recorded_encodings, Sequence) or isinstance(
            recorded_encodings, (str, bytes)
        ):
            raise ValueError("Polynomial frequency bank response encodings must be a sequence.")
        if tuple(recorded_encodings) != bank.response_encodings:
            raise ValueError(
                "Polynomial frequency bank response encodings do not match its entries."
            )
        recorded_bounds = _bounds(
            coordinates.get("common_ka_bounds_dpd"),
            "coordinates.common_ka_bounds_dpd",
        )
        if recorded_bounds != bank.ka_bounds_dpd:
            raise ValueError(
                "Polynomial frequency bank common_ka_bounds_dpd does not match its entries."
            )
        if _finite_float(
            coordinates.get("diameter_tolerance_um"),
            "coordinates.diameter_tolerance_um",
        ) != DIAMETER_TOLERANCE_UM:
            raise ValueError("Polynomial frequency bank diameter tolerance does not match runtime.")
        if bank.validation.get("all_dense_grids_finite_positive") is not True:
            raise ValueError(
                "Polynomial frequency bank is missing dense finite-positive validation."
            )
        for entry in bank.emulators:
            entry_validation = entry.validation
            if entry_validation.get("dense_grid_finite_positive") is not True:
                raise ValueError(
                    "Polynomial frequency emulator lacks dense finite-positive validation "
                    f"at diameter {entry.diameter_um:g} um."
                )
            if int(entry_validation.get("dense_grid_point_count", 0)) < DENSE_VALIDATION_MIN_POINTS:
                raise ValueError(
                    "Polynomial frequency emulator dense validation is too sparse at "
                    f"diameter {entry.diameter_um:g} um."
                )
            entry.dense_finite_positive_validation()
        return bank

    @classmethod
    def load(cls, path: str | Path) -> "DpdPolynomialFrequencyEmulatorBank":
        artifact_path = Path(path)
        with artifact_path.open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def write(self, path: str | Path) -> Path:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_mapping(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return artifact_path


@dataclass(frozen=True)
class KaHoldoutMetric:
    held_out_ka_dpd: float
    point_count: int
    rmse_mhz: float
    max_absolute_error_mhz: float

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "held_out_ka_dpd": self.held_out_ka_dpd,
            "point_count": self.point_count,
            "rmse_mhz": self.rmse_mhz,
            "max_absolute_error_mhz": self.max_absolute_error_mhz,
        }


def _unique_sorted_ka_frequency(
    ka_dpd: Any,
    frequency_mhz: Any,
) -> tuple[np.ndarray, np.ndarray]:
    ka = np.asarray(ka_dpd, dtype=np.float64).reshape(-1)
    frequency = np.asarray(frequency_mhz, dtype=np.float64).reshape(-1)
    if ka.shape != frequency.shape or ka.size == 0:
        raise ValueError("ka_dpd and frequency_mhz must be non-empty arrays of equal length.")
    if not np.all(np.isfinite(ka)) or np.any(ka < 0.0):
        raise ValueError("ka_dpd must contain finite non-negative values.")
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_mhz must contain finite positive values.")
    order = np.argsort(ka, kind="mergesort")
    ka = ka[order]
    frequency = frequency[order]
    unique_ka: list[float] = []
    unique_frequency: list[float] = []
    index = 0
    while index < ka.size:
        current = ka[index]
        duplicate_mask = ka == current
        duplicate_values = frequency[duplicate_mask]
        if np.ptp(duplicate_values) > 1.0e-12:
            raise ValueError(
                f"Duplicate frequency labels at ka={current:g} differ; refusing to smooth labels."
            )
        unique_ka.append(float(current))
        unique_frequency.append(float(duplicate_values[0]))
        index += int(np.count_nonzero(duplicate_mask))
    if len(unique_ka) < 2:
        raise ValueError("PCHIP frequency emulator requires at least two unique ka nodes.")
    return np.asarray(unique_ka, dtype=np.float64), np.asarray(unique_frequency, dtype=np.float64)


def fit_frequency_emulator(
    *,
    diameter_um: float,
    ka_dpd: Any,
    frequency_mhz: Any,
    ka_bounds_dpd: tuple[float, float] | None = None,
    degree: int = 3,
    provenance: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
) -> DpdFrequencyEmulator:
    """Fit a shape-preserving PCHIP to measured frequency squared."""

    del degree
    ka, frequency = _unique_sorted_ka_frequency(ka_dpd, frequency_mhz)
    if ka_bounds_dpd is None:
        ka_bounds_dpd = (float(ka[0]), float(ka[-1]))
    bounds = _bounds(ka_bounds_dpd, "ka_bounds_dpd")
    if ka[0] != bounds[0] or ka[-1] != bounds[1]:
        raise ValueError("PCHIP fit requires endpoint ka labels at ka_bounds_dpd.")

    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError as error:  # pragma: no cover - exercised only in stripped runtimes
        raise RuntimeError("SciPy is required to generate PCHIP frequency emulators.") from error

    interpolator = PchipInterpolator(ka, frequency**2, extrapolate=False)
    interval_coefficients = np.asarray(interpolator.c.T[:, ::-1], dtype=np.float64)
    return DpdFrequencyEmulator(
        diameter_um=diameter_um,
        ka_bounds_dpd=bounds,
        knots_ka_dpd=ka,
        interval_coefficients=interval_coefficients,
        provenance=provenance or {},
        validation=validation or {},
    )


def leave_one_ka_out_metrics(
    *,
    diameter_um: float,
    ka_dpd: Any,
    frequency_mhz: Any,
    ka_bounds_dpd: tuple[float, float] | None = None,
    degree: int = 3,
) -> list[KaHoldoutMetric]:
    """Validate only interior ka interpolation by withholding each interior node."""

    del degree
    ka, frequency = _unique_sorted_ka_frequency(ka_dpd, frequency_mhz)
    if ka_bounds_dpd is None:
        ka_bounds_dpd = (float(ka[0]), float(ka[-1]))
    bounds = _bounds(ka_bounds_dpd, "ka_bounds_dpd")
    if ka[0] != bounds[0] or ka[-1] != bounds[1]:
        raise ValueError("Leave-one-ka-out validation requires endpoint labels at ka_bounds_dpd.")
    if ka.size <= 2:
        return []
    metrics: list[KaHoldoutMetric] = []
    for held_out in ka[1:-1]:
        test_mask = ka == held_out
        emulator = fit_frequency_emulator(
            diameter_um=diameter_um,
            ka_dpd=ka[~test_mask],
            frequency_mhz=frequency[~test_mask],
            ka_bounds_dpd=bounds,
        )
        predicted = emulator.predict_mhz(ka[test_mask])
        errors = predicted - frequency[test_mask]
        metrics.append(
            KaHoldoutMetric(
                held_out_ka_dpd=float(held_out),
                point_count=int(np.count_nonzero(test_mask)),
                rmse_mhz=float(np.sqrt(np.mean(errors**2))),
                max_absolute_error_mhz=float(np.max(np.abs(errors))),
            )
        )
    return metrics


__all__ = [
    "COEFFICIENT_ORDER",
    "DIAMETER_TOLERANCE_UM",
    "DENSE_VALIDATION_MIN_POINTS",
    "DpdFrequencyEmulator",
    "DpdFrequencyEmulatorBank",
    "DpdPolynomialFrequencyEmulator",
    "DpdPolynomialFrequencyEmulatorBank",
    "FREQUENCY_EMULATOR_BANK_SCHEMA",
    "FREQUENCY_RESPONSE_ENCODING",
    "FREQUENCY_SQUARED_RESPONSE_ENCODING",
    "FIT_PRIMARY_LABEL_ADMISSION_POLICY",
    "KaHoldoutMetric",
    "NONMONOTONE_FREQUENCY_POLICY",
    "POLYNOMIAL_ALLOWED_DEGREES",
    "POLYNOMIAL_FREQUENCY_BANK_SCHEMA",
    "fit_frequency_emulator",
    "leave_one_ka_out_metrics",
]
