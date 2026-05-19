"""Candidate constraints for active-learning candidate collections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.contracts import Candidate, as_candidate
from meso_uq.structures.gv.material_parameters import validate_material_parameter_overrides
from meso_uq.structures.gv.parameters import GV_MATERIAL_PARAMETER_NAMES
from meso_uq.structures.gv.sampling.validation import GV_SAMPLING_EXPERIMENTS, validate_explicit_controls

ACTIVE_LEARNING_CANDIDATE_CONSTRAINTS_SCHEMA_VERSION = "meso_uq.active_learning.candidate_constraints.v1"

DEFAULT_ALLOWED_FAMILIES: tuple[str, ...] = ("gv", "emb")
DEFAULT_REQUIRED_PARAMETER_PATHS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "gv": (
        ("experiment",),
        ("material_parameters",),
        ("controls",),
    ),
    "emb": (
        ("experiment",),
    ),
}
DEFAULT_NUMERIC_RANGE_CHECKS: Mapping[tuple[str, ...], tuple[float | None, float | None]] = {
    ("controls",): (None, None),
    ("radGV",): (0.0, None),
    ("height",): (0.0, None),
    ("geometry", "radGV"): (0.0, None),
    ("geometry", "height"): (0.0, None),
}

REASON_DUPLICATE_CANDIDATE_ID = "duplicate_candidate_id"
REASON_INVALID_CANDIDATE = "invalid_candidate"
REASON_INVALID_EXPERIMENT = "invalid_experiment"
REASON_INVALID_FAMILY = "invalid_family"
REASON_MISSING_GV_GEOMETRY = "missing_gv_geometry"
REASON_MIXED_GV_GEOMETRY = "mixed_gv_geometry"
REASON_INVALID_PAYLOAD_PATH = "invalid_payload_path"
REASON_MISSING_REQUIRED_PARAMETER_PATH = "missing_required_parameter_path"
REASON_NON_FINITE_NUMERIC = "non_finite_numeric_value"
REASON_OUT_OF_RANGE = "numeric_value_out_of_range"

_DEFAULT_GV_ALLOWED_EXPERIMENTS = frozenset(GV_SAMPLING_EXPERIMENTS)

_NON_PATH_SENTINEL = object()


@dataclass(frozen=True)
class ActiveLearningCandidateConstraintReason:
    """One validation reason for a rejected candidate."""

    code: str
    message: str


@dataclass(frozen=True)
class ActiveLearningCandidateConstraintReport:
    """Structured validation report for one active-learning candidate batch."""

    valid_candidates: tuple[Candidate, ...]
    rejected_candidate_ids: tuple[str, ...]
    reason_code_counts: Mapping[str, int]
    per_candidate_reasons: Mapping[str, tuple[ActiveLearningCandidateConstraintReason, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTIVE_LEARNING_CANDIDATE_CONSTRAINTS_SCHEMA_VERSION,
            "valid_candidates": [candidate.as_dict() for candidate in self.valid_candidates],
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "reason_code_counts": dict(self.reason_code_counts),
            "per_candidate_reasons": {
                candidate_id: [
                    {"code": reason.code, "message": reason.message}
                    for reason in reasons
                ]
                for candidate_id, reasons in self.per_candidate_reasons.items()
            },
        }


def _coerce_family(payload: Mapping[str, Any], *, allowed_families: set[str], context: str) -> tuple[str, Mapping[str, Any]]:
    raw_family = payload.get("family", payload.get("dpd_family"))
    if raw_family is not None:
        family = str(raw_family).strip().lower()
        if family not in allowed_families:
            raise ValueError(
                f"{context} has unsupported family {family!r}. Supported: {', '.join(sorted(allowed_families))}."
            )
        if family == "gv" and "gv_launch" in payload:
            candidate_payload = payload["gv_launch"]
            if not isinstance(candidate_payload, Mapping):
                raise ValueError(
                    f"{context} declared family='gv' but family payload key 'gv_launch' is not a mapping."
                )
            return family, candidate_payload
        if family == "emb" and "emb_launch" in payload:
            candidate_payload = payload["emb_launch"]
            if not isinstance(candidate_payload, Mapping):
                raise ValueError(
                    f"{context} declared family='emb' but family payload key 'emb_launch' is not a mapping."
                )
            return family, candidate_payload
        return family, payload

    if "gv_launch" in payload:
        if "gv" not in allowed_families:
            raise ValueError(
                f"{context} has unsupported family 'gv'. Supported: {', '.join(sorted(allowed_families))}."
            )
        candidate_payload = payload["gv_launch"]
        if not isinstance(candidate_payload, Mapping):
            raise ValueError("candidate payload key 'gv_launch' must be a mapping.")
        return "gv", candidate_payload
    if "emb_launch" in payload:
        if "emb" not in allowed_families:
            raise ValueError(
                f"{context} has unsupported family 'emb'. Supported: {', '.join(sorted(allowed_families))}."
            )
        candidate_payload = payload["emb_launch"]
        if not isinstance(candidate_payload, Mapping):
            raise ValueError("candidate payload key 'emb_launch' must be a mapping.")
        return "emb", candidate_payload

    family = "gv" if "gv" in allowed_families else next(iter(allowed_families))
    return family, payload


def _coerce_allowed_experiments(
    allowed_experiments: Sequence[str] | None,
) -> set[str] | None:
    if allowed_experiments is None:
        return None
    normalized = {str(item).strip().lower() for item in allowed_experiments if str(item).strip()}
    if not normalized:
        return None
    return normalized


def _coerce_required_paths(
    required_parameter_paths: Mapping[str, Sequence[Sequence[str]]],
) -> Mapping[str, tuple[tuple[str, ...], ...]]:
    normalized: dict[str, tuple[tuple[str, ...], ...]] = {}
    for family, raw_paths in required_parameter_paths.items():
        key = str(family).strip().lower()
        if not key:
            continue
        path_values: list[tuple[str, ...]] = []
        for raw_path in raw_paths:
            path = tuple(str(item).strip() for item in raw_path if str(item).strip())
            if not path:
                continue
            path_values.append(path)
        if path_values:
            normalized[key] = tuple(path_values)
    return normalized


def _coerce_numeric_ranges(
    numeric_range_checks: Mapping[Sequence[str] | str, tuple[float | None, float | None]],
) -> Mapping[tuple[str, ...], tuple[float | None, float | None]]:
    normalized: dict[tuple[str, ...], tuple[float | None, float | None]] = {}
    for raw_path, raw_bounds in numeric_range_checks.items():
        if isinstance(raw_path, str):
            path = tuple(item for item in raw_path.split(".") if item.strip())
        else:
            path = tuple(str(item).strip() for item in raw_path if str(item).strip())
        if not path:
            continue
        if (
            not isinstance(raw_bounds, tuple)
            or len(raw_bounds) != 2
        ):
            raise ValueError(
                "numeric_range_checks values must be 2-tuples of optional lower and upper bounds."
            )
        lower, upper = raw_bounds
        if lower is not None and not isinstance(lower, (int, float)):
            raise ValueError("numeric lower bounds must be numbers or None.")
        if upper is not None and not isinstance(upper, (int, float)):
            raise ValueError("numeric upper bounds must be numbers or None.")
        normalized[path] = (None if lower is None else float(lower), None if upper is None else float(upper))
    return normalized


def _reason_for_missing_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _value_at_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return _NON_PATH_SENTINEL
        if key not in current:
            return _NON_PATH_SENTINEL
        current = current[key]
    return current


def _is_missing_value(value: Any) -> bool:
    if value is _NON_PATH_SENTINEL:
        return True
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value) == 0
    return False


def _flatten_numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("non-numeric payload value")
    if isinstance(value, (int, float)):
        if not isfinite(float(value)):
            raise ValueError("non-finite numeric value")
        return [float(value)]
    if isinstance(value, Mapping):
        result: list[float] = []
        for item in value.values():
            result.extend(_flatten_numeric_values(item))
        return result
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten_numeric_values(item))
        return result
    raise TypeError("non-numeric payload value")


def _record_reason(
    *,
    candidate_id: str,
    code: str,
    message: str,
    code_counts: Counter[str],
    per_candidate_reasons: dict[str, list[ActiveLearningCandidateConstraintReason]],
) -> None:
    code_counts[code] += 1
    per_candidate_reasons.setdefault(candidate_id, [])
    per_candidate_reasons[candidate_id].append(
        ActiveLearningCandidateConstraintReason(code=code, message=message)
    )


def validate_active_learning_candidates(
    candidates: Sequence[object],
    *,
    required_parameter_paths: Mapping[str, Sequence[Sequence[str]]] | None = None,
    numeric_range_checks: Mapping[Sequence[str] | str, tuple[float | None, float | None]]
    | None = None,
    allowed_families: Sequence[str] = DEFAULT_ALLOWED_FAMILIES,
    allowed_experiments: Sequence[str] | None = None,
) -> ActiveLearningCandidateConstraintReport:
    """Return a structured report for candidate validation.

    The validator is intentionally lightweight: no heavy optional dependencies and only
    dictionary-based checks with concise reason-code reporting.
    """

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
        raise ValueError("candidates must be a sequence.")
    if not candidates:
        return ActiveLearningCandidateConstraintReport(
            valid_candidates=(),
            rejected_candidate_ids=(),
            reason_code_counts={},
            per_candidate_reasons={},
        )

    allowed_families_set = {str(item).strip().lower() for item in allowed_families if str(item).strip()}
    if not allowed_families_set:
        raise ValueError("allowed_families must contain at least one value.")
    if not allowed_families_set.issubset(set(DEFAULT_ALLOWED_FAMILIES)):
        raise ValueError(
            "allowed_families must be a subset of {'gv', 'emb'}."
        )

    normalized_required_paths = _coerce_required_paths(
        DEFAULT_REQUIRED_PARAMETER_PATHS
        if required_parameter_paths is None
        else required_parameter_paths
    )
    normalized_numeric_ranges = _coerce_numeric_ranges(
        DEFAULT_NUMERIC_RANGE_CHECKS
        if numeric_range_checks is None
        else numeric_range_checks
    )
    normalized_experiments = _coerce_allowed_experiments(allowed_experiments)

    seen_candidate_ids: set[str] = set()
    valid_candidates: list[Candidate] = []
    rejected_candidate_ids: list[str] = []
    reason_counts: Counter[str] = Counter()
    per_candidate_reasons: dict[str, list[ActiveLearningCandidateConstraintReason]] = {}

    for index, item in enumerate(candidates):
        candidate_id = (
            str(item.get("candidate_id"))
            if isinstance(item, Mapping)
            else f"candidate-{index}"
        )
        if candidate_id == "None":
            candidate_id = f"candidate-{index}"
        candidate: Candidate | None = None

        try:
            candidate = as_candidate(item)
            candidate_id = candidate.candidate_id
        except Exception as exc:
            _record_reason(
                candidate_id=candidate_id,
                code=REASON_INVALID_CANDIDATE,
                message=f"Candidate {candidate_id!r} is not a valid active-learning Candidate: {exc}",
                code_counts=reason_counts,
                per_candidate_reasons=per_candidate_reasons,
            )

        if candidate is None:
            rejected_candidate_ids.append(candidate_id)
            seen_candidate_ids.add(candidate_id)
            continue

        if candidate_id in seen_candidate_ids:
            _record_reason(
                candidate_id=candidate_id,
                code=REASON_DUPLICATE_CANDIDATE_ID,
                message=f"Duplicate candidate_id {candidate_id!r}.",
                code_counts=reason_counts,
                per_candidate_reasons=per_candidate_reasons,
            )
            rejected_candidate_ids.append(candidate_id)
            continue

        seen_candidate_ids.add(candidate_id)
        existing_reason_count = len(per_candidate_reasons.get(candidate_id, ()))
        candidate_allowed_experiments = normalized_experiments

        try:
            family, candidate_payload = _coerce_family(
                candidate.parameters,
                allowed_families=allowed_families_set,
                context=f"Candidate {candidate_id!r}",
            )
        except Exception as exc:
            _record_reason(
                candidate_id=candidate_id,
                code=REASON_INVALID_FAMILY,
                message=str(exc),
                code_counts=reason_counts,
                per_candidate_reasons=per_candidate_reasons,
            )
            rejected_candidate_ids.append(candidate_id)
            continue

        required_paths = normalized_required_paths.get(family, ())
        for path in required_paths:
            value = _value_at_path(candidate_payload, path)
            if _is_missing_value(value):
                _record_reason(
                    candidate_id=candidate_id,
                    code=REASON_MISSING_REQUIRED_PARAMETER_PATH,
                    message=(
                        f"{candidate_id!r} for family {family!r} is missing required parameter path "
                        f"{_reason_for_missing_path(path)!r}."
                    ),
                    code_counts=reason_counts,
                    per_candidate_reasons=per_candidate_reasons,
                )

        if family == "gv":
            if candidate_allowed_experiments is None:
                candidate_allowed_experiments = _DEFAULT_GV_ALLOWED_EXPERIMENTS

            geometry_value = _value_at_path(candidate_payload, ("geometry",))
            nested_radgv_value = _value_at_path(candidate_payload, ("geometry", "radGV"))
            nested_height_value = _value_at_path(candidate_payload, ("geometry", "height"))
            radgv_value = _value_at_path(candidate_payload, ("radGV",))
            height_value = _value_at_path(candidate_payload, ("height",))
            has_explicit_geometry_field = "geometry" in candidate_payload
            has_complete_nested_geometry = (
                not _is_missing_value(nested_radgv_value)
                and not _is_missing_value(nested_height_value)
            )
            has_explicit_legacy_geometry_field = "radGV" in candidate_payload or "height" in candidate_payload
            has_legacy_geometry = (
                not _is_missing_value(radgv_value)
                and not _is_missing_value(height_value)
            )
            if has_explicit_geometry_field and has_explicit_legacy_geometry_field:
                _record_reason(
                    candidate_id=candidate_id,
                    code=REASON_MIXED_GV_GEOMETRY,
                    message=(
                        f"{candidate_id!r} for family 'gv' mixes nested 'geometry' with explicit top-level "
                        "'radGV'/'height' fields; provide exactly one geometry form."
                    ),
                    code_counts=reason_counts,
                    per_candidate_reasons=per_candidate_reasons,
                )
            elif not (has_complete_nested_geometry or has_legacy_geometry):
                _record_reason(
                    candidate_id=candidate_id,
                    code=REASON_MISSING_GV_GEOMETRY,
                    message=(
                        f"{candidate_id!r} for family 'gv' must include either complete nested geometry "
                        "fields 'geometry.radGV' and 'geometry.height' or both legacy geometry fields "
                        "'radGV' and 'height'."
                    ),
                    code_counts=reason_counts,
                    per_candidate_reasons=per_candidate_reasons,
                )

            material_parameters_value = _value_at_path(candidate_payload, ("material_parameters",))
            if not _is_missing_value(material_parameters_value):
                if not isinstance(material_parameters_value, Mapping):
                    _record_reason(
                        candidate_id=candidate_id,
                        code=REASON_INVALID_PAYLOAD_PATH,
                        message=(
                            f"{candidate_id!r} path 'material_parameters' must be a mapping for GV material validation."
                        ),
                        code_counts=reason_counts,
                        per_candidate_reasons=per_candidate_reasons,
                    )
                else:
                    try:
                        validated_material_parameters = validate_material_parameter_overrides(
                            material_parameters_value
                        )
                        for name in GV_MATERIAL_PARAMETER_NAMES:
                            value = float(validated_material_parameters[name])
                            if not isfinite(value) or value <= 0.0:
                                raise ValueError(
                                    f"GV launch material parameter {name!r} must be finite and > 0."
                                )
                    except ValueError as exc:
                        error_message = str(exc)
                        reason_code = REASON_INVALID_PAYLOAD_PATH
                        if error_message.startswith("Missing required GV material parameters:"):
                            reason_code = REASON_MISSING_REQUIRED_PARAMETER_PATH
                        elif "must be finite and" in error_message:
                            reason_code = REASON_OUT_OF_RANGE
                        _record_reason(
                            candidate_id=candidate_id,
                            code=reason_code,
                            message=f"{candidate_id!r} has invalid GV material_parameters: {error_message}",
                            code_counts=reason_counts,
                            per_candidate_reasons=per_candidate_reasons,
                        )

            controls_value = _value_at_path(candidate_payload, ("controls",))
            experiment_value = _value_at_path(candidate_payload, ("experiment",))
            if (
                not _is_missing_value(controls_value)
                and not _is_missing_value(experiment_value)
                and str(experiment_value) in _DEFAULT_GV_ALLOWED_EXPERIMENTS
            ):
                try:
                    validate_explicit_controls(
                        experiment=str(experiment_value),
                        controls=controls_value if isinstance(controls_value, Mapping) else None,
                    )
                except ValueError as exc:
                    _record_reason(
                        candidate_id=candidate_id,
                        code=REASON_INVALID_PAYLOAD_PATH,
                        message=f"{candidate_id!r} has invalid GV controls: {exc}",
                        code_counts=reason_counts,
                        per_candidate_reasons=per_candidate_reasons,
                    )

        experiment_value = _value_at_path(candidate_payload, ("experiment",))
        if candidate_allowed_experiments is not None and not _is_missing_value(experiment_value):
            if family == "gv":
                experiment = str(experiment_value)
            else:
                experiment = str(experiment_value).strip().lower()
            if experiment not in candidate_allowed_experiments:
                _record_reason(
                    candidate_id=candidate_id,
                    code=REASON_INVALID_EXPERIMENT,
                    message=(
                        f"{candidate_id!r} has experiment {experiment_value!r} "
                        "which is not in allowed experiment names."
                    ),
                    code_counts=reason_counts,
                    per_candidate_reasons=per_candidate_reasons,
                )

        for path, bounds in normalized_numeric_ranges.items():
            value = _value_at_path(candidate_payload, path)
            if _is_missing_value(value):
                continue
            try:
                values = _flatten_numeric_values(value)
            except ValueError:
                _record_reason(
                    candidate_id=candidate_id,
                    code=REASON_NON_FINITE_NUMERIC,
                    message=f"{candidate_id!r} path {_reason_for_missing_path(path)!r} contains non-finite values.",
                    code_counts=reason_counts,
                    per_candidate_reasons=per_candidate_reasons,
                )
                continue
            except TypeError:
                _record_reason(
                    candidate_id=candidate_id,
                    code=REASON_INVALID_PAYLOAD_PATH,
                    message=(
                        f"{candidate_id!r} path {_reason_for_missing_path(path)!r} "
                        "must be numeric for numeric validation."
                    ),
                    code_counts=reason_counts,
                    per_candidate_reasons=per_candidate_reasons,
                )
                continue

            if not values:
                _record_reason(
                    candidate_id=candidate_id,
                    code=REASON_INVALID_PAYLOAD_PATH,
                    message=f"{candidate_id!r} path {_reason_for_missing_path(path)!r} must contain numeric values.",
                    code_counts=reason_counts,
                    per_candidate_reasons=per_candidate_reasons,
                )
                continue

            lower_bound, upper_bound = bounds
            for value in values:
                if not isfinite(value):
                    _record_reason(
                        candidate_id=candidate_id,
                        code=REASON_NON_FINITE_NUMERIC,
                        message=f"{candidate_id!r} path {_reason_for_missing_path(path)!r} contains non-finite values.",
                        code_counts=reason_counts,
                        per_candidate_reasons=per_candidate_reasons,
                    )
                    break
                if lower_bound is not None and value < lower_bound:
                    _record_reason(
                        candidate_id=candidate_id,
                        code=REASON_OUT_OF_RANGE,
                        message=(
                            f"{candidate_id!r} path {_reason_for_missing_path(path)!r} has value {value!r} "
                            f"below lower bound {lower_bound!r}."
                        ),
                        code_counts=reason_counts,
                        per_candidate_reasons=per_candidate_reasons,
                    )
                    break
                if upper_bound is not None and value > upper_bound:
                    _record_reason(
                        candidate_id=candidate_id,
                        code=REASON_OUT_OF_RANGE,
                        message=(
                            f"{candidate_id!r} path {_reason_for_missing_path(path)!r} has value {value!r} "
                            f"above upper bound {upper_bound!r}."
                        ),
                        code_counts=reason_counts,
                        per_candidate_reasons=per_candidate_reasons,
                    )
                    break

        if family == "gv":
            for path in (("radGV",), ("height",), ("geometry", "radGV"), ("geometry", "height")):
                value = _value_at_path(candidate_payload, path)
                if _is_missing_value(value):
                    continue
                try:
                    numbers = _flatten_numeric_values(value)
                except (TypeError, ValueError):
                    continue
                for number in numbers:
                    if number == 0.0:
                        _record_reason(
                            candidate_id=candidate_id,
                            code=REASON_OUT_OF_RANGE,
                            message=(
                                f"{candidate_id!r} path {_reason_for_missing_path(path)!r} has value {number!r} "
                                "but GV geometry values must be > 0."
                            ),
                            code_counts=reason_counts,
                            per_candidate_reasons=per_candidate_reasons,
                        )
                        break

        current_reason_count = len(per_candidate_reasons.get(candidate_id, ()))
        if current_reason_count > existing_reason_count:
            rejected_candidate_ids.append(candidate_id)
        else:
            valid_candidates.append(candidate)

    for candidate_id in list(per_candidate_reasons.keys()):
        per_candidate_reasons[candidate_id] = tuple(per_candidate_reasons[candidate_id])

    return ActiveLearningCandidateConstraintReport(
        valid_candidates=tuple(valid_candidates),
        rejected_candidate_ids=tuple(rejected_candidate_ids),
        reason_code_counts=dict(reason_counts),
        per_candidate_reasons=per_candidate_reasons,
    )


__all__ = [
    "ACTIVE_LEARNING_CANDIDATE_CONSTRAINTS_SCHEMA_VERSION",
    "ActiveLearningCandidateConstraintReason",
    "ActiveLearningCandidateConstraintReport",
    "DEFAULT_ALLOWED_FAMILIES",
    "DEFAULT_REQUIRED_PARAMETER_PATHS",
    "DEFAULT_NUMERIC_RANGE_CHECKS",
    "REASON_DUPLICATE_CANDIDATE_ID",
    "REASON_INVALID_CANDIDATE",
    "REASON_INVALID_EXPERIMENT",
    "REASON_INVALID_FAMILY",
    "REASON_MISSING_GV_GEOMETRY",
    "REASON_MIXED_GV_GEOMETRY",
    "REASON_INVALID_PAYLOAD_PATH",
    "REASON_MISSING_REQUIRED_PARAMETER_PATH",
    "REASON_NON_FINITE_NUMERIC",
    "REASON_OUT_OF_RANGE",
    "validate_active_learning_candidates",
]
