from __future__ import annotations

"""Deterministic result-ingestion scaffold for Active Learning simulation outputs."""

import hashlib
import json
import math
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTIVE_LEARNING_RESULT_INGESTION_SCHEMA_VERSION = "meso_uq.active_learning.result_ingestion.v1"
ACTIVE_LEARNING_RESULT_INGESTION_RECORD_SCHEMA_VERSION = "meso_uq.active_learning.result_ingestion.record.v1"
ACTIVE_LEARNING_RESULT_INGESTION_REPORT_SCHEMA_VERSION = "meso_uq.active_learning.result_ingestion.report.v1"
ACTIVE_LEARNING_RESULT_INGESTION_MANIFEST_FILENAME = "result_ingestion_manifest.json"
ACTIVE_LEARNING_RESULT_INGESTION_REPORT_FILENAME = "result_ingestion_report.json"
ACTIVE_LEARNING_RESULT_INGESTION_PLOT_FILENAME = "result_ingestion_validation.png"
ACTIVE_LEARNING_RESULT_INGESTION_PLOT_SIDECAR_FILENAME = "result_ingestion_validation.png.json"
ACTIVE_LEARNING_RESULT_INGESTION_PLOT_DPI = 90

_SUPPORTED_FAMILIES = frozenset({"gv", "emb"})
_SUPPORTED_STATUSES = frozenset({"completed", "failed", "missing", "partial"})
_FAILED_STATUS_TOKENS = frozenset({"error", "failed", "failure"})


def _coerce_nonempty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _coerce_family(value: object, *, candidate_id: str) -> str:
    family = _coerce_nonempty_text(value, field_name=f"candidate {candidate_id} family").lower()
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(f"candidate {candidate_id!r} family must be one of: {', '.join(sorted(_SUPPORTED_FAMILIES))}.")
    return family


def _coerce_iteration_token(iteration: int | str) -> str:
    if isinstance(iteration, int):
        if iteration < 0:
            raise ValueError("iteration must be non-negative.")
        return f"iter_{iteration:04d}"
    raw = _coerce_nonempty_text(iteration, field_name="iteration")
    lowered = raw.lower()
    if lowered.startswith("iter_") or lowered.startswith("iter-"):
        return raw.replace("-", "_")
    if raw.isdigit():
        return f"iter_{int(raw):04d}"
    return f"iter_{raw}"


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _normalize_json(value: Any, context: str) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_normalize_json(item, f"{context}[]") for item in value]
    if isinstance(value, tuple):
        return [_normalize_json(item, f"{context}[]") for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item, f"{context}[{key!r}]") for key, item in value.items()}
    raise ValueError(f"{context} must be JSON-compatible, got {type(value)!r}.")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_normalize_json(payload, "payload"), sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256_short(payload: Mapping[str, Any], *, length: int = 16) -> str:
    if length < 4 or length > 64:
        raise ValueError("hash length must be within [4, 64].")
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def _coerce_mapping(payload: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return {str(key): value for key, value in payload.items()}


def _coerce_optional_mapping(payload: object, *, field_name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    return _coerce_mapping(payload, field_name=field_name)


def _extract_ref_mappings(payload: object) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        source = {str(key): value for key, value in payload.items()}
        if {"dataset_id", "hdf5_path", "manifest_path"}.issubset(source):
            return [source]
        refs: list[dict[str, Any]] = []
        if "campaign" in source:
            refs.extend(_extract_ref_mappings(source["campaign"]))
        if "runs" in source:
            refs.extend(_extract_ref_mappings(source["runs"]))
        if refs:
            return refs
        for value in source.values():
            refs.extend(_extract_ref_mappings(value))
        return refs
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        refs: list[dict[str, Any]] = []
        for item in payload:
            refs.extend(_extract_ref_mappings(item))
        return refs
    return []


def _normalize_raw_curve_refs(
    payload: Mapping[str, Any],
    *,
    family: str,
    candidate_id: str,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    refs_payload = payload.get("expected_hdf5_datasets")
    if refs_payload is None:
        refs_payload = payload.get("expected_hdf5_refs")
    refs = _extract_ref_mappings(refs_payload)
    reason_codes: list[str] = []
    normalized: list[dict[str, str]] = []
    if not refs:
        return (), ("missing_expected_hdf5_refs",)

    for index, ref in enumerate(refs):
        context = f"candidate {candidate_id} expected_hdf5[{index}]"
        dataset_id = str(ref.get("dataset_id", "")).strip()
        hdf5_path = str(ref.get("hdf5_path", "")).strip()
        manifest_path = str(ref.get("manifest_path", "")).strip()
        ref_family = str(ref.get("family", family)).strip().lower() or family
        if not dataset_id:
            reason_codes.append("missing_ref_dataset_id")
            dataset_id = f"{candidate_id}-dataset-{index:03d}"
        if ref_family not in _SUPPORTED_FAMILIES:
            reason_codes.append("invalid_ref_family")
            ref_family = family
        if ref_family != family:
            reason_codes.append("mismatched_ref_family")
        if not hdf5_path:
            reason_codes.append("missing_hdf5_path")
        elif not hdf5_path.lower().endswith((".h5", ".hdf5")):
            reason_codes.append("invalid_hdf5_path_suffix")
        if not manifest_path:
            reason_codes.append("missing_manifest_path")
        elif not manifest_path.lower().endswith(".json"):
            reason_codes.append("invalid_manifest_path_suffix")
        normalized.append(
            {
                "family": family,
                "dataset_id": dataset_id,
                "hdf5_path": hdf5_path,
                "manifest_path": manifest_path,
            }
        )
    return tuple(normalized), tuple(sorted(set(reason_codes)))


def _normalize_reduced_observables(payload: Mapping[str, Any]) -> tuple[dict[str, float], tuple[str, ...]]:
    observables_payload = payload.get("reduced_observables")
    if observables_payload is None:
        observables_payload = payload.get("reduced")
    if observables_payload is None:
        return {}, ("missing_reduced_observables",)
    if not isinstance(observables_payload, Mapping):
        return {}, ("invalid_reduced_observables_mapping",)

    observables: dict[str, float] = {}
    reason_codes: list[str] = []
    for key, value in observables_payload.items():
        name = str(key).strip()
        if not name:
            reason_codes.append("invalid_reduced_observable_name")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            reason_codes.append("invalid_reduced_observable_value")
            continue
        if not math.isfinite(numeric):
            reason_codes.append("invalid_reduced_observable_value")
            continue
        observables[name] = numeric

    if not observables:
        reason_codes.append("missing_reduced_observables")
    return observables, tuple(sorted(set(reason_codes)))


def _extract_experiment(payload: Mapping[str, Any]) -> str:
    for key in ("experiment",):
        if key in payload and str(payload[key]).strip():
            return str(payload[key]).strip()
    normalized_payload = payload.get("normalized_payload")
    if isinstance(normalized_payload, Mapping):
        experiment = str(normalized_payload.get("experiment", "")).strip()
        if experiment:
            return experiment
    for key in ("gv_launch", "emb_launch"):
        sub_payload = payload.get(key)
        if isinstance(sub_payload, Mapping):
            experiment = str(sub_payload.get("experiment", "")).strip()
            if experiment:
                return experiment
    return "unknown"


def _matching_candidate_lineage(payload: object, *, candidate_id: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        lineage = _coerce_mapping(payload, field_name="candidate_lineage")
        lineage_candidate_id = str(lineage.get("candidate_id", "")).strip()
        if lineage_candidate_id and lineage_candidate_id != candidate_id:
            raise ValueError(
                f"candidate_lineage candidate_id {lineage_candidate_id!r} does not match candidate {candidate_id!r}."
            )
        return lineage
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        matches: list[dict[str, Any]] = []
        anonymous: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            lineage = _coerce_mapping(item, field_name=f"candidate_lineage[{index}]")
            lineage_candidate_id = str(lineage.get("candidate_id", "")).strip()
            if lineage_candidate_id == candidate_id:
                matches.append(lineage)
            elif not lineage_candidate_id:
                anonymous.append(lineage)
        if len(matches) > 1:
            raise ValueError(f"candidate_lineage contains duplicate entries for candidate {candidate_id!r}.")
        if matches:
            return matches[0]
        if len(payload) == 1 and anonymous:
            return anonymous[0]
        if payload:
            raise ValueError(f"candidate_lineage does not contain candidate {candidate_id!r}.")
        return {}
    raise ValueError("candidate_lineage must be a mapping or sequence of mappings.")


def _extract_active_learning_metadata(payload: Mapping[str, Any], *, candidate_id: str) -> dict[str, Any]:
    metadata = payload.get("active_learning_metadata")
    if metadata is None:
        candidate_lineage = _matching_candidate_lineage(payload.get("candidate_lineage"), candidate_id=candidate_id)
        metadata = candidate_lineage.get("active_learning_metadata")
    if metadata is None and isinstance(payload.get("metadata"), Mapping):
        legacy_metadata = _coerce_mapping(payload.get("metadata"), field_name="metadata")
        metadata = legacy_metadata.get(
            "active_learning_metadata",
            {key: value for key, value in legacy_metadata.items() if key != "lineage"},
        )
    normalized = _coerce_optional_mapping(metadata, field_name="active_learning_metadata")
    return _normalize_json(normalized, "active_learning_metadata")


def _extract_candidate_lineage(
    payload: Mapping[str, Any],
    *,
    candidate_id: str,
    active_learning_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _matching_candidate_lineage(payload.get("candidate_lineage"), candidate_id=candidate_id)
    if not normalized:
        legacy_lineage = payload.get("lineage")
        if legacy_lineage is None and isinstance(payload.get("metadata"), Mapping):
            legacy_lineage = payload.get("metadata", {}).get("lineage")
        normalized = _coerce_optional_mapping(legacy_lineage, field_name="lineage")

    for key in ("run_id", "iteration", "batch_id", "iteration_id", "parent_iteration_id"):
        if key in payload and key not in normalized:
            normalized[key] = payload[key]
    normalized.setdefault("candidate_id", candidate_id)
    lineage_metadata = _coerce_optional_mapping(
        normalized.get("active_learning_metadata"),
        field_name="candidate_lineage.active_learning_metadata",
    )
    if lineage_metadata and active_learning_metadata and dict(lineage_metadata) != dict(active_learning_metadata):
        raise ValueError(f"candidate {candidate_id!r} active_learning_metadata conflicts with candidate_lineage.")
    if lineage_metadata and not active_learning_metadata:
        active_learning_metadata = lineage_metadata
    normalized["active_learning_metadata"] = dict(active_learning_metadata)
    return _normalize_json(normalized, "candidate_lineage")


def _extract_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata_payload = payload.get("metadata")
    if metadata_payload is None:
        return {}
    return _normalize_json(_coerce_mapping(metadata_payload, field_name="metadata"), "metadata")


def _resolve_status(payload: Mapping[str, Any], *, has_valid_refs: bool, has_reduced: bool) -> tuple[str, tuple[str, ...]]:
    raw_status = str(payload.get("status", "")).strip().lower()
    reason_codes: list[str] = []
    if raw_status in _SUPPORTED_STATUSES:
        status = raw_status
    elif raw_status in _FAILED_STATUS_TOKENS:
        status = "failed"
        reason_codes.append("status_mapped_to_failed")
    elif raw_status:
        status = "partial"
        reason_codes.append("status_unrecognized")
    else:
        if has_valid_refs and has_reduced:
            status = "completed"
        elif not has_valid_refs and not has_reduced:
            status = "missing"
        else:
            status = "partial"
            reason_codes.append("status_derived_partial")

    if status == "completed" and (not has_valid_refs or not has_reduced):
        status = "partial"
        reason_codes.append("completed_but_incomplete_payload")
    if status == "failed":
        reason_codes.append("upstream_failed")
    if status == "missing":
        reason_codes.append("upstream_missing")
    if status == "partial":
        reason_codes.append("partial_completion")
    return status, tuple(sorted(set(reason_codes)))


def _missing_ref_reasons(reason_codes: Sequence[str]) -> tuple[str, ...]:
    missing_prefixes = (
        "missing_expected_hdf5_refs",
        "missing_hdf5_path",
        "missing_manifest_path",
        "missing_ref_dataset_id",
        "invalid_hdf5_path_suffix",
        "invalid_manifest_path_suffix",
        "mismatched_ref_family",
        "invalid_ref_family",
    )
    return tuple(sorted({code for code in reason_codes if code.startswith(missing_prefixes)}))


def _has_valid_raw_curve_refs(raw_curve_refs: Sequence[Mapping[str, str]], reason_codes: Sequence[str]) -> bool:
    return bool(raw_curve_refs) and not _missing_ref_reasons(reason_codes)


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


@dataclass(frozen=True)
class ResultIngestionRecord:
    candidate_id: str
    family: str
    experiment: str
    status: str
    reason_codes: tuple[str, ...]
    raw_curve_refs: tuple[dict[str, str], ...]
    reduced_observables: dict[str, float]
    active_learning_metadata: dict[str, Any] = field(default_factory=dict)
    candidate_lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_RESULT_INGESTION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _coerce_nonempty_text(self.candidate_id, field_name="candidate_id"))
        object.__setattr__(self, "family", _coerce_family(self.family, candidate_id=self.candidate_id))
        object.__setattr__(self, "experiment", _coerce_nonempty_text(self.experiment, field_name="experiment"))
        normalized_status = _coerce_nonempty_text(self.status, field_name="status").lower()
        if normalized_status not in _SUPPORTED_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(_SUPPORTED_STATUSES))}.")
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(self, "reason_codes", tuple(sorted(str(item).strip() for item in self.reason_codes if str(item).strip())))
        object.__setattr__(self, "raw_curve_refs", tuple(dict(item) for item in self.raw_curve_refs))
        object.__setattr__(self, "reduced_observables", dict(self.reduced_observables))
        active_learning_metadata = _normalize_json(
            _coerce_optional_mapping(self.active_learning_metadata, field_name="active_learning_metadata"),
            "active_learning_metadata",
        )
        candidate_lineage = _normalize_json(
            _coerce_optional_mapping(self.candidate_lineage, field_name="candidate_lineage"),
            "candidate_lineage",
        )
        candidate_lineage.setdefault("candidate_id", self.candidate_id)
        if candidate_lineage["candidate_id"] != self.candidate_id:
            raise ValueError(
                f"candidate_lineage candidate_id {candidate_lineage['candidate_id']!r} "
                f"does not match candidate {self.candidate_id!r}."
            )
        lineage_metadata = _coerce_optional_mapping(
            candidate_lineage.get("active_learning_metadata"),
            field_name="candidate_lineage.active_learning_metadata",
        )
        if lineage_metadata and active_learning_metadata and dict(lineage_metadata) != dict(active_learning_metadata):
            raise ValueError(f"candidate {self.candidate_id!r} active_learning_metadata conflicts with candidate_lineage.")
        if lineage_metadata and not active_learning_metadata:
            active_learning_metadata = lineage_metadata
        candidate_lineage["active_learning_metadata"] = dict(active_learning_metadata)
        object.__setattr__(self, "active_learning_metadata", active_learning_metadata)
        object.__setattr__(self, "candidate_lineage", candidate_lineage)
        object.__setattr__(self, "metadata", _normalize_json(dict(self.metadata), "metadata"))

    @property
    def lineage(self) -> dict[str, Any]:
        return dict(self.candidate_lineage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "family": self.family,
            "experiment": self.experiment,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "raw_curve_refs": [dict(item) for item in self.raw_curve_refs],
            "reduced_observables": dict(self.reduced_observables),
            "active_learning_metadata": dict(self.active_learning_metadata),
            "candidate_lineage": dict(self.candidate_lineage),
            "metadata": dict(self.metadata),
        }

    def stable_hash(self, *, length: int = 16) -> str:
        return _sha256_short(self.as_dict(), length=length)


@dataclass(frozen=True)
class ResultIngestionBatch:
    run_id: str
    iteration: str
    records: tuple[ResultIngestionRecord, ...]
    schema_version: str = ACTIVE_LEARNING_RESULT_INGESTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _coerce_nonempty_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "iteration", _coerce_nonempty_text(self.iteration, field_name="iteration"))
        sorted_records = tuple(
            sorted(
                self.records,
                key=lambda item: (item.candidate_id, item.family, item.experiment),
            )
        )
        object.__setattr__(self, "records", sorted_records)

    @property
    def ingestion_hash(self) -> str:
        return _sha256_short(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "iteration": self.iteration,
                "records": [record.as_dict() for record in self.records],
            }
        )

    @property
    def status_counts(self) -> dict[str, int]:
        counter = Counter(record.status for record in self.records)
        return {status: counter.get(status, 0) for status in sorted(_SUPPORTED_STATUSES)}

    @property
    def family_counts(self) -> dict[str, int]:
        counter = Counter(record.family for record in self.records)
        return {family: counter.get(family, 0) for family in sorted(_SUPPORTED_FAMILIES)}

    @property
    def reason_code_counts(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for record in self.records:
            counter.update(record.reason_codes)
        return dict(sorted(counter.items()))

    @property
    def missing_refs(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for record in self.records:
            reasons = _missing_ref_reasons(record.reason_codes)
            if not reasons:
                continue
            entries.append(
                {
                    "candidate_id": record.candidate_id,
                    "family": record.family,
                    "status": record.status,
                    "reason_codes": list(reasons),
                    "raw_curve_refs": [dict(item) for item in record.raw_curve_refs],
                }
            )
        return entries

    @property
    def candidate_lineage(self) -> list[dict[str, Any]]:
        return [dict(record.candidate_lineage) for record in self.records]

    @property
    def reduced_observable_distributions(self) -> dict[str, dict[str, float | int]]:
        values: dict[str, list[float]] = defaultdict(list)
        for record in self.records:
            for observable, value in record.reduced_observables.items():
                values[observable].append(float(value))
        return {name: _distribution(sorted(series)) for name, series in sorted(values.items())}

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "ingestion_hash": self.ingestion_hash,
            "record_count": len(self.records),
            "status_counts": self.status_counts,
            "family_counts": self.family_counts,
            "candidate_lineage": self.candidate_lineage,
            "records": [record.as_dict() for record in self.records],
        }

    def as_report(self) -> dict[str, Any]:
        return {
            "schema_version": ACTIVE_LEARNING_RESULT_INGESTION_REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "ingestion_hash": self.ingestion_hash,
            "record_count": len(self.records),
            "status_counts": self.status_counts,
            "family_counts": self.family_counts,
            "reason_code_counts": self.reason_code_counts,
            "missing_refs": self.missing_refs,
            "candidate_lineage": self.candidate_lineage,
            "reduced_observable_distributions": self.reduced_observable_distributions,
        }


@dataclass(frozen=True)
class ResultIngestionArtifacts:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]


def _manifest_to_record(payload: Mapping[str, Any]) -> ResultIngestionRecord:
    candidate_id = _coerce_nonempty_text(payload.get("candidate_id", ""), field_name="candidate_id")
    family = _coerce_family(payload.get("family", ""), candidate_id=candidate_id)
    experiment = _extract_experiment(payload)
    raw_curve_refs, ref_reasons = _normalize_raw_curve_refs(payload, family=family, candidate_id=candidate_id)
    reduced_observables, reduced_reasons = _normalize_reduced_observables(payload)
    status, status_reasons = _resolve_status(
        payload,
        has_valid_refs=_has_valid_raw_curve_refs(raw_curve_refs, ref_reasons),
        has_reduced=bool(reduced_observables),
    )
    active_learning_metadata = _extract_active_learning_metadata(payload, candidate_id=candidate_id)
    candidate_lineage = _extract_candidate_lineage(
        payload,
        candidate_id=candidate_id,
        active_learning_metadata=active_learning_metadata,
    )

    reason_codes = tuple(sorted(set(ref_reasons + reduced_reasons + status_reasons)))
    return ResultIngestionRecord(
        candidate_id=candidate_id,
        family=family,
        experiment=experiment,
        status=status,
        reason_codes=reason_codes,
        raw_curve_refs=raw_curve_refs,
        reduced_observables=reduced_observables,
        active_learning_metadata=active_learning_metadata,
        candidate_lineage=candidate_lineage,
        metadata=_extract_metadata(payload),
    )


def ingest_active_learning_results(
    candidate_manifests: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    iteration: int | str,
) -> ResultIngestionBatch:
    if not isinstance(candidate_manifests, Sequence):
        raise ValueError("candidate_manifests must be a sequence.")
    if not candidate_manifests:
        raise ValueError("candidate_manifests must contain at least one item.")

    records = tuple(_manifest_to_record(_coerce_mapping(item, field_name="candidate_manifest")) for item in candidate_manifests)
    return ResultIngestionBatch(
        run_id=_coerce_nonempty_text(run_id, field_name="run_id"),
        iteration=_coerce_iteration_token(iteration),
        records=records,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _artifact_dir(output_root: str | Path, run_id: str, iteration: str) -> Path:
    return Path(output_root) / run_id / "iterations" / iteration


def _write_validation_plot(path: Path, report: Mapping[str, Any], *, include_plot: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return
    try:
        import matplotlib.pyplot as pyplot  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    status_counts = _coerce_optional_mapping(report.get("status_counts"), field_name="status_counts")
    family_counts = _coerce_optional_mapping(report.get("family_counts"), field_name="family_counts")
    distributions = _coerce_optional_mapping(
        report.get("reduced_observable_distributions"),
        field_name="reduced_observable_distributions",
    )
    observable_names = sorted(distributions.keys())
    observable_means: list[float] = []
    for name in observable_names:
        payload = distributions[name]
        if isinstance(payload, Mapping):
            try:
                observable_means.append(float(payload.get("mean", 0.0)))
            except (TypeError, ValueError):
                observable_means.append(0.0)
        else:
            observable_means.append(0.0)

    fig, axes = pyplot.subplots(2, 2, figsize=(12, 8))
    try:
        axes[0, 0].bar(list(status_counts.keys()), [int(value) for value in status_counts.values()], color="#4c72b0")
        axes[0, 0].set_title("Completion Status")
        axes[0, 0].set_xlabel("Status")
        axes[0, 0].set_ylabel("Count")

        axes[0, 1].bar(list(family_counts.keys()), [int(value) for value in family_counts.values()], color="#c44e52")
        axes[0, 1].set_title("Family Counts")
        axes[0, 1].set_xlabel("Family")
        axes[0, 1].set_ylabel("Count")

        axes[1, 0].bar(observable_names, observable_means, color="#8172b2")
        axes[1, 0].set_title("Reduced Observable Means")
        axes[1, 0].tick_params(axis="x", rotation=35)

        axes[1, 1].axis("off")
        lines = [
            f"ingestion_hash: {report.get('ingestion_hash', '')}",
            f"record_count: {report.get('record_count', 0)}",
            f"missing_refs: {len(report.get('missing_refs', []))}",
        ]
        y = 0.95
        for line in lines:
            axes[1, 1].text(0.03, y, line, transform=axes[1, 1].transAxes, ha="left", va="top")
            y -= 0.15

        fig.suptitle("AL Result Ingestion Validation")
        fig.tight_layout()
        fig.savefig(path, dpi=ACTIVE_LEARNING_RESULT_INGESTION_PLOT_DPI)
    finally:
        pyplot.close(fig)


def write_result_ingestion_artifacts(
    *,
    output_root: str | Path,
    ingestion: ResultIngestionBatch,
    include_plot: bool = True,
) -> ResultIngestionArtifacts:
    iteration = ingestion.iteration
    artifact_dir = _artifact_dir(output_root, ingestion.run_id, iteration)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = ingestion.as_manifest()
    report = ingestion.as_report()

    manifest_path = artifact_dir / ACTIVE_LEARNING_RESULT_INGESTION_MANIFEST_FILENAME
    report_path = artifact_dir / ACTIVE_LEARNING_RESULT_INGESTION_REPORT_FILENAME
    plot_path = artifact_dir / ACTIVE_LEARNING_RESULT_INGESTION_PLOT_FILENAME
    plot_sidecar_path = artifact_dir / ACTIVE_LEARNING_RESULT_INGESTION_PLOT_SIDECAR_FILENAME

    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    _write_validation_plot(plot_path, report, include_plot=include_plot)
    _write_json(plot_sidecar_path, report)

    return ResultIngestionArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        manifest=manifest,
        report=report,
    )


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + type_code + data + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)


def _build_fallback_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw_pixel = b"\x00\x00\x00\x00\x00"
    idat_data = zlib.compress(raw_pixel)
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", idat_data) + _png_chunk(b"IEND", b"")


_FALLBACK_PNG = _build_fallback_png()


__all__ = [
    "ACTIVE_LEARNING_RESULT_INGESTION_MANIFEST_FILENAME",
    "ACTIVE_LEARNING_RESULT_INGESTION_PLOT_FILENAME",
    "ACTIVE_LEARNING_RESULT_INGESTION_PLOT_SIDECAR_FILENAME",
    "ACTIVE_LEARNING_RESULT_INGESTION_RECORD_SCHEMA_VERSION",
    "ACTIVE_LEARNING_RESULT_INGESTION_REPORT_FILENAME",
    "ACTIVE_LEARNING_RESULT_INGESTION_REPORT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_RESULT_INGESTION_SCHEMA_VERSION",
    "ResultIngestionArtifacts",
    "ResultIngestionBatch",
    "ResultIngestionRecord",
    "ingest_active_learning_results",
    "write_result_ingestion_artifacts",
]
