from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
    EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
    EMB_34UM_DNN_CAUSAL_FORBIDDEN_FINAL_LHS_SOURCES,
    validate_dnn_causal_ensemble_size,
    validate_dnn_causal_force_grid,
)


EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_ingestion.v1"
)


def _load_payload(value: object) -> object:
    if isinstance(value, (str, Path)):
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload
    return value


def _coerce_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _coerce_sequence(value: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(value)


def _normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def _record_text(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return _normalize_text(value)
    return ""


def _has_training_marker(record: Mapping[str, Any]) -> bool:
    for key in ("used_for_training", "training", "is_training", "train", "included_in_training"):
        value = record.get(key)
        if value not in (None, ""):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)
    return False


def _record_category(record: Mapping[str, Any], *, section: str | None = None) -> str:
    category = _normalize_text(record.get("category"))
    if category in {"shared_unseen", "final_lhs", "selection_training"}:
        return category
    if section:
        normalized = _normalize_text(section)
        if normalized in {"shared_unseen", "unseen_test", "shared_unseen_test"}:
            return "shared_unseen"
        if normalized in {"final_lhs", "lhs", "lhs_final", "final_lhs_records"}:
            return "final_lhs"
        if normalized in {"selection_training", "training", "selection", "selection_records", "training_records"}:
            return "selection_training"

    kind = _record_text(record, "kind", "record_kind", "type", "stage", "method")
    if "unseen" in kind or "shared" in kind and "test" in kind:
        return "shared_unseen"
    if "lhs" in kind:
        return "final_lhs"
    if "training" in kind or "selection" in kind:
        return "selection_training"

    source = _record_text(record, "source", "selection_source", "provenance", "origin", "dataset")
    if "fresh_dpd_shared_unseen" in source or "fresh_test" in source or "unseen" in source:
        return "shared_unseen"
    if any(token in source for token in ("samples_all", "replay", "archive", "historical")):
        return "final_lhs"
    if "lhs" in source:
        return "final_lhs"
    if "ensemble_size" in record:
        return "selection_training"

    raise ValueError("record could not be classified as shared_unseen, final_lhs, or selection_training.")


def _iter_records(payload: object, *, section: str | None = None) -> list[dict[str, Any]]:
    payload = _load_payload(payload)
    records: list[dict[str, Any]] = []

    if isinstance(payload, Mapping):
        if "records" in payload and isinstance(payload["records"], Sequence):
            for item in _coerce_sequence(payload["records"], label="records"):
                records.extend(_iter_records(item, section=section))
            return records
        if "candidate_records" in payload and isinstance(payload["candidate_records"], Sequence):
            for item in _coerce_sequence(payload["candidate_records"], label="candidate_records"):
                records.extend(_iter_records(item, section=section))
            return records

        section_keys = (
            ("shared_unseen", "shared_unseen"),
            ("unseen_test", "shared_unseen"),
            ("final_lhs", "final_lhs"),
            ("lhs_records", "final_lhs"),
            ("selection_training", "selection_training"),
            ("training_records", "selection_training"),
        )
        if any(key in payload for key, _ in section_keys):
            for key, category in section_keys:
                section_payload = payload.get(key)
                if section_payload in (None, ""):
                    continue
                if isinstance(section_payload, Mapping) and "records" in section_payload:
                    section_payload = section_payload["records"]
                if isinstance(section_payload, Mapping) and "candidate_records" in section_payload:
                    section_payload = section_payload["candidate_records"]
                for item in _coerce_sequence(section_payload, label=key):
                    records.extend(_iter_records(item, section=category))
            return records

        rec = dict(payload)
        rec.setdefault("category", _record_category(rec, section=section))
        records.append(rec)
        return records

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            records.extend(_iter_records(item, section=section))
        return records

    raise ValueError("records payload must be a mapping, sequence, or JSON path.")


def normalize_emb_34um_dnn_causal_validation_records(source: object) -> list[dict[str, Any]]:
    return _iter_records(source)


def build_emb_34um_dnn_causal_validation_ingestion_report(source: object) -> dict[str, Any]:
    records = normalize_emb_34um_dnn_causal_validation_records(source)
    shared_unseen = [record for record in records if _record_category(record) == "shared_unseen"]
    final_lhs = [record for record in records if _record_category(record) == "final_lhs"]
    selection_training = [record for record in records if _record_category(record) == "selection_training"]

    blockers: list[str] = []

    if not shared_unseen:
        blockers.append("missing shared unseen test set.")
    elif len(shared_unseen) != 100:
        blockers.append(f"shared unseen test set must contain exactly 100 records, got {len(shared_unseen)}.")
    else:
        grids: list[tuple[float, ...]] = []
        for record in shared_unseen:
            grid = record.get("force_grid", record.get("force_grid_points"))
            if grid is None:
                blockers.append("shared unseen test set is missing force_grid.")
                break
            try:
                normalized_grid = validate_dnn_causal_force_grid(grid)
            except ValueError as exc:
                blockers.append(f"shared unseen force grid is invalid: {exc}")
                break
            grids.append(normalized_grid)
            source_text = _record_text(record, "source", "selection_source", "provenance", "origin", "dataset")
            if "fresh_dpd" not in source_text and EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE not in source_text:
                blockers.append("shared unseen test set must come from fresh DPD source.")
                break
            if any(token in source_text for token in ("samples_all", "replay", "archive", "historical")):
                blockers.append("shared unseen test set must not reuse samples_all.dat, replay, archive, or historical data.")
                break
            if _has_training_marker(record):
                blockers.append("shared unseen test set must not be used for training.")
                break
        if not blockers and len({grid for grid in grids}) != 1:
            blockers.append("shared unseen test set must use one exact 8-point force grid.")

    if not final_lhs:
        blockers.append("missing final LHS records.")
    else:
        for record in final_lhs:
            source_text = _record_text(record, "source", "selection_source", "provenance", "origin", "dataset")
            if not source_text:
                blockers.append("final LHS record is missing a fresh DPD source.")
                break
            if any(token in source_text for token in EMB_34UM_DNN_CAUSAL_FORBIDDEN_FINAL_LHS_SOURCES):
                blockers.append(
                    "Final DNN causal LHS must use fresh DPD curves, not samples_all.dat, replay, archive, or historical data."
                )
                break
            if "fresh_dpd" not in source_text and EMB_34UM_DNN_CAUSAL_LHS_SOURCE not in source_text:
                blockers.append("final LHS record must come from fresh DPD source.")
                break

    if not selection_training:
        blockers.append("missing DNN selection/training records.")
    else:
        for record in selection_training:
            if "ensemble_size" not in record:
                blockers.append("DNN selection/training record must include ensemble_size=10.")
                break
            try:
                value = validate_dnn_causal_ensemble_size(record["ensemble_size"])
            except ValueError as exc:
                blockers.append(str(exc))
                break
            if value != EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE:
                blockers.append(f"ensemble_size must be {EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE}.")
                break

    report = {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
        "counts": {
            "shared_unseen": len(shared_unseen),
            "final_lhs": len(final_lhs),
            "selection_training": len(selection_training),
            "total": len(records),
        },
        "blockers": blockers,
        "status": "passed" if not blockers else "blocked",
        "passed": not blockers,
    }
    return report


__all__ = [
    "EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION",
    "build_emb_34um_dnn_causal_validation_ingestion_report",
    "normalize_emb_34um_dnn_causal_validation_records",
]
