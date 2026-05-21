from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_causal_validation.v1"
EMB_34UM_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME = "emb_34um_causal_validation_ingest_manifest.json"
EMB_34UM_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME = "emb_34um_causal_validation_ingest_report.json"
EMB_34UM_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME = "emb_34um_causal_validation_ingestion_summary.csv"
EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_MANIFEST_FILENAME = "emb_34um_causal_validation_replacement_manifest.json"
EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME = "emb_34um_causal_validation_replacement_batch_summary.json"
EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_MANIFEST_FILENAME = "emb_34um_causal_validation_lhs_replacement_manifest.json"
EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_BATCH_SUMMARY_FILENAME = "emb_34um_causal_validation_lhs_replacement_batch_summary.json"
EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_PLAN_FILENAME = "emb_34um_causal_validation_replacement_plan.json"
DEFAULT_SUCCESS_FILENAMES = ("F_Delta.dat",)
DEFAULT_STATUS_FILENAMES = ("runtime_status.json", "result_status.json", "emb_34um_runtime_status.json")


@dataclass(frozen=True)
class Emb34umCausalValidationIngestionArtifacts:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    summary_csv_path: Path
    replacement_plan_path: Path | None
    manifest: dict[str, Any]
    report: dict[str, Any]
    replacement_plan: dict[str, Any] | None


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path!s}.")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _coerce_sequence(value: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(value)


def _as_int(value: object, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    return parsed


def _as_float(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    return parsed


def _candidate_id(record: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "id", "curve_id", "sample_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    raise ValueError("candidate record missing candidate_id.")


def _normalize_method(method: str) -> str:
    text = str(method).strip().lower().replace("_", "-")
    if "-step-" in text:
        base, step_text = text.rsplit("-step-", 1)
        if base and step_text.isdigit():
            text = base
    if text in {"shared", "shared-initial"}:
        return "shared_initial"
    return text


def _stage_from_method(method: str) -> str | None:
    text = str(method).strip().lower().replace("_", "-")
    if "-step-" not in text:
        return None
    base, step_text = text.rsplit("-step-", 1)
    if base in {"al", "lhs"} and step_text.isdigit():
        return f"{base}-step-{int(step_text):02d}"
    return None


def _get_alias(record: Mapping[str, Any], *aliases: str, required: bool = False) -> str:
    for alias in aliases:
        value = record.get(alias)
        if value not in (None, ""):
            return str(value).strip()
    if required:
        raise ValueError(f"Missing required field aliases: {aliases!r}")
    return ""


def _normalized_stage(method: str, step: int | None) -> str:
    original_method = method
    method = _normalize_method(method)
    if method == "validation":
        return "validation"
    if method in {"al", "lhs"}:
        if step is not None:
            return f"{method}-step-{step:02d}"
        stage = _stage_from_method(original_method)
        if stage is not None:
            return stage
        return method
    return method


def _status_from_json(path: Path) -> str | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    status = str(payload.get("status", "")).strip().lower()
    if status in {"completed", "complete", "success", "succeeded"}:
        return "completed"
    if status in {"failed", "failure", "error", "cancelled", "timeout", "quarantine", "quarantined"}:
        return "failed"
    return None


def _mapping_or_empty(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _first_nonempty(*values: object) -> object | None:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _coerce_dynamic_al_record(
    *,
    candidate_manifest: Mapping[str, Any],
    manifest_path: Path,
    seed: int,
    step: int,
    order: int,
) -> dict[str, Any]:
    rendered_payload = _mapping_or_empty(candidate_manifest.get("rendered_payload"))
    request_payload = _mapping_or_empty(rendered_payload.get("request_payload"))
    normalized_payload = _mapping_or_empty(candidate_manifest.get("normalized_payload"))
    request_parameters = _mapping_or_empty(request_payload.get("parameters"))
    normalized_parameters = _mapping_or_empty(normalized_payload.get("parameters"))
    metadata = _mapping_or_empty(candidate_manifest.get("active_learning_metadata"))

    candidate_id = str(
        _first_nonempty(
            request_payload.get("candidate_id"),
            candidate_manifest.get("candidate_id"),
            normalized_payload.get("candidate_id"),
        )
        or ""
    ).strip()
    if not candidate_id:
        raise ValueError(f"{manifest_path} is missing a candidate id.")

    output_root = str(
        _first_nonempty(
            request_payload.get("output_root"),
            normalized_payload.get("output_root"),
            candidate_manifest.get("output_root"),
            metadata.get("output_root"),
        )
        or ""
    ).strip()
    if not output_root:
        raise ValueError(f"{manifest_path} is missing output_root in request/normalized payload.")

    vault_output_root = str(
        _first_nonempty(
            request_payload.get("vault_output_root"),
            normalized_payload.get("vault_output_root"),
            candidate_manifest.get("vault_output_root"),
            metadata.get("vault_output_root"),
        )
        or ""
    ).strip()

    ka_value = _first_nonempty(
        request_parameters.get("ka"),
        normalized_parameters.get("ka"),
        candidate_manifest.get("ka"),
        metadata.get("ka"),
    )
    kb_value = _first_nonempty(
        request_parameters.get("kb"),
        normalized_parameters.get("kb"),
        candidate_manifest.get("kb"),
        metadata.get("kb"),
    )
    if ka_value is None or kb_value is None:
        raise ValueError(f"{manifest_path} is missing ka/kb in request payload, normalized payload, or metadata.")

    resolved_seed = _as_int(_first_nonempty(request_payload.get("seed"), metadata.get("seed"), seed), label="dynamic.seed")
    resolved_step = _as_int(_first_nonempty(request_payload.get("step"), metadata.get("step"), step), label="dynamic.step")

    return {
        "candidate_id": candidate_id,
        "output_root": output_root,
        "vault_output_root": vault_output_root,
        "method": "al",
        "stage": f"al-step-{resolved_step:02d}",
        "seed": resolved_seed,
        "step": resolved_step,
        "order": order,
        "ka": _as_float(ka_value, label="dynamic.ka"),
        "kb": _as_float(kb_value, label="dynamic.kb"),
        "selection_mode": str(
            _first_nonempty(
                request_payload.get("selection_mode"),
                metadata.get("selection_mode"),
                "causal_fresh_only",
            )
        ),
        "fresh_only": _bool_like(_first_nonempty(request_payload.get("fresh_only"), metadata.get("fresh_only"), True)),
        "metadata": metadata,
    }


def _classify_record(
    record: Mapping[str, Any],
    *,
    success_filenames: Sequence[str],
    status_filenames: Sequence[str],
) -> tuple[str, list[str]]:
    output_root = Path(_get_alias(record, "output_root", "output_dir", required=True))
    reasons: list[str] = []
    if any((output_root / name).is_file() for name in success_filenames):
        return "completed", reasons
    for name in status_filenames:
        status = _status_from_json(output_root / name)
        if status == "failed":
            reasons.append(f"status:{name}:failed")
            return "failed", reasons
        if status == "completed":
            reasons.append(f"status:{name}:completed_without_success_file")
    if output_root.exists():
        reasons.append("missing_success_file")
    else:
        reasons.append("missing_output_root")
    return "missing", reasons


def _provenance_tokens(record: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("provenance", "metadata", "source", "source_dataset", "dataset", "lineage", "origin"):
        value = record.get(key)
        if isinstance(value, Mapping):
            parts.append(json.dumps(value, sort_keys=True))
        elif value not in (None, ""):
            parts.append(str(value))
    return " ".join(parts).lower()


def _bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _string_token(value: object) -> str:
    return str(value).strip().lower()


def _manifest_requires_causal_fresh_only(manifest: Mapping[str, Any]) -> bool:
    policy = manifest.get("policy")
    provenance = manifest.get("provenance")
    if not isinstance(policy, Mapping) or not isinstance(provenance, Mapping):
        return False
    return _bool_like(policy.get("fresh_only")) and _string_token(provenance.get("selection_mode")) == "causal_fresh_only"


def _record_allows_fresh_only(record: Mapping[str, Any]) -> bool:
    if _string_token(record.get("selection_mode")) in {"fresh_only", "causal_fresh_only"}:
        return True
    if _bool_like(record.get("fresh_only")):
        return True
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        if _string_token(metadata.get("selection_mode")) in {"fresh_only", "causal_fresh_only"}:
            return True
        if _bool_like(metadata.get("fresh_only")):
            return True
    return False


def _assert_fresh_only(record: Mapping[str, Any], *, allow_legacy_fallback: bool) -> None:
    tokens = _provenance_tokens(record)
    if any(bad in tokens for bad in ("legacy", "final-gate", "final_gate", "reused", "reuse")):
        raise ValueError(
            f"{_candidate_id(record)} rejected for fresh-only: legacy/final-gate reuse is not allowed."
        )
    if _record_allows_fresh_only(record):
        return
    if allow_legacy_fallback and "dpd" in tokens:
        return
    raise ValueError(f"{_candidate_id(record)} rejected: fresh-only requires causal_fresh_only provenance.")


def assert_same_validation_set(records: Sequence[Mapping[str, Any]]) -> None:
    ids = {str(item.get("validation_set_id", "")).strip() for item in records if item.get("validation_set_id") not in (None, "")}
    if len(ids) > 1:
        raise ValueError(f"Validation set mismatch: expected one validation_set_id, got {sorted(ids)!r}.")


def _expected_policy_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    policy = _coerce_mapping(manifest.get("policy", {}), label="policy")
    return {
        "shared_initial": _as_int(policy.get("shared_size", 100), label="policy.shared_size"),
        "validation": _as_int(policy.get("validation_size", 100), label="policy.validation_size"),
        "al": _as_int(policy.get("step_count", 5), label="policy.step_count")
        * _as_int(policy.get("step_size", 100), label="policy.step_size"),
        "lhs": _as_int(policy.get("step_count", 5), label="policy.step_count")
        * _as_int(policy.get("step_size", 100), label="policy.step_size"),
    }


def _expected_stage_policy_counts(manifest: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    policy = _coerce_mapping(manifest.get("policy", {}), label="policy")
    step_count = _as_int(policy.get("step_count", 5), label="policy.step_count")
    step_size = _as_int(policy.get("step_size", 100), label="policy.step_size")
    return {
        ("shared_initial", "shared_initial"): _as_int(policy.get("shared_size", 100), label="policy.shared_size"),
        ("validation", "validation"): _as_int(policy.get("validation_size", 100), label="policy.validation_size"),
        **{
            ("al", f"al-step-{step:02d}"): step_size
            for step in range(1, step_count + 1)
        },
        **{
            ("lhs", f"lhs-step-{step:02d}"): step_size
            for step in range(1, step_count + 1)
        },
    }


def _build_expected_stage_policy_rows(
    manifest: Mapping[str, Any],
    *,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in sorted(seeds):
        for (policy_group, stage), expected in sorted(_expected_stage_policy_counts(manifest).items()):
            rows.append(
                {
                    "seed": seed,
                    "policy_group": policy_group,
                    "stage": stage,
                    "expected": expected,
                }
            )
    return rows


def _replacement_targets_by_completed_replacement(rows: Sequence[Mapping[str, Any]]) -> set[tuple[int, str, str, str]]:
    targets: set[tuple[int, str, str, str]] = set()
    for row in rows:
        if not bool(row.get("replacement")):
            continue
        if str(row.get("status")) != "completed":
            continue
        candidate_for = str(row.get("replacement_for", "")).strip()
        if not candidate_for:
            continue
        targets.add((int(row["seed"]), str(row["policy_group"]), str(row["stage"]), candidate_for))
    return targets


def _coerce_batch_records(batch: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any], ...]:
    raw_records = batch.get("candidate_records", batch.get("records", ()))
    return _coerce_sequence(raw_records, label=f"{label} batch records")


def _records_from_rendered_al_batch(
    batch: Mapping[str, Any],
    *,
    seed: int,
    step: int,
    fresh_only: bool,
    allow_legacy_fallback: bool,
) -> tuple[dict[str, Any], ...]:
    summary_path_text = str(batch.get("expected_batch_summary_path", "")).strip()
    if not summary_path_text:
        return ()
    summary_path = Path(summary_path_text)
    if not summary_path.is_file():
        return ()
    summary = _read_json(summary_path)
    manifest_paths = summary.get("rendered_candidate_manifests", ())
    if not isinstance(manifest_paths, Sequence) or isinstance(manifest_paths, (str, bytes, bytearray)):
        raise ValueError(f"{summary_path} rendered_candidate_manifests must be a sequence.")

    records: list[dict[str, Any]] = []
    for order, manifest_path_text in enumerate(manifest_paths, start=1):
        manifest_path = Path(str(manifest_path_text))
        candidate_manifest = _read_json(manifest_path)
        record = _coerce_dynamic_al_record(
            candidate_manifest=candidate_manifest,
            manifest_path=manifest_path,
            seed=seed,
            step=step,
            order=order,
        )
        if fresh_only:
            _assert_fresh_only(record, allow_legacy_fallback=allow_legacy_fallback)
        records.append(record)
    return tuple(records)


def _coerce_replacement_record(
    record: Mapping[str, Any],
    *,
    step: int,
    seed: int,
    method: str = "al",
) -> dict[str, Any]:
    method = _normalize_method(method)
    if method not in {"al", "lhs"}:
        raise ValueError(f"replacement method must be 'al' or 'lhs', got {method!r}.")
    replacement_candidate_id = _first_nonempty(
        record.get("replacement_candidate_id"),
        record.get("candidate_id"),
    )
    if replacement_candidate_id is None or str(replacement_candidate_id).strip() == "":
        raise ValueError("replacement manifest record is missing replacement_candidate_id.")
    replacement_candidate_id = str(replacement_candidate_id).strip()
    replaced_for = _first_nonempty(
        record.get("replacement_for"),
        record.get("failed_candidate_id"),
        record.get("original_candidate_id"),
    )
    output_root = _first_nonempty(record.get("output_root"), record.get("output_dir"))
    if output_root is None or str(output_root).strip() == "":
        raise ValueError(f"replacement record {replacement_candidate_id!r} is missing output_root.")
    ka_value = _first_nonempty(record.get("ka"), record.get("force_ka"), record.get("lambda_a"), record.get("lambdaA"))
    kb_value = _first_nonempty(record.get("kb"), record.get("force_kb"), record.get("lambda_b"), record.get("lambdaB"))
    if ka_value is None or kb_value is None:
        raise ValueError(f"replacement record {replacement_candidate_id!r} is missing ka/kb.")
    metadata = _mapping_or_empty(record.get("metadata"))
    metadata.setdefault("selection_mode", metadata.get("selection_mode", "causal_fresh_only"))
    metadata["fresh_only"] = _bool_like(metadata.get("fresh_only", True))
    metadata.setdefault("source", metadata.get("source", "replacement"))
    metadata.setdefault("selection_source", metadata.get("selection_source", record.get("replacement_policy", "reserve")))
    stage = f"{method}-step-{step:02d}"
    return {
        "candidate_id": str(replacement_candidate_id),
        "output_root": str(output_root).strip(),
        "vault_output_root": str(_first_nonempty(record.get("vault_output_root"), record.get("vault_output_dir"), "")),
        "method": method,
        "stage": stage,
        "seed": seed,
        "step": step,
        "ka": _as_float(ka_value, label="replacement.ka"),
        "kb": _as_float(kb_value, label="replacement.kb"),
        "selection_mode": str(metadata.get("selection_mode", "causal_fresh_only")),
        "fresh_only": _bool_like(metadata.get("fresh_only", True)),
        "selection_seed": _as_int(_first_nonempty(record.get("selection_seed"), record.get("selection_order"), record.get("selection_pool_rank"), 0), label="replacement.selection_seed"),
        "selection_source": str(metadata.get("selection_source", "reserve")),
        "selection_order": _first_nonempty(record.get("selection_order"), ""),
        "selection_pool_rank": _first_nonempty(record.get("selection_pool_rank"), ""),
        "candidate_pool_id": str(_first_nonempty(record.get("candidate_pool_id"), replacement_candidate_id)),
        "replacement": True,
        "replacement_for": str(replaced_for).strip() if replaced_for is not None else "",
        "replacement_candidate_id": replacement_candidate_id,
        "replacement_policy": str(_first_nonempty(record.get("replacement_policy"), metadata.get("replacement_policy"), "")),
        "metadata": metadata,
    }


def _collect_replacement_records_for_step(
    campaign_root: Path,
    *,
    seed: int,
    step: int,
    method: str = "al",
) -> tuple[dict[str, Any], ...]:
    method = _normalize_method(method)
    if method not in {"al", "lhs"}:
        raise ValueError(f"replacement method must be 'al' or 'lhs', got {method!r}.")
    step_root = campaign_root / f"seed-{seed:03d}" / f"{method}-step-{step:02d}"
    replacement_root = step_root / "replacement"
    if not replacement_root.is_dir():
        return ()
    manifest_filename = (
        EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_MANIFEST_FILENAME
        if method == "al"
        else EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_MANIFEST_FILENAME
    )
    batch_summary_filename = (
        EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME
        if method == "al"
        else EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_BATCH_SUMMARY_FILENAME
    )

    found: dict[tuple[str, str], dict[str, Any]] = {}

    def add_records(payload_path: Path) -> None:
        payload = _read_json(payload_path)
        for index, item in enumerate(
            _coerce_sequence(payload.get("replacement_records", ()), label=f"{payload_path.name} replacement_records")
        ):
            record = _coerce_mapping(item, label=f"{payload_path.name}[{index}]")
            candidate = _coerce_replacement_record(record, step=step, seed=seed, method=method)
            key = (str(candidate["candidate_id"]), str(candidate["replacement_for"]))
            if key not in found:
                found[key] = candidate

    stage_manifest_path = step_root / manifest_filename
    if stage_manifest_path.is_file():
        add_records(stage_manifest_path)

    for batch_summary_path in sorted(replacement_root.glob(f"batch-*/{batch_summary_filename}")):
        add_records(batch_summary_path)

    return tuple(found.values())


def _flatten_manifest_records(
    manifest: Mapping[str, Any],
    *,
    fresh_only: bool,
    campaign_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    reserves: dict[tuple[int, str], list[dict[str, Any]]] = {}
    allow_legacy_fallback = not _manifest_requires_causal_fresh_only(manifest)
    for seed_payload in _coerce_sequence(manifest.get("seeds", ()), label="seeds"):
        seed_map = _coerce_mapping(seed_payload, label="seed payload")
        seed = _as_int(seed_map.get("seed"), label="seed")

        def add_batch(batch: Mapping[str, Any], *, method: str, step: int | None) -> int:
            raw_records = _coerce_batch_records(batch, label=f"seed {seed} {method}")
            for raw in raw_records:
                rec = _coerce_mapping(raw, label="candidate record")
                if fresh_only:
                    _assert_fresh_only(rec, allow_legacy_fallback=allow_legacy_fallback)
                rec_seed = _as_int(rec.get("seed", seed), label="candidate.seed")
                rec_method = _normalize_method(_get_alias(rec, "method", required=False) or method)
                rec_step = rec.get("step", step)
                rec_step_int = _as_int(rec_step, label="candidate.step") if rec_step not in (None, "") else None
                flat = {
                    **rec,
                    "candidate_id": _candidate_id(rec),
                    "seed": rec_seed,
                    "method": rec_method,
                    "step": rec_step_int,
                    "stage": _normalized_stage(rec_method, rec_step_int),
                    "policy_group": "validation" if rec_method == "validation" else ("shared_initial" if rec_method == "shared_initial" else rec_method),
                }
                records.append(flat)
            return len(raw_records)

        add_batch(_coerce_mapping(seed_map.get("shared_initial", {}), label="shared_initial"), method="shared_initial", step=0)
        add_batch(_coerce_mapping(seed_map.get("validation", {}), label="validation"), method="validation", step=0)
        for step_batch in _coerce_sequence(seed_map.get("al_steps", ()), label="al_steps"):
            batch = _coerce_mapping(step_batch, label="al step")
            step_value = _as_int(batch.get("step"), label="al.step")
            if add_batch(batch, method="al", step=step_value) == 0:
                for rec in _records_from_rendered_al_batch(
                    batch,
                    seed=seed,
                    step=step_value,
                    fresh_only=fresh_only,
                    allow_legacy_fallback=allow_legacy_fallback,
                ):
                    records.append(
                        {
                            **rec,
                            "candidate_id": _candidate_id(rec),
                            "seed": seed,
                            "method": "al",
                            "step": step_value,
                            "stage": _normalized_stage("al", step_value),
                            "policy_group": "al",
                        }
                    )

            for replacement_record in _collect_replacement_records_for_step(
                campaign_root,
                seed=seed,
                step=step_value,
            ):
                if fresh_only:
                    _assert_fresh_only(replacement_record, allow_legacy_fallback=allow_legacy_fallback)
                replacement_record["policy_group"] = "al"
                records.append(replacement_record)

        for step_batch in _coerce_sequence(seed_map.get("lhs_steps", ()), label="lhs_steps"):
            batch = _coerce_mapping(step_batch, label="lhs step")
            step_value = _as_int(batch.get("step"), label="lhs.step")
            add_batch(batch, method="lhs", step=step_value)
            for replacement_record in _collect_replacement_records_for_step(
                campaign_root,
                seed=seed,
                step=step_value,
                method="lhs",
            ):
                if fresh_only:
                    _assert_fresh_only(replacement_record, allow_legacy_fallback=allow_legacy_fallback)
                replacement_record["policy_group"] = "lhs"
                records.append(replacement_record)

        reserve_payload = _coerce_mapping(seed_map.get("candidate_reserve", {}), label="candidate_reserve")
        for method in ("al", "lhs", "validation", "shared_initial"):
            method_reserve = reserve_payload.get(method)
            if not isinstance(method_reserve, Mapping):
                continue
            for raw in _coerce_batch_records(method_reserve, label=f"reserve {method}"):
                rec = _coerce_mapping(raw, label="reserve record")
                if fresh_only:
                    _assert_fresh_only(rec, allow_legacy_fallback=allow_legacy_fallback)
                rec_method = _normalize_method(_get_alias(rec, "method", required=False) or method)
                rec_step = rec.get("step")
                rec_step_int = _as_int(rec_step, label="reserve.step") if rec_step not in (None, "") else None
                stage = _normalized_stage(rec_method, rec_step_int)
                reserves.setdefault((seed, stage), []).append(
                    {
                        **rec,
                        "candidate_id": _candidate_id(rec),
                        "seed": seed,
                        "method": rec_method,
                        "step": rec_step_int,
                        "stage": stage,
                    }
                )
    return records, reserves


def _build_replacement_plan(
    records: Sequence[Mapping[str, Any]],
    reserves: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    missing_or_failed = [
        row for row in records if row.get("status") in {"failed", "missing"}
    ]
    plan_rows: list[dict[str, Any]] = []
    reserve_cursors: dict[tuple[int, str], int] = {}
    for row in sorted(missing_or_failed, key=lambda item: (int(item["seed"]), str(item["stage"]), str(item["candidate_id"]))):
        key = (int(row["seed"]), str(row["stage"]))
        pool = sorted(reserves.get(key, ()), key=lambda item: str(item["candidate_id"]))
        cursor = reserve_cursors.get(key, 0)
        if cursor >= len(pool):
            continue
        replacement = pool[cursor]
        reserve_cursors[key] = cursor + 1
        plan_rows.append(
            {
                **replacement,
                "replacement_for": row["candidate_id"],
                "original_candidate_id": row["candidate_id"],
                "reason": row["status"],
            }
        )
    return {
        "schema_version": EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
        "replacement_count": len(plan_rows),
        "records": plan_rows,
    }


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("seed", "stage", "policy_group", "candidate_id", "status", "reason_codes")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row.get("seed"),
                    "stage": row.get("stage"),
                    "policy_group": row.get("policy_group"),
                    "candidate_id": row.get("candidate_id"),
                    "status": row.get("status"),
                    "reason_codes": ";".join(row.get("reason_codes", ())),
                }
            )


def build_emb_34um_causal_validation_ingestion_report(
    *,
    campaign_manifest_path: str | Path,
    success_filenames: Sequence[str] = DEFAULT_SUCCESS_FILENAMES,
    status_filenames: Sequence[str] = DEFAULT_STATUS_FILENAMES,
    fresh_only: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(campaign_manifest_path)
    if manifest.get("schema_version") != EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {manifest.get('schema_version')!r}; "
            f"expected {EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION!r}."
        )
    expected_policy = _expected_policy_counts(manifest)
    campaign_root = Path(str(manifest.get("campaign_root", Path(campaign_manifest_path).parent)))
    flattened, reserves = _flatten_manifest_records(
        manifest,
        fresh_only=fresh_only,
        campaign_root=campaign_root,
    )
    assert_same_validation_set([row for row in flattened if row.get("method") == "validation"])

    preliminary: list[dict[str, Any]] = []
    for row in flattened:
        status, reasons = _classify_record(row, success_filenames=success_filenames, status_filenames=status_filenames)
        preliminary.append({**row, "status": status, "reason_codes": reasons})

    replacement_targets = _replacement_targets_by_completed_replacement(preliminary)
    classified: list[dict[str, Any]] = []
    for row in preliminary:
        status = str(row["status"])
        replacement_for = str(row.get("replacement_for", "")).strip()
        replacement = bool(row.get("replacement"))
        if (
            not replacement
            and status in {"failed", "missing"}
            and (int(row["seed"]), str(row["policy_group"]), str(row["stage"]), str(row["candidate_id"]).strip()) in replacement_targets
        ):
            status = "quarantined"
            row["reason_codes"] = [*row["reason_codes"], "replaced_by_successful_replacement"]
            if replacement_for and f"replacement_for:{replacement_for}" not in row["reason_codes"]:
                row["reason_codes"].append(f"replacement_for:{replacement_for}")
        row["status"] = status
        row["quarantined"] = status == "quarantined"
        classified.append(row)

    by_seed_stage_policy: dict[tuple[int, str, str], dict[str, int]] = {}
    for row in classified:
        key = (int(row["seed"]), str(row["stage"]), str(row["policy_group"]))
        bucket = by_seed_stage_policy.setdefault(
            key, {"completed": 0, "failed": 0, "missing": 0, "quarantined": 0, "total": 0}
        )
        bucket_key = str(row["status"])
        if bucket_key not in bucket:
            raise ValueError(f"Unsupported status {bucket_key!r} in row {row.get('candidate_id')!r}")
        bucket[bucket_key] += 1
        if bucket_key != "quarantined":
            bucket["total"] += 1

    by_seed_policy: dict[tuple[int, str], dict[str, int]] = {}
    for row in classified:
        key = (int(row["seed"]), str(row["policy_group"]))
        bucket = by_seed_policy.setdefault(
            key, {"completed": 0, "failed": 0, "missing": 0, "quarantined": 0, "total": 0}
        )
        bucket_key = str(row["status"])
        if bucket_key not in bucket:
            raise ValueError(f"Unsupported status {bucket_key!r} in row {row.get('candidate_id')!r}")
        bucket[bucket_key] += 1
        if bucket_key != "quarantined":
            bucket["total"] += 1

    blockers: list[str] = []
    seeds = sorted(_as_int(_coerce_mapping(seed_payload, label="seed payload").get("seed"), label="seed") for seed_payload in _coerce_sequence(manifest.get("seeds", ()), label="seeds"))

    expected_stage_policy = _expected_stage_policy_counts(manifest)
    for seed in seeds:
        for policy_group, stage in sorted(expected_stage_policy):
            expected = expected_stage_policy[(policy_group, stage)]
            counts = by_seed_stage_policy.get(
                (seed, stage, policy_group),
                {"completed": 0, "failed": 0, "missing": 0, "quarantined": 0, "total": 0},
            )
            if counts["completed"] >= expected:
                continue
            blockers.append(
                f"seed={seed} policy={policy_group} stage={stage} completed={counts['completed']} expected={expected} shortfall={expected - counts['completed']}"
            )
    for seed in seeds:
        for policy, expected in sorted(expected_policy.items()):
            counts = by_seed_policy.get((seed, policy), {"completed": 0, "failed": 0, "missing": 0, "total": 0})
            if counts["completed"] >= expected:
                continue
            blockers.append(
                f"seed={seed} policy={policy} completed={counts['completed']} expected={expected} shortfall={expected - counts['completed']}"
            )
    if blockers:
        blockers.insert(0, "Completed counts fall short of policy requirements.")

    ingestion_manifest = {
        "schema_version": EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
        "campaign_manifest_path": str(campaign_manifest_path),
        "record_count": len(classified),
        "records": classified,
    }
    report = {
        **ingestion_manifest,
        "expected_policy_counts": expected_policy,
        "expected_counts_by_seed_stage_policy": _build_expected_stage_policy_rows(manifest, seeds=seeds),
        "counts_by_seed_stage_policy": [
            {
                "seed": seed,
                "stage": stage,
                "policy_group": policy,
                **counts,
            }
            for (seed, stage, policy), counts in sorted(by_seed_stage_policy.items())
        ],
        "counts_by_seed_policy": [
            {"seed": seed, "policy_group": policy, **counts}
            for (seed, policy), counts in sorted(by_seed_policy.items())
        ],
        "blockers": blockers,
        "status": "blocked" if blockers else "passed",
        "passed": not blockers,
        "replacement_plan_preview": _build_replacement_plan(classified, reserves),
    }
    return ingestion_manifest, report


def write_emb_34um_causal_validation_ingestion_artifacts(
    *,
    campaign_manifest_path: str | Path,
    output_root: str | Path | None = None,
    write_replacement_plan: bool = False,
    replacement_manifest_output_path: str | Path | None = None,
    success_filenames: Sequence[str] = DEFAULT_SUCCESS_FILENAMES,
    status_filenames: Sequence[str] = DEFAULT_STATUS_FILENAMES,
    fresh_only: bool = True,
) -> Emb34umCausalValidationIngestionArtifacts:
    ingestion_manifest, report = build_emb_34um_causal_validation_ingestion_report(
        campaign_manifest_path=campaign_manifest_path,
        success_filenames=success_filenames,
        status_filenames=status_filenames,
        fresh_only=fresh_only,
    )
    manifest = _read_json(campaign_manifest_path)
    campaign_root = Path(str(manifest.get("campaign_root", Path(campaign_manifest_path).parent)))
    artifact_dir = Path(output_root) if output_root is not None else campaign_root / "ingest"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME
    report_path = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME
    summary_csv_path = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME
    replacement_plan_path: Path | None = None
    replacement_plan: dict[str, Any] | None = None

    _write_json(manifest_path, ingestion_manifest)
    _write_json(report_path, report)
    _write_summary_csv(summary_csv_path, ingestion_manifest["records"])
    if write_replacement_plan:
        replacement_plan = dict(report["replacement_plan_preview"])
        replacement_plan_path = (
            Path(replacement_manifest_output_path)
            if replacement_manifest_output_path is not None
            else artifact_dir / EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_PLAN_FILENAME
        )
        _write_json(replacement_plan_path, replacement_plan)

    return Emb34umCausalValidationIngestionArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        replacement_plan_path=replacement_plan_path,
        manifest=ingestion_manifest,
        report=report,
        replacement_plan=replacement_plan,
    )


__all__ = [
    "EMB_34UM_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION",
    "EMB_34UM_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_MANIFEST_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_MANIFEST_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_BATCH_SUMMARY_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_PLAN_FILENAME",
    "Emb34umCausalValidationIngestionArtifacts",
    "assert_same_validation_set",
    "build_emb_34um_causal_validation_ingestion_report",
    "write_emb_34um_causal_validation_ingestion_artifacts",
]
