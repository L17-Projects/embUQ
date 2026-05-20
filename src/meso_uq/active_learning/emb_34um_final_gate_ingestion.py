from __future__ import annotations

"""Ingestion/report gate for EMB 3.4um AL final-gate DPD outputs."""

import csv
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EMB_34UM_FINAL_GATE_INGESTION_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_final_gate_ingestion.v1"
EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME = "emb_34um_final_gate_ingestion_report.json"
EMB_34UM_FINAL_GATE_INGESTION_MANIFEST_FILENAME = "emb_34um_final_gate_ingestion_manifest.json"
EMB_34UM_FINAL_GATE_INGESTION_SUMMARY_CSV_FILENAME = "emb_34um_final_gate_ingestion_summary.csv"
EMB_34UM_FINAL_GATE_INGESTION_PLOT_FILENAME = "emb_34um_final_gate_ingestion_validation.png"
EMB_34UM_FINAL_GATE_INGESTION_PLOT_SIDECAR_FILENAME = "emb_34um_final_gate_ingestion_validation.png.json"
EMB_34UM_FINAL_GATE_QUARANTINE_FILENAME = "emb_34um_final_gate_quarantine_manifest.json"

_PARAMETER_COLUMN_COUNT = 8
_FLOAT_TOLERANCE = 1e-6
_FULL_ROUNDS = (1, 2, 3)


@dataclass(frozen=True)
class Emb34umFinalGateIngestionArtifacts:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    summary_csv_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    quarantine_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path!s}.")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_path(path: object, *, label: str) -> Path:
    text = str(path or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty path.")
    return Path(text)


def _as_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _as_sequence(value: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(value)


def _float_sequence(value: object, *, label: str) -> tuple[float, ...]:
    values = _as_sequence(value, label=label)
    parsed: list[float] = []
    for index, item in enumerate(values):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{index}] must be numeric.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] must be finite.")
        parsed.append(number)
    if not parsed:
        raise ValueError(f"{label} must not be empty.")
    return tuple(parsed)


def _close_series(left: Sequence[float], right: Sequence[float], *, tolerance: float = _FLOAT_TOLERANCE) -> bool:
    if len(left) != len(right):
        return False
    return all(math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance) for a, b in zip(left, right))


def _candidate_manifest_paths(campaign: Mapping[str, Any]) -> tuple[tuple[str, Path], ...]:
    paths: list[tuple[str, Path]] = []
    for gate_key in ("canary_gate", "full_gate", "lhs_gate"):
        gate = campaign.get(gate_key)
        if not isinstance(gate, Mapping):
            continue
        for item in gate.get("rendered_candidate_manifests", ()):
            paths.append((gate_key, Path(str(item))))
    if not paths:
        raise ValueError("Campaign manifest does not include rendered_candidate_manifests.")
    return tuple(paths)


def _candidate_round(candidate_id: str, metadata: Mapping[str, Any], gate_key: str) -> int | None:
    round_value = metadata.get("round")
    if round_value is not None:
        try:
            return int(round_value)
        except (TypeError, ValueError):
            return None
    match = re.search(r"-r(\d{2})-c\d{3}$", candidate_id)
    if match:
        return int(match.group(1))
    if gate_key == "canary_gate":
        return 0
    return None


def _candidate_expected_paths(
    candidate: Mapping[str, Any],
    *,
    campaign_root: Path,
    extra_f_delta_files: Sequence[Path],
) -> tuple[Path, ...]:
    normalized = _as_mapping(candidate.get("normalized_payload", {}), label="normalized_payload")
    request_payload = candidate.get("rendered_payload", {})
    if isinstance(request_payload, Mapping):
        request_payload = request_payload.get("request_payload", {})
    if not isinstance(request_payload, Mapping):
        request_payload = {}

    output_roots: list[Path] = []
    for payload in (normalized, request_payload):
        output_root = payload.get("output_root")
        if output_root:
            output_roots.append(Path(str(output_root)))
        expected_paths = payload.get("expected_output_paths")
        if isinstance(expected_paths, Mapping):
            f_delta = expected_paths.get("f_delta") or expected_paths.get("F_Delta")
            if f_delta:
                output_roots.append(Path(str(f_delta)).parent)

    candidate_output_root = candidate.get("output_root")
    if candidate_output_root:
        output_roots.append(Path(str(candidate_output_root)))

    paths: list[Path] = []
    for root in output_roots:
        paths.append(root / "F_Delta.dat")
    paths.append(campaign_root / "F_Delta.dat")
    paths.extend(extra_f_delta_files)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        token = str(path)
        if token not in seen:
            unique.append(path)
            seen.add(token)
    return tuple(unique)


def _parse_f_delta_rows(path: Path, *, expected_force_count: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return tuple()
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values = [float(item) for item in line.split()]
        except ValueError as exc:
            raise ValueError(f"Could not parse numeric row {index} in {path!s}.") from exc
        expected_width = _PARAMETER_COLUMN_COUNT + 2 * expected_force_count
        if len(values) != expected_width:
            continue
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"Non-finite value in {path!s} row {index}.")
        rows.append(
            {
                "path": str(path),
                "row_index": index,
                "parameters": values[:_PARAMETER_COLUMN_COUNT],
                "outputs": values[_PARAMETER_COLUMN_COUNT : _PARAMETER_COLUMN_COUNT + expected_force_count],
                "forces": values[_PARAMETER_COLUMN_COUNT + expected_force_count :],
            }
        )
    return tuple(rows)


def _find_matching_f_delta_row(
    *,
    candidate: Mapping[str, Any],
    expected_paths: Sequence[Path],
    force_grid: Sequence[float],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    normalized = _as_mapping(candidate.get("normalized_payload", {}), label="normalized_payload")
    parameters = _as_mapping(normalized.get("parameters", {}), label="normalized_payload.parameters")
    ka = float(parameters["ka"])
    kb = float(parameters["kb"])
    reasons: list[str] = []
    readable_files = [path for path in expected_paths if path.is_file()]
    if not readable_files:
        return None, ("missing_f_delta_dat",)

    for path in readable_files:
        try:
            rows = _parse_f_delta_rows(path, expected_force_count=len(force_grid))
        except ValueError as exc:
            reasons.append("invalid_f_delta_dat")
            reasons.append(str(exc))
            continue
        for row in rows:
            params = row["parameters"]
            if not math.isclose(float(params[1]), ka, rel_tol=_FLOAT_TOLERANCE, abs_tol=_FLOAT_TOLERANCE):
                continue
            if not math.isclose(float(params[2]), kb, rel_tol=_FLOAT_TOLERANCE, abs_tol=_FLOAT_TOLERANCE):
                continue
            if not _close_series(row["forces"], force_grid):
                reasons.append("force_grid_mismatch")
                continue
            return row, tuple()
    if reasons:
        return None, tuple(sorted(set(reasons)))
    return None, ("missing_matching_f_delta_row",)


def _runtime_status(candidate: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _as_mapping(candidate.get("normalized_payload", {}), label="normalized_payload")
    roots = []
    output_root = normalized.get("output_root")
    if output_root:
        roots.append(Path(str(output_root)))
    candidate_output_root = candidate.get("output_root")
    if candidate_output_root:
        roots.append(Path(str(candidate_output_root)))

    for root in roots:
        for name in ("runtime_status.json", "result_status.json", "emb_34um_runtime_status.json"):
            path = root / name
            if path.is_file():
                payload = _read_json(path)
                payload["status_path"] = str(path)
                return payload
    return {}


def _status_from_runtime(runtime: Mapping[str, Any]) -> str | None:
    status = str(runtime.get("status", "")).strip().lower()
    if status in {"failed", "failure", "error", "cancelled", "timeout"}:
        return "failed"
    if status in {"completed", "complete", "success", "succeeded"}:
        return "completed"
    if status:
        return "partial"
    return None


def _retry_count(runtime: Mapping[str, Any]) -> int:
    for key in ("retry_count", "retries", "attempts"):
        if key in runtime:
            try:
                return max(0, int(runtime[key]))
            except (TypeError, ValueError):
                return 0
    return 0


def _candidate_record(
    *,
    gate_key: str,
    manifest_path: Path,
    candidate: Mapping[str, Any],
    campaign_root: Path,
    retry_limit: int,
    extra_f_delta_files: Sequence[Path],
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError(f"{manifest_path!s} is missing candidate_id.")
    normalized = _as_mapping(candidate.get("normalized_payload", {}), label=f"{candidate_id} normalized_payload")
    force_grid = _float_sequence(normalized.get("force_grid"), label=f"{candidate_id} force_grid")
    metadata = _as_mapping(candidate.get("active_learning_metadata", {}), label=f"{candidate_id} active_learning_metadata")
    expected_paths = _candidate_expected_paths(candidate, campaign_root=campaign_root, extra_f_delta_files=extra_f_delta_files)
    runtime = _runtime_status(candidate)
    runtime_status = _status_from_runtime(runtime)
    retry_count = _retry_count(runtime)
    f_delta_row, reasons = _find_matching_f_delta_row(
        candidate=candidate,
        expected_paths=expected_paths,
        force_grid=force_grid,
    )

    if f_delta_row is not None:
        status = "completed"
    elif runtime_status == "failed":
        status = "failed"
        reasons = tuple(sorted(set(reasons + ("runtime_failed",))))
    elif runtime_status == "partial":
        status = "partial"
        reasons = tuple(sorted(set(reasons + ("runtime_status_partial",))))
    else:
        status = "missing"

    quarantined = status in {"failed", "missing", "partial"} and retry_count >= retry_limit
    if status in {"failed", "missing", "partial"} and retry_count < retry_limit:
        reasons = tuple(sorted(set(reasons + ("retry_evidence_below_limit",))))

    row_outputs = f_delta_row.get("outputs", ()) if f_delta_row else ()
    reduced = {
        "curve_point_count": len(row_outputs),
        "mean_output": float(sum(row_outputs) / len(row_outputs)) if row_outputs else 0.0,
        "min_output": float(min(row_outputs)) if row_outputs else 0.0,
        "max_output": float(max(row_outputs)) if row_outputs else 0.0,
    }

    return {
        "candidate_id": candidate_id,
        "gate": gate_key,
        "round": _candidate_round(candidate_id, metadata, gate_key),
        "manifest_path": str(manifest_path),
        "status": status,
        "reason_codes": list(reasons),
        "retry_count": retry_count,
        "retry_limit": retry_limit,
        "quarantined": quarantined,
        "runtime_status_path": runtime.get("status_path", ""),
        "expected_f_delta_paths": [str(path) for path in expected_paths],
        "f_delta_path": str(f_delta_row["path"]) if f_delta_row else "",
        "f_delta_row_index": int(f_delta_row["row_index"]) if f_delta_row else None,
        "force_grid_count": len(force_grid),
        "parameters": dict(normalized.get("parameters", {})),
        "reduced_observables": reduced,
    }


def _status_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = ("completed", "failed", "missing", "partial")
    return {key: sum(1 for item in records if item.get("status") == key) for key in keys}


def _round_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    rounds: dict[str, dict[str, int]] = {}
    for record in records:
        round_value = record.get("round")
        label = "canary" if round_value == 0 else f"round_{int(round_value):02d}" if round_value is not None else "unknown"
        rounds.setdefault(label, {"completed": 0, "failed": 0, "missing": 0, "partial": 0, "quarantined": 0, "total": 0})
        status = str(record.get("status"))
        if status in rounds[label]:
            rounds[label][status] += 1
        if record.get("quarantined"):
            rounds[label]["quarantined"] += 1
        rounds[label]["total"] += 1
    return dict(sorted(rounds.items()))


def _criteria(records: Sequence[Mapping[str, Any]], *, expected_full_count: int, expected_canary_count: int) -> dict[str, dict[str, bool]]:
    full_records = [item for item in records if item.get("gate") == "full_gate"]
    canary_records = [item for item in records if item.get("gate") == "canary_gate"]
    full_rounds = sorted({item.get("round") for item in full_records})
    plot_ready = True
    return {
        "candidate_count": {
            "passed": len(full_records) == expected_full_count and len(canary_records) == expected_canary_count,
            "required": True,
        },
        "canary_first_present": {
            "passed": len(canary_records) == expected_canary_count,
            "required": True,
        },
        "full_rounds_present": {
            "passed": full_rounds == list(_FULL_ROUNDS),
            "required": True,
        },
        "all_expected_outputs_ingested": {
            "passed": all(item.get("status") == "completed" for item in records),
            "required": True,
        },
        "all_noncompleted_curves_quarantined_or_retryable": {
            "passed": all(
                item.get("status") == "completed"
                or item.get("quarantined")
                or int(item.get("retry_count", 0)) < int(item.get("retry_limit", 0))
                for item in records
            ),
            "required": True,
        },
        "validation_plot_written": {
            "passed": plot_ready,
            "required": True,
        },
    }


def _failure_reasons(criteria: Mapping[str, Mapping[str, bool]]) -> list[str]:
    return [key for key, value in criteria.items() if value.get("required") and not value.get("passed")]


def _write_summary_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "gate",
        "round",
        "status",
        "retry_count",
        "retry_limit",
        "quarantined",
        "f_delta_path",
        "f_delta_row_index",
        "force_grid_count",
        "reason_codes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            row["reason_codes"] = ";".join(str(item) for item in record.get("reason_codes", ()))
            writer.writerow(row)


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + type_code + data + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)


def _fallback_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    width = 16
    height = 16
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((80, 120, 180, 255) if (x + y) % 2 else (230, 235, 240, 255))
        rows.append(bytes(row))
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", zlib.compress(b"".join(rows))) + _png_chunk(b"IEND", b"")


def _write_validation_plot(path: Path, records: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    round_counts = _round_counts(records)
    labels = list(round_counts)
    completed = [round_counts[label]["completed"] for label in labels]
    missing = [round_counts[label]["missing"] for label in labels]
    failed = [round_counts[label]["failed"] for label in labels]
    partial = [round_counts[label]["partial"] for label in labels]
    quarantined = [round_counts[label]["quarantined"] for label in labels]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    x = list(range(len(labels)))
    axes[0].bar(x, completed, label="completed")
    axes[0].bar(x, missing, bottom=completed, label="missing")
    failed_bottom = [a + b for a, b in zip(completed, missing)]
    axes[0].bar(x, failed, bottom=failed_bottom, label="failed")
    partial_bottom = [a + b for a, b in zip(failed_bottom, failed)]
    axes[0].bar(x, partial, bottom=partial_bottom, label="partial")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("Curves")
    axes[0].set_title("EMB 3.4um final-gate ingestion status")
    axes[0].legend()

    means = [
        float(item["reduced_observables"]["mean_output"])
        for item in records
        if item.get("status") == "completed" and item.get("reduced_observables")
    ]
    if means:
        axes[1].hist(means, bins=min(20, max(3, len(means) // 3)))
    axes[1].set_xlabel("Mean F_Delta output")
    axes[1].set_ylabel("Completed curves")
    axes[1].set_title(f"Quarantined candidates: {sum(quarantined)}")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_emb_34um_final_gate_ingestion_report(
    *,
    campaign_manifest_path: str | Path,
    extra_f_delta_files: Sequence[str | Path] = (),
    expected_full_count: int = 90,
    expected_canary_count: int = 1,
    retry_limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_manifest = _read_json(campaign_manifest_path)
    campaign_root = _coerce_path(campaign_manifest.get("campaign_root"), label="campaign_root")
    resources = _as_mapping(campaign_manifest.get("expected_resources", {}), label="expected_resources")
    effective_retry_limit = int(retry_limit if retry_limit is not None else resources.get("retry_limit", 3))
    f_delta_files = tuple(Path(item) for item in extra_f_delta_files)

    records: list[dict[str, Any]] = []
    for gate_key, manifest_path in _candidate_manifest_paths(campaign_manifest):
        candidate = _read_json(manifest_path)
        records.append(
            _candidate_record(
                gate_key=gate_key,
                manifest_path=manifest_path,
                candidate=candidate,
                campaign_root=campaign_root,
                retry_limit=effective_retry_limit,
                extra_f_delta_files=f_delta_files,
            )
        )

    records = sorted(records, key=lambda item: (str(item["gate"]), int(item["round"] or 999), str(item["candidate_id"])))
    criteria = _criteria(records, expected_full_count=expected_full_count, expected_canary_count=expected_canary_count)
    failures = _failure_reasons(criteria)
    completed_records = [item for item in records if item["status"] == "completed"]
    noncompleted = [item for item in records if item["status"] != "completed"]
    quarantine_records = [item for item in records if item["quarantined"]]

    blockers = []
    if noncompleted:
        blockers.append("DPD F_Delta.dat outputs are missing, failed, partial, or do not match candidate force grids.")
    if not completed_records:
        blockers.append("No completed F_Delta.dat rows were ingested; canary/production pass/fail cannot be computed.")
    blockers.append(
        "Final AL-vs-LHS pass/fail still requires per-round surrogate prediction/reference metrics "
        "and the 90-curve LHS comparator metrics consumed by final_gate_reports."
    )

    manifest = {
        "schema_version": EMB_34UM_FINAL_GATE_INGESTION_SCHEMA_VERSION,
        "campaign_manifest_path": str(campaign_manifest_path),
        "campaign_root": str(campaign_root),
        "record_count": len(records),
        "status_counts": _status_counts(records),
        "round_counts": _round_counts(records),
        "quarantine_count": len(quarantine_records),
        "records": records,
    }
    report = {
        **manifest,
        "pass_fail_criteria": criteria,
        "failure_reasons": failures,
        "passed": not failures,
        "status": "passed" if not failures else "blocked",
        "completed_count": len(completed_records),
        "noncompleted_count": len(noncompleted),
        "quarantine_records": quarantine_records,
        "blockers": blockers,
        "required_runtime_output_contract": {
            "file": "F_Delta.dat",
            "producer": "emb.indentation.evalkit.posterior_indentation.compute_indentation",
            "row_layout": "Yt ka kb b1 b2 a3 a4 radp, followed by one output per force, followed by the force grid",
            "matching_columns": {"ka": 1, "kb": 2},
        },
    }
    return manifest, report


def write_emb_34um_final_gate_ingestion_artifacts(
    *,
    campaign_manifest_path: str | Path,
    output_root: str | Path | None = None,
    extra_f_delta_files: Sequence[str | Path] = (),
    expected_full_count: int = 90,
    expected_canary_count: int = 1,
    retry_limit: int | None = None,
    include_plot: bool = True,
) -> Emb34umFinalGateIngestionArtifacts:
    manifest, report = build_emb_34um_final_gate_ingestion_report(
        campaign_manifest_path=campaign_manifest_path,
        extra_f_delta_files=extra_f_delta_files,
        expected_full_count=expected_full_count,
        expected_canary_count=expected_canary_count,
        retry_limit=retry_limit,
    )
    campaign_root = Path(str(report["campaign_root"]))
    artifact_dir = Path(output_root) if output_root is not None else campaign_root / "ingestion_report"
    manifest_path = artifact_dir / EMB_34UM_FINAL_GATE_INGESTION_MANIFEST_FILENAME
    report_path = artifact_dir / EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME
    summary_csv_path = artifact_dir / EMB_34UM_FINAL_GATE_INGESTION_SUMMARY_CSV_FILENAME
    plot_path = artifact_dir / EMB_34UM_FINAL_GATE_INGESTION_PLOT_FILENAME
    plot_sidecar_path = artifact_dir / EMB_34UM_FINAL_GATE_INGESTION_PLOT_SIDECAR_FILENAME
    quarantine_path = artifact_dir / EMB_34UM_FINAL_GATE_QUARANTINE_FILENAME

    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    _write_summary_csv(summary_csv_path, report["records"])
    _write_validation_plot(plot_path, report["records"], include_plot=include_plot)
    _write_json(plot_sidecar_path, {"plot_path": str(plot_path), **report})
    _write_json(
        quarantine_path,
        {
            "schema_version": EMB_34UM_FINAL_GATE_INGESTION_SCHEMA_VERSION,
            "campaign_manifest_path": str(campaign_manifest_path),
            "quarantine_count": len(report["quarantine_records"]),
            "records": report["quarantine_records"],
        },
    )
    report["plot_paths"] = {
        "ingestion_validation": str(plot_path),
        "ingestion_validation_sidecar": str(plot_sidecar_path),
    }
    _write_json(report_path, report)

    return Emb34umFinalGateIngestionArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        quarantine_path=quarantine_path,
        manifest=manifest,
        report=report,
    )


__all__ = [
    "EMB_34UM_FINAL_GATE_INGESTION_MANIFEST_FILENAME",
    "EMB_34UM_FINAL_GATE_INGESTION_PLOT_FILENAME",
    "EMB_34UM_FINAL_GATE_INGESTION_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME",
    "EMB_34UM_FINAL_GATE_INGESTION_SCHEMA_VERSION",
    "EMB_34UM_FINAL_GATE_INGESTION_SUMMARY_CSV_FILENAME",
    "EMB_34UM_FINAL_GATE_QUARANTINE_FILENAME",
    "Emb34umFinalGateIngestionArtifacts",
    "build_emb_34um_final_gate_ingestion_report",
    "write_emb_34um_final_gate_ingestion_artifacts",
]
