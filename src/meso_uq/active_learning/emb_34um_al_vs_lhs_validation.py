from __future__ import annotations

"""AL-vs-LHS validation plotting for EMB 3.4um curve rows."""

import csv
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EMB_34UM_AL_VS_LHS_VALIDATION_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_al_vs_lhs_validation.v1"
EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME = "emb_34um_al_vs_lhs_validation_manifest.json"
EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME = "emb_34um_al_vs_lhs_validation_summary.csv"
EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation.png"
EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME = "emb_34um_al_vs_lhs_validation.png.json"

_CURVE_METRIC_ALIASES = ("curve_rel_l2_pct", "rel_l2_pct", "median_curve_rel_l2_pct")
_PREDICTED_ALIASES = (
    "predicted_curve",
    "prediction_curve",
    "predicted_outputs",
    "prediction_outputs",
    "outputs_pred",
    "y_pred",
)
_REFERENCE_ALIASES = (
    "reference_curve",
    "truth_curve",
    "reference_outputs",
    "target_outputs",
    "outputs_ref",
    "y_true",
)
_POINT_PREDICTED_ALIASES = ("predicted", "prediction", "pred", "y_pred")
_POINT_REFERENCE_ALIASES = ("reference", "truth", "target", "y_true")


@dataclass(frozen=True)
class Emb34umAlVsLhsValidationArtifacts:
    artifact_dir: Path
    manifest_path: Path
    summary_csv_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    manifest: dict[str, Any]
    summary_rows: tuple[dict[str, Any], ...]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_rows(rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if isinstance(rows, (str, Path)):
        path = Path(rows)
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                return tuple(dict(row) for row in csv.DictReader(handle))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return _coerce_rows(loaded)
    if isinstance(rows, Mapping):
        for key in ("curve_rows", "rows", "records"):
            value = rows.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return tuple(dict(item) for item in value if isinstance(item, Mapping))
        raise ValueError("row payload mapping must contain curve_rows, rows, or records.")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("rows must be a sequence, mapping, JSON path, or CSV path.")
    return tuple(dict(item) for item in rows if isinstance(item, Mapping))


def _strategy(row: Mapping[str, Any]) -> str | None:
    raw = (
        str(row.get("strategy") or row.get("cohort") or row.get("source") or row.get("gate") or "")
        .strip()
        .lower()
    )
    if raw in {"al", "active_learning", "active-learning", "full", "full_gate"}:
        return "al"
    if raw in {"lhs", "lhs_gate", "lhs_comparator", "comparator"}:
        return "lhs"
    candidate_id = str(row.get("candidate_id") or row.get("curve_id") or "").lower()
    if "-lhs-" in candidate_id or candidate_id.startswith("lhs"):
        return "lhs"
    if "-full-" in candidate_id or "-r01-" in candidate_id or "-r02-" in candidate_id or "-r03-" in candidate_id:
        return "al"
    return None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _parse_number_sequence(value: object) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as exc:
                if not text.endswith("]"):
                    raise ValueError("bracketed curve sequence must be closed.") from exc
                tokens = [token for token in text[1:-1].replace(";", " ").replace(",", " ").split() if token]
                return tuple(_finite_float(token, label="curve value") for token in tokens)
            if not isinstance(loaded, Sequence) or isinstance(loaded, (str, bytes, bytearray)):
                raise ValueError("bracketed curve sequence must decode to a sequence.")
            return tuple(_finite_float(item, label="curve value") for item in loaded)
        tokens = [token for token in text.replace(";", " ").replace(",", " ").split() if token]
        return tuple(_finite_float(token, label="curve value") for token in tokens)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_finite_float(item, label="curve value") for item in value)
    return None


def _first_sequence(row: Mapping[str, Any], aliases: Sequence[str]) -> tuple[float, ...] | None:
    for alias in aliases:
        if alias in row:
            parsed = _parse_number_sequence(row.get(alias))
            if parsed is not None:
                return parsed
    return None


def _first_scalar(row: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    for alias in aliases:
        if alias in row and row.get(alias) not in (None, ""):
            return _finite_float(row.get(alias), label=alias)
    return None


def _curve_rel_l2_pct(predicted: Sequence[float], reference: Sequence[float]) -> float:
    if len(predicted) != len(reference) or not predicted:
        raise ValueError("predicted and reference curves must have the same non-zero length.")
    residual_sq = sum((float(a) - float(b)) ** 2 for a, b in zip(predicted, reference))
    reference_sq = sum(float(item) ** 2 for item in reference)
    if reference_sq <= 0.0:
        raise ValueError("reference curve L2 norm must be positive.")
    return float(100.0 * math.sqrt(residual_sq) / math.sqrt(reference_sq))


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute median of an empty sequence.")
    ordered = sorted(float(item) for item in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _curve_id(row: Mapping[str, Any], *, index: int) -> str:
    for key in ("curve_id", "candidate_id", "sample_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"row_{index:06d}"


def _row_order(row: Mapping[str, Any], *, index: int) -> int:
    for key in ("order", "curve_index", "candidate_index", "f_delta_row_index", "row_index"):
        value = _optional_int(row.get(key))
        if value is not None:
            return value
    return index


def _round_sort_value(value: object) -> int:
    parsed = _optional_int(value)
    return parsed if parsed is not None else 999999


def _extract_curve_records(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    direct_records: list[dict[str, Any]] = []
    grouped_points: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for index, row in enumerate(rows, start=1):
        strategy = _strategy(row)
        if strategy is None:
            skip("missing_strategy")
            continue
        curve_id = _curve_id(row, index=index)
        round_index = _optional_int(row.get("round"))
        order = _row_order(row, index=index)
        metric_value = None
        invalid_curve_metric = False
        for alias in _CURVE_METRIC_ALIASES:
            if alias in row and row.get(alias) not in (None, ""):
                try:
                    metric_value = _finite_float(row.get(alias), label=alias)
                except ValueError:
                    skip("invalid_curve_metric")
                    invalid_curve_metric = True
                break
        if invalid_curve_metric:
            continue
        if metric_value is not None:
            direct_records.append(
                {
                    "strategy": strategy,
                    "round": round_index,
                    "curve_id": curve_id,
                    "order": order,
                    "curve_rel_l2_pct": metric_value,
                }
            )
            continue

        try:
            predicted = _first_sequence(row, _PREDICTED_ALIASES)
            reference = _first_sequence(row, _REFERENCE_ALIASES)
        except ValueError:
            skip("invalid_curve_arrays")
            continue
        if predicted is not None or reference is not None:
            if predicted is None or reference is None:
                skip("incomplete_curve_arrays")
                continue
            try:
                metric_value = _curve_rel_l2_pct(predicted, reference)
            except ValueError:
                skip("invalid_curve_arrays")
                continue
            direct_records.append(
                {
                    "strategy": strategy,
                    "round": round_index,
                    "curve_id": curve_id,
                    "order": order,
                    "curve_rel_l2_pct": metric_value,
                }
            )
            continue

        try:
            predicted_point = _first_scalar(row, _POINT_PREDICTED_ALIASES)
            reference_point = _first_scalar(row, _POINT_REFERENCE_ALIASES)
        except ValueError:
            skip("invalid_point_curve")
            continue
        if predicted_point is None or reference_point is None:
            skip("missing_curve_metric_inputs")
            continue
        key = (strategy, str(round_index), curve_id)
        group = grouped_points.setdefault(
            key,
            {
                "strategy": strategy,
                "round": round_index,
                "curve_id": curve_id,
                "order": order,
                "points": [],
            },
        )
        group["points"].append((row.get("force", row.get("axis", index)), predicted_point, reference_point))

    for group in grouped_points.values():
        try:
            points = sorted(group["points"], key=lambda item: _finite_float(item[0], label="point axis"))
        except ValueError:
            skip("invalid_point_axis")
            continue
        predicted = [float(item[1]) for item in points]
        reference = [float(item[2]) for item in points]
        try:
            metric_value = _curve_rel_l2_pct(predicted, reference)
        except ValueError:
            skip("invalid_point_curve")
            continue
        direct_records.append(
            {
                "strategy": group["strategy"],
                "round": group["round"],
                "curve_id": group["curve_id"],
                "order": group["order"],
                "curve_rel_l2_pct": metric_value,
            }
        )

    records = tuple(
        sorted(direct_records, key=lambda item: (item["strategy"], _round_sort_value(item["round"]), item["order"], item["curve_id"]))
    )
    return records, skipped


def _derive_prefix_counts(al_records: Sequence[Mapping[str, Any]], lhs_records: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    rounds = sorted({int(item["round"]) for item in al_records if item.get("round") is not None})
    if rounds:
        counts: list[int] = []
        for round_index in rounds:
            count = sum(1 for item in al_records if item.get("round") is not None and int(item["round"]) <= round_index)
            if count > len(lhs_records):
                break
            counts.append(count)
        return tuple(counts)
    return (min(len(al_records), len(lhs_records)),) if al_records and lhs_records else tuple()


def _build_summary_rows(
    *,
    curve_records: Sequence[Mapping[str, Any]],
    prefix_curve_counts: Sequence[int] | None,
) -> tuple[dict[str, Any], ...]:
    al_records = [item for item in curve_records if item["strategy"] == "al"]
    lhs_records = [item for item in curve_records if item["strategy"] == "lhs"]
    counts = tuple(int(item) for item in prefix_curve_counts) if prefix_curve_counts is not None else _derive_prefix_counts(al_records, lhs_records)
    rows: list[dict[str, Any]] = []
    for prefix_index, count in enumerate(counts, start=1):
        if count <= 0:
            continue
        if count > len(al_records) or count > len(lhs_records):
            continue
        al_prefix = al_records[:count]
        lhs_prefix = lhs_records[:count]
        if not al_prefix or not lhs_prefix:
            continue
        al_median = _median([float(item["curve_rel_l2_pct"]) for item in al_prefix])
        lhs_median = _median([float(item["curve_rel_l2_pct"]) for item in lhs_prefix])
        max_al_round = max((int(item["round"]) for item in al_prefix if item.get("round") is not None), default=prefix_index)
        rows.append(
            {
                "prefix": prefix_index,
                "al_round_prefix": max_al_round,
                "prefix_curve_count": count,
                "al_curve_count": len(al_prefix),
                "lhs_curve_count": len(lhs_prefix),
                "al_median_curve_rel_l2_pct": al_median,
                "lhs_median_curve_rel_l2_pct": lhs_median,
                "median_delta": al_median - lhs_median,
                "median_improved": al_median < lhs_median,
            }
        )
    return tuple(rows)


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
            row.extend((245, 245, 245, 255) if (x + y) % 2 else (70, 115, 160, 255))
        rows.append(bytes(row))
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", zlib.compress(b"".join(rows))) + _png_chunk(b"IEND", b"")


def _write_plot(path: Path, summary_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not summary_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    x = [int(row["al_round_prefix"]) for row in summary_rows]
    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(x, [float(row["al_median_curve_rel_l2_pct"]) for row in summary_rows], marker="o", label="AL prefix")
    axis.plot(x, [float(row["lhs_median_curve_rel_l2_pct"]) for row in summary_rows], marker="s", label="LHS prefix")
    axis.set_title("EMB 3.4um AL vs LHS validation")
    axis.set_xlabel("AL round prefix")
    axis.set_ylabel("median_curve_rel_l2_pct")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prefix",
        "al_round_prefix",
        "prefix_curve_count",
        "al_curve_count",
        "lhs_curve_count",
        "al_median_curve_rel_l2_pct",
        "lhs_median_curve_rel_l2_pct",
        "median_delta",
        "median_improved",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_emb_34um_al_vs_lhs_validation_report(
    *,
    curve_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    prefix_curve_counts: Sequence[int] | None = None,
    adaptive_acquisition_available: bool = False,
    acquisition_engine_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    rows = _coerce_rows(curve_rows)
    curve_records, skipped = _extract_curve_records(rows)
    summary_rows = _build_summary_rows(curve_records=curve_records, prefix_curve_counts=prefix_curve_counts)
    al_count = sum(1 for item in curve_records if item["strategy"] == "al")
    lhs_count = sum(1 for item in curve_records if item["strategy"] == "lhs")

    blockers: list[str] = []
    limitations: list[str] = []
    if not curve_records:
        blockers.append("No usable curve-level validation metrics were available.")
    if al_count == 0:
        blockers.append("No AL curve rows with curve metrics were available.")
    if lhs_count == 0:
        blockers.append("No LHS comparator curve rows with curve metrics were available.")
    if not summary_rows:
        blockers.append("No paired AL/LHS prefix summary could be computed.")
    if not adaptive_acquisition_available:
        limitations.append(
            "Adaptive acquisition engine was not supplied; AL prefixes are evaluated from provided rows only."
        )

    manifest = {
        "schema_version": EMB_34UM_AL_VS_LHS_VALIDATION_SCHEMA_VERSION,
        "experiment": "indentation",
        "diameter_um": "3.4",
        "primary_metric": "median_curve_rel_l2_pct",
        "source_row_count": len(rows),
        "usable_curve_count": len(curve_records),
        "al_curve_count": al_count,
        "lhs_curve_count": lhs_count,
        "skipped_row_reasons": skipped,
        "prefix_curve_counts": [int(item["prefix_curve_count"]) for item in summary_rows],
        "adaptive_acquisition": {
            "available": bool(adaptive_acquisition_available),
            "engine": str(acquisition_engine_name or ""),
            "status": "available" if adaptive_acquisition_available else "not_available",
            "note": (
                "A real acquisition engine can provide selected AL rows before this validation step."
                if adaptive_acquisition_available
                else "No adaptive acquisition engine was supplied; this module only validates provided AL and LHS rows."
            ),
        },
        "summary_rows": list(summary_rows),
        "curve_records": list(curve_records),
        "blockers": blockers,
        "limitations": limitations,
        "status": "ready" if summary_rows else "blocked",
        "metadata": dict(metadata or {}),
    }
    return manifest, summary_rows


def write_emb_34um_al_vs_lhs_validation_artifacts(
    *,
    curve_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    output_root: str | Path,
    prefix_curve_counts: Sequence[int] | None = None,
    adaptive_acquisition_available: bool = False,
    acquisition_engine_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_plot: bool = True,
) -> Emb34umAlVsLhsValidationArtifacts:
    artifact_dir = Path(output_root)
    manifest_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME
    summary_csv_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME
    plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME
    plot_sidecar_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(
        curve_rows=curve_rows,
        prefix_curve_counts=prefix_curve_counts,
        adaptive_acquisition_available=adaptive_acquisition_available,
        acquisition_engine_name=acquisition_engine_name,
        metadata=metadata,
    )
    _write_summary_csv(summary_csv_path, summary_rows)
    _write_plot(plot_path, summary_rows, include_plot=include_plot)
    manifest["plot_paths"] = {
        "al_vs_lhs_validation": str(plot_path),
        "al_vs_lhs_validation_sidecar": str(plot_sidecar_path),
    }
    _write_json(manifest_path, manifest)
    _write_json(plot_sidecar_path, {"plot_path": str(plot_path), **manifest})

    return Emb34umAlVsLhsValidationArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        summary_csv_path=summary_csv_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        manifest=manifest,
        summary_rows=summary_rows,
    )


__all__ = [
    "EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_SCHEMA_VERSION",
    "EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME",
    "Emb34umAlVsLhsValidationArtifacts",
    "build_emb_34um_al_vs_lhs_validation_report",
    "write_emb_34um_al_vs_lhs_validation_artifacts",
]
