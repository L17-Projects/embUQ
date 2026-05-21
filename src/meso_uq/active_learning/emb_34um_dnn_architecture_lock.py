from __future__ import annotations

"""Architecture-lock manifest/report builder for EMB 3.4um DNN causal protocol."""

import json
import math
import statistics
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND,
    EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
    validate_dnn_causal_ensemble_size,
    validate_dnn_causal_timing_canary,
)

EMB_34UM_DNN_ARCHITECTURE_LOCK_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_architecture_lock.v1"
)

EMB_34UM_DNN_ARCHITECTURE_LOCK_MANIFEST_FILENAME = (
    "emb_34um_dnn_causal_architecture_lock_manifest.json"
)
EMB_34UM_DNN_ARCHITECTURE_LOCK_REPORT_FILENAME = (
    "emb_34um_dnn_causal_architecture_lock_report.json"
)
EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_FILENAME = (
    "emb_34um_dnn_causal_architecture_lock_training_validation_loss.png"
)
EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_architecture_lock_training_validation_loss.png.json"
)

EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE = "dnn_mlp_64_64"
EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_SEEDS = tuple(range(1, EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE + 1))
EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_TRAINING_HYPERPARAMETERS = {
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 200,
}


@dataclass(frozen=True)
class Emb34umDnnArchitectureLockArtifacts:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _coerce_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty.")
    return text


def _coerce_non_negative_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _coerce_positive_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number <= 0:
        raise ValueError(f"{label} must be positive.")
    return number


def _coerce_positive_float(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    if number <= 0:
        raise ValueError(f"{label} must be positive.")
    return number


def _coerce_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _coerce_sequence(value: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(value)


def _coerce_source_evidence(
    source_evidence: Mapping[str, Any] | None,
    *,
    selected_architecture: str,
) -> dict[str, Any]:
    evidence = {
        "selection_backend": EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND,
        "selected_architecture": selected_architecture,
        "recorded_by": "architecture_lock_module",
    }
    if source_evidence is not None:
        evidence.update(_coerce_mapping(source_evidence, label="source_evidence"))
    return evidence


def _coerce_training_hyperparameters(
    training_hyperparameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    hyperparameters = dict(EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_TRAINING_HYPERPARAMETERS)
    if training_hyperparameters is None:
        return hyperparameters
    values = _coerce_mapping(training_hyperparameters, label="training_hyperparameters")
    hyperparameters.update(values)
    # Validate a small set of expected tuning knobs.
    hyperparameters["learning_rate"] = _coerce_positive_float(
        hyperparameters["learning_rate"],
        label="training_hyperparameters.learning_rate",
    )
    hyperparameters["batch_size"] = _coerce_positive_int(
        hyperparameters["batch_size"],
        label="training_hyperparameters.batch_size",
    )
    hyperparameters["epochs"] = _coerce_positive_int(
        hyperparameters["epochs"],
        label="training_hyperparameters.epochs",
    )
    return hyperparameters


def _coerce_ensemble_seeds(seeds: Sequence[int] | None, *, expected: int) -> tuple[int, ...]:
    if seeds is None:
        return tuple(EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_SEEDS)
    normalized = tuple(
        _coerce_non_negative_int(seed, label="ensemble_seed")
        for seed in _coerce_sequence(seeds, label="ensemble_seeds")
    )
    if not normalized:
        raise ValueError("ensemble_seeds must be a non-empty sequence.")
    if len(normalized) != expected:
        raise ValueError(f"ensemble_seeds must contain exactly {expected} entries.")
    unique = sorted(set(normalized))
    if len(unique) != len(normalized):
        raise ValueError("ensemble_seeds must be unique.")
    return tuple(normalized)


def _coerce_training_history(training_history: Sequence[Mapping[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    if training_history is None:
        return ()
    rows = _coerce_sequence(training_history, label="training_history")
    normalized: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows, start=1):
        row = _coerce_mapping(raw_row, label=f"training_history[{index}]")
        normalized.append(
            {
                "epoch": _coerce_non_negative_int(
                    row.get("epoch", index),
                    label=f"training_history[{index}].epoch",
                ),
                "train_loss": _coerce_positive_float(
                    row.get("train_loss"),
                    label=f"training_history[{index}].train_loss",
                ),
                "validation_loss": _coerce_positive_float(
                    row.get("validation_loss"),
                    label=f"training_history[{index}].validation_loss",
                ),
            }
        )
    return tuple(normalized)


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + type_code
        + data
        + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)
    )


def _build_fallback_png() -> bytes:
    width = 16
    height = 16
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    rows = bytearray()
    for row_index in range(height):
        rows.append(0)
        for column_index in range(width):
            rows.extend(
                (233, 236, 239, 255)
                if (row_index + column_index) % 2 == 0
                else (180, 190, 198, 255)
            )
    idat_data = zlib.compress(bytes(rows))
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", idat_data) + _png_chunk(b"IEND", b"")


_FALLBACK_PNG = _build_fallback_png()


def _build_training_history_summary(history: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not history:
        return {
            "epoch_count": 0,
            "train_loss_final": 0.0,
            "validation_loss_final": 0.0,
            "train_loss_min": 0.0,
            "validation_loss_min": 0.0,
            "validation_gap_final": 0.0,
        }
    train_losses = tuple(float(item["train_loss"]) for item in history)
    validation_losses = tuple(float(item["validation_loss"]) for item in history)
    train_best = min(train_losses)
    validation_best = min(validation_losses)
    return {
        "epoch_count": len(history),
        "train_loss_final": train_losses[-1],
        "validation_loss_final": validation_losses[-1],
        "train_loss_min": train_best,
        "validation_loss_min": validation_best,
        "validation_gap_final": validation_losses[-1] - train_losses[-1],
    }


def _plot_training_validation_loss(
    path: Path,
    *,
    training_history: Sequence[Mapping[str, Any]],
    include_plot: bool,
) -> dict[str, Any]:
    summary = _build_training_history_summary(training_history)
    if not include_plot or not training_history:
        path.write_bytes(_FALLBACK_PNG)
        return {"status": "fallback_png", "reason": "disabled_or_no_history", **summary}
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return {"status": "fallback_png", "reason": "missing_matplotlib", **summary}

    x_axis = [float(item["epoch"]) for item in training_history]
    y_train = [float(item["train_loss"]) for item in training_history]
    y_validation = [float(item["validation_loss"]) for item in training_history]
    fig, axis = plt.subplots(1, 1, figsize=(7.0, 4.0))
    axis.plot(x_axis, y_train, marker="o", label="training loss")
    axis.plot(x_axis, y_validation, marker="s", label="validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("DNN architecture lock training evidence")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"status": "rendered", **summary}


def build_emb_34um_dnn_causal_architecture_lock_manifest(
    *,
    selected_architecture: str = EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE,
    ensemble_size: int = EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    ensemble_seeds: Sequence[int] | None = None,
    training_hyperparameters: Mapping[str, Any] | None = None,
    timing_canary: Mapping[str, Any] | None = None,
    source_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_architecture = _coerce_text(selected_architecture, label="selected_architecture")
    if selected_architecture != EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE:
        raise ValueError(
            "selected_architecture must use the fixed DNN architecture "
            f"{EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE!r}."
        )

    validated_ensemble_size = validate_dnn_causal_ensemble_size(ensemble_size)
    seeds = _coerce_ensemble_seeds(ensemble_seeds, expected=validated_ensemble_size)
    hyperparameters = _coerce_training_hyperparameters(training_hyperparameters)
    canary = validate_dnn_causal_timing_canary(dict(timing_canary or {}))
    evidence = _coerce_source_evidence(
        source_evidence,
        selected_architecture=selected_architecture,
    )
    return {
        "schema_version": EMB_34UM_DNN_ARCHITECTURE_LOCK_SCHEMA_VERSION,
        "selected_architecture": selected_architecture,
        "ensemble_size": validated_ensemble_size,
        "ensemble_seeds": list(seeds),
        "training_hyperparameters": hyperparameters,
        "timing_canary": canary,
        "source_evidence": evidence,
    }


def write_emb_34um_dnn_causal_architecture_lock_artifacts(
    output_root: str | Path,
    *,
    selected_architecture: str = EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE,
    ensemble_size: int = EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    ensemble_seeds: Sequence[int] | None = None,
    training_hyperparameters: Mapping[str, Any] | None = None,
    timing_canary: Mapping[str, Any] | None = None,
    source_evidence: Mapping[str, Any] | None = None,
    training_history: Sequence[Mapping[str, Any]] | None = None,
    include_plot: bool = True,
) -> Emb34umDnnArchitectureLockArtifacts:
    output_root = Path(output_root)
    manifest = build_emb_34um_dnn_causal_architecture_lock_manifest(
        selected_architecture=selected_architecture,
        ensemble_size=ensemble_size,
        ensemble_seeds=ensemble_seeds,
        training_hyperparameters=training_hyperparameters,
        timing_canary=timing_canary,
        source_evidence=source_evidence,
    )
    normalized_history = _coerce_training_history(training_history)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / EMB_34UM_DNN_ARCHITECTURE_LOCK_MANIFEST_FILENAME
    report_path = output_root / EMB_34UM_DNN_ARCHITECTURE_LOCK_REPORT_FILENAME
    plot_path = output_root / EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_FILENAME
    plot_sidecar_path = output_root / EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_SIDECAR_FILENAME

    loss_summary = _build_training_history_summary(normalized_history)
    plot_sidecar = _plot_training_validation_loss(
        plot_path,
        training_history=normalized_history,
        include_plot=include_plot,
    )
    plot_sidecar.update(
        {
            "plot": str(plot_path.name),
            "evidence_summary": {
                "selected_architecture": manifest["selected_architecture"],
                "ensemble_size": manifest["ensemble_size"],
                "training_hyperparameters": manifest["training_hyperparameters"],
                **loss_summary,
            },
        }
    )

    report = {
        "schema_version": EMB_34UM_DNN_ARCHITECTURE_LOCK_SCHEMA_VERSION,
        "selected_architecture": manifest["selected_architecture"],
        "ensemble_signature": {
            "count": manifest["ensemble_size"],
            "seeds_checksum": statistics.fsum(float(seed) for seed in manifest["ensemble_seeds"]),
            "seeds": manifest["ensemble_seeds"],
        },
        "timing_canary": manifest["timing_canary"],
        "training_history_summary": loss_summary,
        "manifest_path": str(manifest_path),
        "plot_path": str(plot_path),
        "plot_sidecar": str(plot_sidecar_path),
        "source_evidence": manifest["source_evidence"],
    }
    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    _write_json(plot_sidecar_path, plot_sidecar)

    return Emb34umDnnArchitectureLockArtifacts(
        artifact_dir=output_root,
        manifest_path=manifest_path,
        report_path=report_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        manifest=manifest,
        report=report,
    )


__all__ = [
    "Emb34umDnnArchitectureLockArtifacts",
    "EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_SEEDS",
    "EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_TRAINING_HYPERPARAMETERS",
    "EMB_34UM_DNN_ARCHITECTURE_LOCK_MANIFEST_FILENAME",
    "EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_FILENAME",
    "EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_ARCHITECTURE_LOCK_REPORT_FILENAME",
    "EMB_34UM_DNN_ARCHITECTURE_LOCK_SCHEMA_VERSION",
    "EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE",
    "build_emb_34um_dnn_causal_architecture_lock_manifest",
    "write_emb_34um_dnn_causal_architecture_lock_artifacts",
]
