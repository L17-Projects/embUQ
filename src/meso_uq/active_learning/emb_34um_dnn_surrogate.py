from __future__ import annotations

"""DNN surrogate helpers for the EMB 3.4um causal validation workflow."""

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_FORBIDDEN_SELECTOR_TOKENS,
    validate_dnn_causal_ensemble_size,
    validate_dnn_causal_force_grid,
)

EMB_34UM_DNN_SURROGATE_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_surrogate.v1"
EMB_34UM_DNN_SURROGATE_BACKEND = "fixed_architecture_dnn"
EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE = "dnn_mlp_64_64"
EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT = 0.25
EMB_34UM_DNN_SURROGATE_MEMBER_PREFIX = "emb_34um_dnn_surrogate_member"

EMB_34UM_DNN_SURROGATE_MANIFEST_FILENAME = "emb_34um_dnn_surrogate_training_manifest.json"
EMB_34UM_DNN_SURROGATE_REPORT_FILENAME = "emb_34um_dnn_surrogate_training_report.json"


@dataclass(frozen=True)
class DnnSurrogateLongRow:
    curve_id: str
    ka_log10: float
    kb_log10: float
    force: float
    force_norm: float
    observable: float


@dataclass(frozen=True)
class Emb34umDnnSurrogateFit:
    architecture: str
    backend: str
    ensemble_seeds: tuple[int, ...]
    ensemble_checkpoints: tuple[Path, ...]
    ensemble_member_count: int
    train_runtime_seconds: float
    score_runtime_seconds: float
    feature_mean: tuple[float, float, float]
    feature_scale: tuple[float, float, float]
    target_mean: float
    target_scale: float
    force_min: float
    force_max: float
    force_grid: tuple[float, ...]
    device: str
    train_row_count: int
    validation_row_count: int
    model_path: Path
    training_history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _EnsembleMember:
    seed: int
    checkpoint_path: Path
    train_loss_history: tuple[float, ...]
    validation_loss_history: tuple[float, ...]
    train_runtime_seconds: float


def _coerce_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string.")
    return text


def _coerce_float(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _coerce_positive_float(value: object, *, label: str) -> float:
    number = _coerce_float(value, label=label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return number


def _coerce_non_negative_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _coerce_positive_int(value: object, *, label: str) -> int:
    number = _coerce_non_negative_int(value, label=label)
    if number <= 0:
        raise ValueError(f"{label} must be positive.")
    return number


def _coerce_sequence(values: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(values)


def _coerce_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _coerce_float_sequence(values: object, *, label: str) -> tuple[float, ...]:
    parsed = _coerce_sequence(values, label=label)
    if not parsed:
        raise ValueError(f"{label} must not be empty.")
    return tuple(_coerce_float(item, label=f"{label}[{index}]") for index, item in enumerate(parsed, start=1))


def _row_label(index: int) -> str:
    return f"rows[{index}]"


def _coerce_curve_id(row: Mapping[str, Any], *, row_index: int) -> str:
    for key in ("curve_id", "candidate_id", "sample_id", "source"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return f"curve_{row_index:06d}"


def _coerce_ka_kb(row: Mapping[str, Any], *, row_index: int) -> tuple[float, float]:
    parameters = row.get("parameters")
    ka = None
    kb = None
    if isinstance(parameters, Mapping):
        if "ka" in parameters:
            ka = _coerce_positive_float(parameters["ka"], label=f"{_row_label(row_index)}.parameters.ka")
        if "kb" in parameters:
            kb = _coerce_positive_float(parameters["kb"], label=f"{_row_label(row_index)}.parameters.kb")
    if ka is None:
        for key in ("ka", "Yt", "yt"):
            if key in row:
                ka = _coerce_positive_float(row[key], label=f"{_row_label(row_index)}.{key}")
                break
    if kb is None:
        for key in ("kb", "kb_scale"):
            if key in row:
                kb = _coerce_positive_float(row[key], label=f"{_row_label(row_index)}.{key}")
                break
    if ka is None:
        raise ValueError(f"{_row_label(row_index)} is missing ka.")
    if kb is None:
        raise ValueError(f"{_row_label(row_index)} is missing kb.")
    return ka, kb


def _coerce_force_grid_from_row(row: Mapping[str, Any], *, row_index: int) -> tuple[float, ...]:
    for key in ("force_grid", "force", "forces", "axis", "force_axis"):
        if key in row:
            return _coerce_float_sequence(row[key], label=f"{_row_label(row_index)}.{key}")
    raise ValueError(f"{_row_label(row_index)} is missing force grid.")


def _coerce_observable(row: Mapping[str, Any], *, row_index: int) -> tuple[float, ...]:
    for key in (
        "reference_curve",
        "output_curve",
        "outputs",
        "displacement",
        "curve",
        "y",
        "target_curve",
        "curve_values",
    ):
        if key in row:
            return _coerce_float_sequence(row[key], label=f"{_row_label(row_index)}.{key}")
    raise ValueError(f"{_row_label(row_index)} is missing target curve.")


def _normalize_force(force: float, *, force_min: float, force_max: float) -> float:
    width = force_max - force_min
    if math.isclose(width, 0.0):
        return 0.0
    return (force - force_min) / width


def _assert_no_lightweight_selector_tokens(value: str) -> None:
    normalized = str(value).strip().lower()
    for token in EMB_34UM_DNN_CAUSAL_FORBIDDEN_SELECTOR_TOKENS:
        if token in normalized:
            raise ValueError("DNN causal stage must not use lightweight NumPy/ridge selector paths.")


def _check_architecture(architecture: str) -> str:
    value = _coerce_text(architecture, label="architecture")
    _assert_no_lightweight_selector_tokens(value)
    if value != EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE:
        raise ValueError(
            "selected architecture is fixed for causal DNN stage: "
            f"{EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE!r}."
        )
    return value


def _coerce_ensemble_seeds(seeds: Sequence[int] | None, *, size: int) -> tuple[int, ...]:
    if seeds is None:
        return tuple(range(1, size + 1))
    normalized = tuple(
        _coerce_non_negative_int(seed, label=f"ensemble_seeds[{index}]")
        for index, seed in enumerate(_coerce_sequence(seeds, label="ensemble_seeds"), start=1)
    )
    if len(normalized) != size:
        raise ValueError(f"ensemble_seeds must contain exactly {size} values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("ensemble_seeds must be unique.")
    return normalized


def convert_completed_curves_to_long_rows(
    completed_curve_rows: Sequence[Mapping[str, Any]],
    *,
    force_grid: Sequence[float] | None = None,
) -> tuple[DnnSurrogateLongRow, ...]:
    source_rows = _coerce_sequence(completed_curve_rows, label="completed_curve_rows")
    if not source_rows:
        return ()

    requested_grid = _coerce_float_sequence(force_grid, label="force_grid") if force_grid is not None else None
    expected_grid = requested_grid
    long_rows: list[DnnSurrogateLongRow] = []

    for index, raw_row in enumerate(source_rows, start=1):
        row = _coerce_mapping(raw_row, label=_row_label(index))
        ka, kb = _coerce_ka_kb(row, row_index=index)
        row_force_grid = _coerce_force_grid_from_row(row, row_index=index)
        observable = _coerce_observable(row, row_index=index)

        if len(row_force_grid) != len(observable):
            raise ValueError(f"{_row_label(index)} force and observable lengths differ.")

        if expected_grid is None:
            expected_grid = row_force_grid
        elif len(expected_grid) != len(row_force_grid) or any(
            not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in zip(row_force_grid, expected_grid)
        ):
            raise ValueError("all completed curves must share the same force grid.")

        force_min = float(expected_grid[0])
        force_max = float(expected_grid[-1])
        ka_log10 = math.log10(ka)
        kb_log10 = math.log10(kb)
        curve_id = _coerce_curve_id(row, row_index=index)

        for force, value in zip(expected_grid, observable):
            long_rows.append(
                DnnSurrogateLongRow(
                    curve_id=curve_id,
                    ka_log10=ka_log10,
                    kb_log10=kb_log10,
                    force=float(force),
                    force_norm=_normalize_force(float(force), force_min=force_min, force_max=force_max),
                    observable=float(value),
                )
            )

    if expected_grid is None:
        return ()
    validate_dnn_causal_force_grid(expected_grid)
    return tuple(long_rows)


def _rows_to_arrays(rows: Sequence[DnnSurrogateLongRow]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not rows:
        raise ValueError("rows must not be empty.")
    x = np.asarray([(row.ka_log10, row.kb_log10, row.force_norm) for row in rows], dtype=float)
    y = np.asarray([row.observable for row in rows], dtype=float)
    curve_ids = tuple(row.curve_id for row in rows)
    return x, y, curve_ids


def _scaler_from_rows(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float, float]:
    feature_mean = tuple(float(value) for value in features.mean(axis=0))
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    target_mean = float(targets.mean())
    target_scale = float(targets.std())
    if not math.isfinite(target_scale) or math.isclose(target_scale, 0.0):
        target_scale = 1.0
    return feature_mean, tuple(float(value) for value in feature_scale), target_mean, target_scale


def _normalize_features(
    features: np.ndarray,
    *,
    feature_mean: tuple[float, float, float],
    feature_scale: tuple[float, float, float],
) -> np.ndarray:
    return (features - np.asarray(feature_mean, dtype=float)) / np.asarray(feature_scale, dtype=float)


def _denormalize_outputs(normalized: np.ndarray, *, target_mean: float, target_scale: float) -> np.ndarray:
    return normalized * target_scale + target_mean


def _coerce_device(device: str | None) -> str:
    normalized = str(device or "cpu").strip().lower()
    if normalized in {"", "auto", "cpu"}:
        return "cpu"
    if normalized in {"gpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    raise ValueError("device must be one of cpu, auto, gpu, cuda, cuda:N.")


def _coerce_output_root(output_root: Path | str | None) -> Path:
    if output_root is None:
        raise ValueError("output_root is required.")
    return Path(output_root)


def _train_test_split_indices(count: int, seed: int, *, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("at least two rows are required for train/validation split.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0, 1).")

    indices = np.arange(count)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    validation_count = int(round(count * validation_fraction))
    validation_count = max(1, min(validation_count, count - 1))
    return indices[:-validation_count], indices[-validation_count:]


def _build_dnn_model(seed: int) -> Any:
    import torch

    from meso_uq.surrogate.model import MLP, init_weights

    torch.manual_seed(int(seed))
    model = MLP(3, 1, [64, 64])
    model.apply(init_weights)
    return model


def _save_checkpoint(
    model: Any,
    *,
    feature_mean: tuple[float, float, float],
    feature_scale: tuple[float, float, float],
    target_mean: float,
    target_scale: float,
    path: Path,
) -> None:
    from meso_uq.surrogate.model import save_model_states

    path.parent.mkdir(parents=True, exist_ok=True)
    save_model_states(
        model,
        xshift=feature_mean,
        xscale=feature_scale,
        yshift=target_mean,
        yscale=target_scale,
        path=str(path),
    )


def _load_checkpoint(path: Path) -> tuple[Any, tuple[float, float, float], tuple[float, float, float], float, float]:
    from meso_uq.surrogate.model import load_model_states

    model, xshift, xscale, yshift, yscale = load_model_states(str(path))
    return (
        model,
        tuple(float(value) for value in xshift),
        tuple(float(value) for value in xscale),
        float(yshift),
        float(yscale),
    )


def _train_single_member(
    *,
    rows: Sequence[DnnSurrogateLongRow],
    seed: int,
    architecture: str,
    output_root: Path,
    index: int,
    feature_mean: tuple[float, float, float],
    feature_scale: tuple[float, float, float],
    target_mean: float,
    target_scale: float,
    validation_fraction: float,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    device: str,
) -> tuple[_EnsembleMember, Any]:
    if architecture != EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE:
        raise ValueError(f"unsupported architecture {architecture!r}.")

    all_features, all_targets, _ = _rows_to_arrays(rows)
    train_index, validation_index = _train_test_split_indices(
        len(all_features),
        int(seed),
        validation_fraction=validation_fraction,
    )

    normalized_features = _normalize_features(
        all_features,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    normalized_targets = (all_targets - target_mean) / target_scale

    x_train = normalized_features[train_index]
    y_train = normalized_targets[train_index]
    x_valid = normalized_features[validation_index]
    y_valid = normalized_targets[validation_index]

    import torch

    resolved_device = torch.device(device)
    torch.manual_seed(int(seed))
    model = _build_dnn_model(seed).to(resolved_device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1)
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=max(1, min(int(batch_size), len(train_dataset))),
        generator=generator,
    )
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32, device=resolved_device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32, device=resolved_device).unsqueeze(1)

    train_losses: list[float] = []
    validation_losses: list[float] = []

    for _ in range(max(1, int(epochs))):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(resolved_device)
            y_batch = y_batch.to(resolved_device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1
        train_losses.append(float(running_loss / batch_count) if batch_count else 0.0)

        model.eval()
        with torch.no_grad():
            y_valid_pred = model(x_valid_tensor)
            validation_losses.append(float(criterion(y_valid_pred, y_valid_tensor).item()))

    checkpoint_path = output_root / "members" / f"{EMB_34UM_DNN_SURROGATE_MEMBER_PREFIX}_{index:02d}_seed_{seed}.pt"
    _save_checkpoint(
        model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        path=checkpoint_path,
    )
    return (
        _EnsembleMember(
            seed=int(seed),
            checkpoint_path=checkpoint_path,
            train_loss_history=tuple(train_losses),
            validation_loss_history=tuple(validation_losses),
            train_runtime_seconds=0.0,
        ),
        model,
    )


def _predict_member_rows(model: Any, rows: np.ndarray, *, scaler: Any) -> np.ndarray:
    import torch

    model.eval()
    feature_mean, feature_scale, target_mean, target_scale = scaler
    normalized = _normalize_features(rows, feature_mean=feature_mean, feature_scale=feature_scale)
    device = next(model.parameters()).device
    x = torch.tensor(normalized, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(x).squeeze(1).detach().cpu().numpy()
    return _denormalize_outputs(y, target_mean=target_mean, target_scale=target_scale)


def _load_ensemble_members(
    fit: Emb34umDnnSurrogateFit,
) -> tuple[tuple[Any, tuple[float, float, float], tuple[float, float, float], float, float], ...]:
    if not fit.ensemble_checkpoints:
        raise ValueError("ensemble_checkpoints is empty.")
    return tuple(_load_checkpoint(Path(path)) for path in fit.ensemble_checkpoints)


def load_ensemble_member_predictions(
    fit: Emb34umDnnSurrogateFit,
    force_values: Sequence[float],
    *,
    ka: float,
    kb: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    members = _load_ensemble_members(fit)
    force_axis = tuple(float(value) for value in force_values)
    if not force_axis:
        return (), ()

    ka_log10 = math.log10(_coerce_positive_float(ka, label="ka"))
    kb_log10 = math.log10(_coerce_positive_float(kb, label="kb"))
    x = np.asarray(
        [
            (
                ka_log10,
                kb_log10,
                _normalize_force(force, force_min=fit.force_min, force_max=fit.force_max),
            )
            for force in force_axis
        ],
        dtype=float,
    )

    predictions = [
        np.asarray(
            _predict_member_rows(model, x, scaler=(feature_mean, feature_scale, target_mean, target_scale)),
            dtype=float,
        )
        for model, feature_mean, feature_scale, target_mean, target_scale in members
    ]
    if not predictions:
        return (), ()
    stack = np.stack(predictions, axis=0)
    return (
        tuple(float(item) for item in stack.mean(axis=0)),
        tuple(tuple(float(item) for item in row) for row in stack),
    )


def predict_candidate_curves(
    fit: Emb34umDnnSurrogateFit,
    candidate_records: Sequence[Mapping[str, Any]],
    force_grid: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], ...]:
    if not candidate_records:
        return ()
    force_axis = validate_dnn_causal_force_grid(force_grid) if force_grid is not None else fit.force_grid
    predictions: list[tuple[float, ...]] = []
    for index, raw_row in enumerate(_coerce_sequence(candidate_records, label="candidate_records"), start=1):
        row = _coerce_mapping(raw_row, label=f"candidate_records[{index}]")
        ka = _coerce_positive_float(row.get("ka", row.get("Yt", 1.0)), label=f"candidate_records[{index}].ka")
        kb = _coerce_positive_float(row.get("kb", row.get("kb_scale", 1.0)), label=f"candidate_records[{index}].kb")
        curve, _ = load_ensemble_member_predictions(fit, force_axis, ka=ka, kb=kb)
        predictions.append(curve)
    return tuple(predictions)


def _build_training_report(fit: Emb34umDnnSurrogateFit, *, timing_canary: bool) -> dict[str, Any]:
    return {
        "schema_version": EMB_34UM_DNN_SURROGATE_SCHEMA_VERSION,
        "backend": fit.backend,
        "architecture": fit.architecture,
        "ensemble_size": fit.ensemble_member_count,
        "ensemble_seeds": list(fit.ensemble_seeds),
        "ensemble_checkpoints": [str(path) for path in fit.ensemble_checkpoints],
        "train_runtime_seconds": fit.train_runtime_seconds,
        "score_runtime_seconds": fit.score_runtime_seconds,
        "device": fit.device,
        "train_row_count": fit.train_row_count,
        "validation_row_count": fit.validation_row_count,
        "force_grid": list(fit.force_grid),
        "feature_mean": list(fit.feature_mean),
        "feature_scale": list(fit.feature_scale),
        "target_mean": float(fit.target_mean),
        "target_scale": float(fit.target_scale),
        "timing_canary": {
            "enabled": bool(timing_canary),
            "ensemble_size": fit.ensemble_member_count,
        },
        "training_history": [dict(item) for item in fit.training_history],
    }


def train_emb_34um_dnn_surrogate_ensemble(
    completed_curve_rows: Sequence[Mapping[str, Any]] | Sequence[DnnSurrogateLongRow],
    *,
    architecture: str = EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE,
    ensemble_size: int = 10,
    ensemble_seeds: Sequence[int] | None = None,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1.0e-3,
    validation_fraction: float = 0.1,
    force_grid: Sequence[float] | None = None,
    output_root: Path | str | None = None,
    device: str | None = None,
    seed_offset: int = 0,
    timing_canary: bool = False,
) -> tuple[Emb34umDnnSurrogateFit, dict[str, Any]]:
    architecture = _check_architecture(architecture)
    ensemble_size = validate_dnn_causal_ensemble_size(ensemble_size)
    output_root = _coerce_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_device = _coerce_device(device)

    raw_rows = _coerce_sequence(completed_curve_rows, label="completed_curve_rows")
    if not raw_rows:
        raise ValueError("completed_curve_rows must not be empty.")

    resolved_force_grid = validate_dnn_causal_force_grid(force_grid or EMB_34UM_DNN_CAUSAL_FORCE_GRID)
    if isinstance(raw_rows[0], DnnSurrogateLongRow):
        train_rows = tuple(raw_rows)  # type: ignore[assignment]
    else:
        train_rows = convert_completed_curves_to_long_rows(raw_rows, force_grid=resolved_force_grid)  # type: ignore[arg-type]
    if len(train_rows) < 2:
        raise ValueError("Need at least two curve points to train a DNN surrogate.")

    seeds = _coerce_ensemble_seeds(ensemble_seeds, size=ensemble_size)
    features, targets, _ = _rows_to_arrays(train_rows)
    feature_mean, feature_scale, target_mean, target_scale = _scaler_from_rows(features, targets)
    _, validation_index = _train_test_split_indices(
        len(train_rows),
        int(seeds[0]) + int(seed_offset),
        validation_fraction=validation_fraction,
    )

    members: list[_EnsembleMember] = []
    training_history: list[dict[str, Any]] = []

    for index, seed in enumerate(seeds, start=1):
        effective_seed = int(seed) + int(seed_offset)
        start = perf_counter()
        member, model = _train_single_member(
            rows=train_rows,
            seed=effective_seed,
            architecture=architecture,
            output_root=output_root,
            index=index,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            validation_fraction=validation_fraction,
            batch_size=_coerce_positive_int(batch_size, label="batch_size"),
            learning_rate=_coerce_positive_float(learning_rate, label="learning_rate"),
            epochs=_coerce_positive_int(epochs, label="epochs"),
            device=selected_device,
        )
        runtime = perf_counter() - start
        del model

        stored_member = _EnsembleMember(
            seed=int(seed),
            checkpoint_path=member.checkpoint_path,
            train_loss_history=member.train_loss_history,
            validation_loss_history=member.validation_loss_history,
            train_runtime_seconds=float(runtime),
        )
        members.append(stored_member)
        training_history.append(
            {
                "member_index": index,
                "seed": int(seed),
                "effective_seed": effective_seed,
                "checkpoint_path": str(stored_member.checkpoint_path),
                "train_runtime_seconds": float(runtime),
                "train_loss_history": list(stored_member.train_loss_history),
                "validation_loss_history": list(stored_member.validation_loss_history),
                "train_loss_initial": float(stored_member.train_loss_history[0]) if stored_member.train_loss_history else 0.0,
                "train_loss_final": float(stored_member.train_loss_history[-1]) if stored_member.train_loss_history else 0.0,
                "validation_loss_initial": float(stored_member.validation_loss_history[0]) if stored_member.validation_loss_history else 0.0,
                "validation_loss_final": float(stored_member.validation_loss_history[-1]) if stored_member.validation_loss_history else 0.0,
                "train_loss_min": min(stored_member.train_loss_history) if stored_member.train_loss_history else 0.0,
                "validation_loss_min": min(stored_member.validation_loss_history) if stored_member.validation_loss_history else 0.0,
            }
        )

    fit = Emb34umDnnSurrogateFit(
        architecture=architecture,
        backend=EMB_34UM_DNN_SURROGATE_BACKEND,
        ensemble_seeds=seeds,
        ensemble_checkpoints=tuple(member.checkpoint_path for member in members),
        ensemble_member_count=len(members),
        train_runtime_seconds=sum(member.train_runtime_seconds for member in members),
        score_runtime_seconds=0.0,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        force_min=float(resolved_force_grid[0]),
        force_max=float(resolved_force_grid[-1]),
        force_grid=resolved_force_grid,
        device=selected_device,
        train_row_count=len(train_rows),
        validation_row_count=len(validation_index),
        model_path=output_root / f"{EMB_34UM_DNN_SURROGATE_MEMBER_PREFIX}_trained.pt",
        training_history=tuple(training_history),
    )

    all_train_losses = tuple(loss for member in members for loss in member.train_loss_history)
    all_validation_losses = tuple(loss for member in members for loss in member.validation_loss_history)
    report = _build_training_report(fit, timing_canary=timing_canary)
    report["runtime_summary"] = {
        "train_loss_count": len(all_train_losses),
        "validation_loss_count": len(all_validation_losses),
        "train_loss_mean": statistics.mean(all_train_losses) if all_train_losses else 0.0,
        "validation_loss_mean": statistics.mean(all_validation_losses) if all_validation_losses else 0.0,
        "member_runtime_seconds": [member.train_runtime_seconds for member in members],
    }

    _write_training_artifacts(output_root, fit, report)
    return fit, report


def train_emb_34um_dnn_surrogate(*args: Any, **kwargs: Any) -> tuple[Emb34umDnnSurrogateFit, dict[str, Any]]:
    return train_emb_34um_dnn_surrogate_ensemble(*args, **kwargs)


def score_candidate_curves(
    fit: Emb34umDnnSurrogateFit,
    candidate_records: Sequence[Mapping[str, Any]],
    force_grid: Sequence[float],
    *,
    existing_points: Sequence[tuple[float, float]] | None = None,
    diversity_weight: float = EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
) -> tuple[Mapping[str, Any], ...]:
    from meso_uq.active_learning.emb_34um_dnn_acquisition import score_candidate_pool

    return score_candidate_pool(
        fit=fit,
        candidate_records=candidate_records,
        force_grid=_coerce_float_sequence(force_grid, label="force_grid"),
        existing_points=existing_points,
        diversity_weight=diversity_weight,
    )


def train_and_score(
    completed_curve_rows: Sequence[Mapping[str, Any]] | Sequence[DnnSurrogateLongRow],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    force_grid: Sequence[float] | None = None,
    **train_kwargs: Any,
) -> tuple[Emb34umDnnSurrogateFit, dict[str, Any], tuple[Mapping[str, Any], ...]]:
    train_rows = (
        convert_completed_curves_to_long_rows(completed_curve_rows, force_grid=force_grid)  # type: ignore[arg-type]
        if completed_curve_rows and not isinstance(completed_curve_rows[0], DnnSurrogateLongRow)  # type: ignore[index]
        else completed_curve_rows
    )
    fit, report = train_emb_34um_dnn_surrogate_ensemble(train_rows, force_grid=force_grid, **train_kwargs)
    scored = score_candidate_curves(fit, candidate_records, force_grid=fit.force_grid)
    return fit, report, scored


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _fit_to_jsonable_dict(fit: Emb34umDnnSurrogateFit) -> dict[str, Any]:
    return {
        "architecture": fit.architecture,
        "backend": fit.backend,
        "ensemble_seeds": list(fit.ensemble_seeds),
        "ensemble_checkpoints": [str(path) for path in fit.ensemble_checkpoints],
        "ensemble_member_count": fit.ensemble_member_count,
        "train_runtime_seconds": fit.train_runtime_seconds,
        "score_runtime_seconds": fit.score_runtime_seconds,
        "feature_mean": list(fit.feature_mean),
        "feature_scale": list(fit.feature_scale),
        "target_mean": fit.target_mean,
        "target_scale": fit.target_scale,
        "force_min": fit.force_min,
        "force_max": fit.force_max,
        "force_grid": list(fit.force_grid),
        "device": fit.device,
        "train_row_count": fit.train_row_count,
        "validation_row_count": fit.validation_row_count,
        "model_path": str(fit.model_path),
        "training_history": [dict(item) for item in fit.training_history],
    }


def _write_training_artifacts(output_root: Path, fit: Emb34umDnnSurrogateFit, report: Mapping[str, Any]) -> None:
    write_json(output_root / EMB_34UM_DNN_SURROGATE_MANIFEST_FILENAME, report)
    write_json(
        output_root / EMB_34UM_DNN_SURROGATE_REPORT_FILENAME,
        {
            "schema_version": EMB_34UM_DNN_SURROGATE_SCHEMA_VERSION,
            "fit": _fit_to_jsonable_dict(fit),
            "report": dict(report),
        },
    )


__all__ = [
    "EMB_34UM_DNN_SURROGATE_BACKEND",
    "EMB_34UM_DNN_SURROGATE_SCHEMA_VERSION",
    "EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE",
    "EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT",
    "EMB_34UM_DNN_SURROGATE_MANIFEST_FILENAME",
    "EMB_34UM_DNN_SURROGATE_REPORT_FILENAME",
    "DnnSurrogateLongRow",
    "Emb34umDnnSurrogateFit",
    "convert_completed_curves_to_long_rows",
    "train_emb_34um_dnn_surrogate",
    "train_emb_34um_dnn_surrogate_ensemble",
    "load_ensemble_member_predictions",
    "predict_candidate_curves",
    "score_candidate_curves",
    "write_json",
]
