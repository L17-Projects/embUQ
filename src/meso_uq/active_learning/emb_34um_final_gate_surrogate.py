from __future__ import annotations

from dataclasses import dataclass
import math
import re
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np


EMB_34UM_FINAL_GATE_SURROGATE_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_final_gate_surrogate.v1"

EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES = (
    "linear_rf8",
    "poly2_rf8",
    "poly3_rf4",
)
EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS = (11, 17, 29)
EMB_34UM_FINAL_GATE_SURROGATE_DIVERSITY_WEIGHT = 0.25


@dataclass(frozen=True)
class SurrogateLongRow:
    """Normalized per-force-point representation used by the surrogates."""

    curve_id: str
    ka_log10: float
    kb_log10: float
    force: float
    force_norm: float
    displacement: float


@dataclass(frozen=True)
class Emb34umFinalGateSurrogateFit:
    """Selected fit object for downstream scoring and report payloads."""

    selected_architecture: str
    selected_seed: int
    backend: str
    runtime_seconds: float
    feature_mean: tuple[float, float, float]
    feature_scale: tuple[float, float, float]
    target_mean: float
    target_scale: float
    force_min: float
    force_max: float
    models: tuple[Any, ...]
    architecture_scores: tuple[Mapping[str, Any], ...]
    train_row_count: int
    validation_row_count: int


@dataclass(frozen=True)
class _RidgeRandomFeatureModel:
    """Single ridge-regression member with optional random features."""

    architecture: str
    seed: int
    degree: int
    random_features: int
    feature_mean: tuple[float, float, float]
    feature_scale: tuple[float, float, float]
    target_mean: float
    target_scale: float
    random_weight: tuple[tuple[float, ...], ...]
    random_bias: tuple[float, ...]
    coefficients: tuple[float, ...]

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError("features must be 2D")
        if x.shape[1] != 3:
            raise ValueError("features must be shape (n, 3)")

        normalized = (x - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        phi = _build_design_matrix(
            normalized,
            degree=self.degree,
            random_weight=np.asarray(self.random_weight),
            random_bias=np.asarray(self.random_bias),
        )
        pred_scaled = phi @ np.asarray(self.coefficients)
        return float(self.target_mean) + float(self.target_scale) * pred_scaled


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
    number = int(value)
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _coerce_mapping(payload: object, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(payload)


def _coerce_sequence(payload: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(payload)


def _coerce_float_sequence(payload: object, *, label: str) -> tuple[float, ...]:
    values = _coerce_sequence(payload, label=label)
    parsed: list[float] = []
    for item in values:
        parsed.append(_coerce_float(item, label=f"{label} entry"))
    if not parsed:
        raise ValueError(f"{label} must not be empty.")
    return tuple(parsed)


def _coerce_curve_id(row_index: int, row: Mapping[str, Any]) -> str:
    for key in ("curve_id", "candidate_id", "sample_id", "source"):
        value = row.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    return f"curve_{row_index:06d}"


def _extract_controls(row: Mapping[str, Any]) -> tuple[float, float]:
    params = row.get("parameters")
    if isinstance(params, Mapping):
        if "ka" in params and "kb" in params:
            return (
                _coerce_positive_float(params["ka"], label="ka"),
                _coerce_positive_float(params["kb"], label="kb"),
            )
    if isinstance(params, Sequence) and len(params) >= 3:
        return (
            _coerce_positive_float(params[1], label="ka"),
            _coerce_positive_float(params[2], label="kb"),
        )

    for key in ("ka", "Yt", "yt", "Y", "kt"):
        if key in row:
            ka = _coerce_positive_float(row[key], label="ka")
            break
    else:
        raise ValueError("row is missing ka.")

    for key in ("kb", "kb_scale", "mu"):
        if key in row:
            kb = _coerce_positive_float(row[key], label="kb")
            break
    else:
        raise ValueError("row is missing kb.")

    return ka, kb


def _extract_force_grid(row: Mapping[str, Any]) -> tuple[float, ...]:
    for key in ("force_grid", "forces", "force", "axis"):
        if key in row:
            return _coerce_float_sequence(row[key], label=key)
    raise ValueError("row is missing force_grid.")


def _extract_curve(row: Mapping[str, Any]) -> tuple[float, ...]:
    for key in ("reference_curve", "output_curve", "outputs", "displacement", "y", "reference"):
        if key in row:
            return _coerce_float_sequence(row[key], label=key)
    raise ValueError("row is missing reference_curve/output_curve.")


def _normalize_force(force: float, force_min: float, force_max: float) -> float:
    width = force_max - force_min
    if math.isclose(width, 0.0):
        return 0.0
    return (force - force_min) / width


def convert_completed_curves_to_long_rows(
    completed_curve_rows: Sequence[Mapping[str, Any]],
    *,
    force_grid: Sequence[float] | None = None,
) -> tuple[SurrogateLongRow, ...]:
    rows = _coerce_sequence(completed_curve_rows, label="completed_curve_rows")
    if not rows:
        return ()

    provided_grid = tuple(_coerce_float_sequence(force_grid, label="force_grid")) if force_grid is not None else None
    normalized_rows: list[SurrogateLongRow] = []
    expected_grid = provided_grid

    for index, raw_row in enumerate(rows, start=1):
        row = _coerce_mapping(raw_row, label=f"completed_curve_rows[{index}]")
        ka, kb = _extract_controls(row)
        force_axis = _extract_force_grid(row)
        curve = _extract_curve(row)

        if len(force_axis) != len(curve):
            raise ValueError(f"{_coerce_curve_id(index, row)}: force and curve lengths differ.")

        if expected_grid is None:
            expected_grid = force_axis
        elif len(force_axis) != len(expected_grid) or any(
            not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(force_axis, expected_grid)
        ):
            raise ValueError("all completed curves must use the same force grid.")

        lower = float(expected_grid[0])
        upper = float(expected_grid[-1])
        ka_log10 = math.log10(ka)
        kb_log10 = math.log10(kb)
        for force, value in zip(force_axis, curve):
            normalized_rows.append(
                SurrogateLongRow(
                    curve_id=_coerce_curve_id(index, row),
                    ka_log10=ka_log10,
                    kb_log10=kb_log10,
                    force=float(force),
                    force_norm=_normalize_force(float(force), lower, upper),
                    displacement=float(value),
                )
            )

    return tuple(normalized_rows)


def _rows_to_arrays(rows: Sequence[SurrogateLongRow]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not rows:
        raise ValueError("rows are empty.")
    features = np.array(
        [(row.ka_log10, row.kb_log10, row.force_norm) for row in rows],
        dtype=float,
    )
    targets = np.array([row.displacement for row in rows], dtype=float)
    curve_ids = tuple(row.curve_id for row in rows)
    return features, targets, curve_ids


def _fit_scaler(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    target_mean = float(targets.mean())
    target_scale = float(targets.std())
    if not math.isfinite(target_scale) or target_scale == 0.0:
        target_scale = 1.0
    return feature_mean, feature_scale, target_mean, target_scale


_ARCH_RE = re.compile(r"^(?:(?P<prefix>poly|rf|random)(?P<degree>\d+))?(?:(?:_|-)?rf(?P<rf>\d+))?$")


def _parse_architecture(name: str) -> tuple[str, int, int]:
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        raise ValueError("architecture name must be non-empty.")
    if text.startswith("linear"):
        if text == "linear":
            return text, 1, 8
        if text.startswith("linear_rf"):
            return text, 1, int(text.replace("linear_rf", "").strip("_"))
        if text.startswith("linear_"):
            suffix = text.replace("linear_", "")
            if suffix.startswith("rf") and suffix[2:].isdigit():
                return text, 1, int(suffix[2:])
        return text, 1, 8
    if text.isdigit():
        # Backwards-compatible fallback: "8" means random feature width 8.
        return f"random{text}", 1, int(text)

    match = _ARCH_RE.match(text)
    if match is None:
        raise ValueError(f"unsupported architecture name {name!r}")

    prefix = match.group("prefix") or "linear"
    degree_text = match.group("degree")
    rf_text = match.group("rf")
    degree = 1
    if prefix == "poly":
        degree = int(degree_text) if degree_text else 1
    random_features = int(rf_text) if rf_text else (int(degree_text) if prefix in {"rf", "random"} and degree_text else 8)
    return f"{text}", degree, random_features


def _parse_architecture_names(architecture_names: Sequence[str]) -> tuple[tuple[str, int, int], ...]:
    raw = _coerce_sequence(architecture_names, label="architecture_names")
    if not raw:
        raise ValueError("architecture_names must be non-empty.")
    parsed: list[tuple[str, int, int]] = []
    for raw_name in raw:
        parsed.append(_parse_architecture(str(raw_name)))
    return tuple(parsed)


def _build_design_matrix(
    scaled_features: np.ndarray,
    *,
    degree: int,
    random_weight: np.ndarray,
    random_bias: np.ndarray,
) -> np.ndarray:
    if scaled_features.ndim != 2 or scaled_features.shape[1] != 3:
        raise ValueError("scaled_features must be shape (n, 3).")

    x1, x2, x3 = scaled_features[:, 0], scaled_features[:, 1], scaled_features[:, 2]
    terms: list[np.ndarray] = [np.ones_like(x1), x1, x2, x3]

    if degree >= 2:
        terms.extend([x1 * x1, x2 * x2, x3 * x3, x1 * x2, x1 * x3, x2 * x3])
    if degree >= 3:
        terms.extend(
            [
                x1**3,
                x2**3,
                x3**3,
                x1 * x1 * x2,
                x1 * x1 * x3,
                x1 * x2 * x2,
                x2 * x2 * x3,
                x1 * x3 * x3,
                x2 * x3 * x3,
                x1 * x2 * x3,
            ]
        )
    if random_weight.size:
        terms.append(np.tanh(scaled_features @ random_weight + random_bias))

    return np.column_stack(terms)


def _fit_single_model(
    rows: Sequence[SurrogateLongRow],
    *,
    degree: int,
    random_features: int,
    seed: int,
    architecture: str,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
    alpha: float,
) -> _RidgeRandomFeatureModel:
    x, y, _ = _rows_to_arrays(rows)
    normalized = (x - feature_mean) / feature_scale
    rng = np.random.default_rng(seed)
    if random_features > 0:
        random_weight = rng.normal(size=(3, random_features))
        random_bias = rng.normal(size=(random_features,))
    else:
        random_weight = np.zeros((3, 0), dtype=float)
        random_bias = np.zeros((0,), dtype=float)

    phi = _build_design_matrix(
        normalized,
        degree=degree,
        random_weight=random_weight,
        random_bias=random_bias,
    )
    y_scaled = (y - target_mean) / target_scale
    rhs = phi.T @ y_scaled
    lhs = phi.T @ phi + alpha * np.eye(phi.shape[1], dtype=float)
    lhs[0, 0] -= alpha
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return _RidgeRandomFeatureModel(
        architecture=architecture,
        seed=int(seed),
        degree=int(degree),
        random_features=int(random_features),
        feature_mean=tuple(float(item) for item in feature_mean),
        feature_scale=tuple(float(item) for item in feature_scale),
        target_mean=float(target_mean),
        target_scale=float(target_scale),
        random_weight=tuple(tuple(float(v) for v in row) for row in random_weight),
        random_bias=tuple(float(v) for v in random_bias),
        coefficients=tuple(float(v) for v in coefficients),
    )


def _predict_with_ensemble(models: Sequence[_RidgeRandomFeatureModel], features: np.ndarray) -> np.ndarray:
    if not models:
        raise ValueError("ensemble has no models.")
    return np.stack([model.predict(features) for model in models], axis=0)


def _evaluate_curves(
    models: Sequence[_RidgeRandomFeatureModel],
    rows: Sequence[SurrogateLongRow],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], tuple[str, ...]]:
    if not rows:
        raise ValueError("rows are empty.")

    by_curve: dict[str, list[SurrogateLongRow]] = {}
    for row in rows:
        by_curve.setdefault(row.curve_id, []).append(row)

    residuals: list[float] = []
    predicted_curves: list[tuple[float, ...]] = []
    reference_curves: list[tuple[float, ...]] = []
    curve_ids: list[str] = []

    for curve_id in sorted(by_curve):
        curve_rows = sorted(by_curve[curve_id], key=lambda item: item.force)
        x = np.array([(row.ka_log10, row.kb_log10, row.force_norm) for row in curve_rows], dtype=float)
        reference = np.array([row.displacement for row in curve_rows], dtype=float)
        curve_prediction = _predict_with_ensemble(models, x).mean(axis=0)

        residual_sq = float(np.sum((curve_prediction - reference) ** 2))
        ref_norm_sq = float(np.sum(reference**2))
        if ref_norm_sq <= 0.0:
            rel = float("inf")
        else:
            rel = float(100.0 * math.sqrt(residual_sq / ref_norm_sq))

        residuals.append(rel)
        predicted_curves.append(tuple(float(v) for v in curve_prediction))
        reference_curves.append(tuple(float(v) for v in reference))
        curve_ids.append(curve_id)

    return tuple(float(v) for v in residuals), tuple(predicted_curves), tuple(reference_curves), tuple(curve_ids)


def _evaluate_metrics(models: Sequence[_RidgeRandomFeatureModel], rows: Sequence[SurrogateLongRow]) -> dict[str, Any]:
    residuals, predicted_curves, reference_curves, _ = _evaluate_curves(models, rows)
    if not residuals:
        raise ValueError("No validation curves available.")
    return {
        "median_curve_rel_l2_pct": float(statistics.median(residuals)),
        "mean_curve_rel_l2_pct": float(statistics.mean(residuals)),
        "max_curve_rel_l2_pct": float(max(residuals)),
        "residuals": residuals,
        "predicted_curves": predicted_curves,
        "reference_curves": reference_curves,
    }


def _coerce_fit_like(ensemble: Mapping[str, Any] | Emb34umFinalGateSurrogateFit) -> Emb34umFinalGateSurrogateFit:
    if isinstance(ensemble, Emb34umFinalGateSurrogateFit):
        return ensemble
    if isinstance(ensemble, Mapping):
        payload = dict(ensemble)
        if "fit" in payload and isinstance(payload["fit"], Emb34umFinalGateSurrogateFit):
            return payload["fit"]
        if "ensemble" in payload and isinstance(payload["ensemble"], Emb34umFinalGateSurrogateFit):
            return payload["ensemble"]
        return Emb34umFinalGateSurrogateFit(
            selected_architecture=str(payload.get("selected_architecture", payload.get("architecture", "unknown"))),
            selected_seed=int(payload.get("selected_seed", 0)),
            backend=str(payload.get("backend", "numpy")),
            runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
            feature_mean=tuple(float(item) for item in payload.get("feature_mean", (0.0, 0.0, 0.0))),
            feature_scale=tuple(float(item) for item in payload.get("feature_scale", (1.0, 1.0, 1.0))),
            target_mean=float(payload.get("target_mean", 0.0)),
            target_scale=float(payload.get("target_scale", 1.0)),
            force_min=float(payload.get("force_min", 0.0)),
            force_max=float(payload.get("force_max", 1.0)),
            models=tuple(payload.get("models", ())),
            architecture_scores=tuple(payload.get("architecture_scores", ())),
            train_row_count=int(payload.get("train_row_count", 0)),
            validation_row_count=int(payload.get("validation_row_count", 0)),
        )
    raise ValueError("ensemble must be a fit object or dict payload.")


def train_emb_34um_surrogate_ensemble(
    records: Sequence[Mapping[str, Any]] | Sequence[SurrogateLongRow],
    validation_records: Sequence[Mapping[str, Any]] | Sequence[SurrogateLongRow],
    seeds: Sequence[int],
    architecture_names: Sequence[str],
) -> dict[str, Any]:
    """Train a deterministic numpy ensemble on training/validation records."""

    raw_records = _coerce_sequence(records, label="records")
    if not raw_records:
        raise ValueError("records must be non-empty.")
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    if isinstance(raw_records[0], SurrogateLongRow):
        train_rows = tuple(raw_records)  # type: ignore[assignment]
        first_curve = train_rows[0].curve_id
        train_force_grid = tuple(row.force for row in train_rows if row.curve_id == first_curve)
    else:
        train_rows = convert_completed_curves_to_long_rows(raw_records)  # type: ignore[arg-type]
        train_force_grid = _extract_force_grid(_coerce_mapping(raw_records[0], label="records[0]"))

    if not train_rows:
        raise ValueError("No valid training rows.")

    if validation_records:
        if isinstance(validation_records[0], SurrogateLongRow):
            validation_rows = tuple(validation_records)  # type: ignore[arg-type]
        else:
            validation_rows = convert_completed_curves_to_long_rows(_coerce_sequence(validation_records, label="validation_records"), force_grid=train_force_grid)
    else:
        validation_rows = train_rows

    parsed_seeds = tuple(_coerce_non_negative_int(seed, label=f"seeds[{index}]") for index, seed in enumerate(seeds, start=1))
    architectures = _parse_architecture_names(_coerce_sequence(architecture_names, label="architecture_names"))

    x_train, y_train, _ = _rows_to_arrays(train_rows)
    feature_mean, feature_scale, target_mean, target_scale = _fit_scaler(x_train, y_train)

    start = time.perf_counter()
    best_score = float("inf")
    selected_architecture = architectures[0][0]
    selected_seed = parsed_seeds[0]
    selected_models: tuple[_RidgeRandomFeatureModel, ...] = ()
    architecture_scores: list[Mapping[str, Any]] = []

    for architecture, degree, random_features in architectures:
        models: list[_RidgeRandomFeatureModel] = []
        for seed in parsed_seeds:
            models.append(
                _fit_single_model(
                    train_rows,
                    degree=degree,
                    random_features=random_features,
                    seed=seed,
                    architecture=architecture,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    target_mean=target_mean,
                    target_scale=target_scale,
                    alpha=1.0e-6,
                )
            )

        metrics = _evaluate_metrics(models, validation_rows)
        mean_metric = float(metrics["mean_curve_rel_l2_pct"])
        architecture_scores.append(
            {
                "architecture": architecture,
                "degree": degree,
                "random_features": random_features,
                "mean_curve_rel_l2_pct": mean_metric,
            }
        )
        if mean_metric < best_score:
            best_score = mean_metric
            selected_architecture = architecture
            selected_seed = parsed_seeds[0]
            selected_models = tuple(models)

    if not selected_models:
        raise RuntimeError("Failed to train any model.")

    fit = Emb34umFinalGateSurrogateFit(
        selected_architecture=selected_architecture,
        selected_seed=selected_seed,
        backend="numpy",
        runtime_seconds=float(time.perf_counter() - start),
        feature_mean=tuple(float(item) for item in feature_mean),
        feature_scale=tuple(float(item) for item in feature_scale),
        target_mean=float(target_mean),
        target_scale=float(target_scale),
        force_min=float(min(row.force for row in train_rows)),
        force_max=float(max(row.force for row in train_rows)),
        models=selected_models,
        architecture_scores=tuple(architecture_scores),
        train_row_count=len(train_rows),
        validation_row_count=len(validation_rows),
    )

    metrics = _evaluate_metrics(fit.models, validation_rows)  # type: ignore[arg-type]
    return {
        "fit": fit,
        "median_curve_rel_l2_pct": metrics["median_curve_rel_l2_pct"],
        "mean_curve_rel_l2_pct": metrics["mean_curve_rel_l2_pct"],
        "max_curve_rel_l2_pct": metrics["max_curve_rel_l2_pct"],
        "residuals": metrics["residuals"],
        "predicted_curves": metrics["predicted_curves"],
        "reference_curves": metrics["reference_curves"],
        "model_selection": {
            "rerun": True,
            "architecture": fit.selected_architecture,
            "backend": fit.backend,
            "notes": "deterministic numpy polynomial/random-feature ridge ensemble",
        },
        "architecture_scores": tuple(architecture_scores),
    }


def _candidate_score_row(
    fit: Emb34umFinalGateSurrogateFit,
    ka: float,
    kb: float,
    force_grid: Sequence[float],
    *,
    existing_points: Sequence[tuple[float, float]] | None,
    diversity_weight: float,
) -> tuple[float, float, tuple[float, ...], dict[str, Any]]:
    force_min = float(fit.force_min)
    force_max = float(fit.force_max)
    points = np.array(
        [
            [
                math.log10(_coerce_positive_float(ka, label="ka")),
                math.log10(_coerce_positive_float(kb, label="kb")),
                _normalize_force(float(force), force_min, force_max),
            ]
            for force in force_grid
        ],
        dtype=float,
    )
    preds = _predict_with_ensemble(fit.models, points)
    predicted = np.mean(preds, axis=0)
    disagreement = float(np.mean(np.std(preds, axis=0)))
    point = (math.log10(ka), math.log10(kb))

    if existing_points:
        nearest = min(math.dist(point, candidate) for candidate in existing_points)
    else:
        nearest = 0.0

    acquisition_score = disagreement + float(diversity_weight) * float(nearest)
    return acquisition_score, disagreement, tuple(float(item) for item in predicted), {
        "ka": float(ka),
        "kb": float(kb),
        "ensemble_disagreement": disagreement,
        "acquisition_score": acquisition_score,
    }


def _normalize_existing_points(
    points: Sequence[tuple[float, float]] | None,
) -> tuple[tuple[float, float], ...]:
    if not points:
        return ()
    normalized: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError("existing_points entries must be (ka, kb).")
        normalized.append(
            (
                math.log10(_coerce_positive_float(point[0], label="existing point ka")),
                math.log10(_coerce_positive_float(point[1], label="existing point kb")),
            )
        )
    return tuple(normalized)


def score_emb_34um_candidate_pool(
    ensemble: Mapping[str, Any] | Emb34umFinalGateSurrogateFit,
    candidate_records: Sequence[Mapping[str, Any]],
    force_grid: Sequence[float],
    existing_points: Sequence[tuple[float, float]] | None,
    diversity_weight: float = EMB_34UM_FINAL_GATE_SURROGATE_DIVERSITY_WEIGHT,
) -> tuple[Mapping[str, Any], ...]:
    """Score candidate rows with disagreement, diversity-aware acquisition score, and prediction."""

    fit = _coerce_fit_like(ensemble)
    candidates = _coerce_sequence(candidate_records, label="candidate_records")
    force_axis = _coerce_float_sequence(force_grid, label="force_grid")
    normalized_existing = _normalize_existing_points(existing_points)

    scored: list[dict[str, Any]] = []
    for index, raw_row in enumerate(candidates, start=1):
        row = _coerce_mapping(raw_row, label=f"candidate_records[{index}]")
        ka = _coerce_positive_float(row.get("ka", row.get("Yt", 1.0)), label="ka")
        kb = _coerce_positive_float(row.get("kb", row.get("mu", row.get("kb_scale", 1.0))), label="kb")
        acquisition, disagreement, predicted_curve, row_payload = _candidate_score_row(
            fit,
            ka=ka,
            kb=kb,
            force_grid=force_axis,
            existing_points=normalized_existing,
            diversity_weight=diversity_weight,
        )
        merged = dict(row)
        merged.update(row_payload)
        merged["predicted_curve"] = predicted_curve
        scored.append(merged)

    scored.sort(
        key=lambda item: (
            -float(item["acquisition_score"]),
            -float(item["ensemble_disagreement"]),
            float(item["ka"]),
            float(item["kb"]),
        )
    )
    return tuple(scored)


def _row_point(row: Mapping[str, Any]) -> tuple[float, float]:
    ka = _coerce_positive_float(row.get("ka", 0.0), label="ka")
    kb = _coerce_positive_float(row.get("kb", 0.0), label="kb")
    return (math.log10(ka), math.log10(kb))


def select_scored_candidates(
    scored: Sequence[Mapping[str, Any]],
    count: int,
    exploration_count: int = 0,
) -> tuple[Mapping[str, Any], ...]:
    """Select top candidates with deterministic score/diversity ordering."""
    rows = _coerce_sequence(scored, label="scored")
    if count < 0:
        raise ValueError("count must be non-negative.")
    if exploration_count < 0:
        raise ValueError("exploration_count must be non-negative.")
    if not rows or (count == 0 and exploration_count == 0):
        return ()

    ordered = [
        _coerce_mapping(item, label=f"scored[{index}]")
        for index, item in enumerate(rows, start=1)
    ]
    ordered.sort(
        key=lambda item: (
            -float(item.get("acquisition_score", 0.0)),
            -float(item.get("ensemble_disagreement", 0.0)),
            float(item.get("ka", 0.0)),
            float(item.get("kb", 0.0)),
        )
    )

    target_total = int(count) + int(exploration_count)
    selected_rows: list[dict[str, Any]] = []
    selected_points: list[tuple[float, float]] = []

    while len(selected_rows) < min(count, len(ordered)):
        raw = ordered.pop(0)
        item = dict(raw)
        selected_rows.append(item)
        selected_points.append(_row_point(item))

    while len(selected_rows) < target_total and ordered:
        best_index = -1
        best_score = (-math.inf, -math.inf)
        for index, item in enumerate(ordered):
            point = _row_point(item)
            if not selected_points:
                diversity = 0.0
            else:
                diversity = min(math.dist(point, picked) for picked in selected_points)
            acquisition = float(item.get("acquisition_score", 0.0))
            score = (diversity, acquisition)

            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            break
        raw = ordered.pop(best_index)
        item = dict(raw)
        selected_rows.append(item)
        selected_points.append(_row_point(item))

    return tuple(selected_rows)


def train_surrogate_ensemble(
    *,
    completed_curve_rows: Sequence[Mapping[str, Any]] | Sequence[SurrogateLongRow],
    architectures: Sequence[tuple[int, ...]] | None = None,
    ensemble_seeds: Sequence[int] | None = None,
    validation_fraction: float | None = None,
    force_grid: Sequence[float] | None = None,
    **_kwargs: Any,
    ) -> tuple[Emb34umFinalGateSurrogateFit, dict[str, Any]]:
    """Compatibility wrapper for the legacy API."""
    if architectures is None:
        architecture_names = list(EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES)
    else:
        architecture_names = []
        for idx, item in enumerate(_coerce_sequence(architectures, label="architectures"), start=1):
            if isinstance(item, str):
                architecture_names.append(item)
                continue
            if isinstance(item, Sequence) and item:
                architecture_names.append(str(item[0]))
                continue
            raise ValueError(f"architectures[{idx}] must be a string or tuple.")
    if ensemble_seeds is None:
        ensemble_seeds = EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS

    report = train_emb_34um_surrogate_ensemble(
        completed_curve_rows if completed_curve_rows else (),
        validation_records=completed_curve_rows,
        seeds=ensemble_seeds,
        architecture_names=architecture_names,
    )
    fit = _coerce_fit_like(report)
    return fit, {k: report[k] for k in (
        "median_curve_rel_l2_pct",
        "mean_curve_rel_l2_pct",
        "max_curve_rel_l2_pct",
        "residuals",
        "predicted_curves",
        "reference_curves",
        "model_selection",
    )}


def score_candidate_pool_with_ensemble(
    *,
    candidate_pool: Sequence[Mapping[str, Any]],
    fit: Emb34umFinalGateSurrogateFit,
    force_grid: Sequence[float],
    top_n: int,
    existing_points: Sequence[tuple[float, float]] | None = None,
    diversity_weight: float = EMB_34UM_FINAL_GATE_SURROGATE_DIVERSITY_WEIGHT,
    source: str = "acquisition",
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    scored = score_emb_34um_candidate_pool(
        fit,
        candidate_records=candidate_pool,
        force_grid=force_grid,
        existing_points=existing_points,
        diversity_weight=diversity_weight,
    )
    all_scores = tuple(dict(item, source=source) for item in scored)
    selected = select_scored_candidates(all_scores, count=top_n, exploration_count=0)
    return all_scores, selected


def select_candidates_with_diversity(
    candidate_rows: Sequence[Mapping[str, Any]],
    top_n: int,
    *,
    existing_points: Sequence[tuple[float, float]] | None = None,
    diversity_weight: float = EMB_34UM_FINAL_GATE_SURROGATE_DIVERSITY_WEIGHT,
) -> tuple[Mapping[str, Any], ...]:
    del existing_points, diversity_weight
    if top_n <= 0:
        return ()
    return select_scored_candidates(candidate_rows, count=top_n, exploration_count=0)


def evaluate_surrogate_round_metrics(
    *,
    fit: Emb34umFinalGateSurrogateFit,
    validation_curve_rows: Sequence[SurrogateLongRow],
) -> dict[str, Any]:
    return _evaluate_metrics(fit.models, validation_curve_rows)  # type: ignore[arg-type]


def build_final_gate_round_payload(
    *,
    round_index: int,
    fit: Emb34umFinalGateSurrogateFit,
    metrics: Mapping[str, Any],
    scored_candidates: Sequence[Mapping[str, Any]] | None = None,
    selected_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    scored = scored_candidates or ()
    selected = selected_candidates or ()
    return {
        "round": _coerce_non_negative_int(round_index, label="round_index"),
        "al": {
            "median_curve_rel_l2_pct": float(_coerce_float(metrics.get("median_curve_rel_l2_pct"), label="median_curve_rel_l2_pct")),
            "mean_curve_rel_l2_pct": float(_coerce_float(metrics.get("mean_curve_rel_l2_pct"), label="mean_curve_rel_l2_pct")),
            "max_curve_rel_l2_pct": float(_coerce_float(metrics.get("max_curve_rel_l2_pct"), label="max_curve_rel_l2_pct")),
            "residuals": tuple(float(v) for v in metrics.get("residuals", ())),
            "predicted_curves": tuple(tuple(float(v) for v in curve) for curve in metrics.get("predicted_curves", ())),
            "reference_curves": tuple(tuple(float(v) for v in curve) for curve in metrics.get("reference_curves", ())),
        },
        "acquisition_scores": tuple(float(item.get("acquisition_score", item.get("ensemble_disagreement", 0.0))) for item in scored),
        "selected_candidate_scores": tuple(
            float(item.get("acquisition_score", item.get("ensemble_disagreement", 0.0))) for item in selected
        ),
        "model_selection": {
            "rerun": True,
            "architecture": str(fit.selected_architecture),
            "backend": str(fit.backend),
            "notes": "deterministic numpy surrogate for report compatibility",
        },
    }


def build_validation_rows_for_al_vs_lhs(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    strategy: str = "al",
    round_index: int | None = None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for order, row in enumerate(_coerce_sequence(candidate_rows, label="candidate_rows"), start=1):
        source = _coerce_mapping(row, label=f"candidate_rows[{order}]")
        rows.append(
            {
                "strategy": strategy,
                "curve_id": f"{strategy}-r{int(round_index or 0):02d}-c{order:03d}",
                "round": int(round_index) if round_index is not None else 0,
                "order": int(order),
                "source": strategy,
                "ka": _coerce_positive_float(source.get("ka", source.get("Yt", 1.0)), label="ka"),
                "kb": _coerce_positive_float(source.get("kb", source.get("mu", 1.0)), label="kb"),
                "predicted_curve": tuple(float(item) for item in (source.get("predicted_curve") or ())),
                "reference_curve": tuple(float(item) for item in (source.get("reference_curve") or ())),
                "acquisition_score": float(source.get("acquisition_score", source.get("ensemble_disagreement", 0.0))),
                "ensemble_disagreement": float(source.get("ensemble_disagreement", 0.0)),
            }
        )
    return tuple(rows)


__all__ = [
    "EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES",
    "EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS",
    "EMB_34UM_FINAL_GATE_SURROGATE_DIVERSITY_WEIGHT",
    "EMB_34UM_FINAL_GATE_SURROGATE_SCHEMA_VERSION",
    "SurrogateLongRow",
    "Emb34umFinalGateSurrogateFit",
    "convert_completed_curves_to_long_rows",
    "train_emb_34um_surrogate_ensemble",
    "score_emb_34um_candidate_pool",
    "select_scored_candidates",
    "train_surrogate_ensemble",
    "score_candidate_pool_with_ensemble",
    "select_candidates_with_diversity",
    "evaluate_surrogate_round_metrics",
    "build_final_gate_round_payload",
    "build_validation_rows_for_al_vs_lhs",
]
