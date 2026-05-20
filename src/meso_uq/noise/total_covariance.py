from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from meso_uq.noise.covariance import CovarianceBuildResult


_TOTAL_COMPONENT = "total_covariance"


def _finite_float(value: float, label: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be finite numeric scalars; got {value!r}.") from exc
    if not isfinite(value):
        raise ValueError(f"{label} must be finite; got {value}.")
    return value


def _nonnegative_float(value: float, label: str) -> float:
    value = _finite_float(value, label)
    if value < 0.0:
        raise ValueError(f"{label} must be >= 0.0; got {value}.")
    return value


def _positive_float(value: float, label: str) -> float:
    value = _finite_float(value, label)
    if value <= 0.0:
        raise ValueError(f"{label} must be positive; got {value}.")
    return value


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _normalize_name(value: str, label: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError(f"{label} must be non-empty.")
    return name


def _coerce_matrix(value: Sequence[Sequence[float]] | np.ndarray, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be finite numeric scalars.") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be a square 2D covariance matrix; got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} values must be finite.")
    return matrix


def _ensure_symmetric(matrix: np.ndarray, label: str, tolerance: float) -> None:
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} covariance must be symmetric.")


def _ensure_psd(matrix: np.ndarray, label: str, tolerance: float) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        eigenvalues = np.linalg.eigvalsh(0.5 * matrix + 0.5 * matrix.T)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{label} covariance eigenvalues must be finite.")
    min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    if min_eigenvalue < -tolerance:
        raise ValueError(f"{label} covariance must be positive semidefinite; min eigenvalue {min_eigenvalue}.")


def _ensure_finite_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} values must be finite after covariance assembly.")
    return matrix


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    if np.allclose(diagonal, 0.0):
        return np.eye(covariance.shape[0], dtype=float)
    scale = np.sqrt(np.maximum(diagonal, 0.0))
    denominator = np.outer(scale, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if isfinite(value) else None


def _matrix_summary(matrix: np.ndarray) -> dict[str, Any]:
    symmetric = 0.5 * matrix + 0.5 * matrix.T
    eigenvalues = np.linalg.eigvalsh(symmetric)
    condition_number = np.linalg.cond(matrix) if matrix.size and not np.allclose(matrix, 0.0) else float("nan")
    return {
        "shape": list(matrix.shape),
        "diagonal_min": _finite_or_none(np.min(np.diag(matrix))),
        "diagonal_max": _finite_or_none(np.max(np.diag(matrix))),
        "min_eigenvalue": _finite_or_none(np.min(eigenvalues)),
        "max_eigenvalue": _finite_or_none(np.max(eigenvalues)),
        "matrix_rank": int(np.linalg.matrix_rank(matrix)) if matrix.size else 0,
        "condition_number": _finite_or_none(condition_number),
    }


def _cholesky_with_optional_jitter(covariance: np.ndarray, jitter: float, max_jitter: float) -> tuple[np.ndarray | None, float]:
    if np.allclose(covariance, 0.0):
        return None, 0.0
    symmetric = 0.5 * covariance + 0.5 * covariance.T
    try:
        return np.linalg.cholesky(symmetric), 0.0
    except np.linalg.LinAlgError:
        pass
    if max_jitter <= 0.0:
        return None, 0.0
    candidate = jitter if jitter > 0.0 else min(max_jitter, 1e-12)
    eye = np.eye(symmetric.shape[0], dtype=float)
    while candidate <= max_jitter * (1.0 + 1e-12):
        try:
            return np.linalg.cholesky(symmetric + candidate * eye), candidate
        except np.linalg.LinAlgError:
            candidate *= 10.0
    return None, 0.0


@dataclass(frozen=True)
class TotalCovarianceConfig:
    jitter: float = 0.0
    max_jitter: float = 0.0
    symmetry_tolerance: float = 1e-12
    psd_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        jitter = _nonnegative_float(self.jitter, "jitter")
        max_jitter = _nonnegative_float(self.max_jitter, "max_jitter")
        symmetry_tolerance = _positive_float(self.symmetry_tolerance, "symmetry_tolerance")
        psd_tolerance = _positive_float(self.psd_tolerance, "psd_tolerance")
        if max_jitter < jitter:
            raise ValueError("max_jitter must be >= jitter.")
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "max_jitter", max_jitter)
        object.__setattr__(self, "symmetry_tolerance", symmetry_tolerance)
        object.__setattr__(self, "psd_tolerance", psd_tolerance)


@dataclass(frozen=True)
class CovarianceTerm:
    name: str
    covariance: Sequence[Sequence[float]] | np.ndarray
    included: bool = True
    aliases: tuple[str, ...] = ()
    children: Mapping[str, Sequence[Sequence[float]] | np.ndarray] | None = None
    summary: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        name = _normalize_name(self.name, "covariance term name")
        aliases = tuple(_normalize_name(alias, "covariance term alias") for alias in self.aliases)
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"Covariance term '{name}' has duplicate aliases.")
        if name in aliases:
            raise ValueError(f"Covariance term '{name}' duplicates one of its aliases.")
        covariance = _coerce_matrix(self.covariance, name)
        children: dict[str, np.ndarray] = {}
        if self.children:
            for child_name, child_covariance in self.children.items():
                normalized = _normalize_name(child_name, "covariance child component name")
                if normalized in children:
                    raise ValueError(f"Covariance term '{name}' has duplicate child component '{normalized}'.")
                children[normalized] = _coerce_matrix(child_covariance, normalized)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "included", bool(self.included))
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "summary", {} if self.summary is None else dict(self.summary))


@dataclass(frozen=True)
class TotalCovarianceResult:
    covariance: CovarianceBuildResult
    covariance_components: Mapping[str, np.ndarray]
    child_covariance_components: Mapping[str, np.ndarray]
    variance_components: Mapping[str, tuple[float, ...]]
    included_term_names: tuple[str, ...]
    excluded_term_names: tuple[str, ...]
    standard_deviation: tuple[float, ...]
    summary: Mapping[str, Any]


def covariance_term_from_diagonal(
    name: str,
    diagonal_variance: Sequence[float],
    *,
    included: bool = True,
    aliases: tuple[str, ...] = (),
    children: Mapping[str, Sequence[Sequence[float]] | np.ndarray] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> CovarianceTerm:
    diagonal = np.asarray(_finite_vector(diagonal_variance, f"{name} diagonal variance"), dtype=float)
    if np.any(diagonal < 0.0):
        raise ValueError(f"{name} diagonal variance values must be >= 0.0.")
    return CovarianceTerm(
        name=name,
        covariance=np.diag(diagonal),
        included=included,
        aliases=aliases,
        children=children,
        summary=summary,
    )


def assemble_total_covariance(
    terms: Sequence[CovarianceTerm],
    config: TotalCovarianceConfig | None = None,
) -> TotalCovarianceResult:
    config = TotalCovarianceConfig() if config is None else config
    values = tuple(terms)
    if not values:
        raise ValueError("assemble_total_covariance requires at least one covariance term.")

    shape = values[0].covariance.shape
    identifiers: dict[str, str] = {}
    child_names: dict[str, str] = {}
    covariance_components: dict[str, np.ndarray] = {}
    child_components: dict[str, np.ndarray] = {}
    included_names: list[str] = []
    excluded_names: list[str] = []
    term_summaries: dict[str, Any] = {}

    for term in values:
        matrix = np.asarray(term.covariance, dtype=float)
        if matrix.shape != shape:
            raise ValueError(f"Covariance term '{term.name}' shape {matrix.shape} does not match {shape}.")
        _ensure_symmetric(matrix, term.name, config.symmetry_tolerance)
        _ensure_psd(matrix, term.name, config.psd_tolerance)

        for identifier in (term.name, *term.aliases):
            if identifier in identifiers:
                raise ValueError(
                    f"Duplicate covariance term identifier '{identifier}' from '{term.name}' and '{identifiers[identifier]}'."
                )
            if identifier in child_names:
                raise ValueError(
                    f"Covariance term identifier '{identifier}' duplicates child component from '{child_names[identifier]}'."
                )
            identifiers[identifier] = term.name

        for child_name, child_matrix in (term.children or {}).items():
            child = np.asarray(child_matrix, dtype=float)
            if child.shape != shape:
                raise ValueError(f"Child component '{child_name}' shape {child.shape} does not match {shape}.")
            _ensure_symmetric(child, child_name, config.symmetry_tolerance)
            if child_name in identifiers:
                raise ValueError(f"Child component '{child_name}' duplicates covariance term identifier.")
            if child_name in child_names:
                raise ValueError(
                    f"Duplicate child component '{child_name}' from '{term.name}' and '{child_names[child_name]}'."
                )
            child_names[child_name] = term.name
            child_components[child_name] = child

        covariance_components[term.name] = matrix
        term_summaries[term.name] = {
            "included": bool(term.included),
            "aliases": list(term.aliases),
            "child_component_names": list((term.children or {}).keys()),
            "matrix": _matrix_summary(matrix),
            **dict(term.summary or {}),
        }
        if term.included:
            included_names.append(term.name)
        else:
            excluded_names.append(term.name)

    with np.errstate(over="ignore", invalid="ignore"):
        total = sum(
            (covariance_components[name] for name in included_names),
            np.zeros(shape, dtype=float),
        )
    _ensure_finite_matrix(total, _TOTAL_COMPONENT)
    _ensure_symmetric(total, _TOTAL_COMPONENT, config.symmetry_tolerance)
    _ensure_psd(total, _TOTAL_COMPONENT, config.psd_tolerance)
    active = bool(included_names and not np.allclose(total, 0.0))
    cholesky, jitter_added = _cholesky_with_optional_jitter(total, config.jitter, config.max_jitter)
    if active and cholesky is None:
        if config.max_jitter <= 0.0:
            raise ValueError("Total covariance matrix is not positive definite and no jitter is allowed.")
        raise ValueError("Total covariance matrix is not positive definite within max_jitter.")
    if cholesky is not None and jitter_added > 0.0:
        total = total + jitter_added * np.eye(total.shape[0], dtype=float)
    correlation = _correlation_from_covariance(total)
    covariance_components[_TOTAL_COMPONENT] = total
    variance_components = {name: tuple(float(value) for value in np.diag(component)) for name, component in covariance_components.items()}
    standard_deviation = tuple(float(sqrt(max(value, 0.0))) for value in np.diag(total))
    summary = {
        "schema_version": 1,
        "component_name": _TOTAL_COMPONENT,
        "included_term_names": list(included_names),
        "excluded_term_names": list(excluded_names),
        "child_component_names": list(child_components.keys()),
        "term_summaries": term_summaries,
        "shape": list(total.shape),
        "active": active,
        "diagonal_min": _finite_or_none(np.min(np.diag(total))),
        "diagonal_max": _finite_or_none(np.max(np.diag(total))),
        "correlation_min": _finite_or_none(np.min(correlation)),
        "correlation_max": _finite_or_none(np.max(correlation)),
        "min_eigenvalue": _matrix_summary(total)["min_eigenvalue"],
        "max_eigenvalue": _matrix_summary(total)["max_eigenvalue"],
        "matrix_rank": _matrix_summary(total)["matrix_rank"],
        "condition_number": _matrix_summary(total)["condition_number"],
        "jitter_added": float(jitter_added),
        "cholesky_success": cholesky is not None,
    }
    covariance_result = CovarianceBuildResult(
        covariance=total,
        correlation=correlation,
        cholesky=cholesky,
        jitter_added=jitter_added,
        summary=summary,
    )
    return TotalCovarianceResult(
        covariance=covariance_result,
        covariance_components=covariance_components,
        child_covariance_components=child_components,
        variance_components=variance_components,
        included_term_names=tuple(included_names),
        excluded_term_names=tuple(excluded_names),
        standard_deviation=standard_deviation,
        summary=summary,
    )


__all__ = [
    "CovarianceTerm",
    "TotalCovarianceConfig",
    "TotalCovarianceResult",
    "assemble_total_covariance",
    "covariance_term_from_diagonal",
]
