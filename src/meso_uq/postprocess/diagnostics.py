from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

POSTERIOR_LOG_COLUMNS = {"logLikelihood", "logPrior", "logPosterior"}

PHASE1_POSTERIOR_FIGURE_POLICY: dict[str, object] = {
    "policy_id": "phase1-posterior-duplication-audit-v1",
    "filtering": "disabled",
    "figure_source": "raw_posterior_samples",
    "annotation": "enabled",
    "rationale": (
        "Phase 1 posterior figures must preserve full sample multiplicity. "
        "Duplicate-particle and chain-leader diagnostics are reported as annotations instead of filtering."
    ),
}


def posterior_parameter_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name not in POSTERIOR_LOG_COLUMNS]


def duplicate_particle_metrics(
    frame: pd.DataFrame,
    *,
    parameter_columns: Iterable[str] | None = None,
    top_k: int = 10,
) -> dict[str, float | int]:
    params = list(parameter_columns) if parameter_columns is not None else posterior_parameter_columns(frame)
    if not params:
        raise ValueError("No parameter columns available for duplicate-particle diagnostics.")

    multiplicity = frame.value_counts(subset=params, dropna=False)
    sample_count = int(multiplicity.sum())
    unique_count = int(len(multiplicity))
    duplicated = multiplicity[multiplicity > 1]
    duplicate_particle_count = int(len(duplicated))
    duplicate_sample_count = int(duplicated.sum()) if duplicate_particle_count else 0
    top_duplicate_count = int(duplicated.max()) if duplicate_particle_count else 0
    top_k_sum = int(duplicated.nlargest(max(top_k, 1)).sum()) if duplicate_particle_count else 0

    def _mass(count: int) -> float:
        if sample_count <= 0:
            return 0.0
        return float(count / sample_count)

    return {
        "sample_count": sample_count,
        "unique_particle_count": unique_count,
        "duplicate_particle_count": duplicate_particle_count,
        "duplicate_sample_count": duplicate_sample_count,
        "duplicate_mass_total": _mass(duplicate_sample_count),
        "top_duplicate_count": top_duplicate_count,
        "top_duplicate_mass": _mass(top_duplicate_count),
        "top_10_duplicate_mass": _mass(top_k_sum),
    }


def duplicate_mass_comparison(
    posterior_metrics: dict[str, float | int],
    chain_leader_metrics: dict[str, float | int] | None,
) -> dict[str, float | None]:
    if chain_leader_metrics is None:
        return {
            "top_duplicate_mass_delta": None,
            "top_10_duplicate_mass_delta": None,
            "duplicate_mass_total_delta": None,
        }
    return {
        "top_duplicate_mass_delta": float(chain_leader_metrics["top_duplicate_mass"]) - float(posterior_metrics["top_duplicate_mass"]),
        "top_10_duplicate_mass_delta": float(chain_leader_metrics["top_10_duplicate_mass"]) - float(posterior_metrics["top_10_duplicate_mass"]),
        "duplicate_mass_total_delta": float(chain_leader_metrics["duplicate_mass_total"]) - float(posterior_metrics["duplicate_mass_total"]),
    }


def mean_or_nan(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))
