from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence


REFERENCE_CPP_PATH = Path(
    "extern/korali/source/modules/problem/bayesian/reference/reference.cpp"
)
STDEV_EPSILON = 1.0e-11
LOG2PI = math.log(2.0 * math.pi)
REQUIRED_MARKERS = (
    "double sse = compute_normalized_sse(refEvals, stdDevs, _referenceData);",
    "if (stdDevs[i] < STDEV_EPSILON) stdDevs[i] = STDEV_EPSILON;",
    "loglike -= 0.5 * (Nd * _log2pi + sse);",
)


def read_vendored_reference_normal_source(repo_root: Path) -> str:
    source = (repo_root / REFERENCE_CPP_PATH).read_text(encoding="utf-8")
    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise ValueError(
            "Vendored Korali reference.cpp is missing required Normal-likelihood markers: "
            + ", ".join(missing)
        )
    return source


def korali_reference_normal_loglikelihood(
    reference_data: Sequence[float],
    reference_evaluations: Sequence[float],
    standard_deviations: Sequence[float],
) -> float:
    if not (
        len(reference_data)
        == len(reference_evaluations)
        == len(standard_deviations)
    ):
        raise ValueError("reference_data, reference_evaluations, and standard_deviations must have matching length.")

    sse = 0.0
    for observed, predicted, sigma in zip(
        reference_data,
        reference_evaluations,
        standard_deviations,
        strict=True,
    ):
        sigma_value = float(sigma)
        if sigma_value < 0.0:
            raise ValueError("Korali Normal likelihood rejects negative standard deviations.")
        numerator = float(observed) - float(predicted)
        if sigma_value == 0.0:
            if numerator == 0.0:
                diff = math.nan
            else:
                diff = math.copysign(math.inf, numerator)
        else:
            diff = numerator / sigma_value
        sse += diff * diff

    loglike = 0.0
    for sigma in standard_deviations:
        safe_sigma = float(sigma)
        if safe_sigma < STDEV_EPSILON:
            safe_sigma = STDEV_EPSILON
        loglike -= math.log(safe_sigma)

    loglike -= 0.5 * (len(reference_data) * LOG2PI + sse)
    return loglike
