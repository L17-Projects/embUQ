from __future__ import annotations

from math import erf, sqrt
from typing import Iterable

import numpy as np


def _ndtr(x: np.ndarray) -> np.ndarray:
    vec = np.vectorize(lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return vec(x)


def _validate_shapes(mean_samples: np.ndarray, std_samples: np.ndarray | None) -> np.ndarray | None:
    mean_samples = np.asarray(mean_samples, dtype=float)
    if mean_samples.ndim != 2:
        raise ValueError(f"Expected mean_samples with shape (n_samples, n_points), got {mean_samples.shape}")
    if std_samples is None:
        return None
    std_samples = np.asarray(std_samples, dtype=float)
    if std_samples.shape != mean_samples.shape:
        raise ValueError(f"mean_samples and std_samples must have the same shape, got {mean_samples.shape} and {std_samples.shape}")
    return np.maximum(std_samples, 0.0)


def gaussian_mixture_quantiles(mean_samples: np.ndarray, std_samples: np.ndarray, quantiles: Iterable[float], *, n_bisection_steps: int = 60, tail_sigma: float = 10.0) -> np.ndarray:
    std_samples = _validate_shapes(mean_samples, std_samples)
    mean_samples = np.asarray(mean_samples, dtype=float)
    quantiles = np.asarray(list(quantiles), dtype=float)
    if quantiles.ndim != 1 or quantiles.size == 0:
        raise ValueError("quantiles must be a non-empty 1D iterable")
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError(f"quantiles must lie in [0, 1], got {quantiles}")

    out = np.empty((quantiles.size, mean_samples.shape[1]), dtype=float)
    for j in range(mean_samples.shape[1]):
        mu = mean_samples[:, j]
        sd = std_samples[:, j]
        finite = np.isfinite(mu) & np.isfinite(sd)
        if not np.any(finite):
            out[:, j] = np.nan
            continue
        mu = mu[finite]
        sd = sd[finite]
        if np.all(sd == 0.0):
            out[:, j] = np.quantile(mu, quantiles)
            continue
        positive = sd > 0.0
        mu_pos = mu[positive]
        sd_pos = sd[positive]
        mu_zero = mu[~positive]
        lower = np.min(np.where(positive, mu - tail_sigma * sd, mu))
        upper = np.max(np.where(positive, mu + tail_sigma * sd, mu))

        def mixture_cdf(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=float)
            cdf = np.zeros_like(values)
            if mu_pos.size:
                z = (values[None, :] - mu_pos[:, None]) / sd_pos[:, None]
                cdf += np.mean(_ndtr(z), axis=0) * (mu_pos.size / mu.size)
            if mu_zero.size:
                cdf += np.mean(mu_zero[:, None] <= values[None, :], axis=0) * (mu_zero.size / mu.size)
            return cdf

        lo = np.full(quantiles.size, lower, dtype=float)
        hi = np.full(quantiles.size, upper, dtype=float)
        for _ in range(n_bisection_steps):
            mid = 0.5 * (lo + hi)
            cdf_mid = mixture_cdf(mid)
            lo = np.where(cdf_mid < quantiles, mid, lo)
            hi = np.where(cdf_mid >= quantiles, mid, hi)
        out[:, j] = 0.5 * (lo + hi)
    return out


def compute_interval_statistics(mean_samples: np.ndarray, std_samples: np.ndarray | None = None, *, percentiles: Iterable[int] = (90,), include_observation_noise: bool = True) -> dict[str, np.ndarray]:
    mean_samples = np.asarray(mean_samples, dtype=float)
    std_samples = _validate_shapes(mean_samples, std_samples)
    probs = {0.5}
    percentile_list = []
    for p in percentiles:
        lower = (100.0 - float(p)) / 200.0
        upper = 1.0 - lower
        probs.add(lower)
        probs.add(upper)
        percentile_list.append((p, lower, upper))
    ordered_probs = np.array(sorted(probs), dtype=float)
    stats = {"mean": np.mean(mean_samples, axis=0)}
    if include_observation_noise and std_samples is not None:
        predictive_var = np.var(mean_samples, axis=0) + np.mean(std_samples**2, axis=0)
        quantile_values = gaussian_mixture_quantiles(mean_samples, std_samples, ordered_probs)
    else:
        predictive_var = np.var(mean_samples, axis=0)
        quantile_values = np.quantile(mean_samples, ordered_probs, axis=0)
    quantile_map = {prob: quantile_values[idx] for idx, prob in enumerate(ordered_probs)}
    stats["median"] = quantile_map[0.5]
    stats["std"] = np.sqrt(predictive_var)
    for p, lower, upper in percentile_list:
        stats[f"ci_{int(p)}_lower"] = quantile_map[lower]
        stats[f"ci_{int(p)}_upper"] = quantile_map[upper]
    return stats
