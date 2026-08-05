from __future__ import annotations

import math
import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from meso_uq.config.models import EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE
from meso_uq.workflow_acceleration import get_fixed_parameters

LEGACY_YT_KB_SURROGATE_PARAMETERIZATION = "legacy_yt_kb"
DIRECT_KA_KB_SURROGATE_PARAMETERIZATION = "direct_ka_kb"
GENERIC_DIRECT_PARAMETER_ORDER = ("ka", "kb", "d0", "sigma")


def is_generic_direct_phase1_contract(config: Mapping[str, object]) -> bool:
    return str(config.get("phase1_contract_mode") or "") == EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE


def surrogate_parameterization_for_experiment(experiment: Any) -> str:
    value = getattr(experiment, "surrogate_parameterization", LEGACY_YT_KB_SURROGATE_PARAMETERIZATION)
    normalized = str(value).strip().lower()
    if normalized not in {
        LEGACY_YT_KB_SURROGATE_PARAMETERIZATION,
        DIRECT_KA_KB_SURROGATE_PARAMETERIZATION,
    }:
        raise ValueError(f"Unsupported EMB surrogate parameterization '{value}'.")
    return normalized


@lru_cache(maxsize=None)
def load_emb_runtime_defaults(project_root: str | Path, modality: str = "indentation") -> dict[str, Any]:
    path = Path(project_root).resolve() / "emb" / modality / "src" / "parameters-default.emb.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid runtime defaults at {path}; expected a mapping.")
    return dict(payload)


def emb_elastic_stiffness_unit_n_per_m(defaults: Mapping[str, Any], *, label: str = "EMB") -> float:
    required = ("ul", "kbol", "t0", "energyFactor")
    missing = [key for key in required if key not in defaults]
    if missing:
        raise ValueError(f"{label} runtime defaults are missing unit constants: {missing}")
    ul = float(defaults["ul"])
    kbol = float(defaults["kbol"])
    t0 = float(defaults["t0"])
    energy_factor = float(defaults["energyFactor"])
    if any(not math.isfinite(value) or value <= 0.0 for value in (ul, kbol, t0, energy_factor)):
        raise ValueError(f"{label} runtime defaults contain invalid unit constants.")
    ue = energy_factor * kbol * t0
    return ue / ul**2


def emb_dpd_ka_to_chi_n_per_m(ka: float, defaults: Mapping[str, Any], *, label: str = "EMB") -> float:
    if "fscale" not in defaults:
        raise ValueError(f"{label} runtime defaults are missing fscale for ka/chi conversion.")
    fscale = float(defaults["fscale"])
    if not math.isfinite(fscale) or fscale <= 0.0:
        raise ValueError(f"{label} runtime defaults contain invalid fscale={fscale}.")
    return float(ka) * emb_elastic_stiffness_unit_n_per_m(defaults, label=label) / fscale


def emb_chi_n_per_m_to_dpd_ka(chi_n_per_m: float, defaults: Mapping[str, Any], *, label: str = "EMB") -> float:
    if "fscale" not in defaults:
        raise ValueError(f"{label} runtime defaults are missing fscale for chi/ka conversion.")
    fscale = float(defaults["fscale"])
    if not math.isfinite(fscale) or fscale <= 0.0:
        raise ValueError(f"{label} runtime defaults contain invalid fscale={fscale}.")
    return float(chi_n_per_m) * fscale / emb_elastic_stiffness_unit_n_per_m(defaults, label=label)


def _ka_per_yt(defaults: Mapping[str, Any], *, modality: str) -> float:
    required = ("shell_th", "nu", "fscale", "th_fac", "yt_fac")
    missing = [key for key in required if key not in defaults]
    if missing:
        raise ValueError(
            f"{modality} runtime defaults are missing compatibility constants for ka/Yt conversion: {missing}"
        )
    shell_th = float(defaults["th_fac"]) * float(defaults["shell_th"])
    yt_fac = float(defaults["yt_fac"])
    nu = float(defaults["nu"])
    fscale = float(defaults["fscale"])
    if any(not math.isfinite(value) or value <= 0.0 for value in (shell_th, yt_fac, fscale)):
        raise ValueError(f"{modality} runtime defaults contain invalid ka/Yt conversion constants.")
    if not math.isfinite(nu) or math.isclose(1.0 - nu, 0.0):
        raise ValueError(f"{modality} runtime defaults contain an invalid Poisson ratio for ka/Yt conversion.")
    stiffness_unit = emb_elastic_stiffness_unit_n_per_m(defaults, label=modality)
    return fscale * yt_fac * shell_th / (2.0 * (1.0 - nu)) / stiffness_unit


def ka_to_legacy_yt(ka: float, *, modality: str, project_root: str | Path) -> float:
    defaults = load_emb_runtime_defaults(project_root, modality)
    return float(ka) / _ka_per_yt(defaults, modality=modality)


def direct_parameters_to_legacy_vector(
    params: Any,
    *,
    config: Mapping[str, object],
    modality: str,
    project_root: str | Path,
) -> np.ndarray:
    vector = np.asarray(params, dtype=np.float32).reshape(-1)
    if vector.shape != (4,):
        raise ValueError(
            "Generic direct EMB parameterization expects [ka, kb, d0, sigma], "
            f"got shape {vector.shape}."
        )
    fixed = get_fixed_parameters(config)
    yt = ka_to_legacy_yt(float(vector[0]), modality=modality, project_root=project_root)
    return np.asarray(
        [
            yt,
            float(vector[1]),
            float(fixed.get("b1", 0.0)),
            float(fixed.get("b2", 0.0)),
            float(fixed.get("a3", 0.0)),
            float(fixed.get("a4", 0.0)),
            float(vector[2]),
            float(vector[3]),
        ],
        dtype=np.float32,
    )


def direct_parameters_to_legacy_batch(
    batch_params: Any,
    *,
    config: Mapping[str, object],
    modality: str,
    project_root: str | Path,
) -> np.ndarray:
    batch = np.asarray(batch_params, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[1] != 4:
        raise ValueError(
            "Generic direct EMB batch parameterization expects Batch Parameters "
            f"with shape [batch, 4], got {batch.shape}."
        )
    return np.vstack(
        [
            direct_parameters_to_legacy_vector(
                row,
                config=config,
                modality=modality,
                project_root=project_root,
            )
            for row in batch
        ]
    )


def adapt_sample_for_legacy_surrogate(
    sample: Mapping[str, Any],
    *,
    config: Mapping[str, object],
    modality: str,
    project_root: str | Path,
) -> dict[str, Any]:
    return {
        "Parameters": direct_parameters_to_legacy_vector(
            sample["Parameters"],
            config=config,
            modality=modality,
            project_root=project_root,
        ).tolist()
    }


def adapt_batch_sample_for_legacy_surrogate(
    sample: Mapping[str, Any],
    *,
    config: Mapping[str, object],
    modality: str,
    project_root: str | Path,
) -> dict[str, Any]:
    return {
        "Batch Parameters": direct_parameters_to_legacy_batch(
            sample["Batch Parameters"],
            config=config,
            modality=modality,
            project_root=project_root,
        )
    }


def resolve_direct_compression_surrogate_surface() -> tuple[Any, Any, Any]:
    module = importlib.import_module("emb.compression.evalkit.posterior_compression")
    required = (
        "preload_compression_surrogate_direct",
        "compute_compression_surrogate_direct",
        "compute_compression_surrogate_batch_direct",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "Direct compression ka/kb surrogate surface is not available. "
            "Expected emb.compression.evalkit.posterior_compression to export "
            f"{required}; missing {missing}."
        )
    return (
        getattr(module, "preload_compression_surrogate_direct"),
        getattr(module, "compute_compression_surrogate_direct"),
        getattr(module, "compute_compression_surrogate_batch_direct"),
    )


__all__ = [
    "DIRECT_KA_KB_SURROGATE_PARAMETERIZATION",
    "GENERIC_DIRECT_PARAMETER_ORDER",
    "LEGACY_YT_KB_SURROGATE_PARAMETERIZATION",
    "adapt_batch_sample_for_legacy_surrogate",
    "adapt_sample_for_legacy_surrogate",
    "direct_parameters_to_legacy_batch",
    "direct_parameters_to_legacy_vector",
    "emb_chi_n_per_m_to_dpd_ka",
    "emb_dpd_ka_to_chi_n_per_m",
    "emb_elastic_stiffness_unit_n_per_m",
    "is_generic_direct_phase1_contract",
    "ka_to_legacy_yt",
    "load_emb_runtime_defaults",
    "resolve_direct_compression_surrogate_surface",
    "surrogate_parameterization_for_experiment",
]
