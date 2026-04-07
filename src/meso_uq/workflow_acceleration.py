"""
Helpers for optional GPU-accelerated workflow paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

FIXABLE_PARAMETER_ORDER = ("b1", "b2", "a3", "a4")
FULL_VARIABLE_ORDER = ("Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma")
HIERARCHICAL_VARIABLE_ORDER = tuple(name for name in FULL_VARIABLE_ORDER if name != "sigma")


def require_single_rank(comm, context: str) -> None:
    if comm.Get_size() != 1:
        raise ValueError(
            f"{context} requires a single MPI rank. Launch it with plain python or mpirun -np 1."
        )


def configure_korali_conduit(
    engine,
    *,
    mpi_ranks: int,
    ranks_per_worker: int = 1,
    concurrent_jobs: int = 1,
) -> None:
    if isinstance(engine, dict) and "Conduit" not in engine:
        engine["Conduit"] = {}
    if mpi_ranks <= 1:
        if isinstance(engine, dict) and not engine["Conduit"]:
            engine.pop("Conduit")
        return
    if mpi_ranks > 1:
        engine["Conduit"]["Type"] = "Distributed"
        engine["Conduit"]["Ranks Per Worker"] = ranks_per_worker
        return


def to_korali_path(path: str, base_dir: Optional[str] = None) -> str:
    if not path:
        return path
    if not Path(path).is_absolute():
        return path
    if base_dir is None:
        base_dir = str(Path.cwd())
    try:
        return os.path.relpath(path, start=base_dir)
    except ValueError:
        return path


def subsample_parameters(
    param_samples: np.ndarray,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    if num_samples is None or num_samples >= len(param_samples):
        return param_samples

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(param_samples), size=num_samples, replace=False)
    return param_samples[indices]


def default_variable_names(num_params: int) -> list[str]:
    if num_params == 4:
        return ["Yt", "kb", "d0", "sigma"]
    if num_params == 8:
        return ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]
    return [f"param_{i}" for i in range(num_params)]


def get_fixed_parameters(config: Mapping[str, object]) -> dict[str, float]:
    fixed_params = config.get("fixed_params") or {}
    if not isinstance(fixed_params, Mapping):
        raise ValueError(f"Expected fixed_params to be a mapping, got {type(fixed_params).__name__}")
    return {
        name: float(fixed_params[name])
        for name in FIXABLE_PARAMETER_ORDER
        if name in fixed_params
    }


def active_variable_names(config: Mapping[str, object]) -> list[str]:
    fixed_params = get_fixed_parameters(config)
    return [name for name in FULL_VARIABLE_ORDER if name not in fixed_params]


def active_hierarchical_variable_names(config: Mapping[str, object]) -> list[str]:
    return [name for name in active_variable_names(config) if name != "sigma"]


def expand_reduced_parameters(
    param_samples: np.ndarray,
    fixed_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Expand reduced 4-parameter samples to the full 8-parameter layout.

    The reduced workflow stores `[Yt, kb, d0, sigma]`. The full surrogate expects
    `[Yt, kb, b1, b2, a3, a4, d0, sigma]`.
    """
    params = np.asarray(param_samples, dtype=np.float32)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError(f"Expected reduced parameter array of shape [batch, 4], got {params.shape}")

    if fixed_params is None:
        fixed_params = {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}

    expanded = np.column_stack(
        [
            params[:, 0],
            params[:, 1],
            np.full(params.shape[0], float(fixed_params["b1"]), dtype=np.float32),
            np.full(params.shape[0], float(fixed_params["b2"]), dtype=np.float32),
            np.full(params.shape[0], float(fixed_params["a3"]), dtype=np.float32),
            np.full(params.shape[0], float(fixed_params["a4"]), dtype=np.float32),
            params[:, 2],
            params[:, 3],
        ]
    )
    return expanded


def expand_parameter_vector(
    params: Sequence[float] | np.ndarray,
    fixed_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Normalize legacy/full/reduced samples to the full 8-parameter layout."""
    arr = np.asarray(params, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D parameter vector, got shape {arr.shape}")
    if arr.shape[0] == 8:
        return arr
    if arr.shape[0] == 7:
        return np.array(
            [arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], 0.0, arr[6]],
            dtype=np.float32,
        )
    if arr.shape[0] == 4:
        return expand_reduced_parameters(arr.reshape(1, -1), fixed_params=fixed_params)[0]
    raise ValueError(f"Expected 4, 7, or 8 parameters, got shape {arr.shape}")


def phase1_prior_specs(
    config: Mapping[str, object],
    *,
    prior_d0: Optional[Sequence[float]] = None,
    prior_sigma: Optional[Sequence[float]] = None,
) -> list[tuple[str, Sequence[float]]]:
    def _bounds_for(name: str) -> Sequence[float]:
        if name == "d0":
            return prior_d0 if prior_d0 is not None else config.get("prior_d0", [0.0, 0.5])
        if name == "sigma":
            return prior_sigma if prior_sigma is not None else config["prior_sigma"]
        return config[f"prior_{name}"]

    return [(name, _bounds_for(name)) for name in active_variable_names(config)]


def phase2_hyperprior_specs(config: Mapping[str, object]) -> list[tuple[str, Sequence[float], Sequence[float]]]:
    def _required(key: str):
        if key not in config:
            raise KeyError(f"Missing required hyperprior bound '{key}'")
        return config[key]

    def _bounds_for(name: str) -> tuple[Sequence[float], Sequence[float]]:
        if name == "d0":
            return (
                config.get("hyperprior_mu_d0", [0.0, 0.5]),
                config.get("hyperprior_sigma_d0", [0.0, 0.3]),
            )
        return (
            _required(f"hyperprior_mu_{name}"),
            _required(f"hyperprior_sigma_{name}"),
        )

    return [(name, *_bounds_for(name)) for name in active_hierarchical_variable_names(config)]


def write_propagation_state(
    output_dir: str,
    param_samples: np.ndarray,
    reference_evaluations: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
    variable_names: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if variable_names is None:
        variable_names = default_variable_names(param_samples.shape[1])

    samples = []
    for sample_id in range(param_samples.shape[0]):
        sample = {
            "Sample Id": int(sample_id),
            "Current Generation": 1,
            "Parameters": param_samples[sample_id].tolist(),
            "Reference Evaluations": reference_evaluations[sample_id].tolist(),
        }
        if standard_deviation is not None:
            sample["Standard Deviation"] = standard_deviation[sample_id].tolist()
        samples.append(sample)

    data = {
        "Type": "Experiment",
        "Current Generation": 1,
        "Is Finished": True,
        "Store Sample Information": True,
        "Problem": {"Type": "Propagation"},
        "Solver": {
            "Type": "Executor",
            "Executions Per Generation": int(param_samples.shape[0]),
            "Sample Database": param_samples.tolist(),
        },
        "Results": {
            "Sample Database": param_samples.tolist(),
        },
        "Variables": [{"Name": name} for name in variable_names],
        "Samples": samples,
    }
    if metadata is not None:
        data["Metadata"] = metadata

    latest_path = output_path / "latest"
    generation_path = output_path / "gen00000001.json"
    with latest_path.open("w") as handle:
        json.dump(data, handle, indent=2)
    with generation_path.open("w") as handle:
        json.dump(data, handle, indent=2)

    return latest_path
