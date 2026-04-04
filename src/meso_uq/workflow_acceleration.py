"""
Helpers for optional GPU-accelerated workflow paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


def require_single_rank(comm, context: str) -> None:
    if comm.Get_size() != 1:
        raise ValueError(
            f"{context} requires a single MPI rank. Launch it with plain python or mpirun -np 1."
        )


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
