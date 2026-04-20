#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from meso_uq.surrogate.bnn import VariationalBNNPredictor

_ARTIFACT_CANDIDATES = (
    "microbubble_force_BNN.pt",
    "microbubble_force_BNN.pth",
)


def _resolve_artifact_path(base_dir: str) -> str:
    for filename in _ARTIFACT_CANDIDATES:
        candidate = Path(base_dir) / filename
        if candidate.exists():
            return str(candidate)
    discovered = sorted(
        {
            *Path(base_dir).glob("*BNN*.pt"),
            *Path(base_dir).glob("*BNN*.pth"),
            *Path(base_dir).glob("*bnn*.pt"),
            *Path(base_dir).glob("*bnn*.pth"),
        }
    )
    if len(discovered) == 1:
        return str(discovered[0])
    if len(discovered) > 1:
        options = ", ".join(str(path.name) for path in discovered)
        raise FileNotFoundError(
            f"Found multiple compression BNN artifacts under {base_dir}. "
            f"Pass explicit path through config or keep one artifact. Candidates: {options}"
        )
    joined = ", ".join(_ARTIFACT_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find compression BNN artifact under {base_dir}. Expected one of: {joined}"
    )


class Surrogate:
    def __init__(self, base_dir: str, device: str = "cpu") -> None:
        artifact_path = _resolve_artifact_path(base_dir)
        self.predictor = VariationalBNNPredictor(artifact_path, device=device)

    def evaluate_compression(
        self,
        x: List[float],
        disp: List[float],
        *,
        predictive_mc_samples: int = 32,
        predictive_mc_chunk_size: int = 8,
    ) -> tuple[List[float], List[float]]:
        if len(x) != 6:
            raise ValueError(f"Expected 6 compression parameters, got {len(x)}")
        if not disp:
            return [], []

        Yt, kb, b1, b2, a3, a4 = x
        disp_arr = np.asarray(disp, dtype=np.float32)
        n_disp = disp_arr.shape[0]
        inputs = np.column_stack(
            [
                np.full(n_disp, Yt, dtype=np.float32),
                np.full(n_disp, kb, dtype=np.float32),
                np.full(n_disp, b1, dtype=np.float32),
                np.full(n_disp, b2, dtype=np.float32),
                np.full(n_disp, a3, dtype=np.float32),
                np.full(n_disp, a4, dtype=np.float32),
                disp_arr,
            ]
        )
        mean, std = self.predictor.predict_mean_std(
            inputs,
            predictive_mc_samples=predictive_mc_samples,
            predictive_mc_chunk_size=predictive_mc_chunk_size,
        )
        return np.maximum(0.0, mean).tolist(), std.tolist()

    def evaluate_compression_batch(
        self,
        x: np.ndarray,
        disp: List[float],
        d0: Optional[np.ndarray] = None,
        chunk_size: int = 2048,
        *,
        predictive_mc_samples: int = 32,
        predictive_mc_chunk_size: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        params = np.asarray(x, dtype=np.float32)
        if params.ndim != 2 or params.shape[1] != 6:
            raise ValueError(f"Expected parameter array of shape [batch, 6], got {params.shape}")
        disp_arr = np.asarray(disp, dtype=np.float32)
        if disp_arr.ndim != 1:
            raise ValueError(f"Expected 1D displacement array, got shape {disp_arr.shape}")
        if d0 is None:
            d0_arr = np.zeros(params.shape[0], dtype=np.float32)
        else:
            d0_arr = np.asarray(d0, dtype=np.float32)
            if d0_arr.shape != (params.shape[0],):
                raise ValueError(f"Expected d0 shape {(params.shape[0],)}, got {d0_arr.shape}")
        if params.shape[0] == 0:
            n_disp = disp_arr.shape[0]
            return (
                np.zeros((0, n_disp), dtype=np.float32),
                np.zeros((0, n_disp), dtype=np.float32),
            )

        disp_corr = np.maximum(disp_arr[None, :] - d0_arr[:, None], 0.0)
        n_particles, n_disp = params.shape[0], disp_arr.shape[0]
        theta = np.repeat(params[:, None, :], n_disp, axis=1)
        inputs = np.concatenate([theta, disp_corr[:, :, None]], axis=2).reshape(-1, 7)
        mean_flat, std_flat = self.predictor.predict_mean_std(
            inputs,
            predictive_mc_samples=predictive_mc_samples,
            predictive_mc_chunk_size=predictive_mc_chunk_size,
        )
        mean = np.maximum(0.0, mean_flat.reshape(n_particles, n_disp))
        std = std_flat.reshape(n_particles, n_disp)
        return mean, std
