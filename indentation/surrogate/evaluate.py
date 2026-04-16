#!/usr/bin/env python3

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
import torch

from meso_uq.surrogate import load_model_states

_TORCH_BATCH_PROFILE_ENV = "HUQ_TORCH_BATCH_PROFILE_JSONL"


def _append_jsonl(path: str, payload: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")


class Surrogate:
    def __init__(self, base_dir: str, device: str = "cpu") -> None:
        model_path = None
        for filename in ("microbubble_displacement_BEST.pkl", "microbubble_disp_BEST.pkl"):
            candidate = os.path.join(base_dir, filename)
            if os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            raise FileNotFoundError(
                f"Could not find indentation surrogate weights under {base_dir}"
            )
        self.model, self.xshift, self.xscale, self.yshift, self.yscale = load_model_states(
            model_path
        )
        self.xshift = np.array(self.xshift)
        self.xscale = np.array(self.xscale)
        self.yshift = np.array(self.yshift)
        self.yscale = np.array(self.yscale)
        self.device = torch.device(device)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.xshift_t = torch.as_tensor(self.xshift, dtype=torch.float32, device=self.device)
        self.xscale_t = torch.as_tensor(self.xscale, dtype=torch.float32, device=self.device)
        self.yshift_t = torch.as_tensor(self.yshift, dtype=torch.float32, device=self.device)
        self.yscale_t = torch.as_tensor(self.yscale, dtype=torch.float32, device=self.device)
        if hasattr(self.model, "register_buffer"):
            self.model.register_buffer("mesouq_xshift", self.xshift_t, persistent=False)
            self.model.register_buffer("mesouq_xscale", self.xscale_t, persistent=False)
            self.model.register_buffer("mesouq_yshift", self.yshift_t, persistent=False)
            self.model.register_buffer("mesouq_yscale", self.yscale_t, persistent=False)
        self.base_dir = base_dir
        self._pinned_params_buf: Optional[torch.Tensor] = None
        self._pinned_d0_buf: Optional[torch.Tensor] = None

    def _ensure_pinned_bufs(self, n: int) -> None:
        if self.device.type != "cuda":
            return
        if self._pinned_params_buf is None or self._pinned_params_buf.shape[0] < n:
            self._pinned_params_buf = torch.empty((n, 6), dtype=torch.float32).pin_memory()
            self._pinned_d0_buf = torch.empty((n,), dtype=torch.float32).pin_memory()

    def evaluate_indentation(self, x: List[float], forces: List[float]) -> List[float]:
        assert len(x) == 6
        if not forces:
            return []
        Yt, kb, b1, b2, a3, a4 = x
        force_arr = np.asarray(forces)
        n_force = len(force_arr)
        inputs = np.column_stack(
            [
                np.full(n_force, Yt),
                np.full(n_force, kb),
                np.full(n_force, b1),
                np.full(n_force, b2),
                np.full(n_force, a3),
                np.full(n_force, a4),
                force_arr,
            ]
        )
        x_rescaled = (inputs - self.xshift) / self.xscale
        X = torch.as_tensor(x_rescaled, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            y = self.model(X)
            y = y * self.yscale_t + self.yshift_t
        return torch.clamp(y[:, 0], min=0.0).detach().cpu().numpy().tolist()

    def evaluate_indentation_batch(
        self,
        x: np.ndarray,
        forces: List[float],
        d0: Optional[np.ndarray] = None,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        """Evaluate indentation displacements for a batch of particles."""
        params = np.asarray(x, dtype=np.float32)
        if params.ndim != 2 or params.shape[1] != 6:
            raise ValueError(f"Expected parameter array of shape [batch, 6], got {params.shape}")
        force_arr = np.asarray(forces, dtype=np.float32)
        if force_arr.ndim != 1:
            raise ValueError(f"Expected 1D force array, got shape {force_arr.shape}")
        if d0 is None:
            d0_arr = np.zeros(params.shape[0], dtype=np.float32)
        else:
            d0_arr = np.asarray(d0, dtype=np.float32)
            if d0_arr.shape != (params.shape[0],):
                raise ValueError(f"Expected d0 shape {(params.shape[0],)}, got {d0_arr.shape}")
        if params.shape[0] == 0:
            return np.zeros((0, force_arr.shape[0]), dtype=np.float32)

        requested_chunk_size = params.shape[0] if chunk_size <= 0 else chunk_size
        while True:
            try:
                return self._evaluate_indentation_batch_chunked(
                    params=params,
                    force_arr=force_arr,
                    d0_arr=d0_arr,
                    chunk_size=requested_chunk_size,
                )
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower()
                if not is_oom or self.device.type != "cuda" or requested_chunk_size <= 1:
                    raise
                torch.cuda.empty_cache()
                requested_chunk_size = max(1, requested_chunk_size // 2)

    def _evaluate_indentation_batch_chunked(
        self,
        params: np.ndarray,
        force_arr: np.ndarray,
        d0_arr: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        profile_path = os.getenv(_TORCH_BATCH_PROFILE_ENV)
        profile_enabled = bool(profile_path) and self.device.type == "cuda"
        total_start = time.perf_counter()
        h2d_seconds = 0.0
        compute_seconds = 0.0
        d2h_seconds = 0.0

        force_t = torch.as_tensor(force_arr, dtype=torch.float32, device=self.device)
        outputs = []

        with torch.inference_mode():
            for start in range(0, params.shape[0], chunk_size):
                stop = min(start + chunk_size, params.shape[0])
                batch = stop - start

                if self.device.type == "cuda":
                    self._ensure_pinned_bufs(batch)
                    self._pinned_params_buf[:batch].copy_(torch.from_numpy(params[start:stop]))
                    self._pinned_d0_buf[:batch].copy_(torch.from_numpy(d0_arr[start:stop]))
                    theta_t = self._pinned_params_buf[:batch].to(self.device, non_blocking=True)
                    d0_t = self._pinned_d0_buf[:batch].to(self.device, non_blocking=True)
                else:
                    theta_t = torch.as_tensor(
                        params[start:stop], dtype=torch.float32, device=self.device
                    )
                    d0_t = torch.as_tensor(
                        d0_arr[start:stop], dtype=torch.float32, device=self.device
                    )

                inputs = torch.cat(
                    [
                        theta_t.unsqueeze(1).expand(batch, force_t.shape[0], 6),
                        force_t.unsqueeze(0).expand(batch, -1).unsqueeze(-1),
                    ],
                    dim=-1,
                ).reshape(batch * force_t.shape[0], 7)
                inputs = (inputs - self.xshift_t) / self.xscale_t
                y = self.model(inputs)
                y = y * self.yscale_t + self.yshift_t
                batch_out = torch.clamp(
                    y[:, 0].reshape(batch, force_t.shape[0]) + d0_t.unsqueeze(1), min=0.0
                )
                outputs.append(batch_out.cpu())

        result = torch.cat(outputs, dim=0).numpy()

        if profile_enabled:
            total_seconds = time.perf_counter() - total_start
            _append_jsonl(
                profile_path,
                "{"
                f'"kind":"indentation_gpu_batch",'
                f'"model_dir":"{self.base_dir}",'
                f'"device":"{self.device}",'
                f'"particles":{int(params.shape[0])},'
                f'"forces":{int(force_arr.shape[0])},'
                f'"chunk_size":{int(chunk_size)},'
                f'"chunks":{int((params.shape[0] + chunk_size - 1) // chunk_size)},'
                f'"h2d_seconds":{h2d_seconds:.9f},'
                f'"compute_seconds":{compute_seconds:.9f},'
                f'"d2h_seconds":{d2h_seconds:.9f},'
                f'"total_seconds":{total_seconds:.9f}'
                "}",
            )

        return result
