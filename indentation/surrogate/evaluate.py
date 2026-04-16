#!/usr/bin/env python3

import os

import numpy as np
import torch

from meso_uq.surrogate import load_model_states


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

    def evaluate_indentation(self, x, forces):
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
