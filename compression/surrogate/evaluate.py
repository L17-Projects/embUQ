#!/usr/bin/env python3

import os

import numpy as np
import torch

from meso_uq.surrogate import load_model_states


class Surrogate:
    def __init__(self, base_dir: str, device: str = "cpu") -> None:
        self.cp, self.cp_xshift, self.cp_xscale, self.cp_yshift, self.cp_yscale = load_model_states(
            os.path.join(base_dir, "microbubble_force_BEST.pkl")
        )
        self.cp_xscale = np.array(self.cp_xscale)
        self.cp_xshift = np.array(self.cp_xshift)
        self.cp_yscale = np.array(self.cp_yscale)
        self.cp_yshift = np.array(self.cp_yshift)
        self.device = torch.device(device)
        if hasattr(self.cp, "to"):
            self.cp = self.cp.to(self.device)
        if hasattr(self.cp, "eval"):
            self.cp.eval()
        self.cp_xscale_t = torch.as_tensor(self.cp_xscale, dtype=torch.float32, device=self.device)
        self.cp_xshift_t = torch.as_tensor(self.cp_xshift, dtype=torch.float32, device=self.device)
        self.cp_yscale_t = torch.as_tensor(self.cp_yscale, dtype=torch.float32, device=self.device)
        self.cp_yshift_t = torch.as_tensor(self.cp_yshift, dtype=torch.float32, device=self.device)
        if hasattr(self.cp, "register_buffer"):
            self.cp.register_buffer("mesouq_cp_xscale", self.cp_xscale_t, persistent=False)
            self.cp.register_buffer("mesouq_cp_xshift", self.cp_xshift_t, persistent=False)
            self.cp.register_buffer("mesouq_cp_yscale", self.cp_yscale_t, persistent=False)
            self.cp.register_buffer("mesouq_cp_yshift", self.cp_yshift_t, persistent=False)

    def evaluate_compression(self, x, disp):
        assert len(x) == 6
        if not disp:
            return []
        Yt, kb, b1, b2, a3, a4 = x
        disp_arr = np.asarray(disp)
        n_disp = len(disp_arr)
        inputs = np.column_stack(
            [
                np.full(n_disp, Yt),
                np.full(n_disp, kb),
                np.full(n_disp, b1),
                np.full(n_disp, b2),
                np.full(n_disp, a3),
                np.full(n_disp, a4),
                disp_arr,
            ]
        )
        x_rescaled = (inputs - self.cp_xshift) / self.cp_xscale
        X = torch.as_tensor(x_rescaled, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            y = self.cp(X)
            y = y * self.cp_yscale_t + self.cp_yshift_t
        return torch.clamp(y[:, 0], min=0.0).detach().cpu().numpy().tolist()
