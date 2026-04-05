#!/usr/bin/env python3

import os

import numpy as np
import torch

from meso_uq.surrogate import load_model_states


class Surrogate:
    def __init__(self, base_dir: str) -> None:
        self.cp, self.cp_xshift, self.cp_xscale, self.cp_yshift, self.cp_yscale = load_model_states(
            os.path.join(base_dir, "microbubble_force_BEST.pkl")
        )
        self.cp_xscale = np.array(self.cp_xscale)
        self.cp_xshift = np.array(self.cp_xshift)
        self.cp_yscale = np.array(self.cp_yscale)
        self.cp_yshift = np.array(self.cp_yshift)

    def evaluate_compression(self, x, disp):
        assert len(x) == 6
        if not disp:
            return []
        Yt, kb, b1, b2, a3, a4 = x
        disp_arr = np.asarray(disp)
        n_disp = len(disp_arr)
        inputs = np.column_stack([
            np.full(n_disp, Yt),
            np.full(n_disp, kb),
            np.full(n_disp, b1),
            np.full(n_disp, b2),
            np.full(n_disp, a3),
            np.full(n_disp, a4),
            disp_arr,
        ])
        x_rescaled = (inputs - self.cp_xshift) / self.cp_xscale
        X = torch.from_numpy(x_rescaled).float()
        with torch.no_grad():
            y = self.cp(X)
            y = y * self.cp_yscale + self.cp_yshift
        return np.maximum(0.0, y[:, 0].numpy()).tolist()
