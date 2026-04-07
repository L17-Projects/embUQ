#!/usr/bin/env python3

import os

import numpy as np
import torch

from meso_uq.surrogate import load_model_states


class Surrogate:
    def __init__(self, base_dir: str) -> None:
        model_path = None
        for filename in ("microbubble_displacement_BEST.pkl", "microbubble_disp_BEST.pkl"):
            candidate = os.path.join(base_dir, filename)
            if os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            raise FileNotFoundError(f"Could not find indentation surrogate weights under {base_dir}")
        self.model, self.xshift, self.xscale, self.yshift, self.yscale = load_model_states(model_path)
        self.xshift = np.array(self.xshift)
        self.xscale = np.array(self.xscale)
        self.yshift = np.array(self.yshift)
        self.yscale = np.array(self.yscale)

    def evaluate_indentation(self, x, forces):
        assert len(x) == 6
        if not forces:
            return []
        Yt, kb, b1, b2, a3, a4 = x
        force_arr = np.asarray(forces)
        n_force = len(force_arr)
        inputs = np.column_stack([
            np.full(n_force, Yt),
            np.full(n_force, kb),
            np.full(n_force, b1),
            np.full(n_force, b2),
            np.full(n_force, a3),
            np.full(n_force, a4),
            force_arr,
        ])
        x_rescaled = (inputs - self.xshift) / self.xscale
        X = torch.from_numpy(x_rescaled).float()
        with torch.no_grad():
            y = self.model(X)
            y = y * self.yscale + self.yshift
        return np.maximum(0.0, y[:, 0].numpy()).tolist()
