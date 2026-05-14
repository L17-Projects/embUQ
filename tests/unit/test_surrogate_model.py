"""Tests for meso_uq.surrogate.model — MLP, save/load, weight init."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from meso_uq.surrogate.model import (
    MLP,
    _install_legacy_pickle_aliases,
    init_weights,
    load_model_states,
    save_model_states,
)


# ---------------------------------------------------------------------------
# MLP construction & forward
# ---------------------------------------------------------------------------


def test_mlp_forward_shape() -> None:
    model = MLP(input_dims=4, output_dims=1, hl_dims=[16, 8])
    x = torch.randn(5, 4)
    y = model(x)
    assert y.shape == (5, 1)


def test_mlp_single_hidden_layer() -> None:
    model = MLP(input_dims=2, output_dims=3, hl_dims=[8])
    x = torch.randn(10, 2)
    y = model(x)
    assert y.shape == (10, 3)


def test_mlp_deterministic_with_seed() -> None:
    torch.manual_seed(42)
    m1 = MLP(input_dims=2, output_dims=1, hl_dims=[8])
    torch.manual_seed(42)
    m2 = MLP(input_dims=2, output_dims=1, hl_dims=[8])
    x = torch.randn(3, 2)
    assert torch.allclose(m1(x), m2(x))


# ---------------------------------------------------------------------------
# init_weights
# ---------------------------------------------------------------------------


def test_init_weights_sets_bias_to_001() -> None:
    model = MLP(input_dims=2, output_dims=1, hl_dims=[8])
    model.apply(init_weights)
    for layer in model.layers:
        if isinstance(layer, torch.nn.Linear):
            assert torch.allclose(layer.bias, torch.full_like(layer.bias, 0.01))


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    model = MLP(input_dims=3, output_dims=1, hl_dims=[8])
    model.apply(init_weights)
    xshift = [1.0, 2.0, 3.0]
    xscale = [0.5, 0.5, 0.5]
    yshift = [0.0]
    yscale = [1.0]

    path = tmp_path / "model.pkl"
    save_model_states(model, xshift=xshift, xscale=xscale, yshift=yshift, yscale=yscale, path=str(path))
    assert path.exists()

    loaded_model, lxs, lxc, lys, lyc = load_model_states(str(path))
    assert lxs == xshift
    assert lxc == xscale
    assert lys == yshift
    assert lyc == yscale

    x = torch.randn(2, 3)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded_model(x))


# ---------------------------------------------------------------------------
# _install_legacy_pickle_aliases
# ---------------------------------------------------------------------------


def test_install_legacy_pickle_aliases_creates_sys_modules_entries() -> None:
    _install_legacy_pickle_aliases()
    assert "learning.model" in sys.modules
    assert sys.modules["learning.model"].MLP is MLP
