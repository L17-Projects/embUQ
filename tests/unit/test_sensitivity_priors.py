"""Tests for meso_uq.sensitivity.priors (previously 0% coverage)."""
from __future__ import annotations

import pytest

from meso_uq.sensitivity.priors import UniformPrior, VarConfig, build_problem


def test_uniform_prior_stores_bounds():
    p = UniformPrior(0.0, 1.0)
    assert p.low() == 0.0
    assert p.high() == 1.0


def test_uniform_prior_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="a < b"):
        UniformPrior(1.0, 0.0)


def test_uniform_prior_rejects_equal_bounds():
    with pytest.raises(ValueError, match="a < b"):
        UniformPrior(5.0, 5.0)


def test_var_config_delegates_to_prior():
    p = UniformPrior(2.0, 8.0)
    vc = VarConfig(name="Yt", prior=p)
    assert vc.name == "Yt"
    assert vc.low() == 2.0
    assert vc.high() == 8.0


def test_build_problem_correct_structure():
    configs = [
        VarConfig("Yt", UniformPrior(1e6, 1e8)),
        VarConfig("kb", UniformPrior(1e3, 1e5)),
    ]
    problem = build_problem(configs)
    assert problem["num_vars"] == 2
    assert problem["names"] == ["Yt", "kb"]
    assert problem["bounds"] == [[1e6, 1e8], [1e3, 1e5]]


def test_build_problem_single_var():
    configs = [VarConfig("sigma", UniformPrior(0.01, 0.1))]
    problem = build_problem(configs)
    assert problem["num_vars"] == 1
    assert problem["names"] == ["sigma"]
