from pathlib import Path

import numpy as np
import pytest

from meso_uq.workflow_acceleration import (
    configure_korali_conduit,
    default_variable_names,
    expand_reduced_parameters,
    subsample_parameters,
    to_korali_path,
    write_propagation_state,
)


def test_default_variable_names_for_4_and_8_params():
    assert default_variable_names(4) == ["Yt", "kb", "d0", "sigma"]
    assert default_variable_names(8) == ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]


def test_expand_reduced_parameters_shape_and_positions():
    samples = np.array([[10.0, 20.0, 0.5, 0.1]])
    expanded = expand_reduced_parameters(samples)
    assert expanded.shape == (1, 8)
    assert expanded[0, 0] == pytest.approx(10.0)
    assert expanded[0, 1] == pytest.approx(20.0)
    assert expanded[0, 6] == pytest.approx(0.5)
    assert expanded[0, 7] == pytest.approx(0.1)


def test_subsample_parameters_respects_size():
    arr = np.arange(400, dtype=float).reshape(100, 4)
    result = subsample_parameters(arr, num_samples=30, seed=0)
    assert result.shape == (30, 4)


def test_to_korali_path_relativizes_absolute_path(tmp_path):
    abs_path = str(tmp_path / "results" / "phase_1")
    rel = to_korali_path(abs_path, base_dir=str(tmp_path))
    assert rel == "results/phase_1"


def test_configure_korali_conduit_switches_single_vs_multi_rank_modes():
    engine = {}
    configure_korali_conduit(engine, mpi_ranks=1, ranks_per_worker=2, concurrent_jobs=3)
    assert "Conduit" not in engine

    engine = {}
    configure_korali_conduit(engine, mpi_ranks=4, ranks_per_worker=2, concurrent_jobs=3)
    assert engine["Conduit"]["Type"] == "Distributed"
    assert engine["Conduit"]["Ranks Per Worker"] == 2


def test_write_propagation_state_creates_expected_files(tmp_path):
    params = np.random.default_rng(0).random((5, 4)).astype(np.float32)
    evals = np.random.default_rng(1).random((5, 10)).astype(np.float32)
    latest = write_propagation_state(str(tmp_path / "out"), params, evals)
    assert latest.name == "latest"
    assert latest.exists()
    assert (tmp_path / "out" / "gen00000001.json").exists()
    assert isinstance(latest, Path)
