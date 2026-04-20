"""Additional coverage tests for meso_uq.workflow_acceleration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from meso_uq.workflow_acceleration import (
    default_variable_names,
    expand_parameter_vector,
    subsample_parameters,
    write_propagation_state,
)


# ---------------------------------------------------------------------------
# subsample_parameters
# ---------------------------------------------------------------------------


def test_subsample_parameters_returns_all_when_num_samples_none() -> None:
    params = np.arange(20, dtype=np.float32).reshape(5, 4)
    result = subsample_parameters(params)
    np.testing.assert_array_equal(result, params)


def test_subsample_parameters_returns_all_when_num_samples_exceeds_len() -> None:
    params = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = subsample_parameters(params, num_samples=10)
    np.testing.assert_array_equal(result, params)


def test_subsample_parameters_subsamples_deterministically() -> None:
    params = np.arange(40, dtype=np.float32).reshape(10, 4)
    r1 = subsample_parameters(params, num_samples=3, seed=42)
    r2 = subsample_parameters(params, num_samples=3, seed=42)
    np.testing.assert_array_equal(r1, r2)
    assert r1.shape == (3, 4)


# ---------------------------------------------------------------------------
# default_variable_names
# ---------------------------------------------------------------------------


def test_default_variable_names_4() -> None:
    assert default_variable_names(4) == ["Yt", "kb", "d0", "sigma"]


def test_default_variable_names_8() -> None:
    assert default_variable_names(8) == ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]


def test_default_variable_names_other() -> None:
    names = default_variable_names(3)
    assert names == ["param_0", "param_1", "param_2"]


# ---------------------------------------------------------------------------
# expand_parameter_vector edge cases
# ---------------------------------------------------------------------------


def test_expand_parameter_vector_rejects_2d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        expand_parameter_vector(np.ones((2, 4)))


def test_expand_parameter_vector_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="4, 7, or 8"):
        expand_parameter_vector([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# write_propagation_state
# ---------------------------------------------------------------------------


def test_write_propagation_state_creates_files(tmp_path: Path) -> None:
    params = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    refs = np.array([[10.0, 20.0]], dtype=np.float32)

    result = write_propagation_state(str(tmp_path / "output"), params, refs)

    assert result.exists()
    gen_path = tmp_path / "output" / "gen00000001.json"
    assert gen_path.exists()

    with open(result) as f:
        data = json.load(f)
    assert data["Is Finished"] is True
    assert len(data["Samples"]) == 1
    assert data["Variables"][0]["Name"] == "Yt"


def test_write_propagation_state_with_std_and_metadata(tmp_path: Path) -> None:
    params = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    refs = np.array([[10.0]], dtype=np.float32)
    std = np.array([[0.5]], dtype=np.float32)

    result = write_propagation_state(
        str(tmp_path / "out"),
        params,
        refs,
        standard_deviation=std,
        variable_names=["a", "b", "c", "d"],
        metadata={"model": "test"},
    )

    with open(result) as f:
        data = json.load(f)
    assert data["Samples"][0]["Standard Deviation"] == [0.5]
    assert data["Variables"][0]["Name"] == "a"
    assert data["Metadata"]["model"] == "test"
