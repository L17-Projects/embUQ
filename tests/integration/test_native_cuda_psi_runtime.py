from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.support.native_cuda_psi import ConditionalPrior, ParameterSource, SubProblemSamples
from tests.support.native_cuda_runtime import (
    NativeCudaRuntimeUnavailable,
    expected_batch_log_likelihoods,
    expected_subproblem_log_likelihoods,
    launch_psi_native_cuda_kernel,
)


pytestmark = [pytest.mark.cuda, pytest.mark.korali]


def _assert_close_or_negative_infinity(actual: float, expected: float) -> None:
    if expected == -math.inf:
        assert actual == -math.inf
    else:
        assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


def _launch_or_skip(**kwargs):
    try:
        return launch_psi_native_cuda_kernel(**kwargs)
    except NativeCudaRuntimeUnavailable as exc:
        pytest.skip(f"NativeCuda runtime harness requires CUDA driver and NVRTC on a GPU node: {exc}")


def test_psi_native_cuda_kernel_launch_matches_reference_math() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    priors = (
        ConditionalPrior(
            kind="normal",
            sample_dimension=0,
            parameter_a=ParameterSource.variable(0),
            parameter_b=ParameterSource.variable(1),
        ),
        ConditionalPrior(
            kind="uniform",
            sample_dimension=1,
            parameter_a=ParameterSource.constant(-1.0),
            parameter_b=ParameterSource.variable(2),
        ),
    )
    subproblems = (
        SubProblemSamples(
            coordinates_by_dimension=((0.0, 0.5, 1.0), (-0.5, 0.5, 1.5)),
            base_log_weights=(0.0, -0.25, -0.5),
        ),
        SubProblemSamples(
            coordinates_by_dimension=((1.0, 1.5), (-0.75, 0.25)),
            base_log_weights=(-0.1, -0.2),
        ),
    )
    batch_parameters = (
        (0.25, 1.2, 2.0),
        (1.0, 0.8, -2.0),
        (0.5, 0.0, 2.0),
    )
    batch_log_priors = (0.0, 0.0, math.nan)

    actual = _launch_or_skip(
        repo_root=repo_root,
        priors=priors,
        subproblems=subproblems,
        batch_parameters=batch_parameters,
        batch_log_priors=batch_log_priors,
    )

    expected_partials = expected_subproblem_log_likelihoods(
        priors=priors,
        subproblems=subproblems,
        batch_parameters=batch_parameters,
    )
    expected_batch = expected_batch_log_likelihoods(
        priors=priors,
        subproblems=subproblems,
        batch_parameters=batch_parameters,
        batch_log_priors=batch_log_priors,
    )

    assert actual.compute_capability[0] >= 1
    assert len(actual.subproblem_log_likelihoods) == len(expected_partials)
    for actual_row, expected_row in zip(actual.subproblem_log_likelihoods, expected_partials, strict=True):
        for actual_value, expected_value in zip(actual_row, expected_row, strict=True):
            _assert_close_or_negative_infinity(actual_value, expected_value)

    for actual_value, expected_value in zip(actual.batch_log_likelihoods, expected_batch, strict=True):
        _assert_close_or_negative_infinity(actual_value, expected_value)


def test_psi_native_cuda_kernel_launch_handles_no_dynamic_priors() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    priors = ()
    subproblems = (
        SubProblemSamples(
            coordinates_by_dimension=(),
            base_log_weights=(0.0, -0.25, -0.5),
        ),
        SubProblemSamples(
            coordinates_by_dimension=(),
            base_log_weights=(-0.1, -0.2),
        ),
    )
    batch_parameters = ((0.0,), (1.0,))

    actual = _launch_or_skip(
        repo_root=repo_root,
        priors=priors,
        subproblems=subproblems,
        batch_parameters=batch_parameters,
    )

    expected_partials = expected_subproblem_log_likelihoods(
        priors=priors,
        subproblems=subproblems,
        batch_parameters=batch_parameters,
    )
    expected_batch = expected_batch_log_likelihoods(
        priors=priors,
        subproblems=subproblems,
        batch_parameters=batch_parameters,
    )

    assert actual.compute_capability[0] >= 1
    for actual_row, expected_row in zip(actual.subproblem_log_likelihoods, expected_partials, strict=True):
        for actual_value, expected_value in zip(actual_row, expected_row, strict=True):
            _assert_close_or_negative_infinity(actual_value, expected_value)
    for actual_value, expected_value in zip(actual.batch_log_likelihoods, expected_batch, strict=True):
        _assert_close_or_negative_infinity(actual_value, expected_value)
