from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.support.native_cuda_psi import (
    KERNEL_ENTRYPOINT,
    ConditionalPrior,
    ParameterSource,
    SubProblemSamples,
    batch_log_likelihood,
    extract_psi_native_cuda_kernel_source,
    read_vendored_psi_native_cuda_kernel_source,
    subproblem_log_likelihood,
)


pytestmark = pytest.mark.korali


def test_extracts_vendored_psi_native_cuda_kernel_source() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    kernel_source = read_vendored_psi_native_cuda_kernel_source(repo_root)

    assert KERNEL_ENTRYPOINT in kernel_source
    assert "const double *batchParameters" in kernel_source
    assert "const double * const *subProblemCoordinates" in kernel_source
    assert "priorKinds[priorId] == 0" in kernel_source
    assert "parameterB <= parameterA" in kernel_source
    assert "blockMax + log(sharedSum[0])" in kernel_source


def test_kernel_source_extraction_rejects_missing_or_malformed_source() -> None:
    with pytest.raises(ValueError, match="Could not find kPsiNativeCudaKernelSource"):
        extract_psi_native_cuda_kernel_source("constexpr const char *other = R\"CUDA()CUDA\";")

    malformed = '''
constexpr const char *kPsiNativeCudaKernelSource = R"CUDA(
extern "C" __global__
void psiBatchLogLikelihoodKernel() {}
)CUDA";
'''
    with pytest.raises(ValueError, match="missing required markers"):
        extract_psi_native_cuda_kernel_source(malformed)


def test_reference_normal_prior_matches_hand_computed_logsumexp() -> None:
    prior = ConditionalPrior(
        kind="normal",
        sample_dimension=0,
        parameter_a=ParameterSource.variable(0),
        parameter_b=ParameterSource.constant(2.0),
    )
    subproblem = SubProblemSamples(
        coordinates_by_dimension=((1.0, 2.0, 3.0),),
        base_log_weights=(0.0, math.log(0.5), math.log(0.25)),
    )

    actual = subproblem_log_likelihood(
        priors=(prior,),
        parameters=(1.5,),
        subproblem=subproblem,
    )
    log_values = []
    for sample, base in zip(subproblem.coordinates_by_dimension[0], subproblem.base_log_weights, strict=True):
        delta = (sample - 1.5) / 2.0
        log_values.append(base - 0.9189385332046727 - math.log(2.0) - 0.5 * delta * delta)
    expected = max(log_values) + math.log(sum(math.exp(value - max(log_values)) for value in log_values))

    assert actual == pytest.approx(expected)


def test_reference_supports_mixed_priors_and_multiple_subproblems() -> None:
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

    likelihoods = batch_log_likelihood(
        priors=priors,
        subproblems=subproblems,
        batch_parameters=((0.25, 1.2, 2.0), (1.0, 0.8, -2.0)),
    )

    assert math.isfinite(likelihoods[0])
    assert likelihoods[1] == -math.inf


@pytest.mark.parametrize(
    "prior",
    [
        ConditionalPrior(
            kind="normal",
            sample_dimension=0,
            parameter_a=ParameterSource.constant(0.0),
            parameter_b=ParameterSource.variable(0),
        ),
        ConditionalPrior(
            kind="uniform",
            sample_dimension=0,
            parameter_a=ParameterSource.variable(0),
            parameter_b=ParameterSource.variable(1),
        ),
    ],
)
def test_reference_rejects_invalid_sigma_and_uniform_ranges(prior: ConditionalPrior) -> None:
    parameters = (-1.0, -2.0) if prior.kind == "uniform" else (0.0,)
    subproblem = SubProblemSamples(
        coordinates_by_dimension=((0.0, 0.1),),
        base_log_weights=(0.0, 0.0),
    )

    assert batch_log_likelihood(
        priors=(prior,),
        subproblems=(subproblem,),
        batch_parameters=(parameters,),
    ) == [-math.inf]


def test_reference_handles_nonfinite_weights_priors_and_empty_batches() -> None:
    prior = ConditionalPrior(
        kind="normal",
        sample_dimension=0,
        parameter_a=ParameterSource.variable(0),
        parameter_b=ParameterSource.variable(1),
    )
    subproblem = SubProblemSamples(
        coordinates_by_dimension=((0.0, 1.0, 2.0),),
        base_log_weights=(math.nan, -0.1, -math.inf),
    )

    assert batch_log_likelihood(
        priors=(prior,),
        subproblems=(subproblem,),
        batch_parameters=((0.5, 1.0),),
        batch_log_priors=(math.nan,),
    ) == [-math.inf]
    assert batch_log_likelihood(
        priors=(prior,),
        subproblems=(subproblem,),
        batch_parameters=((0.5, math.nan),),
    ) == [-math.inf]
    assert batch_log_likelihood(
        priors=(prior,),
        subproblems=(subproblem,),
        batch_parameters=(),
    ) == []


def test_reference_treats_empty_subproblem_sample_set_as_infeasible() -> None:
    prior = ConditionalPrior(
        kind="normal",
        sample_dimension=0,
        parameter_a=ParameterSource.constant(0.0),
        parameter_b=ParameterSource.constant(1.0),
    )
    subproblem = SubProblemSamples(coordinates_by_dimension=((),), base_log_weights=())

    assert batch_log_likelihood(
        priors=(prior,),
        subproblems=(subproblem,),
        batch_parameters=((0.0,),),
    ) == [-math.inf]


def test_reference_handles_50k_like_population_shape_without_cuda() -> None:
    sample_count = 50_000
    coordinates = tuple((i % 251) / 50.0 for i in range(sample_count))
    base_weights = tuple(-0.001 * (i % 17) for i in range(sample_count))
    prior = ConditionalPrior(
        kind="normal",
        sample_dimension=0,
        parameter_a=ParameterSource.variable(0),
        parameter_b=ParameterSource.constant(1.5),
    )
    subproblem = SubProblemSamples(
        coordinates_by_dimension=(coordinates,),
        base_log_weights=base_weights,
    )

    likelihoods = batch_log_likelihood(
        priors=(prior,),
        subproblems=(subproblem,),
        batch_parameters=((1.0,), (2.0,), (3.0,)),
    )

    assert len(likelihoods) == 3
    assert all(math.isfinite(value) for value in likelihoods)
    assert likelihoods[1] > likelihoods[0]
