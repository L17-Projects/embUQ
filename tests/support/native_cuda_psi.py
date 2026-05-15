from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence


KORALI_PSI_CPP_PATH = Path("extern/korali/source/modules/problem/hierarchical/psi/psi.cpp")
KERNEL_SYMBOL = "kPsiNativeCudaKernelSource"
KERNEL_ENTRYPOINT = "psiBatchLogLikelihoodKernel"
NEGATIVE_INFINITY = float("-inf")


_KERNEL_PATTERN = re.compile(
    r'constexpr\s+const\s+char\s+\*\s*kPsiNativeCudaKernelSource\s*=\s*R"CUDA\(\n(?P<source>.*?)\n\)CUDA";',
    re.DOTALL,
)
_REQUIRED_KERNEL_MARKERS = (
    'extern "C" __global__',
    "void psiBatchLogLikelihoodKernel(",
    "priorKinds[priorId] == 0",
    "parameterB <= 0.0",
    "parameterB <= parameterA",
    "sampleValue < parameterA",
    "sampleValue > parameterB",
    "sharedMax[256]",
    "sharedSum[256]",
    "subProblemLogLikelihoods[(size_t)subProblemId * batchSize + candidateId]",
)


@dataclass(frozen=True)
class ParameterSource:
    position: int | None = None
    value: float | None = None

    @classmethod
    def variable(cls, position: int) -> "ParameterSource":
        return cls(position=position)

    @classmethod
    def constant(cls, value: float) -> "ParameterSource":
        return cls(value=float(value))

    def resolve(self, parameters: Sequence[float]) -> float:
        if self.position is None:
            if self.value is None:
                raise ValueError("ParameterSource must define either position or value.")
            return self.value
        return float(parameters[self.position])


@dataclass(frozen=True)
class ConditionalPrior:
    kind: Literal["normal", "uniform"]
    sample_dimension: int
    parameter_a: ParameterSource
    parameter_b: ParameterSource


@dataclass(frozen=True)
class SubProblemSamples:
    coordinates_by_dimension: tuple[tuple[float, ...], ...]
    base_log_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        sample_count = len(self.base_log_weights)
        for dimension, coordinates in enumerate(self.coordinates_by_dimension):
            if len(coordinates) != sample_count:
                raise ValueError(
                    f"coordinates_by_dimension[{dimension}] has {len(coordinates)} samples, "
                    f"expected {sample_count}."
                )


def extract_psi_native_cuda_kernel_source(cpp_source: str) -> str:
    match = _KERNEL_PATTERN.search(cpp_source)
    if match is None:
        raise ValueError(f"Could not find {KERNEL_SYMBOL} raw CUDA source.")

    kernel_source = match.group("source")
    missing = [marker for marker in _REQUIRED_KERNEL_MARKERS if marker not in kernel_source]
    if missing:
        raise ValueError(f"{KERNEL_SYMBOL} is missing required markers: {', '.join(missing)}")
    return kernel_source


def read_vendored_psi_native_cuda_kernel_source(repo_root: Path) -> str:
    return extract_psi_native_cuda_kernel_source((repo_root / KORALI_PSI_CPP_PATH).read_text(encoding="utf-8"))


def conditional_log_density(prior: ConditionalPrior, parameters: Sequence[float], sample_value: float) -> float:
    parameter_a = prior.parameter_a.resolve(parameters)
    parameter_b = prior.parameter_b.resolve(parameters)

    if prior.kind == "normal":
        if parameter_b <= 0.0:
            return NEGATIVE_INFINITY
        try:
            delta = (sample_value - parameter_a) / parameter_b
            return -0.9189385332046727 - math.log(parameter_b) - 0.5 * delta * delta
        except ValueError:
            return math.nan

    if prior.kind == "uniform":
        if parameter_b <= parameter_a or sample_value < parameter_a or sample_value > parameter_b:
            return NEGATIVE_INFINITY
        try:
            return -math.log(parameter_b - parameter_a)
        except ValueError:
            return math.nan

    raise ValueError(f"Unsupported prior kind: {prior.kind!r}.")


def subproblem_log_likelihood(
    *,
    priors: Sequence[ConditionalPrior],
    parameters: Sequence[float],
    subproblem: SubProblemSamples,
) -> float:
    log_values: list[float] = []
    for sample_id, base_log_weight in enumerate(subproblem.base_log_weights):
        log_value = float(base_log_weight)
        if not math.isfinite(log_value):
            continue

        for prior in priors:
            sample_value = subproblem.coordinates_by_dimension[prior.sample_dimension][sample_id]
            log_density = conditional_log_density(prior, parameters, sample_value)
            if not math.isfinite(log_density):
                log_value = NEGATIVE_INFINITY
                break
            log_value += log_density

        if math.isfinite(log_value):
            log_values.append(log_value)

    if not log_values:
        return NEGATIVE_INFINITY

    maximum = max(log_values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in log_values))


def batch_log_likelihood(
    *,
    priors: Sequence[ConditionalPrior],
    subproblems: Sequence[SubProblemSamples],
    batch_parameters: Sequence[Sequence[float]],
    batch_log_priors: Sequence[float] | None = None,
) -> list[float]:
    if batch_log_priors is None:
        batch_log_priors = [0.0] * len(batch_parameters)
    if len(batch_log_priors) != len(batch_parameters):
        raise ValueError("batch_log_priors must match batch_parameters length.")

    likelihoods: list[float] = []
    for parameters, log_prior in zip(batch_parameters, batch_log_priors, strict=True):
        if not math.isfinite(float(log_prior)):
            likelihoods.append(NEGATIVE_INFINITY)
            continue

        total = 0.0
        feasible = True
        for subproblem in subproblems:
            partial = subproblem_log_likelihood(
                priors=priors,
                parameters=parameters,
                subproblem=subproblem,
            )
            if not math.isfinite(partial):
                feasible = False
                break
            total += partial
        likelihoods.append(total if feasible else NEGATIVE_INFINITY)
    return likelihoods
