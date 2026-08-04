#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import korali
import yaml
from mpi4py import MPI

project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(project_root, "emb", "compression"))
sys.path.insert(0, os.path.join(project_root, "emb", "compression", "evalkit"))

from emb.compression.evalkit.tools import datedPrint
from meso_uq.config import resolve_inference_config_path
from meso_uq.config.models import EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE
from meso_uq.site_runtime import get_site_runtime_paths
from meso_uq.experiments import load_experiments
from meso_uq.workflow_acceleration import (
    apply_korali_random_seed,
    configure_korali_conduit,
    phase1_prior_specs,
    phase2_hyperprior_specs,
    require_single_rank,
    to_korali_path,
    validate_korali_random_seed,
)

VALID_PHASE2_BACKENDS = ("cpu-mpi", "native-cuda")
LEGACY_PHASE2_SIGMA_PRIOR = (0.0, 10.0)


def _detect_native_cuda_batch_support(project_root_path: str | Path) -> tuple[bool | None, str]:
    """Detect whether the canonical Korali build exposes native CUDA batching."""
    try:
        paths = get_site_runtime_paths(project_root_path)
    except Exception as exc:
        return None, f"site runtime root unavailable: {exc}"
    build_options_path = paths.korali_build_dir / "meson-info" / "intro-buildoptions.json"
    if not build_options_path.exists():
        return None, f"build options not found at {build_options_path}"

    try:
        payload = json.loads(build_options_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"failed reading {build_options_path}: {exc}"

    for item in payload:
        if item.get("name") == "native_cuda_batch":
            return bool(item.get("value")), f"{build_options_path}:native_cuda_batch"
    return None, f"native_cuda_batch option missing in {build_options_path}"


def _resolve_phase2_backend(
    config: dict[str, object], explicit_backend: str | None, *, profile_hint: str | None = None
) -> str:
    if explicit_backend is not None:
        backend = explicit_backend
    elif isinstance(config.get("phase2_backend"), str):
        backend = str(config["phase2_backend"])
    else:
        backend = "native-cuda" if profile_hint == "production" else "cpu-mpi"

    normalized = backend.strip().lower().replace("_", "-")
    if normalized not in VALID_PHASE2_BACKENDS:
        raise ValueError(
            f"Unsupported phase2_backend '{backend}'. Expected one of {VALID_PHASE2_BACKENDS}."
        )
    return normalized


def _phase2_sigma_prior_bounds(config: dict[str, object]) -> object:
    if config.get("phase2_prior_sigma") is not None:
        return config["phase2_prior_sigma"]
    if (
        str(config.get("phase1_contract_mode") or "") == EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE
        and config.get("prior_sigma") is not None
    ):
        return config["prior_sigma"]
    return LEGACY_PHASE2_SIGMA_PRIOR


def _phase2_conditional_prior_plan(config: dict[str, object]) -> list[tuple[str, str, object]]:
    hyperpairs = {
        name: (mu_bounds, sigma_bounds)
        for name, mu_bounds, sigma_bounds in phase2_hyperprior_specs(config)
    }
    plan: list[tuple[str, str, object]] = []
    for name, prior_bounds in phase1_prior_specs(config):
        if name in hyperpairs:
            plan.append((name, "normal", hyperpairs[name]))
            continue
        if name == "sigma":
            plan.append((name, "uniform", _phase2_sigma_prior_bounds(config)))
        else:
            plan.append((name, "uniform", prior_bounds))
    return plan


def _expected_phase1_variable_name(name: str) -> str:
    return "[Sigma]" if name == "sigma" else name


def _validate_loaded_phase1_sub_problem(
    exp_name: str,
    conditional_plan: list[tuple[str, str, object]],
    *,
    phase1_state_path: Path | None = None,
) -> None:
    if phase1_state_path is None or not phase1_state_path.exists():
        return
    try:
        state = json.loads(phase1_state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Phase 1 state for {exp_name} is not valid JSON: {phase1_state_path}") from exc
    variables = state.get("Variables") or []
    if not isinstance(variables, list):
        raise RuntimeError(
            f"Phase 1 state for {exp_name} has malformed Variables in {phase1_state_path}."
        )
    actual_names = [str(variable.get("Name")) for variable in variables]
    expected_names = [_expected_phase1_variable_name(name) for name, _kind, _bounds in conditional_plan]
    if actual_names != expected_names:
        raise RuntimeError(
            f"Phase 2 conditional-prior order does not match Phase 1 variables for {exp_name}: "
            f"phase1={actual_names}, phase2={expected_names}."
        )


def run_hierarchical_inference(
    profiling: bool = False,
    config_path: str | None = None,
    output_dir: str = "_setup",
    phase2_backend: str | None = None,
    korali_random_seed: int | None = None,
):
    korali_random_seed = validate_korali_random_seed(
        korali_random_seed, field_name="--korali-random-seed"
    )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = Path(project_root, output_root)
    output_root = output_root.resolve()

    def ensure_output_dir(path: str) -> None:
        if rank == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    if config_path is None:
        config_path = str(resolve_inference_config_path(project_root, experiment="compression"))
    with open(config_path, "rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)

    profile_hint = "production" if "/production/" in str(config_path) else None
    resolved_backend = _resolve_phase2_backend(config, phase2_backend, profile_hint=profile_hint)

    experiments = [exp for exp in load_experiments(config, Path(project_root)) if exp.enabled]
    conditional_plan = _phase2_conditional_prior_plan(config)
    e = korali.Experiment()
    applied_seed = apply_korali_random_seed(e, korali_random_seed)
    e["Problem"]["Type"] = "Hierarchical/Psi"

    sub_idx = 0
    for exp in experiments:
        for diameter_um in exp.diameters:
            exp_name = exp.dataset_name(diameter_um)
            phase1_state_path = output_root / "results_phase_1" / exp_name / "latest"
            sub_problem = korali.Experiment()
            found = sub_problem.loadState(str(phase1_state_path))
            if not found:
                raise FileNotFoundError(
                    f"Phase 2 could not load Phase 1 state for {exp_name}: {phase1_state_path}"
                )
            _validate_loaded_phase1_sub_problem(
                exp_name,
                conditional_plan,
                phase1_state_path=phase1_state_path,
            )
            e["Problem"]["Sub Experiments"][sub_idx] = sub_problem
            sub_idx += 1

    var_idx = 0
    dist_idx = 0
    conditional_names = []
    for name, kind, bounds in conditional_plan:
        conditional_name = f"Conditional {name}"
        if kind == "normal":
            mu_bounds, sigma_bounds = bounds
            e["Variables"][var_idx]["Name"] = f"mu_{name}"
            e["Variables"][var_idx]["Prior Distribution"] = f"Uniform mu_{name}"
            var_idx += 1
            e["Variables"][var_idx]["Name"] = f"sigma_{name}"
            e["Variables"][var_idx]["Prior Distribution"] = f"Uniform sigma_{name}"
            var_idx += 1

            e["Distributions"][dist_idx]["Name"] = conditional_name
            e["Distributions"][dist_idx]["Type"] = "Univariate/Normal"
            e["Distributions"][dist_idx]["Mean"] = f"mu_{name}"
            e["Distributions"][dist_idx]["Standard Deviation"] = f"sigma_{name}"
            conditional_names.append(conditional_name)
            dist_idx += 1

            e["Distributions"][dist_idx]["Name"] = f"Uniform mu_{name}"
            e["Distributions"][dist_idx]["Type"] = "Univariate/Uniform"
            e["Distributions"][dist_idx]["Minimum"] = mu_bounds[0]
            e["Distributions"][dist_idx]["Maximum"] = mu_bounds[1]
            dist_idx += 1

            e["Distributions"][dist_idx]["Name"] = f"Uniform sigma_{name}"
            e["Distributions"][dist_idx]["Type"] = "Univariate/Uniform"
            e["Distributions"][dist_idx]["Minimum"] = sigma_bounds[0]
            e["Distributions"][dist_idx]["Maximum"] = sigma_bounds[1]
            dist_idx += 1
            continue

        if kind != "uniform":
            raise ValueError(f"Unsupported Phase 2 conditional-prior kind '{kind}' for '{name}'.")
        e["Distributions"][dist_idx]["Name"] = conditional_name
        e["Distributions"][dist_idx]["Type"] = "Univariate/Uniform"
        e["Distributions"][dist_idx]["Minimum"] = bounds[0]
        e["Distributions"][dist_idx]["Maximum"] = bounds[1]
        conditional_names.append(conditional_name)
        dist_idx += 1

    e["Problem"]["Conditional Priors"] = conditional_names

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = config["hbi_pop_size"]
    e["Solver"]["Burn In"] = config["hbi_burn_in"]
    e["Solver"]["Target Coefficient Of Variation"] = config["hbi_target_cov"]
    e["Solver"]["Covariance Scaling"] = config["hbi_covariance_scaling"]
    if config.get("hbi_max_gen", -1) > 0:
        e["Solver"]["Termination Criteria"]["Max Generations"] = config["hbi_max_gen"]

    e["Console Output"]["Verbosity"] = "Detailed"
    results_phase_2 = output_root / "results_phase_2"
    e["File Output"]["Path"] = to_korali_path(str(results_phase_2), base_dir=project_root)
    e["File Output"]["Use Multiple Files"] = False
    ensure_output_dir(str(results_phase_2))

    k = korali.Engine()
    if resolved_backend == "native-cuda":
        require_single_rank(comm, "Phase 2 native-cuda backend")
        native_cuda_supported, support_source = _detect_native_cuda_batch_support(project_root)
        if native_cuda_supported is False:
            raise RuntimeError(
                "phase2_backend=native-cuda requested, but Korali native_cuda_batch is disabled "
                f"({support_source})."
            )
        if rank == 0:
            if native_cuda_supported is True:
                datedPrint(f"[HBI] native-cuda support detected from {support_source}")
            else:
                datedPrint(
                    "[HBI] native-cuda support could not be verified from repo-local Meson metadata; "
                    "continuing with backend=NativeCuda."
                )
        k["Conduit"]["Type"] = "Sequential"
        e["Problem"]["Use Batch Evaluation"] = True
        e["Problem"]["Batch Evaluation Backend"] = "NativeCuda"
    else:
        k.setMPIComm(MPI.COMM_WORLD)
        configure_korali_conduit(k, mpi_ranks=comm.Get_size(), ranks_per_worker=1, concurrent_jobs=1)

    if profiling:
        k["Profiling"]["Detail"] = "Full"
        k["Profiling"]["Frequency"] = 0.5
    if rank == 0:
        datedPrint(
            f"[HBI] Starting TMCMC with population size={config['hbi_pop_size']} "
            f"phase2_backend={resolved_backend}"
        )
        if applied_seed is not None:
            datedPrint(f"[HBI] Random Seed: {applied_seed}")
    k.run(e)


def main(argv: list[str]) -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="_setup")
    parser.add_argument(
        "--phase2-backend",
        choices=VALID_PHASE2_BACKENDS,
        default=None,
        help="Phase 2 backend: production defaults to native-cuda; validation defaults to cpu-mpi.",
    )
    parser.add_argument(
        "--korali-random-seed",
        type=int,
        default=None,
        help=(
            "Positive nonzero Korali Random Seed. Omit to preserve Korali/default "
            "stochastic initialization."
        ),
    )
    args = parser.parse_args(argv)
    run_hierarchical_inference(
        profiling=args.profiling,
        config_path=args.config,
        output_dir=args.output_dir,
        phase2_backend=args.phase2_backend,
        korali_random_seed=args.korali_random_seed,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
