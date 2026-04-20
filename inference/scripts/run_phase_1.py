#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import korali
import yaml
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "compression"))
sys.path.insert(0, str(PROJECT_ROOT / "compression" / "evalkit"))
sys.path.insert(0, str(PROJECT_ROOT / "indentation"))
sys.path.insert(0, str(PROJECT_ROOT / "indentation" / "evalkit"))

from compression.evalkit.posterior_compression import (
    compute_compression,
    compute_compression_surrogate,
    compute_compression_surrogate_batch,
    preload_compression_surrogate,
)
from compression.evalkit.tools import datedPrint, prepareCompression
from indentation.evalkit.posterior_indentation import (
    compute_indentation_surrogate,
    compute_indentation_surrogate_batch,
    preload_indentation_surrogate,
)
from indentation.evalkit.prepare_env import prepareIndentation
from meso_uq.config import resolve_inference_config_path
from meso_uq.experiments import load_experiments
from meso_uq.workflow_acceleration import (
    configure_device_conduit,
    configure_korali_conduit,
    phase1_prior_specs,
    to_korali_path,
)


def _align_reference_data(ref_points, ref_data, exp_name, rank):
    if len(ref_points) != len(ref_data):
        min_len = min(len(ref_points), len(ref_data))
        if rank == 0:
            datedPrint(
                f"[Korali] WARNING: Reference points/data length mismatch for {exp_name} "
                f"(points={len(ref_points)}, data={len(ref_data)}). Trimming to {min_len}."
            )
        ref_points = ref_points[:min_len]
        ref_data = ref_data[:min_len]
    return ref_points, ref_data


def _resolve_config_path(config_path: str | None) -> Path:
    if config_path is None:
        return resolve_inference_config_path(PROJECT_ROOT, experiment="compression")
    candidate = Path(config_path).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _resolve_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    return output_path.resolve()


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _prepare_experiment_environment(experiments, rank: int) -> None:
    if rank != 0:
        return
    for exp in experiments:
        for diameter_um in exp.diameters:
            datedPrint(f"[Setup] Preparing {exp.name} environment for {diameter_um} μm")
            if exp.name == "compression":
                prepareCompression(diameter_um)
            elif exp.name == "indentation":
                prepareIndentation(
                    diameter_um,
                    data_dir=str(exp.data_dir),
                    data_prefix=exp.data_prefix,
                    data_file=str(exp.data_file(diameter_um)),
                )
            else:
                raise ValueError(f"Unsupported experiment type '{exp.name}'")


def _apply_compression_dry_run(experiments, rank: int) -> None:
    if rank != 0:
        return
    sample_param = {"numsteps": 100, "numsteps_eq": 100}
    for exp in experiments:
        if exp.name != "compression":
            continue
        for diameter_um in exp.diameters:
            filenames = [
                PROJECT_ROOT
                / f"_init_compression_{diameter_um}um"
                / "parameter"
                / "parameters-default00001.yaml",
                PROJECT_ROOT
                / f"_init_compression_{diameter_um}um"
                / "parameter"
                / "parameters-default00001eq.yaml",
            ]
            for filename in filenames:
                if not filename.exists():
                    datedPrint(f"[Dry run] WARNING: parameter template not found: {filename}")
                    continue
                with open(filename, "r") as handle:
                    parameters = yaml.load(handle, Loader=yaml.CLoader)
                for key, value in sample_param.items():
                    parameters[key] = int(value)
                with open(filename, "w") as handle:
                    yaml.dump(parameters, handle)


def run_inference(
    restart: bool = False,
    profiling: bool = False,
    dry_run: bool = False,
    config_path: str = None,
    output_dir: str = "_setup",
    device: str = "cpu",
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config_path_resolved = _resolve_config_path(config_path)
    with open(config_path_resolved, "rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)
    os.environ["HUQ_INFERENCE_CONFIG"] = str(config_path_resolved)

    output_root = _resolve_output_dir(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pop_size = config["pop_size"]
    max_gen = config["max_gen"]
    target_cov = config["target_cov"]
    covariance_scaling = config["covariance_scaling"]
    use_surrogate = config.get("use_surrogate", True)

    experiments = [exp for exp in load_experiments(config, PROJECT_ROOT) if exp.enabled]

    if use_surrogate:
        preload_map = {
            "compression": preload_compression_surrogate,
            "indentation": preload_indentation_surrogate,
        }
        if rank == 0:
            datedPrint("[Korali] Preloading surrogate models")
        for exp in experiments:
            preload_fn = preload_map.get(exp.name)
            if preload_fn is None:
                raise ValueError(
                    f"No surrogate preload function registered for experiment '{exp.name}'"
                )
            for diameter_um in exp.diameters:
                preload_fn(diameter_um, device=device)
        if device == "cpu":
            comm.Barrier()

    k = korali.Engine()
    if device == "cpu":
        k.setMPIComm(MPI.COMM_WORLD)
    configure_device_conduit(k, device=device, mpi_ranks=comm.Get_size())

    compute_surrogate_map = {
        "compression": compute_compression_surrogate,
        "indentation": compute_indentation_surrogate,
    }
    compute_batch_map = {
        "compression": compute_compression_surrogate_batch,
        "indentation": compute_indentation_surrogate_batch,
    }
    compute_mirheo_map = {
        "compression": compute_compression,
    }

    def resolve_compute_model(exp_name: str):
        if use_surrogate:
            if exp_name not in compute_surrogate_map:
                raise ValueError(f"No surrogate model registered for experiment '{exp_name}'")
            return compute_surrogate_map[exp_name]
        if exp_name not in compute_mirheo_map:
            raise NotImplementedError(f"Mirheo model not available for experiment '{exp_name}'")
        return compute_mirheo_map[exp_name]

    if restart:
        phase1_root = output_root / "results_phase_1"
        e_list = []
        for exp in experiments:
            for diameter_um in exp.diameters:
                exp_name = exp.dataset_name(diameter_um)
                experiment_root = phase1_root / exp_name
                e = korali.Experiment()
                e["File Output"]["Path"] = to_korali_path(
                    str(experiment_root), base_dir=str(PROJECT_ROOT)
                )
                found = e.loadState(str(experiment_root / "latest"))
                if not found:
                    raise FileNotFoundError(
                        f"No previous state found for {exp_name} under {experiment_root}"
                    )
                e["File Output"]["Use Multiple Files"] = False
                compute_model = resolve_compute_model(exp.name)
                reference_points = exp.get_reference_points(diameter_um)
                reference_data = e["Problem"].get("Reference Data")
                if reference_data is not None:
                    reference_points, reference_data = _align_reference_data(
                        reference_points,
                        list(reference_data),
                        exp_name,
                        rank,
                    )
                    e["Problem"]["Reference Data"] = reference_data
                if device == "gpu" and use_surrogate:
                    batch_fn = compute_batch_map[exp.name]
                    e["Problem"]["Use Batch Evaluation"] = True
                    e["Problem"]["Batch Computational Model"] = (
                        lambda s, d=diameter_um, pts=reference_points, dev=device, fn=batch_fn: fn(
                            s, pts, d, device=dev
                        )
                    )
                    e["Problem"]["Computational Model"] = (
                        lambda sampleData, d=diameter_um, model=compute_model, pts=reference_points: model(
                            sampleData, pts, d
                        )
                    )
                else:
                    e["Problem"]["Computational Model"] = (
                        lambda sampleData, d=diameter_um, model=compute_model, pts=reference_points: model(
                            sampleData, pts, d
                        )
                    )
                e_list.append(e)
                if rank == 0:
                    model_label = "surrogate" if use_surrogate else "Mirheo"
                    datedPrint(
                        f"[Korali] Resuming {exp_name} with {model_label} model from {experiment_root}"
                    )
    else:
        with _working_directory(PROJECT_ROOT):
            _prepare_experiment_environment(experiments, rank)
            comm.Barrier()
            if dry_run:
                _apply_compression_dry_run(experiments, rank)

        if rank == 0:
            datedPrint(f"[Korali] Number of ranks: {comm.Get_size()}")
            datedPrint(f"[Korali] Population size: {pop_size}")
            datedPrint(f"[Korali] Max generations: {max_gen}")
            datedPrint(f"[Korali] Target COV: {target_cov}")
            datedPrint(f"[Korali] Covariance scaling: {covariance_scaling}")
            datedPrint("[Korali] Experiments:")
            for exp in experiments:
                datedPrint(f"  - {exp.name}: diameters={exp.diameters}")

        comm.Barrier()

        e_list = []
        for exp in experiments:
            for diameter_um in exp.diameters:
                exp_name = exp.dataset_name(diameter_um)
                if rank == 0:
                    datedPrint(f"[Korali] Setting up experiment: {exp_name}")
                e = korali.Experiment()
                compute_model = resolve_compute_model(exp.name)
                if rank == 0 and not e_list:
                    model_label = "surrogate" if use_surrogate else "Mirheo"
                    datedPrint(f"[Korali] Using {model_label} model for all experiments")
                reference_points = exp.get_reference_points(diameter_um)
                reference_data = exp.get_reference_data(diameter_um)
                reference_points, reference_data = _align_reference_data(
                    reference_points,
                    reference_data,
                    exp_name,
                    rank,
                )
                if device == "gpu" and use_surrogate:
                    batch_fn = compute_batch_map[exp.name]
                    e["Problem"]["Use Batch Evaluation"] = True
                    e["Problem"]["Batch Computational Model"] = (
                        lambda s, d=diameter_um, pts=reference_points, dev=device, fn=batch_fn: fn(
                            s, pts, d, device=dev
                        )
                    )
                    e["Problem"]["Computational Model"] = (
                        lambda sampleData, d=diameter_um, model=compute_model, pts=reference_points: model(
                            sampleData, pts, d
                        )
                    )
                else:
                    e["Problem"]["Computational Model"] = (
                        lambda sampleData, d=diameter_um, model=compute_model, pts=reference_points: model(
                            sampleData, pts, d
                        )
                    )
                e["Problem"]["Type"] = "Bayesian/Reference"
                e["Problem"]["Likelihood Model"] = "Normal"
                e["Problem"]["Reference Data"] = reference_data
                e["Solver"]["Type"] = "Sampler/TMCMC"
                e["Solver"]["Population Size"] = pop_size
                e["Solver"]["Target Coefficient Of Variation"] = target_cov
                e["Solver"]["Covariance Scaling"] = covariance_scaling
                if max_gen > 0:
                    e["Solver"]["Termination Criteria"]["Max Generations"] = max_gen

                prior_specs = phase1_prior_specs(
                    config,
                    prior_d0=exp.prior_d0 or config.get("prior_d0", [0.0, 0.5]),
                    prior_sigma=exp.prior_sigma or config["prior_sigma"],
                )
                for i, (name, bounds) in enumerate(prior_specs):
                    e["Distributions"][i]["Name"] = f"Prior {name}"
                    e["Distributions"][i]["Type"] = "Univariate/Uniform"
                    e["Distributions"][i]["Minimum"] = bounds[0]
                    e["Distributions"][i]["Maximum"] = bounds[1]
                    e["Variables"][i]["Name"] = "[Sigma]" if name == "sigma" else name
                    e["Variables"][i]["Prior Distribution"] = f"Prior {name}"

                e["File Output"]["Frequency"] = 1
                e["File Output"]["Use Multiple Files"] = False
                e["File Output"]["Path"] = to_korali_path(
                    str(output_root / "results_phase_1" / exp_name),
                    base_dir=str(PROJECT_ROOT),
                )
                e["Console Output"]["Frequency"] = 1
                e["Console Output"]["Verbosity"] = "Detailed"
                e["Store Sample Information"] = True
                e_list.append(e)

    if profiling:
        k["Profiling"]["Detail"] = "Full"
        k["Profiling"]["Frequency"] = 0.5

    with _working_directory(PROJECT_ROOT):
        k.run(e_list)


def main(argv):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="_setup")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu: Distributed MPI conduit; gpu: Sequential GPU-batch conduit (single rank)",
    )
    args = parser.parse_args()

    run_inference(
        restart=args.restart,
        profiling=args.profiling,
        dry_run=args.dry_run,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
