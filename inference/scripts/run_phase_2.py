#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import korali
import yaml
from mpi4py import MPI

project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(project_root, "compression"))
sys.path.insert(0, os.path.join(project_root, "compression", "evalkit"))

from compression.evalkit.tools import datedPrint
from meso_uq.config import resolve_inference_config_path
from meso_uq.experiments import load_experiments
from meso_uq.workflow_acceleration import (
    configure_korali_conduit,
    phase2_hyperprior_specs,
    to_korali_path,
)


def run_hierarchical_inference(profiling: bool = False, config_path: str = None, output_dir: str = "_setup"):
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
    with open(config_path, "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    experiments = [exp for exp in load_experiments(config, Path(project_root)) if exp.enabled]
    e = korali.Experiment()
    e["Problem"]["Type"] = "Hierarchical/Psi"

    sub_idx = 0
    for exp in experiments:
        for diameter_um in exp.diameters:
            exp_name = exp.dataset_name(diameter_um)
            sub_problem = korali.Experiment()
            sub_problem.loadState(str(output_root / "results_phase_1" / exp_name / "latest"))
            e["Problem"]["Sub Experiments"][sub_idx] = sub_problem
            sub_idx += 1

    hyperpairs = phase2_hyperprior_specs(config)

    var_idx = 0
    dist_idx = 0
    conditional_names = []
    for name, mu_bounds, sigma_bounds in hyperpairs:
        e["Variables"][var_idx]["Name"] = f"mu_{name}"
        e["Variables"][var_idx]["Prior Distribution"] = f"Uniform mu_{name}"
        var_idx += 1
        e["Variables"][var_idx]["Name"] = f"sigma_{name}"
        e["Variables"][var_idx]["Prior Distribution"] = f"Uniform sigma_{name}"
        var_idx += 1

        e["Distributions"][dist_idx]["Name"] = f"Conditional {name}"
        e["Distributions"][dist_idx]["Type"] = "Univariate/Normal"
        e["Distributions"][dist_idx]["Mean"] = f"mu_{name}"
        e["Distributions"][dist_idx]["Standard Deviation"] = f"sigma_{name}"
        conditional_names.append(f"Conditional {name}")
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

    e["Distributions"][dist_idx]["Name"] = "Conditional sigma"
    e["Distributions"][dist_idx]["Type"] = "Univariate/Uniform"
    e["Distributions"][dist_idx]["Minimum"] = 0.0
    e["Distributions"][dist_idx]["Maximum"] = 10.0
    conditional_names.append("Conditional sigma")
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
    k.setMPIComm(MPI.COMM_WORLD)
    configure_korali_conduit(k, mpi_ranks=comm.Get_size(), ranks_per_worker=1, concurrent_jobs=1)
    if profiling:
        k["Profiling"]["Detail"] = "Full"
        k["Profiling"]["Frequency"] = 0.5
    if rank == 0:
        datedPrint(f"[HBI] Starting TMCMC with population size: {config['hbi_pop_size']}")
    k.run(e)


def main(argv):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="_setup")
    args = parser.parse_args()
    run_hierarchical_inference(profiling=args.profiling, config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
