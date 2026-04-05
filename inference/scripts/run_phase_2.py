#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import korali
import yaml
from mpi4py import MPI

project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(project_root, "compression"))
sys.path.insert(0, os.path.join(project_root, "compression", "evalkit"))

from compression.evalkit.tools import datedPrint
from meso_uq.config import resolve_inference_config_path
from meso_uq.experiments import load_experiments


def run_hierarchical_inference(profiling: bool = False, config_path: str = None, output_dir: str = "_setup"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
            sub_problem.loadState(f"{output_dir}/results_phase_1/{exp_name}/latest")
            e["Problem"]["Sub Experiments"][sub_idx] = sub_problem
            sub_idx += 1

    hyperpairs = [
        ("Yt", config["hyperprior_mu_Yt"], config["hyperprior_sigma_Yt"]),
        ("kb", config["hyperprior_mu_kb"], config["hyperprior_sigma_kb"]),
        ("b1", config["hyperprior_mu_b1"], config["hyperprior_sigma_b1"]),
        ("b2", config["hyperprior_mu_b2"], config["hyperprior_sigma_b2"]),
        ("a3", config["hyperprior_mu_a3"], config["hyperprior_sigma_a3"]),
        ("a4", config["hyperprior_mu_a4"], config["hyperprior_sigma_a4"]),
        ("d0", config.get("hyperprior_mu_d0", [0.0, 0.5]), config.get("hyperprior_sigma_d0", [0.0, 0.3])),
    ]

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
    e["File Output"]["Path"] = f"{output_dir}/results_phase_2/"
    ensure_output_dir(f"{output_dir}/results_phase_2/")

    k = korali.Engine()
    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 1
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
