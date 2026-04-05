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


def run_phase_3a(profiling: bool = False, config_path: str = None, output_dir: str = "_setup"):
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
    phase2_path = f"{output_dir}/results_phase_2/latest"
    if not os.path.exists(phase2_path):
        if rank == 0:
            datedPrint(f"[Phase 3a] ERROR: Phase 2 results not found at {phase2_path}")
        sys.exit(1)

    e = korali.Experiment()
    psi = korali.Experiment()
    psi.loadState(phase2_path)
    e["Problem"]["Type"] = "Hierarchical/ThetaNew"
    e["Problem"]["Psi Experiment"] = psi

    priors = [config["prior_Yt"], config["prior_kb"], config["prior_b1"], config["prior_b2"], config["prior_a3"], config["prior_a4"], config.get("prior_d0", [0.0, 0.5]), config["prior_sigma"]]
    names = ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]
    for i, (name, bounds) in enumerate(zip(names, priors)):
        e["Distributions"][i]["Name"] = f"Uniform {i}"
        e["Distributions"][i]["Type"] = "Univariate/Uniform"
        e["Distributions"][i]["Minimum"] = bounds[0]
        e["Distributions"][i]["Maximum"] = bounds[1]
        e["Variables"][i]["Name"] = "[Sigma]" if name == "sigma" else name
        e["Variables"][i]["Prior Distribution"] = f"Uniform {i}"

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = config.get("phase3a_pop_size", 10000)
    e["Solver"]["Target Coefficient Of Variation"] = config.get("phase3a_target_cov", 0.6)
    e["Solver"]["Covariance Scaling"] = config.get("phase3a_covariance_scaling", 0.02)
    e["Solver"]["Burn In"] = 1
    e["Solver"]["Max Chain Length"] = 1
    if config.get("phase3a_max_gen", -1) > 0:
        e["Solver"]["Termination Criteria"]["Max Generations"] = config["phase3a_max_gen"]

    e["Console Output"]["Verbosity"] = "Detailed"
    e["File Output"]["Path"] = f"{output_dir}/results_phase_3a/"
    e["File Output"]["Frequency"] = 1
    e["Console Output"]["Frequency"] = 1
    e["Store Sample Information"] = True
    ensure_output_dir(f"{output_dir}/results_phase_3a/")

    k = korali.Engine()
    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 1
    if profiling:
        k["Profiling"]["Detail"] = "Full"
        k["Profiling"]["Frequency"] = 0.5
    if rank == 0:
        total_sets = sum(len(exp.diameters) for exp in experiments)
        datedPrint(f"[Phase 3a] Starting joint posterior sampling across {total_sets} datasets")
    k.run(e)


def main(argv):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="_setup")
    args = parser.parse_args()
    run_phase_3a(profiling=args.profiling, config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
