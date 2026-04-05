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
sys.path.insert(0, os.path.join(project_root, "indentation"))
sys.path.insert(0, os.path.join(project_root, "indentation", "evalkit"))

from compression.evalkit.posterior_compression import compute_compression, compute_compression_surrogate, preload_compression_surrogate
from compression.evalkit.tools import datedPrint, prepareCompression
from meso_uq.config import resolve_inference_config_path
from meso_uq.experiments import load_experiments
from indentation.evalkit.posterior_indentation import compute_indentation_surrogate, preload_indentation_surrogate
from indentation.evalkit.prepare_env import prepareIndentation


def _align_reference_data(ref_points, ref_data, exp_name, rank):
    if len(ref_points) != len(ref_data):
        min_len = min(len(ref_points), len(ref_data))
        if rank == 0:
            datedPrint(f"[Korali] WARNING: Reference points/data length mismatch for {exp_name} (points={len(ref_points)}, data={len(ref_data)}). Trimming to {min_len}.")
        ref_points = ref_points[:min_len]
        ref_data = ref_data[:min_len]
    return ref_points, ref_data


def run_inference(restart: bool = False, profiling: bool = False, dry_run: bool = False, config_path: str = None, output_dir: str = "_setup"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if config_path is None:
        config_path = str(resolve_inference_config_path(project_root, experiment="compression"))
    with open(config_path, "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    os.environ["HUQ_INFERENCE_CONFIG"] = config_path

    use_surrogate = config.get("use_surrogate", True)
    experiments = [exp for exp in load_experiments(config, Path(project_root)) if exp.enabled]

    if use_surrogate:
        preload_map = {"compression": preload_compression_surrogate, "indentation": preload_indentation_surrogate}
        if rank == 0:
            datedPrint("[Korali] Preloading surrogate models")
        for exp in experiments:
            preload_fn = preload_map.get(exp.name)
            if preload_fn is None:
                continue
            for diameter_um in exp.diameters:
                preload_fn(diameter_um)
        comm.Barrier()

    k = korali.Engine()
    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 1 if use_surrogate else 2

    compute_surrogate_map = {"compression": compute_compression_surrogate, "indentation": compute_indentation_surrogate}
    compute_mirheo_map = {"compression": compute_compression}
    prepare_map = {"compression": prepareCompression, "indentation": prepareIndentation}

    def resolve_compute_model(exp_name: str):
        if use_surrogate:
            return compute_surrogate_map[exp_name]
        return compute_mirheo_map[exp_name]

    if rank == 0:
        for exp in experiments:
            prepare_fn = prepare_map.get(exp.name)
            if prepare_fn is None:
                continue
            for diameter_um in exp.diameters:
                datedPrint(f"[Setup] Preparing {exp.name} environment for {diameter_um} μm")
                if exp.name == "indentation":
                    prepare_fn(diameter_um, data_dir=str(exp.data_dir), data_prefix=exp.data_prefix, data_file=str(exp.data_file(diameter_um)))
                else:
                    prepare_fn(diameter_um)
    comm.Barrier()

    eList = []
    for exp in experiments:
        for diameter_um in exp.diameters:
            exp_name = exp.dataset_name(diameter_um)
            e = korali.Experiment()
            reference_points = exp.get_reference_points(diameter_um)
            reference_data = exp.get_reference_data(diameter_um)
            reference_points, reference_data = _align_reference_data(reference_points, reference_data, exp_name, rank)
            model = resolve_compute_model(exp.name)
            e["Problem"]["Type"] = "Bayesian/Reference"
            e["Problem"]["Likelihood Model"] = "Normal"
            e["Problem"]["Reference Data"] = reference_data
            e["Problem"]["Computational Model"] = lambda sampleData, d=diameter_um, pts=reference_points, m=model: m(sampleData, pts, d)
            e["Solver"]["Type"] = "Sampler/TMCMC"
            e["Solver"]["Population Size"] = config["pop_size"]
            e["Solver"]["Target Coefficient Of Variation"] = config["target_cov"]
            e["Solver"]["Covariance Scaling"] = config["covariance_scaling"]
            if config["max_gen"] > 0:
                e["Solver"]["Termination Criteria"]["Max Generations"] = config["max_gen"]
            priors = [config["prior_Yt"], config["prior_kb"], config["prior_b1"], config["prior_b2"], config["prior_a3"], config["prior_a4"], exp.prior_d0 or config.get("prior_d0", [0.0, 0.5]), exp.prior_sigma or config["prior_sigma"]]
            names = ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]
            for i, (name, bounds) in enumerate(zip(names, priors)):
                e["Distributions"][i]["Name"] = f"Prior {name}"
                e["Distributions"][i]["Type"] = "Univariate/Uniform"
                e["Distributions"][i]["Minimum"] = bounds[0]
                e["Distributions"][i]["Maximum"] = bounds[1]
                e["Variables"][i]["Name"] = "[Sigma]" if name == "sigma" else name
                e["Variables"][i]["Prior Distribution"] = f"Prior {name}"
            e["File Output"]["Frequency"] = 1
            e["File Output"]["Path"] = f"{output_dir}/results_phase_1/{exp_name}"
            e["Console Output"]["Frequency"] = 1
            e["Console Output"]["Verbosity"] = "Detailed"
            e["Store Sample Information"] = True
            eList.append(e)
    if profiling:
        k["Profiling"]["Detail"] = "Full"
        k["Profiling"]["Frequency"] = 0.5
    k.run(eList)


def main(argv):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="_setup")
    args = parser.parse_args()
    run_inference(restart=args.restart, profiling=args.profiling, dry_run=args.dry_run, config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
