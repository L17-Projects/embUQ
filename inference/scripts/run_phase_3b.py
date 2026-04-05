#!/usr/bin/env python3

import gc
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

from compression.evalkit.posterior_compression import compute_compression_surrogate
from compression.evalkit.tools import datedPrint
from meso_uq.config import resolve_inference_config_path
from meso_uq.experiments import load_experiments
from indentation.evalkit.posterior_indentation import compute_indentation_surrogate


def _extract_reference_data(sub):
    try:
        ref_data = sub["Problem"]["Reference Data"]
    except Exception:
        return None
    try:
        return list(ref_data)
    except Exception:
        try:
            return [ref_data[i] for i in range(len(ref_data))]
        except Exception:
            return None


def _align_sub_reference(sub, ref_points, exp_name, rank):
    ref_data = _extract_reference_data(sub)
    if ref_data is None:
        return ref_points
    if len(ref_points) != len(ref_data):
        min_len = min(len(ref_points), len(ref_data))
        if rank == 0:
            datedPrint(f"[Phase 3b] WARNING: Reference points/data length mismatch for {exp_name} (points={len(ref_points)}, data={len(ref_data)}). Trimming to {min_len}.")
        sub["Problem"]["Reference Data"] = ref_data[:min_len]
        return ref_points[:min_len]
    return ref_points


def run_phase_3b_dataset(experiment_name: str, diameter_um: float, reference_points: list, compute_model, pop_size: int, max_gen: int, target_cov: float, output_dir: str, profiling: bool = False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    exp_name = f"{experiment_name}_{diameter_um}um"

    def ensure_output_dir(path: str) -> None:
        if rank == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    k = korali.Engine()
    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 1

    psi = korali.Experiment()
    sub = korali.Experiment()
    psi.loadState(f"{output_dir}/results_phase_2/latest")
    sub.loadState(f"{output_dir}/results_phase_1/{exp_name}/latest")
    reference_points = _align_sub_reference(sub, reference_points, exp_name, rank)
    sub["Problem"]["Computational Model"] = lambda sampleData, d=diameter_um, pts=reference_points, model=compute_model: model(sampleData, pts, d)

    e = korali.Experiment()
    e["File Output"]["Path"] = f"{output_dir}/results_phase_3b/{exp_name}/"
    ensure_output_dir(f"{output_dir}/results_phase_3b/{exp_name}/")
    e["Problem"]["Type"] = "Hierarchical/Theta"
    e["Problem"]["Psi Experiment"] = psi
    e["Problem"]["Sub Experiment"] = sub
    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = pop_size
    e["Solver"]["Burn In"] = 1
    e["Solver"]["Target Coefficient Of Variation"] = target_cov
    if max_gen > 0:
        e["Solver"]["Termination Criteria"]["Max Generations"] = max_gen
    e["Console Output"]["Verbosity"] = "Detailed"
    e["Console Output"]["Frequency"] = 1
    e["File Output"]["Frequency"] = 1
    e["Store Sample Information"] = True
    if profiling:
        k["Profiling"]["Detail"] = "Full"
        k["Profiling"]["Frequency"] = 0.5
    k.run(e)
    del e, psi, sub, k
    gc.collect()
    comm.Barrier()


def run_phase_3b(profiling: bool = False, config_path: str = None, output_dir: str = "_setup"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if config_path is None:
        config_path = str(resolve_inference_config_path(project_root, experiment="compression"))
    with open(config_path, "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    experiments = [exp for exp in load_experiments(config, Path(project_root)) if exp.enabled]
    if not os.path.exists(f"{output_dir}/results_phase_2/latest"):
        if rank == 0:
            datedPrint("[Phase 3b] ERROR: Phase 2 results not found!")
        sys.exit(1)
    compute_surrogate_map = {"compression": compute_compression_surrogate, "indentation": compute_indentation_surrogate}
    for exp in experiments:
        model = compute_surrogate_map[exp.name]
        for diameter_um in exp.diameters:
            run_phase_3b_dataset(
                experiment_name=exp.name,
                diameter_um=diameter_um,
                reference_points=exp.get_reference_points(diameter_um),
                compute_model=model,
                pop_size=config.get("phase3b_pop_size", 10000),
                max_gen=config.get("phase3b_max_gen", -1),
                target_cov=config.get("phase3b_target_cov", 0.6),
                output_dir=output_dir,
                profiling=profiling,
            )


def main(argv):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="_setup")
    args = parser.parse_args()
    run_phase_3b(profiling=args.profiling, config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
