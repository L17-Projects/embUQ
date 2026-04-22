#!/usr/bin/env python3

from __future__ import annotations

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

from compression.evalkit.posterior_compression import (
    compute_compression_surrogate,
    compute_compression_surrogate_batch,
    preload_compression_surrogate,
)
from compression.evalkit.tools import datedPrint
from indentation.evalkit.posterior_indentation import (
    compute_indentation_surrogate,
    compute_indentation_surrogate_batch,
    preload_indentation_surrogate,
)
from meso_uq.config import resolve_inference_config_path
from meso_uq.experiments import load_experiments
from meso_uq.workflow_acceleration import (
    configure_device_conduit,
    configure_gpu_batch_sub_experiment,
    configure_korali_conduit,
    to_korali_path,
)


def _resolve_surrogate_backend(config: dict) -> str:
    surrogate_cfg = config.get("surrogate", {})
    if surrogate_cfg is None:
        surrogate_cfg = {}
    if not isinstance(surrogate_cfg, dict):
        raise ValueError("Expected 'surrogate' config section to be a mapping.")
    backend = str(surrogate_cfg.get("backend", "dnn")).strip().lower()
    if backend not in {"dnn", "bnn"}:
        raise ValueError(f"Unsupported surrogate backend '{backend}'. Expected 'dnn' or 'bnn'.")
    return backend


def _resolve_config_path(config_path: str | None) -> Path:
    if config_path is None:
        return Path(resolve_inference_config_path(project_root, experiment="compression")).resolve()
    candidate = Path(config_path).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = Path(project_root, candidate)
    return candidate.resolve()


def _resolve_output_root(output_dir: str | Path) -> Path:
    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = Path(project_root, output_root)
    return output_root.resolve()


def _extract_reference_data(sub):
    try:
        ref_data = sub["Problem"]["Reference Data"]
    except Exception:
        return None
    if ref_data is None:
        return None

    # Prefer bounded index-based extraction when length is available.
    try:
        n_ref = len(ref_data)
    except Exception:
        n_ref = None

    if n_ref is not None:
        try:
            return [ref_data[i] for i in range(n_ref)]
        except Exception:
            return None

    try:
        return list(ref_data)
    except Exception:
        return None


def _align_sub_reference(sub, ref_points, exp_name, rank):
    ref_data = _extract_reference_data(sub)
    if ref_data is None:
        return ref_points
    if len(ref_points) != len(ref_data):
        min_len = min(len(ref_points), len(ref_data))
        if rank == 0:
            datedPrint(
                f"[Phase 3b] WARNING: Reference points/data length mismatch for {exp_name} (points={len(ref_points)}, data={len(ref_data)}). Trimming to {min_len}."
            )
        sub["Problem"]["Reference Data"] = ref_data[:min_len]
        return ref_points[:min_len]
    return ref_points


def run_phase_3b_dataset(
    experiment_name: str,
    diameter_um: float,
    reference_points: list,
    compute_model,
    pop_size: int,
    max_gen: int,
    target_cov: float,
    output_root: Path,
    profiling: bool = False,
    device: str = "cpu",
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    exp_name = f"{experiment_name}_{diameter_um}um"

    def ensure_output_dir(path: str) -> None:
        if rank == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    if rank == 0:
        datedPrint(f"[Phase 3b] Setting up {exp_name}")

    phase2_latest = output_root / "results_phase_2" / "latest"
    phase1_latest = output_root / "results_phase_1" / exp_name / "latest"
    if not phase2_latest.exists():
        raise FileNotFoundError(f"Phase 2 results not found: {phase2_latest}")
    if not phase1_latest.exists():
        raise FileNotFoundError(f"Phase 1 results not found for {exp_name}: {phase1_latest}")

    k = korali.Engine()
    if device == "cpu":
        k.setMPIComm(MPI.COMM_WORLD)
    configure_device_conduit(k, device=device, mpi_ranks=comm.Get_size())

    psi = korali.Experiment()
    sub = korali.Experiment()
    psi_found = psi.loadState(str(phase2_latest))
    sub_found = sub.loadState(str(phase1_latest))
    if not psi_found:
        raise RuntimeError(f"Failed to load Phase 2 state from {phase2_latest}")
    if not sub_found:
        raise RuntimeError(f"Failed to load Phase 1 state from {phase1_latest}")

    reference_points = _align_sub_reference(sub, reference_points, exp_name, rank)
    batch_fn_map = {
        "compression": compute_compression_surrogate_batch,
        "indentation": compute_indentation_surrogate_batch,
    }
    if device == "gpu":
        batch_fn = batch_fn_map[experiment_name]
        configure_gpu_batch_sub_experiment(
            sub,
            batch_model_fn=lambda s, d=diameter_um, pts=reference_points, dev=device, fn=batch_fn: fn(
                s, pts, d, device=dev
            ),
            single_model_fn=lambda s, d=diameter_um, pts=reference_points, dev=device, m=compute_model: m(
                s, pts, d, device=dev
            ),
        )
    else:
        sub["Problem"]["Computational Model"] = (
            lambda sampleData, d=diameter_um, pts=reference_points, dev=device, model=compute_model: model(
                sampleData, pts, d, device=dev
            )
        )

    if rank == 0:
        datedPrint(f"[Phase 3b] Loaded Phase 1 and Phase 2 states for {exp_name}")

    e = korali.Experiment()
    experiment_output = output_root / "results_phase_3b" / exp_name
    e["File Output"]["Path"] = to_korali_path(str(experiment_output), base_dir=str(project_root))
    ensure_output_dir(str(experiment_output))
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
    e["File Output"]["Use Multiple Files"] = False
    e["Store Sample Information"] = True
    if profiling:
        k["Profiling"]["Detail"] = "Full"
        k["Profiling"]["Frequency"] = 0.5

    if rank == 0:
        datedPrint(f"[Phase 3b] Starting TMCMC for {exp_name} with {comm.Get_size()} MPI ranks")

    k.run(e)

    if rank == 0:
        datedPrint(f"[Phase 3b] Completed sampling for {exp_name}")

    del e, psi, sub, k
    gc.collect()
    comm.Barrier()

    if rank == 0:
        datedPrint(f"[Phase 3b] Memory cleanup completed for {exp_name}")


def run_phase_3b(
    profiling: bool = False,
    config_path: str = None,
    output_dir: str = "_setup",
    device: str = "cpu",
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    config_path_resolved = _resolve_config_path(config_path)
    with open(config_path_resolved, "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    os.environ["HUQ_INFERENCE_CONFIG"] = str(config_path_resolved)
    output_root = _resolve_output_root(output_dir)
    experiments = [exp for exp in load_experiments(config, Path(project_root)) if exp.enabled]
    surrogate_backend = _resolve_surrogate_backend(config)
    preload_map = {
        "compression": preload_compression_surrogate,
        "indentation": preload_indentation_surrogate,
    }
    for exp in experiments:
        preload_fn = preload_map.get(exp.name)
        if preload_fn is None:
            raise ValueError(f"No surrogate preload function registered for experiment '{exp.name}'")
        for diameter_um in exp.diameters:
            preload_fn(diameter_um, device=device, backend=surrogate_backend)
    phase3b_pop_size = config.get("phase3b_pop_size", 10000)
    phase3b_max_gen = config.get("phase3b_max_gen", -1)
    phase3b_target_cov = config.get("phase3b_target_cov", 0.6)

    if rank == 0:
        datedPrint("[Phase 3b] Starting dataset-specific posterior sampling")
        datedPrint(f"[Phase 3b] Population size: {phase3b_pop_size} per dataset")
        datedPrint(f"[Phase 3b] Max generations: {phase3b_max_gen}")
        datedPrint(f"[Phase 3b] Target CoV: {phase3b_target_cov}")
        datedPrint(f"[Phase 3b] Output root: {output_root}")
        total_sets = sum(len(exp.diameters) for exp in experiments)
        datedPrint(f"[Phase 3b] Datasets: {total_sets}")

    phase2_latest = output_root / "results_phase_2" / "latest"
    if not phase2_latest.exists():
        if rank == 0:
            datedPrint(f"[Phase 3b] ERROR: Phase 2 results not found: {phase2_latest}")
        sys.exit(1)

    for exp in experiments:
        for diameter_um in exp.diameters:
            phase1_latest = (
                output_root / "results_phase_1" / exp.dataset_name(diameter_um) / "latest"
            )
            if not phase1_latest.exists():
                if rank == 0:
                    datedPrint(
                        f"[Phase 3b] ERROR: Phase 1 results not found for {exp.name} {diameter_um} μm: {phase1_latest}"
                    )
                sys.exit(1)

    if rank == 0:
        datedPrint("[Phase 3b] Verified prerequisite Phase 1 and Phase 2 results")

    compute_surrogate_map = {
        "compression": compute_compression_surrogate,
        "indentation": compute_indentation_surrogate,
    }
    for exp in experiments:
        model = compute_surrogate_map[exp.name]
        for diameter_um in exp.diameters:
            run_phase_3b_dataset(
                experiment_name=exp.name,
                diameter_um=diameter_um,
                reference_points=exp.get_reference_points(diameter_um),
                compute_model=model,
                pop_size=phase3b_pop_size,
                max_gen=phase3b_max_gen,
                target_cov=phase3b_target_cov,
                output_root=output_root,
                profiling=profiling,
                device=device,
            )


def main(argv):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="_setup")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu: Distributed MPI conduit; gpu: Sequential GPU-batch conduit (single rank)",
    )
    args = parser.parse_args()
    run_phase_3b(
        profiling=args.profiling,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
