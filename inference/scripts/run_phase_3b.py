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
sys.path.insert(0, os.path.join(project_root, "emb", "compression"))
sys.path.insert(0, os.path.join(project_root, "emb", "compression", "evalkit"))
sys.path.insert(0, os.path.join(project_root, "emb", "indentation"))
sys.path.insert(0, os.path.join(project_root, "emb", "indentation", "evalkit"))

from emb.compression.evalkit.posterior_compression import (
    compute_compression_surrogate,
    compute_compression_surrogate_batch,
    preload_compression_surrogate,
)
from emb.compression.evalkit.tools import datedPrint
from emb.indentation.evalkit.posterior_indentation import (
    compute_indentation_surrogate,
    compute_indentation_surrogate_batch,
    preload_indentation_surrogate,
)
from meso_uq.config import resolve_inference_config_path
from meso_uq.experiments import load_experiments
from meso_uq.inference.emb_parameterization import (
    DIRECT_KA_KB_SURROGATE_PARAMETERIZATION,
    adapt_batch_sample_for_legacy_surrogate,
    adapt_sample_for_legacy_surrogate,
    is_generic_direct_phase1_contract,
    resolve_direct_compression_surrogate_surface,
    surrogate_parameterization_for_experiment,
)
from meso_uq.inference.emb_resonance import (
    compute_emb_resonance,
    compute_emb_resonance_batch,
    preload_emb_resonance,
)
from meso_uq.workflows.legacy import resolve_legacy_surrogate_backend
from meso_uq.workflow_acceleration import (
    apply_korali_random_seed,
    configure_device_conduit,
    configure_gpu_batch_sub_experiment,
    configure_korali_conduit,
    to_korali_path,
    validate_korali_random_seed,
)


def _resolve_surrogate_backend(config: dict) -> str:
    return resolve_legacy_surrogate_backend(config)


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


def _select_phase3b_targets(
    experiments,
    dataset_name: str | None = None,
    diameter: float | None = None,
    experiment_names: list[str] | None = None,
    diameters: list[float] | None = None,
):
    if dataset_name is not None and diameter is not None:
        raise ValueError("Use either dataset_name or diameter, not both.")
    if dataset_name is not None and (experiment_names or diameters):
        raise ValueError("Use either dataset_name or config target filters, not both.")
    if diameter is not None and diameters:
        raise ValueError("Use either diameter or phase3b target diameters, not both.")

    allowed_experiments = None if not experiment_names else {str(name) for name in experiment_names}
    allowed_diameters = None if not diameters else {float(value) for value in diameters}

    selected: list[tuple[object, float]] = []
    for exp in experiments:
        if allowed_experiments is not None and exp.name not in allowed_experiments:
            continue
        for diameter_um in exp.diameters:
            current_dataset = exp.dataset_name(diameter_um)
            if dataset_name is not None and current_dataset != dataset_name:
                continue
            if diameter is not None and abs(float(diameter_um) - float(diameter)) >= 1e-9:
                continue
            if allowed_diameters is not None and not any(
                abs(float(diameter_um) - allowed) < 1e-9 for allowed in allowed_diameters
            ):
                continue
            selected.append((exp, float(diameter_um)))

    if dataset_name is not None and not selected:
        raise ValueError(f"Dataset '{dataset_name}' not found in enabled experiments.")
    if diameter is not None and not selected:
        raise ValueError(f"Diameter '{diameter}' not found in enabled experiments.")
    if dataset_name is None and diameter is None and (allowed_experiments or allowed_diameters) and not selected:
        raise ValueError("No Phase 3b targets matched the configured phase3b target filters.")
    return selected


def _optional_string_list(value, *, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError(f"{name} must be a list, not a string.")
    try:
        return [str(item) for item in value]
    except TypeError as exc:
        raise ValueError(f"{name} must be a list.") from exc


def _optional_float_list(value, *, name: str) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError(f"{name} must be a list of numbers, not a string.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a list of numbers.") from exc


def _copy_adapted_sample_outputs(target: dict, source: dict) -> None:
    for key, value in source.items():
        if key in {"Parameters", "Batch Parameters"}:
            continue
        target[key] = value


def _wrap_legacy_surrogate_model(delegate, *, config: dict[str, object], modality: str):
    def _wrapped(sample, controls, diameter_um, device="cpu"):
        adapted = adapt_sample_for_legacy_surrogate(
            sample,
            config=config,
            modality=modality,
            project_root=project_root,
        )
        delegate(adapted, controls, diameter_um, device=device)
        _copy_adapted_sample_outputs(sample, adapted)

    return _wrapped


def _wrap_legacy_surrogate_batch(delegate, *, config: dict[str, object], modality: str):
    def _wrapped(sample, controls, diameter_um, device="cuda"):
        adapted = adapt_batch_sample_for_legacy_surrogate(
            sample,
            config=config,
            modality=modality,
            project_root=project_root,
        )
        delegate(adapted, controls, diameter_um, device=device)
        _copy_adapted_sample_outputs(sample, adapted)

    return _wrapped


def run_phase_3b_dataset(
    experiment_name: str,
    diameter_um: float,
    reference_points: list,
    compute_model,
    pop_size: int,
    max_gen: int,
    target_cov: float,
    output_root: Path,
    covariance_scaling: float = 0.04,
    profiling: bool = False,
    device: str = "cpu",
    compute_batch_model=None,
    dataset_name: str | None = None,
    korali_random_seed: int | None = None,
    restart: bool = False,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    exp_name = dataset_name or f"{experiment_name}_{diameter_um}um"

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
    if device == "gpu":
        if compute_batch_model is None:
            batch_fn_map = {
                "compression": compute_compression_surrogate_batch,
                "indentation": compute_indentation_surrogate_batch,
                "resonance": compute_emb_resonance_batch,
            }
            batch_fn = batch_fn_map[experiment_name]
        else:
            batch_fn = compute_batch_model
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

    experiment_output = output_root / "results_phase_3b" / exp_name
    e = korali.Experiment()
    e["File Output"]["Path"] = to_korali_path(str(experiment_output), base_dir=str(project_root))
    ensure_output_dir(str(experiment_output))
    if restart:
        latest = experiment_output / "latest"
        found = e.loadState(str(latest))
        if not found:
            raise FileNotFoundError(f"No previous Phase 3b state found for {exp_name}: {latest}")
        if rank == 0:
            datedPrint(
                f"[Phase 3b] Restarting {exp_name} from generation {e['Current Generation']}"
            )
    applied_seed = apply_korali_random_seed(e, korali_random_seed)
    e["File Output"]["Path"] = to_korali_path(str(experiment_output), base_dir=str(project_root))
    e["Problem"]["Type"] = "Hierarchical/Theta"
    e["Problem"]["Psi Experiment"] = psi
    e["Problem"]["Sub Experiment"] = sub
    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = pop_size
    e["Solver"]["Burn In"] = 1
    e["Solver"]["Target Coefficient Of Variation"] = target_cov
    e["Solver"]["Covariance Scaling"] = covariance_scaling
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
        if applied_seed is not None:
            datedPrint(f"[Phase 3b] Random Seed for {exp_name}: {applied_seed}")

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
    dataset_name: str | None = None,
    diameter: float | None = None,
    korali_random_seed: int | None = None,
    restart: bool = False,
):
    korali_random_seed = validate_korali_random_seed(
        korali_random_seed, field_name="--korali-random-seed"
    )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    config_path_resolved = _resolve_config_path(config_path)
    with open(config_path_resolved, "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    os.environ["HUQ_INFERENCE_CONFIG"] = str(config_path_resolved)
    output_root = _resolve_output_root(output_dir)
    experiments = [exp for exp in load_experiments(config, Path(project_root)) if exp.enabled]
    surrogate_backend = _resolve_surrogate_backend(config)
    is_generic_direct_phase1 = is_generic_direct_phase1_contract(config)
    target_experiments = None
    target_diameters = None
    if dataset_name is None and diameter is None:
        target_experiments = _optional_string_list(
            config.get("phase3b_target_experiments"),
            name="phase3b_target_experiments",
        )
        target_diameters = _optional_float_list(
            config.get("phase3b_target_diameters"),
            name="phase3b_target_diameters",
        )
    selected_targets = _select_phase3b_targets(
        experiments,
        dataset_name=dataset_name,
        diameter=diameter,
        experiment_names=target_experiments,
        diameters=target_diameters,
    )
    for exp, diameter_um in selected_targets:
        parameterization = surrogate_parameterization_for_experiment(exp)
        if is_generic_direct_phase1 and parameterization == DIRECT_KA_KB_SURROGATE_PARAMETERIZATION:
            if exp.name != "compression":
                raise NotImplementedError(
                    "Direct ka/kb surrogate parameterization is only prepared for compression lanes. "
                    "Use surrogate_parameterization=legacy_yt_kb for non-compression experiments."
                )
            preload_fn, _compute_direct_fn, _compute_direct_batch_fn = resolve_direct_compression_surrogate_surface()
        else:
            preload_map = {
                "compression": preload_compression_surrogate,
                "indentation": preload_indentation_surrogate,
                "resonance": preload_emb_resonance,
            }
            preload_fn = preload_map.get(exp.name)
        if preload_fn is None:
            raise ValueError(f"No surrogate preload function registered for experiment '{exp.name}'")
        preload_fn(diameter_um, device=device, backend=surrogate_backend)
    phase3b_pop_size = config.get("phase3b_pop_size", 10000)
    phase3b_max_gen = config.get("phase3b_max_gen", -1)
    phase3b_target_cov = config.get("phase3b_target_cov", 0.6)
    phase3b_covariance_scaling = config.get("phase3b_covariance_scaling", 0.04)

    if rank == 0:
        datedPrint("[Phase 3b] Starting dataset-specific posterior sampling")
        datedPrint(f"[Phase 3b] Population size: {phase3b_pop_size} per dataset")
        datedPrint(f"[Phase 3b] Max generations: {phase3b_max_gen}")
        datedPrint(f"[Phase 3b] Target CoV: {phase3b_target_cov}")
        datedPrint(f"[Phase 3b] Covariance scaling: {phase3b_covariance_scaling}")
        datedPrint(f"[Phase 3b] Output root: {output_root}")
        datedPrint(f"[Phase 3b] Datasets: {len(selected_targets)}")

    phase2_latest = output_root / "results_phase_2" / "latest"
    if not phase2_latest.exists():
        if rank == 0:
            datedPrint(f"[Phase 3b] ERROR: Phase 2 results not found: {phase2_latest}")
        sys.exit(1)

    for exp, diameter_um in selected_targets:
        phase1_latest = output_root / "results_phase_1" / exp.dataset_name(diameter_um) / "latest"
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
        "resonance": compute_emb_resonance,
    }
    compute_batch_map = {
        "compression": compute_compression_surrogate_batch,
        "indentation": compute_indentation_surrogate_batch,
        "resonance": compute_emb_resonance_batch,
    }
    for target_index, (exp, diameter_um) in enumerate(selected_targets):
        parameterization = surrogate_parameterization_for_experiment(exp)
        if is_generic_direct_phase1 and parameterization == DIRECT_KA_KB_SURROGATE_PARAMETERIZATION:
            if exp.name != "compression":
                raise NotImplementedError(
                    "Direct ka/kb surrogate parameterization is only prepared for compression lanes. "
                    "Use surrogate_parameterization=legacy_yt_kb for non-compression experiments."
                )
            _preload_direct_fn, model, batch_model = resolve_direct_compression_surrogate_surface()
        else:
            model = compute_surrogate_map[exp.name]
            batch_model = compute_batch_map[exp.name]
        if (
            is_generic_direct_phase1
            and parameterization != DIRECT_KA_KB_SURROGATE_PARAMETERIZATION
            and exp.name in {"compression", "indentation"}
        ):
            model = _wrap_legacy_surrogate_model(
                model,
                config=config,
                modality=exp.name,
            )
            batch_model = _wrap_legacy_surrogate_batch(
                batch_model,
                config=config,
                modality=exp.name,
            )
        run_phase_3b_dataset(
            experiment_name=exp.name,
            diameter_um=diameter_um,
            reference_points=exp.get_reference_points(diameter_um),
            compute_model=model,
            compute_batch_model=batch_model,
            pop_size=phase3b_pop_size,
            max_gen=phase3b_max_gen,
            target_cov=phase3b_target_cov,
            covariance_scaling=phase3b_covariance_scaling,
            output_root=output_root,
            profiling=profiling,
            device=device,
            dataset_name=exp.dataset_name(diameter_um),
            korali_random_seed=(
                None if korali_random_seed is None else korali_random_seed + target_index
            ),
            restart=restart,
        )


def main(argv):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="_setup")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu: Distributed MPI conduit; gpu: Sequential GPU-batch conduit (single rank)",
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument(
        "--korali-random-seed",
        type=int,
        default=None,
        help=(
            "Positive nonzero Korali Random Seed. When multiple Phase 3b datasets "
            "are selected, this is a base seed incremented by selected-target order."
        ),
    )
    args = parser.parse_args()
    run_phase_3b(
        profiling=args.profiling,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        dataset_name=args.dataset_name,
        diameter=args.diameter,
        korali_random_seed=args.korali_random_seed,
        restart=args.restart,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
