#!/usr/bin/env python3

from __future__ import annotations

import inspect
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import korali
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight test environments
    korali = None

try:
    from mpi4py import MPI
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight test environments
    class _FallbackComm:
        def Get_rank(self) -> int:
            return 0

        def Barrier(self) -> None:
            return None

        def send(self, *_args, **_kwargs) -> None:
            return None

        def recv(self, *_args, **_kwargs):
            raise RuntimeError("mpi4py is required for multi-rank communication.")

    class _FallbackMPI:
        COMM_WORLD = _FallbackComm()

    MPI = _FallbackMPI()

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
sys.path.insert(0, os.path.join(here, "../../src"))

from meso_uq.noise.legacy import (
    legacy_compression_direct_likelihood,
    legacy_compression_surrogate_batch_likelihood,
    legacy_compression_surrogate_likelihood,
)
from meso_uq.inference.emb_parameterization import (
    direct_parameters_to_legacy_batch,
    direct_parameters_to_legacy_vector,
)
from meso_uq.workflow_acceleration import (
    expand_parameter_vector,
    expand_reduced_parameters,
    get_fixed_parameters,
)
from meso_uq.workflows.legacy import (
    resolve_legacy_inference_config_path,
    resolve_legacy_project_root,
    resolve_legacy_surrogate_runtime,
    resolve_legacy_surrogate_trained_dir,
)

_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_SURROGATE_CACHE: Dict[Tuple[str, float, str, str], Any] = {}
_DIRECT_SURROGATE_CACHE: Dict[Tuple[str, float, str, str, str], Any] = {}


def _get_worker_comm():
    if korali is not None:
        try:
            return korali.getWorkerMPIComm()
        except Exception:
            pass
    return MPI.COMM_WORLD


@lru_cache(maxsize=1)
def _resolve_project_root() -> str:
    return str(
        resolve_legacy_project_root(
            marker_parts=("emb", "compression", "src"),
            start_path=os.getcwd(),
            anchor_file=__file__,
            max_parent_depth=3,
            exists=os.path.exists,
        )
    )


def _resolve_config_path(project_root: str) -> Path:
    return resolve_legacy_inference_config_path(project_root, "compression")


def _load_config(project_root: str) -> Dict[str, Any]:
    config_path = _resolve_config_path(project_root)
    key = str(config_path)
    if key not in _CONFIG_CACHE:
        with open(config_path, "rb") as f:
            _CONFIG_CACHE[key] = yaml.load(f, Loader=yaml.CLoader)
    return _CONFIG_CACHE[key]


def _resolve_surrogate_runtime(config: Dict[str, Any]) -> Tuple[str, int, int]:
    return resolve_legacy_surrogate_runtime(config).as_tuple()


def _build_surrogate(
    project_root: str, diameter_um: float, device: str = "cpu", backend: str = "dnn"
) -> Any:
    surrogate_path = os.fspath(
        resolve_legacy_surrogate_trained_dir(project_root, "compression", diameter_um)
    )
    if backend == "dnn":
        from emb.compression.surrogate.evaluate import Surrogate

        return Surrogate(surrogate_path, device=device)
    if backend == "bnn":
        from emb.compression.surrogate.evaluate_bnn import Surrogate

        return Surrogate(surrogate_path, device=device)
    raise ValueError(f"Unsupported surrogate backend '{backend}'.")


def _resolve_direct_surrogate_trained_dir(project_root: str, diameter_um: float) -> Path:
    try:
        from meso_uq.experiments import load_experiments

        config = _load_config(project_root)
        experiments = load_experiments(config, Path(project_root))
        matches = [
            exp
            for exp in experiments
            if exp.enabled
            and exp.name == "compression"
            and getattr(exp, "surrogate_parameterization", "") == "direct_ka_kb"
            and any(abs(float(candidate) - float(diameter_um)) < 1.0e-9 for candidate in exp.diameters)
        ]
        if len(matches) == 1:
            exp = matches[0]
            diameter_label = exp._lookup_diameter_mapping(exp.diameter_labels, diameter_um) or str(diameter_um)
            return exp.surrogate_dir / f"{diameter_label}um" / "trained"
        if len(matches) > 1:
            raise ValueError(
                f"Direct compression surrogate path is ambiguous for diameter {diameter_um}: "
                f"{[exp.dataset_name(diameter_um) for exp in matches]}"
            )
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
    return resolve_legacy_surrogate_trained_dir(project_root, "compression", diameter_um)


def _build_direct_surrogate(
    project_root: str, diameter_um: float, device: str = "cpu", backend: str = "dnn"
) -> Any:
    if backend != "dnn":
        raise NotImplementedError("Direct ka/kb compression surrogates currently support backend='dnn' only.")
    surrogate_path = os.fspath(_resolve_direct_surrogate_trained_dir(project_root, diameter_um))
    from emb.compression.surrogate.evaluate import Surrogate

    return Surrogate(surrogate_path, device=device)


def _get_surrogate(
    project_root: str, diameter_um: float, device: str = "cpu", backend: str = "dnn"
) -> Any:
    key = (project_root, diameter_um, device, backend)
    if key not in _SURROGATE_CACHE:
        _SURROGATE_CACHE[key] = _build_surrogate(
            project_root, diameter_um, device=device, backend=backend
        )
    return _SURROGATE_CACHE[key]


def _get_direct_surrogate(
    project_root: str, diameter_um: float, device: str = "cpu", backend: str = "dnn"
) -> Any:
    surrogate_dir = str(_resolve_direct_surrogate_trained_dir(project_root, diameter_um))
    key = (project_root, diameter_um, device, backend, surrogate_dir)
    if key not in _DIRECT_SURROGATE_CACHE:
        _DIRECT_SURROGATE_CACHE[key] = _build_direct_surrogate(
            project_root, diameter_um, device=device, backend=backend
        )
    return _DIRECT_SURROGATE_CACHE[key]


def preload_compression_surrogate(
    diameter_um: float, device: str = "cpu", backend: str = "dnn"
) -> None:
    _get_surrogate(_resolve_project_root(), diameter_um, device=device, backend=backend)


def preload_compression_surrogate_direct(
    diameter_um: float, device: str = "cpu", backend: str = "dnn"
) -> None:
    _get_direct_surrogate(_resolve_project_root(), diameter_um, device=device, backend=backend)


def _load_run_equil():
    from emb.compression.src.equil import run_equil

    return run_equil


def compute_compression_surrogate(
    sample: Dict[str, Any], displ: List[float], diameter_um: float, device: str = "cpu"
) -> None:
    project_root = _resolve_project_root()
    config = _load_config(project_root)
    if config.get("debug", 0) >= 1:
        print(f"Running from function {inspect.currentframe().f_code.co_name} in script {__file__}")
    params = expand_parameter_vector(
        sample["Parameters"], fixed_params=get_fixed_parameters(config)
    )
    Yt, kb, b1, b2, a3, a4, d0, sigma = params.tolist()
    backend, predictive_mc_samples, predictive_mc_chunk_size = _resolve_surrogate_runtime(config)
    surrogate = _get_surrogate(project_root, diameter_um, device=device, backend=backend)
    displ_corrected = [max(0.0, d - d0) for d in displ]
    if backend == "dnn":
        forces = surrogate.evaluate_compression(x=[Yt, kb, b1, b2, a3, a4], disp=displ_corrected)
        legacy_compression_surrogate_likelihood(forces, sigma).assign_to_sample(sample)
        return

    force_mean, force_std = surrogate.evaluate_compression(
        x=[Yt, kb, b1, b2, a3, a4],
        disp=displ_corrected,
        predictive_mc_samples=predictive_mc_samples,
        predictive_mc_chunk_size=predictive_mc_chunk_size,
    )
    force_mean_arr = np.asarray(force_mean, dtype=np.float64)
    force_std_arr = np.asarray(force_std, dtype=np.float64)
    legacy_compression_surrogate_likelihood(
        force_mean_arr,
        sigma,
        surrogate_standard_deviation=force_std_arr,
    ).assign_to_sample(sample)


def compute_compression_surrogate_direct(
    sample: Dict[str, Any],
    displ: List[float],
    diameter_um: float,
    device: str = "cpu",
    backend: str = "dnn",
) -> None:
    project_root = _resolve_project_root()
    config = _load_config(project_root)
    params = np.asarray(sample["Parameters"], dtype=np.float32).reshape(-1)
    if params.shape != (4,):
        raise ValueError(f"Direct compression surrogate expects [ka, kb, d0, sigma], got shape {params.shape}.")
    legacy_params = direct_parameters_to_legacy_vector(
        params,
        config=config,
        modality="compression",
        project_root=project_root,
    )
    Yt, kb, b1, b2, a3, a4, d0, sigma = legacy_params.tolist()
    surrogate = _get_direct_surrogate(
        project_root, diameter_um, device=device, backend=backend
    )
    displ_corrected = [max(0.0, d - d0) for d in displ]
    forces = surrogate.evaluate_compression(x=[Yt, kb, b1, b2, a3, a4], disp=displ_corrected)
    legacy_compression_surrogate_likelihood(forces, sigma).assign_to_sample(sample)


def compute_compression_surrogate_batch(
    sample: Dict[str, Any],
    displ: List[float],
    diameter_um: float,
    device: str = "cuda",
    particle_batch_size: int = 2048,
) -> None:
    """Batch surrogate evaluation for GPU-batch TMCMC."""
    project_root = _resolve_project_root()
    config = _load_config(project_root)
    fixed_params = get_fixed_parameters(config)
    backend, predictive_mc_samples, predictive_mc_chunk_size = _resolve_surrogate_runtime(config)
    batch_params = np.asarray(sample["Batch Parameters"], dtype=np.float32)
    if batch_params.ndim != 2:
        raise ValueError(f"Expected 2D batch params, got {batch_params.shape}")
    if batch_params.shape[1] == 8:
        theta, d0, sigma = batch_params[:, :6], batch_params[:, 6], batch_params[:, 7]
    elif batch_params.shape[1] == 7:
        theta = batch_params[:, :6]
        d0 = np.zeros(batch_params.shape[0], dtype=np.float32)
        sigma = batch_params[:, 6]
    elif batch_params.shape[1] == 4:
        expanded = expand_reduced_parameters(batch_params, fixed_params=fixed_params or None)
        theta, d0, sigma = expanded[:, :6], expanded[:, 6], expanded[:, 7]
    else:
        raise ValueError(f"Expected 4, 7, or 8 params, got {batch_params.shape[1]}")
    surrogate = _get_surrogate(project_root, diameter_um, device=device, backend=backend)
    if backend == "dnn":
        forces = surrogate.evaluate_compression_batch(
            theta, disp=displ, d0=d0, chunk_size=particle_batch_size
        )
        legacy_compression_surrogate_batch_likelihood(forces, sigma).assign_to_sample(sample)
        return

    force_mean, force_std = surrogate.evaluate_compression_batch(
        theta,
        disp=displ,
        d0=d0,
        chunk_size=particle_batch_size,
        predictive_mc_samples=predictive_mc_samples,
        predictive_mc_chunk_size=predictive_mc_chunk_size,
    )
    legacy_compression_surrogate_batch_likelihood(
        force_mean,
        sigma,
        surrogate_standard_deviation=force_std,
    ).assign_to_sample(sample)


def compute_compression_surrogate_batch_direct(
    sample: Dict[str, Any],
    displ: List[float],
    diameter_um: float,
    device: str = "cuda",
    particle_batch_size: int = 2048,
    backend: str = "dnn",
) -> None:
    project_root = _resolve_project_root()
    config = _load_config(project_root)
    batch_params = np.asarray(sample["Batch Parameters"], dtype=np.float32)
    if batch_params.ndim != 2 or batch_params.shape[1] != 4:
        raise ValueError(
            "Direct compression surrogate expects Batch Parameters with shape [batch, 4] "
            f"for [ka, kb, d0, sigma], got {batch_params.shape}."
        )
    legacy_params = direct_parameters_to_legacy_batch(
        batch_params,
        config=config,
        modality="compression",
        project_root=project_root,
    )
    theta = legacy_params[:, :6]
    d0 = legacy_params[:, 6]
    sigma = legacy_params[:, 7]
    surrogate = _get_direct_surrogate(
        project_root, diameter_um, device=device, backend=backend
    )
    forces = surrogate.evaluate_compression_batch(
        theta,
        disp=displ,
        d0=d0,
        chunk_size=particle_batch_size,
    )
    legacy_compression_surrogate_batch_likelihood(forces, sigma).assign_to_sample(sample)


def compute_compression(
    sample: Dict[str, Any],
    displ: List[float],
    diameter_um: float,
    init_compression_path: Optional[str] = None,
) -> None:
    project_root = os.fspath(
        resolve_legacy_project_root(
            marker_parts=("emb", "compression", "src"),
            start_path=os.getcwd(),
            anchor_file=__file__,
            max_parent_depth=3,
            exists=os.path.exists,
        )
    )
    with open(
        os.path.join(
            project_root, "inference/configs/production/inference_config_compression.yaml"
        ),
        "rb",
    ) as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    params = expand_parameter_vector(
        sample["Parameters"], fixed_params=get_fixed_parameters(config)
    )
    Yt, kb, b1, b2, a3, a4, d0_offset, sig = params.tolist()
    theta = [Yt, kb, b1, b2, a3, a4]
    comm = _get_worker_comm()
    rank = comm.Get_rank()
    sample["Reference Evaluations"] = []
    sample["Standard Deviation"] = []
    out_names = ["" for _ in range(len(displ))]
    n_ref = 0
    list_simu_path = []
    measured_forces = []
    folder = os.path.join(project_root, f"_out/compression_{diameter_um}um") + "/"
    last_d = 0.0
    source_compression_path = os.path.join(project_root, "emb", "compression", "src") + "/"
    if init_compression_path is None:
        init_compression_path = (
            os.path.join(project_root, f"_init_compression_{diameter_um}um") + "/"
        )
    elif not init_compression_path.endswith("/"):
        init_compression_path = init_compression_path + "/"
    run_equil = _load_run_equil()
    for d in displ:
        if rank == 0:
            name = f"n{n_ref}_{np.random.randint(0, 99999):05d}/"
            comm.send(name, dest=1, tag=0)
        elif rank == 1:
            name = comm.recv(source=0, tag=0)
        simu_path = folder + name
        simnum = "00001"
        list_simu_path.append(simu_path)
        if rank == 0:
            prepare_simulation_parameters(
                source_compression_path,
                init_compression_path,
                simu_path,
                simnum,
                d - last_d,
                theta,
                diameter_um,
            )
        comm.Barrier()
        run_equil(
            source_path=init_compression_path,
            simu_path=simu_path,
            simnum=simnum,
            equil=False,
            restart=True if n_ref >= 1 else False,
            restart_path=list_simu_path[-2] if n_ref >= 1 else None,
            comm=comm,
        )
        comm.Barrier()
        df_canti = pd.read_csv(simu_path + "pinning/cantilever.csv", delimiter=",")
        df_plate = pd.read_csv(simu_path + "pinning/plate.csv", delimiter=",")
        forces_num = df_canti.fz.values - df_plate.fz.values
        measured_forces.append(np.mean(forces_num))
        last_d = d
        n_ref += 1
    legacy_compression_direct_likelihood(measured_forces, sig).assign_to_sample(sample)


def adjust_simu_params(
    sample_param: Dict[str, float], filename_1_simu: str, filename_2_simu: str
) -> None:
    for fname in [filename_1_simu, filename_2_simu]:
        with open(fname, "r") as file:
            parameters = yaml.load(file, Loader=yaml.CLoader)
        for p in sample_param:
            parameters[p] = float(sample_param[p])
        with open(fname, "w") as file:
            yaml.dump(parameters, file)


def prepare_simulation_parameters(
    source_compression_path: str,
    init_compression_path: str,
    simu_path: str,
    simnum: str,
    displacement: float,
    theta: List[float],
    diameter_um: float,
) -> None:
    sys.path.insert(0, source_compression_path)
    from parameters import write_parameters

    os.system(f"mkdir -p {simu_path}")
    os.system(f"mkdir -p {simu_path}/mesh/")
    os.system(f"mkdir -p {simu_path}/force/")
    os.system(f"mkdir -p {simu_path}/stats/")
    os.system(f"mkdir -p {simu_path}/restart/")
    os.system(f"mkdir -p {simu_path}/pinning/")
    os.system(f"cp {source_compression_path}mesh/cantilever.off {simu_path}mesh/cantilever.off")
    os.system(f"cp {source_compression_path}mesh/rigid_coords.txt {simu_path}mesh/rigid_coords.txt")
    os.system(
        f"cp {source_compression_path}mesh/rigid_coords_reflected.txt {simu_path}mesh/rigid_coords_reflected.txt"
    )
    os.system(f"cp -r {init_compression_path}parameter/ {simu_path}")
    os.system(f"cp -r {source_compression_path}gas_vesicle {simu_path}")
    os.system(f"cp -r {source_compression_path}microbubble {simu_path}")
    Yt, kb, b1, b2, a3, a4 = theta
    filename_1_simu = simu_path + "parameter/parameters-default" + simnum + ".yaml"
    filename_2_simu = simu_path + "parameter/parameters-default" + simnum + "eq.yaml"
    filename_3_simu = simu_path + "parameter/parameters.prms" + simnum + ".yaml"
    filename_4_simu = simu_path + "parameter/parameters" + simnum + ".yaml"
    adjust_simu_params(
        {"disp": displacement, "Yt": Yt, "Yl": Yt, "b1": b1, "b2": b2, "a3": a3, "a4": a4},
        filename_1_simu,
        filename_2_simu,
    )
    write_parameters(source_path=init_compression_path, simu_path=simu_path, simnum=simnum)
    adjust_simu_params(
        {"kb": kb, "b1": b1, "b2": b2, "a3": a3, "a4": a4}, filename_3_simu, filename_4_simu
    )
