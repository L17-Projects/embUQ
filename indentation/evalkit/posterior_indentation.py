#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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

from indentation.evalkit.tools import dated_print
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
_DUMP_FLAG: bool | None = None


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
            marker_parts=("indentation", "src"),
            start_path=os.getcwd(),
            anchor_file=__file__,
            exists=os.path.exists,
        )
    )


def _resolve_config_path(project_root: str) -> Path:
    return resolve_legacy_inference_config_path(project_root, "indentation")


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
        resolve_legacy_surrogate_trained_dir(project_root, "indentation", diameter_um)
    )
    if backend == "dnn":
        from indentation.surrogate.evaluate import Surrogate

        return Surrogate(surrogate_path, device=device)
    if backend == "bnn":
        from indentation.surrogate.evaluate_bnn import Surrogate

        return Surrogate(surrogate_path, device=device)
    raise ValueError(f"Unsupported surrogate backend '{backend}'.")


def _get_surrogate(
    project_root: str, diameter_um: float, device: str = "cpu", backend: str = "dnn"
) -> Any:
    key = (project_root, diameter_um, device, backend)
    if key not in _SURROGATE_CACHE:
        _SURROGATE_CACHE[key] = _build_surrogate(
            project_root, diameter_um, device=device, backend=backend
        )
    return _SURROGATE_CACHE[key]


def preload_indentation_surrogate(
    diameter_um: float, device: str = "cpu", backend: str = "dnn"
) -> None:
    _get_surrogate(_resolve_project_root(), diameter_um, device=device, backend=backend)


def _get_dump_flag() -> bool:
    global _DUMP_FLAG
    if _DUMP_FLAG is None:
        _DUMP_FLAG = _load_config(_resolve_project_root()).get("dump", False)
    return _DUMP_FLAG


def compute_indentation_surrogate(
    sample: Dict[str, Any], forces: List[float], diameter_um: float, device: str = "cpu"
) -> None:
    project_root = _resolve_project_root()
    dump = _get_dump_flag()
    config = _load_config(project_root)
    params = expand_parameter_vector(
        sample["Parameters"], fixed_params=get_fixed_parameters(config)
    )
    Yt, kb, b1, b2, a3, a4, d0, sigma = params.tolist()
    backend, predictive_mc_samples, predictive_mc_chunk_size = _resolve_surrogate_runtime(config)
    surrogate = _get_surrogate(project_root, diameter_um, device=device, backend=backend)
    if backend == "dnn":
        displacements = surrogate.evaluate_indentation(x=[Yt, kb, b1, b2, a3, a4], forces=forces)
        displacements = np.maximum(0.0, np.asarray(displacements) + d0)
        sample["Reference Evaluations"] = displacements.tolist()
        sample["Standard Deviation"] = (sigma * displacements).tolist()
        return

    disp_mean, disp_std = surrogate.evaluate_indentation(
        x=[Yt, kb, b1, b2, a3, a4],
        forces=forces,
        predictive_mc_samples=predictive_mc_samples,
        predictive_mc_chunk_size=predictive_mc_chunk_size,
    )
    disp_mean_arr = np.maximum(0.0, np.asarray(disp_mean, dtype=np.float64) + d0)
    disp_std_arr = np.asarray(disp_std, dtype=np.float64)
    obs_std_arr = sigma * np.abs(disp_mean_arr)
    total_std_arr = np.sqrt(np.square(disp_std_arr) + np.square(obs_std_arr))
    comm = _get_worker_comm()
    if dump and comm.Get_rank() == 0:
        print(
            f"[Korali] Indentation surrogate [D={diameter_um}um] | Yt={Yt:.0f}, kb={kb:.0f}, b1={b1:.2f}, b2={b2:.2f}, a3={a3:.2f}, a4={a4:.2f}, d0={d0:.4f}"
        )
    sample["Reference Evaluations"] = disp_mean_arr.tolist()
    sample["Standard Deviation"] = total_std_arr.tolist()


def compute_indentation_surrogate_batch(
    sample: Dict[str, Any],
    forces: List[float],
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
        displacements = surrogate.evaluate_indentation_batch(
            theta, forces=forces, d0=d0, chunk_size=particle_batch_size
        )
        sample["Batch Reference Evaluations"] = displacements.tolist()
        sample["Batch Standard Deviation"] = (sigma[:, None] * displacements).tolist()
        return

    disp_mean, disp_std = surrogate.evaluate_indentation_batch(
        theta,
        forces=forces,
        d0=d0,
        chunk_size=particle_batch_size,
        predictive_mc_samples=predictive_mc_samples,
        predictive_mc_chunk_size=predictive_mc_chunk_size,
    )
    obs_std = sigma[:, None] * np.abs(disp_mean)
    total_std = np.sqrt(np.square(disp_std) + np.square(obs_std))
    sample["Batch Reference Evaluations"] = disp_mean.tolist()
    sample["Batch Standard Deviation"] = total_std.tolist()


def adjust_simu_params(sample_param, filename_1_simu, filename_2_simu):
    for fname in [filename_1_simu, filename_2_simu]:
        with open(fname, "r") as file:
            parameters = yaml.load(file, Loader=yaml.CLoader)
        for p in sample_param:
            parameters[p] = float(sample_param[p])
        with open(fname, "w") as file:
            yaml.dump(parameters, file)


def prepare_simulation_parameters(
    source_indentation_path: str,
    init_indentation_path: str,
    simu_path: str,
    simnum: str,
    displacement: float,
    theta: List[float],
    diameter_um: float,
) -> None:
    from indentation.src.parameters import write_parameters

    os.system(f"mkdir -p {simu_path}")
    os.system(f"mkdir -p {simu_path}/mesh/")
    os.system(f"mkdir -p {simu_path}/force/")
    os.system(f"mkdir -p {simu_path}/stats/")
    os.system(f"mkdir -p {simu_path}/restart/")
    os.system(f"cp -r {init_indentation_path}parameter/ {simu_path}")
    os.system(f"cp -r {source_indentation_path}/microbubble {simu_path}")
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
    write_parameters(source_path=init_indentation_path, simu_path=simu_path, simnum=simnum)
    adjust_simu_params(
        {"kb": kb, "b1": b1, "b2": b2, "a3": a3, "a4": a4}, filename_3_simu, filename_4_simu
    )


def compute_indentation(  # pragma: no cover
    sample,
    X,
    *,
    project_root=None,
    diameter_um=None,
    init_indentation_path=None,
):
    """Run Mirheo DPD indentation simulations at the given sample parameters.

    All Mirheo/h5py imports are deferred to inside this function so that the
    module can be imported in environments where those packages are absent.
    Requires: mirheo, h5py, mpi4py.
    """
    import fnmatch
    import os
    from datetime import datetime
    from pathlib import Path

    import h5py
    import numpy as np
    import yaml
    from mpi4py import MPI

    from indentation.src.equil import run_equil
    from indentation.src.parameters import write_parameters

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (
        repo_root / "inference" / "configs" / "production" / "inference_config_indentation.yaml"
    )
    with open(cfg_path, "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    out = config["out"]
    dump = bool(config.get("dump", False))

    source_indentation_path = repo_root / "indentation" / "src"

    if init_indentation_path is None:
        if project_root is None or diameter_um is None:
            init_indentation_path = "init_indentation/"
        else:
            init_indentation_path = (
                os.path.join(project_root, f"_init_indentation_{diameter_um}um") + "/"
            )
    elif not init_indentation_path.endswith("/"):
        init_indentation_path = init_indentation_path + "/"

    from meso_uq.workflow_acceleration import expand_parameter_vector

    full_params = expand_parameter_vector(sample["Parameters"])
    Yt, kb, b1, b2, a3, a4, _d0, sig = full_params  # d0 not used by Mirheo
    theta = [Yt, kb, b1, b2, a3, a4]
    filename_param = ""
    for p in theta:
        filename_param += "%.2f" % (p) + "_"

    comm = _get_worker_comm()

    rank = comm.Get_rank()

    sample["Reference Evaluations"] = []
    sample["Standard Deviation"] = []

    n_ref = 0
    list_simu_path = []
    diam_vert = []
    std_diam_vert = []
    cnt = 0

    if rank == 0 and dump:
        os.system(f"mkdir -p {out}/Yt{Yt:.2f}_fulltraj/")

    for Xi in X:
        Xi = -Xi  # compression

        folder = f"{out}/indentation/"
        if rank == 0:
            name = "n%d_%05d/" % (n_ref, np.random.randint(0, 99999))
            name = filename_param + name
            comm.send(name, dest=1, tag=0)
        elif rank == 1:
            name = comm.recv(source=0, tag=0)

        simu_path = folder + name
        simnum = "%05d" % 1
        list_simu_path.append(simu_path)

        if rank == 0:
            os.system(f"mkdir -p {simu_path}")
            os.system(f"mkdir -p {simu_path}/mesh/")
            os.system(f"mkdir -p {simu_path}/force/")
            os.system(f"mkdir -p {simu_path}/stats/")
            os.system(f"mkdir -p {simu_path}/restart/")
            os.system(f"mkdir -p {simu_path}/logs/")
            os.system(f"mkdir -p {simu_path}/anchor/")
            os.system(f"mkdir -p {simu_path}/particles/")
            os.system(f"mkdir -p {simu_path}parameter/")

            filename_1 = init_indentation_path + "parameter/parameters-default" + simnum + ".yaml"
            filename_2 = init_indentation_path + "parameter/parameters-default" + simnum + "eq.yaml"
            os.system(f"cp {filename_1} {simu_path}parameter/")
            os.system(f"cp {filename_2} {simu_path}parameter/")
            filename_1_simu = simu_path + "parameter/parameters-default" + simnum + ".yaml"
            filename_2_simu = simu_path + "parameter/parameters-default" + simnum + "eq.yaml"

            os.system(f"cp -r {source_indentation_path}/microbubble {simu_path}")

            sample_param = {"Yt": Yt, "Yl": Yt, "force": Xi}
            adjust_simu_params(sample_param, filename_1_simu, filename_2_simu)
            write_parameters(source_path=init_indentation_path, simu_path=simu_path, simnum=simnum)

            sample_param = {"kb": kb, "b1": b1, "b2": b2, "a3": a3, "a4": a4}
            filename_3_simu = simu_path + "parameter/parameters.prms" + simnum + ".yaml"
            filename_4_simu = simu_path + "parameter/parameters" + simnum + ".yaml"
            adjust_simu_params(sample_param, filename_3_simu, filename_4_simu)

        comm.Barrier()

        hostname = os.uname()[1]
        if rank == 0:
            gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            dated_print(
                f"[Korali] [{hostname}] [GPU nº{gpu_env}] | Running "
                f"Yt={Yt}, kb={kb}, b1={b1}, b2={b2}, a3={a3}, a4={a4}, force={float(Xi)}"
            )

        flag = 1
        nb_tries = 1
        MAX_TRIES = 5
        while flag == 1:
            if nb_tries > MAX_TRIES:
                raise RuntimeError(
                    f"run_equil failed after {MAX_TRIES} attempts for simu_path={simu_path!r}"
                )
            try:
                flag = run_equil(
                    source_path=init_indentation_path,
                    simu_path=simu_path,
                    simnum=simnum,
                    equil=False if n_ref >= 1 else True,
                    restart=True if n_ref >= 1 else False,
                    restart_path=list_simu_path[-2] if n_ref >= 1 else None,
                    vacuum=False,
                    comm=comm,
                )
            except Exception as e:
                dated_print(f"[Mirheo] Error (try {nb_tries}): {e}")
                nb_tries += 1

        comm.Barrier()

        filename_default = init_indentation_path + "parameter/parameters-default" + simnum + ".yaml"
        with open(filename_default, "rb") as f:
            parameters_default = yaml.load(f, Loader=yaml.CLoader)

        stslik = parameters_default["stslik"]
        pts_sampling = int(0.9 * stslik)

        ind_min, ind_max = np.loadtxt(simu_path + "/ind_poles.txt")
        ind_min = int(ind_min)
        ind_max = int(ind_max)

        xyzpath = simu_path + "/particles/"
        xyz_files = np.sort(os.listdir(xyzpath))
        xyz_files = fnmatch.filter(xyz_files, "emb*.h5")

        time_steps = []
        pos_top = []
        pos_bot = []
        for xyz in xyz_files:
            r = h5py.File(xyzpath + xyz, "r")["position"][:]
            pos_top.append(r[ind_max, 2])
            pos_bot.append(r[ind_min, 2])
            time_steps.append(cnt)
            cnt += 1

        pos_top = np.array(pos_top)
        pos_bot = np.array(pos_bot)

        final_dist = float(np.mean(pos_top[-pts_sampling:] - pos_bot[-pts_sampling:]))
        std_dist = float(np.std(pos_top[-pts_sampling:] - pos_bot[-pts_sampling:]))

        if rank == 0:
            dated_print(
                f"[Korali] [{hostname}] [GPU nº{gpu_env}] | "
                f"Yt={Yt}, kb={kb}, b1={b1}, b2={b2}, a3={a3}, a4={a4}, "
                f"force={float(Xi)}, short diameter={final_dist}, sig={sig}"
            )

        diam_vert.append(final_dist)
        std_diam_vert.append(std_dist)
        sample["Reference Evaluations"] += [final_dist]
        sample["Standard Deviation"] += [sig * final_dist]
        n_ref += 1

        if dump and rank == 0:
            os.system(f"cp -r {simu_path}trj_eq/sim{simnum}/* {out}/Yt{Yt:.2f}_fulltraj/")

        comm.Barrier()

    # Output data (uses last simu_path/simnum from the loop)
    filename_simu = simu_path + "parameter/parameters00001.yaml"
    with open(filename_simu, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    ka = data["ka"]

    filename_default = simu_path + "parameter/parameters-default" + simnum + ".yaml"
    with open(filename_default, "rb") as f:
        parameters_default = yaml.load(f, Loader=yaml.CLoader)
    radp = parameters_default["radp"]

    if rank == 0:
        with open(f"{out}/F_Delta.dat", "a") as f:
            params = np.array([Yt, ka, kb, b1, b2, a3, a4, radp])
            np.savetxt(
                f,
                np.concatenate([params, np.array(diam_vert), np.array(X)]).reshape(1, -1),
            )

    # Clean up
    if rank == 0:
        for path in list_simu_path:
            os.system(f"rm -rf {path}restart/")
            os.system(f"rm -f {path}commands.txt")
            os.system(f"rm -f {path}posq.txt")
            os.system(f"rm -f {path}run_HPC.sbatch")
            os.system(f"rm -f {path}ind_poles.txt")
            if not dump:
                os.system(f"rm -rf {path}trj_eq/")
                os.system(f"rm -rf {path}anchor/")
                os.system(f"rm -rf {path}force/")
                os.system(f"rm -rf {path}mesh/")
                os.system(f"rm -rf {path}microbubble/")
                os.system(f"rm -rf {path}particles/")
                os.system(f"rm -rf {path}pin/")
                os.system(f"rm -rf {path}stats/")


def Delta_Reissner_1pole(ka, kb, F, R0):
    return R0 * F / (8.0 * np.sqrt(ka * kb))
