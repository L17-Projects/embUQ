#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import korali
import numpy as np
import yaml
from mpi4py import MPI

from indentation.evalkit.tools import dated_print
from indentation.src.equil import run_equil
from indentation.src.parameters import write_parameters

_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_SURROGATE_CACHE: Dict[Tuple[str, float], Any] = {}
_SURROGATE_PATH_ADDED = False
_DUMP_FLAG: bool | None = None


@lru_cache(maxsize=1)
def _resolve_project_root() -> str:
    cwd = os.getcwd()
    for possible_root in [cwd, os.path.dirname(cwd), os.path.dirname(os.path.dirname(cwd))]:
        if os.path.exists(os.path.join(possible_root, "indentation", "src")):
            return possible_root
    raise RuntimeError(f"Could not find project root (indentation/src) from {cwd}")


def _resolve_config_path(project_root: str) -> Path:
    override = os.getenv("HUQ_INFERENCE_CONFIG") or os.getenv("CONFIG_PATH")
    if override:
        candidate = Path(override)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = Path(project_root, override)
        if candidate.exists():
            return candidate
    for path in [
        Path(project_root, "inference/configs/production/inference_config_indentation.yaml"),
        Path("../../inference/configs/production/inference_config_indentation.yaml"),
        Path("inference/configs/production/inference_config_indentation.yaml"),
    ]:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find inference_config_indentation.yaml")


def _load_config(project_root: str) -> Dict[str, Any]:
    config_path = _resolve_config_path(project_root)
    key = str(config_path)
    if key not in _CONFIG_CACHE:
        with open(config_path, "rb") as f:
            _CONFIG_CACHE[key] = yaml.load(f, Loader=yaml.CLoader)
    return _CONFIG_CACHE[key]


def _build_surrogate(project_root: str, diameter_um: float) -> Any:
    global _SURROGATE_PATH_ADDED
    if not _SURROGATE_PATH_ADDED:
        sys.path.insert(0, os.path.join(project_root, "indentation", "surrogate"))
        _SURROGATE_PATH_ADDED = True
    surrogate_path = os.path.join(project_root, f"indentation/surrogate/diameters/{diameter_um}um/trained")
    from evaluate import Surrogate
    return Surrogate(surrogate_path)


def _get_surrogate(project_root: str, diameter_um: float) -> Any:
    key = (project_root, diameter_um)
    if key not in _SURROGATE_CACHE:
        _SURROGATE_CACHE[key] = _build_surrogate(project_root, diameter_um)
    return _SURROGATE_CACHE[key]


def preload_indentation_surrogate(diameter_um: float) -> None:
    _get_surrogate(_resolve_project_root(), diameter_um)


def _get_dump_flag() -> bool:
    global _DUMP_FLAG
    if _DUMP_FLAG is None:
        _DUMP_FLAG = _load_config(_resolve_project_root()).get("dump", False)
    return _DUMP_FLAG


def compute_indentation_surrogate(sample: Dict[str, Any], forces: List[float], diameter_um: float) -> None:
    project_root = _resolve_project_root()
    dump = _get_dump_flag()
    params = sample["Parameters"]
    if len(params) == 8:
        Yt, kb, b1, b2, a3, a4, d0, sigma = params
    elif len(params) == 7:
        Yt, kb, b1, b2, a3, a4, sigma = params
        d0 = 0.0
    else:
        raise ValueError(f"Expected 7 or 8 parameters, got {len(params)}")
    surrogate = _get_surrogate(project_root, diameter_um)
    displacements = surrogate.evaluate_indentation(x=[Yt, kb, b1, b2, a3, a4], forces=forces)
    displacements = np.maximum(0.0, np.asarray(displacements) + d0).tolist()
    try:
        comm = korali.getWorkerMPIComm()
    except Exception:
        comm = MPI.COMM_WORLD
    if dump and comm.Get_rank() == 0:
        print(f"[Korali] Indentation surrogate [D={diameter_um}um] | Yt={Yt:.0f}, kb={kb:.0f}, b1={b1:.2f}, b2={b2:.2f}, a3={a3:.2f}, a4={a4:.2f}, d0={d0:.4f}")
    sample["Reference Evaluations"] = displacements
    sample["Standard Deviation"] = (sigma * np.asarray(displacements)).tolist()


def adjust_simu_params(sample_param, filename_1_simu, filename_2_simu):
    for fname in [filename_1_simu, filename_2_simu]:
        with open(fname, 'r') as file:
            parameters = yaml.load(file, Loader=yaml.CLoader)
        for p in sample_param:
            parameters[p] = float(sample_param[p])
        with open(fname, 'w') as file:
            yaml.dump(parameters, file)


def prepare_simulation_parameters(source_indentation_path: str, init_indentation_path: str, simu_path: str, simnum: str, displacement: float, theta: List[float], diameter_um: float) -> None:
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
    adjust_simu_params({"disp": displacement, "Yt": Yt, "Yl": Yt, "b1": b1, "b2": b2, "a3": a3, "a4": a4}, filename_1_simu, filename_2_simu)
    write_parameters(source_path=init_indentation_path, simu_path=simu_path, simnum=simnum)
    adjust_simu_params({"kb": kb, "b1": b1, "b2": b2, "a3": a3, "a4": a4}, filename_3_simu, filename_4_simu)


def compute_indentation(sample, X, *, project_root=None, diameter_um=None, init_indentation_path=None):
    raise NotImplementedError("Direct Mirheo indentation is intentionally deferred in this public import slice; use compute_indentation_surrogate for the current workflow path.")


def Delta_Reissner_1pole(ka, kb, F, R0):
    return (R0 * F / (8.0 * np.sqrt(ka * kb)))
