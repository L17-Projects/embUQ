#!/usr/bin/env python3

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

DATA_PREFIX = "indentation_data_"


def dated_print(message: str):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} - {message}")


def datedPrint(msg: str) -> None:
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp} - {msg}")


def _resolve_data_file(diameter_um: float, data_dir: Optional[str] = None, data_prefix: str = DATA_PREFIX) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    resolved_dir = data_dir or os.path.join(here, "data")
    filename = f"{data_prefix}{diameter_um}um.dat"
    return os.path.join(resolved_dir, filename)


def getReferencePoints(diameter_um: float, data_dir: Optional[str] = None, data_prefix: str = DATA_PREFIX) -> List[float]:
    data_file = _resolve_data_file(diameter_um, data_dir=data_dir, data_prefix=data_prefix)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Indentation reference data not found: {data_file}")
    return list(np.loadtxt(data_file, skiprows=1, ndmin=2)[:, 0])


def getReferenceData(diameter_um: float, data_dir: Optional[str] = None, data_prefix: str = DATA_PREFIX) -> List[float]:
    data_file = _resolve_data_file(diameter_um, data_dir=data_dir, data_prefix=data_prefix)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Indentation reference data not found: {data_file}")
    return list(np.loadtxt(data_file, skiprows=1, ndmin=2)[:, 1])


def convertToForceFromDPDUnits(f: float, diameter_um: float, init_dir: Optional[str] = None) -> float:
    init_indentation_dir = f"_init_indentation_{diameter_um}um"
    param_subpath = "parameter/parameters-default00001.yaml"
    search_dirs = []
    if init_dir is not None:
        search_dirs.append(init_dir)
    search_dirs.append(".")
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_based_root = os.path.dirname(os.path.dirname(file_dir))
    search_dirs.append(file_based_root)
    filename_default = None
    for base_dir in search_dirs:
        candidate = os.path.join(base_dir, init_indentation_dir, param_subpath)
        if os.path.exists(candidate):
            filename_default = candidate
            break
    if filename_default is None:
        raise FileNotFoundError("Could not find indentation parameter template")
    with open(filename_default, "rb") as file:
        parameters_default = yaml.load(file, Loader=yaml.CLoader)
    rho_water = parameters_default["rho_water"]
    rhow = parameters_default["rhow"]
    energyFactor = parameters_default["energyFactor"]
    kbol = parameters_default["kbol"]
    t0 = parameters_default["t0"]
    fscale = parameters_default["fscale"]
    ul = parameters_default["ul"]
    ue = energyFactor * kbol * t0
    um = rho_water * ul**3 / rhow
    ut = np.sqrt(um * ul**2 / ue)
    f = f * (um * ul / ut**2)
    f = f / fscale
    return f * 1e9


def prepareIndentation(diameter_um: float, data_dir: Optional[str] = None, data_prefix: str = DATA_PREFIX, data_file: Optional[str] = None, init_path: Optional[str] = None) -> None:
    expected_dpd = _resolve_data_file(diameter_um, data_dir=data_dir, data_prefix=data_prefix)
    resolved = data_file or expected_dpd
    if os.path.exists(resolved) and resolved.lower().endswith(".csv"):
        csv_path = resolved
        dpd_path = Path(expected_dpd)
        from indentation.evalkit.convert_reference_data import convert_reference_csv
        convert_reference_csv(input_path=Path(csv_path), diameter_um=diameter_um, output_path=dpd_path, init_path=init_path)
        resolved = expected_dpd
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Indentation reference data not found: {resolved}")
    datedPrint(f"[Setup] Indentation data ready for {diameter_um} um")
