#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import yaml


def datedPrint(msg: str) -> None:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} - {msg}")
    sys.stdout.flush()


def prepareCompression(diameter_um: float) -> None:
    cwd = os.getcwd()
    project_root = None
    for possible_root in [cwd, os.path.dirname(cwd), os.path.dirname(os.path.dirname(cwd))]:
        if os.path.exists(os.path.join(possible_root, "compression", "src")):
            project_root = possible_root
            break
    if project_root is None:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        file_based_root = os.path.dirname(os.path.dirname(file_dir))
        if os.path.exists(os.path.join(file_based_root, "compression", "src")):
            project_root = file_based_root
    if project_root is None:
        raise RuntimeError(f"Could not find project root (compression/src) from {cwd}")

    diameter_to_file = {
        2.1: os.path.join(project_root, "compression/evalkit/data/data_1.csv"),
        2.9: os.path.join(project_root, "compression/evalkit/data/data_2.csv"),
        3.0: os.path.join(project_root, "compression/evalkit/data/data_3.csv"),
    }
    if diameter_um not in diameter_to_file:
        raise ValueError(f"No data file mapped for diameter {diameter_um} μm")
    data_file = diameter_to_file[diameter_um]
    source_compression_path = os.path.join(project_root, "compression", "src") + "/"
    init_compression_path = os.path.join(project_root, f"_init_compression_{diameter_um}um") + "/"

    sys.path.insert(0, source_compression_path)
    from generate import generate_sim as generate_sim_compression
    from parameters import write_parameters

    os.makedirs(init_compression_path, exist_ok=True)
    os.makedirs(f"{source_compression_path}logs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    generate_sim_compression(
        source_path=source_compression_path,
        simu_path=init_compression_path,
        par=[["buck", "10.0", "10.0", "1"]],
        obj="emb",
        forward=None,
        hysteresis=None,
        parallel=True,
        g=1,
        N=1,
        first=None,
        numJobs=1,
    )

    os.makedirs(f"{init_compression_path}microbubble", exist_ok=True)
    os.makedirs(f"{init_compression_path}mesh", exist_ok=True)
    os.system(f"cp {source_compression_path}microbubble/sphere_icosphere.py {init_compression_path}microbubble/")
    write_parameters(source_path=init_compression_path, simu_path=init_compression_path, simnum="00001")

    param_file = f"{init_compression_path}parameter/parameters-default00001.yaml"
    with open(param_file, "r") as f:
        params = yaml.load(f, Loader=yaml.CLoader)
    ul = params.get("ul", 1e-7)
    radius_physical = (diameter_um / 2.0) * 1e-6
    radius_dpd = radius_physical / ul
    params["radp"] = radius_dpd
    diameter_dpd = 2 * radius_dpd
    params["Lx"] = float(np.ceil(diameter_dpd + 5.0))
    params["Ly"] = float(np.ceil(diameter_dpd + 5.0))
    params["Lz"] = float(np.ceil(diameter_dpd + 10.0))
    with open(param_file, "w") as f:
        yaml.dump(params, f)

    generateCompressionData(data_file, init_compression_path)
    convertToDPDUnits(data_file[:-4] + "_interp.dat", init_compression_path, diameter_um)


def generateCompressionData(file: str, init_compression_path: str) -> None:
    data_exp = np.loadtxt(file, skiprows=3, delimiter=",", comments="#").reshape(-1, 2)
    np.savetxt(file[:-4] + "_interp.dat", np.column_stack((data_exp[3:, 0], data_exp[3:, 1])), header="Force [nN]  Displacement [nm]")


def convertToDPDUnits(file: str, init_compression_path: str, diameter_um: float) -> None:
    filename_default = init_compression_path + "parameter/parameters-default00001.yaml"
    with open(filename_default, "rb") as f:
        parameters_default = yaml.load(f, Loader=yaml.CLoader)
    rho_water = parameters_default["rho_water"]
    rhow = parameters_default["rhow"]
    energyFactor = parameters_default["energyFactor"]
    kbol = parameters_default["kbol"]
    t0 = parameters_default["t0"]
    ul = parameters_default["ul"]
    ue = energyFactor * kbol * t0
    um = rho_water * ul**3 / rhow
    ut = np.sqrt(um * ul**2 / ue)
    fscale = parameters_default["fscale"]
    data = np.loadtxt(file, skiprows=1, comments="#", ndmin=2)
    data *= 1e-9
    data[:, 1] = data[:, 1] / ul
    data[:, 0] = data[:, 0] / (um * ul / ut**2)
    data[:, 0] *= fscale
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    output_file = os.path.join(data_dir, f"compression_data_{diameter_um}um.dat")
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(output_file, data[:, [1, 0]], header="Displacement [DPD units]  Force [DPD units]")


def getReferencePoints(diameter_um: float) -> List[float]:
    here = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(here, "data", f"compression_data_{diameter_um}um.dat")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Reference data file not found: {data_file}")
    return list(np.loadtxt(data_file, skiprows=1, ndmin=2)[:, 0])


def getReferenceData(diameter_um: float) -> List[float]:
    here = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(here, "data", f"compression_data_{diameter_um}um.dat")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Reference data file not found: {data_file}")
    return list(np.loadtxt(data_file, skiprows=1, ndmin=2)[:, 1])


def convertToForceFromDPDUnits(f: float, diameter_um: float, init_dir: Optional[str] = None) -> float:
    init_compression_dir = f"_init_compression_{diameter_um}um"
    param_subpath = "parameter/parameters-default00001.yaml"
    search_dirs = [init_dir] if init_dir is not None else []
    search_dirs += [".", os.path.dirname(os.path.dirname(os.path.realpath(__file__)))]
    filename_default = None
    for base_dir in search_dirs:
        candidate = os.path.join(base_dir, init_compression_dir, param_subpath)
        if os.path.exists(candidate):
            filename_default = candidate
            break
    if filename_default is None:
        raise FileNotFoundError("Could not find compression parameter template")
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
