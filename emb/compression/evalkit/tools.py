#!/usr/bin/env python3

import os
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

DATA_PREFIX = "compression_data_"
RAW_DATA_BY_DIAMETER = {
    2.1: "data_1.csv",
    2.9: "data_2.csv",
    3.0: "data_3.csv",
}


def datedPrint(msg: str) -> None:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} - {msg}")
    sys.stdout.flush()


def _find_project_root() -> str:
    cwd = os.getcwd()
    project_root = None
    for possible_root in [
        cwd,
        os.path.dirname(cwd),
        os.path.dirname(os.path.dirname(cwd)),
        os.path.dirname(os.path.dirname(os.path.dirname(cwd))),
    ]:
        if os.path.exists(os.path.join(possible_root, "emb", "compression", "src")):
            project_root = possible_root
            break
    if project_root is None:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        file_based_root = os.path.dirname(os.path.dirname(os.path.dirname(file_dir)))
        if os.path.exists(os.path.join(file_based_root, "emb", "compression", "src")):
            project_root = file_based_root
    if project_root is None:
        raise RuntimeError(f"Could not find project root (emb/compression/src) from {cwd}")
    return project_root


def _resolve_compression_paths(
    diameter_um: float,
    project_root: str,
    data_dir: Optional[str],
    data_prefix: str,
    data_file: Optional[str],
    init_path: Optional[str],
) -> tuple[str, str, str]:
    if diameter_um not in RAW_DATA_BY_DIAMETER:
        raise ValueError(f"No data file mapped for diameter {diameter_um} μm")

    data_root = Path(data_dir or os.path.join(project_root, "emb", "compression", "evalkit", "data"))
    if not data_root.is_absolute():
        data_root = Path(project_root) / data_root

    expected_dpd = data_root / f"{data_prefix}{diameter_um}um.dat"
    if data_file is None:
        resolved_data_file = expected_dpd
    else:
        resolved_data_file = Path(data_file)
        if not resolved_data_file.is_absolute():
            resolved_data_file = Path(project_root) / resolved_data_file

    explicit_csv = data_file is not None and resolved_data_file.suffix.lower() == ".csv"
    raw_data_file = resolved_data_file if explicit_csv else data_root / RAW_DATA_BY_DIAMETER[diameter_um]
    if explicit_csv and not raw_data_file.exists():
        raise FileNotFoundError(f"Compression raw reference data not found: {raw_data_file}")
    if not explicit_csv and not raw_data_file.exists():
        fallback = Path(project_root) / "emb" / "compression" / "evalkit" / "data" / RAW_DATA_BY_DIAMETER[diameter_um]
        raw_data_file = fallback
    if not raw_data_file.exists():
        raise FileNotFoundError(f"Compression raw reference data not found: {raw_data_file}")

    target_data_file = expected_dpd if resolved_data_file.suffix.lower() == ".csv" else resolved_data_file

    if init_path is None:
        init_root = Path(project_root) / f"_init_compression_{diameter_um}um"
    else:
        init_root = Path(init_path).expanduser()
        if not init_root.is_absolute():
            init_root = Path(project_root) / init_root
    return str(raw_data_file), str(target_data_file), str(init_root.resolve()) + "/"


def _write_yaml_atomic(path: str, payload: dict) -> None:
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.dump(payload, f)
    os.replace(tmp, target)


def _load_source_module(module_name: str, source_compression_path: str):
    module_path = Path(source_compression_path) / f"{module_name}.py"
    unique_name = f"_mesouq_compression_{module_name}_{abs(hash(str(module_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load compression source module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepareCompression(
    diameter_um: float,
    data_dir: Optional[str] = None,
    data_prefix: str = DATA_PREFIX,
    data_file: Optional[str] = None,
    init_path: Optional[str] = None,
) -> str:
    project_root = _find_project_root()
    raw_data_file, target_data_file, init_compression_path = _resolve_compression_paths(
        diameter_um,
        project_root,
        data_dir=data_dir,
        data_prefix=data_prefix,
        data_file=data_file,
        init_path=init_path,
    )
    source_compression_path = os.path.join(project_root, "emb", "compression", "src") + "/"

    generate_module = _load_source_module("generate", source_compression_path)
    parameters_module = _load_source_module("parameters", source_compression_path)

    os.makedirs(init_compression_path, exist_ok=True)
    os.makedirs(f"{source_compression_path}logs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    generate_module.generate_sim(
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
    parameters_module.write_parameters(
        source_path=init_compression_path,
        simu_path=init_compression_path,
        simnum="00001",
    )

    param_file = f"{init_compression_path}parameter/parameters-default00001.yaml"
    with open(param_file, "r") as f:
        params = yaml.load(f, Loader=yaml.CLoader)
    if params is None:
        raise ValueError(f"Compression parameter template is empty: {param_file}")
    ul = params.get("ul", 1e-7)
    radius_physical = (diameter_um / 2.0) * 1e-6
    radius_dpd = radius_physical / ul
    params["radp"] = radius_dpd
    diameter_dpd = 2 * radius_dpd
    params["Lx"] = float(np.ceil(diameter_dpd + 5.0))
    params["Ly"] = float(np.ceil(diameter_dpd + 5.0))
    params["Lz"] = float(np.ceil(diameter_dpd + 10.0))
    _write_yaml_atomic(param_file, params)

    if os.path.exists(target_data_file):
        datedPrint(f"[Setup] Compression data ready for {diameter_um} um")
    else:
        reference_root = os.path.join(init_compression_path, "reference_data")
        os.makedirs(reference_root, exist_ok=True)
        interpolated_file = os.path.join(
            reference_root,
            Path(raw_data_file).with_suffix("").name + "_interp.dat",
        )
        generateCompressionData(raw_data_file, init_compression_path, output_file=interpolated_file)
        convertToDPDUnits(
            interpolated_file,
            init_compression_path,
            diameter_um,
            output_file=target_data_file,
        )
    return init_compression_path


def generateCompressionData(file: str, init_compression_path: str, output_file: Optional[str] = None) -> None:
    data_exp = np.loadtxt(file, skiprows=3, delimiter=",", comments="#").reshape(-1, 2)
    if output_file is None:
        output_file = file[:-4] + "_interp.dat"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_file, np.column_stack((data_exp[3:, 0], data_exp[3:, 1])), header="Force [nN]  Displacement [nm]")


def convertToDPDUnits(
    file: str,
    init_compression_path: str,
    diameter_um: float,
    output_file: Optional[str] = None,
) -> None:
    filename_default = init_compression_path + "parameter/parameters-default00001.yaml"
    with open(filename_default, "rb") as f:
        parameters_default = yaml.load(f, Loader=yaml.CLoader)
    if parameters_default is None:
        raise ValueError(f"Compression parameter template is empty: {filename_default}")
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
    if output_file is None:
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(here, "data")
        output_file = os.path.join(data_dir, f"compression_data_{diameter_um}um.dat")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
