#!/usr/bin/env python3
"""
Convert indentation reference data from physical units to DPD units.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

DATA_PREFIX = "indentation_data_"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_parameters_file(diameter_um: float, init_path: str | None) -> Path:
    project_root = _project_root()
    candidates = []
    if init_path:
        init_path = os.path.expanduser(init_path)
        init_path = os.path.abspath(init_path)
        if os.path.isdir(init_path):
            candidates.append(Path(init_path) / "parameter" / "parameters-default00001.yaml")
        else:
            candidates.append(Path(init_path))
    else:
        candidates.append(project_root / f"_init_compression_{diameter_um}um" / "parameter" / "parameters-default00001.yaml")
        candidates.append(project_root / "sampling" / f"_init_compression_{diameter_um}um" / "parameter" / "parameters-default00001.yaml")
        candidates.append(project_root / "compression" / "src" / "parameters-default.emb.yaml")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find parameters file. Provide --init-path or create _init_compression_<D>um/parameter/parameters-default00001.yaml")


def _load_scaling(params_file: Path) -> Tuple[float, float, float, float]:
    with params_file.open("rb") as handle:
        params = yaml.load(handle, Loader=yaml.CLoader)
    rho_water = params["rho_water"]
    rhow = params["rhow"]
    energy_factor = params["energyFactor"]
    kbol = params["kbol"]
    t0 = params["t0"]
    ul = params["ul"]
    fscale = params.get("fscale", 1.0)
    ue = energy_factor * kbol * t0
    um = rho_water * ul**3 / rhow
    ut = math.sqrt(um * ul**2 / ue)
    return ul, um, ut, fscale


def _convert_to_dpd(data: np.ndarray, ul: float, um: float, ut: float, fscale: float) -> np.ndarray:
    data_si = data * 1e-9
    forces = data_si[:, 0] / (um * ul / ut**2)
    forces *= fscale
    displacements = data_si[:, 1] / ul
    return np.column_stack((forces, displacements))


def _load_csv(input_path: Path, input_order: str) -> np.ndarray:
    raw = np.loadtxt(input_path, delimiter=",", comments="#", ndmin=2)
    if raw.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {input_path}")
    data = raw[:, :2]
    if input_order == "displacement-force":
        data = data[:, [1, 0]]
    return data


def convert_reference_csv(input_path: Path, diameter_um: float, output_path: Path | None = None, input_order: str = "force-displacement", data_prefix: str = DATA_PREFIX, init_path: str | None = None) -> Path:
    params_file = _resolve_parameters_file(diameter_um, init_path)
    ul, um, ut, fscale = _load_scaling(params_file)
    data = _load_csv(input_path, input_order)
    converted = _convert_to_dpd(data, ul, um, ut, fscale)
    if output_path is None:
        output_path = _project_root() / "indentation" / "evalkit" / "data" / f"{data_prefix}{diameter_um}um.dat"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, converted, header="Force [DPD units]  Displacement [DPD units]")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert indentation reference data from CSV (nN, nm) to DPD units.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--diameter", type=float, required=True)
    parser.add_argument("--input-order", choices=["force-displacement", "displacement-force"], default="force-displacement")
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-prefix", default=DATA_PREFIX)
    parser.add_argument("--init-path", default=None)
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    output_path = convert_reference_csv(input_path=input_path, diameter_um=args.diameter, output_path=output_path, input_order=args.input_order, data_prefix=args.data_prefix, init_path=args.init_path)
    print(f"Saved DPD data to: {output_path}")


if __name__ == "__main__":
    main()
