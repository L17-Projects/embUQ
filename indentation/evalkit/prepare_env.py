import os
from pathlib import Path
from typing import Optional

import numpy as np

from .tools import datedPrint as _datedPrint
from .tools import getReferenceData as _getReferenceData
from .tools import getReferencePoints as _getReferencePoints
from .tools import prepareIndentation as _prepareIndentation

_DATA_PREFIX = "indentation_data_"
_DROP_FIRST_POINTS = {3.2: 3, 3.4: 5, 5.8: 3}


def _resolve_data_file(diameter_um: float, data_dir: Optional[str], data_prefix: Optional[str]) -> Path:
    here = os.path.dirname(os.path.abspath(__file__))
    resolved_dir = data_dir or os.path.join(here, "data")
    prefix = data_prefix or _DATA_PREFIX
    return Path(resolved_dir) / f"{prefix}{diameter_um}um.dat"


def _best_split_index(data: np.ndarray, min_points: int = 5) -> int:
    if data.shape[0] <= 2 * min_points:
        return data.shape[0]
    force = data[:, 0]
    disp = data[:, 1]
    best_idx = min_points
    best_sse = None
    for idx in range(min_points, data.shape[0] - min_points):
        a1, b1 = np.polyfit(force[:idx], disp[:idx], 1)
        a2, b2 = np.polyfit(force[idx:], disp[idx:], 1)
        sse = np.sum((disp[:idx] - (a1 * force[:idx] + b1)) ** 2) + np.sum((disp[idx:] - (a2 * force[idx:] + b2)) ** 2)
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_idx = idx
    return best_idx


def _apply_filters(diameter_um: float, data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data
    dkey = round(diameter_um, 1)
    drop = _DROP_FIRST_POINTS.get(dkey)
    if drop:
        data = data[drop:]
    if dkey == 3.4:
        idx = _best_split_index(data)
        data = data[: idx + 1]
        if data.shape[0] > 2:
            data = data[:-2]
    return data


def _filter_reference_data(diameter_um: float, data_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Indentation reference data not found: {data_path}")
    raw = np.loadtxt(data_path, skiprows=1, ndmin=2)
    filtered = _apply_filters(diameter_um, raw)
    if filtered.shape[0] != raw.shape[0]:
        np.savetxt(data_path, filtered, header="Force [DPD units]  Displacement [DPD units]")
        _datedPrint(f"[Setup] Filtered indentation data for {diameter_um} um: {raw.shape[0]} -> {filtered.shape[0]}")


def getReferencePoints(diameter_um: float = None):
    if diameter_um is None:
        raise ValueError("diameter_um must be specified for indentation")
    return _getReferencePoints(diameter_um)


def getReferenceData(diameter_um: float = None):
    if diameter_um is None:
        raise ValueError("diameter_um must be specified for indentation")
    return _getReferenceData(diameter_um)


def prepareIndentation(diameter_um: float, data_dir: str = None, data_prefix: str = None, data_file: str = None):
    _prepareIndentation(diameter_um, data_dir=data_dir, data_prefix=data_prefix, data_file=data_file)
    _filter_reference_data(diameter_um, _resolve_data_file(diameter_um, data_dir, data_prefix))
