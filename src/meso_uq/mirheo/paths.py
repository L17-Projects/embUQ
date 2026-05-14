"""Path utilities for Mirheo equil.py simulation drivers."""
from __future__ import annotations

import py_compile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_EQUIL_PATHS: dict[str, Path] = {
    "compression": _REPO_ROOT / "emb" / "compression" / "src" / "equil.py",
    "indentation": _REPO_ROOT / "emb" / "indentation" / "src" / "equil.py",
}


def find_equil_script(experiment: str) -> Path:
    """Return the absolute path to the equil.py driver for *experiment*.

    Parameters
    ----------
    experiment:
        One of ``"compression"`` or ``"indentation"``.

    Raises
    ------
    ValueError
        If *experiment* is not a recognised experiment name.
    FileNotFoundError
        If the equil.py file is not present on disk.
    """
    if experiment not in _EQUIL_PATHS:
        raise ValueError(
            f"Unknown experiment {experiment!r}. Expected one of: {sorted(_EQUIL_PATHS)}"
        )
    path = _EQUIL_PATHS[experiment]
    if not path.exists():
        raise FileNotFoundError(f"equil.py not found for experiment {experiment!r}: {path}")
    return path


def verify_equil_drivers() -> dict[str, str]:
    """Syntax-check both equil.py drivers and return their paths.

    Uses :mod:`py_compile` to verify the files are syntactically valid Python
    without importing them (which would require Mirheo to be installed).

    Returns
    -------
    dict[str, str]
        Mapping of experiment name to absolute path string.

    Raises
    ------
    FileNotFoundError
        If either driver file is missing.
    py_compile.PyCompileError
        If either driver file has a syntax error.
    """
    result: dict[str, str] = {}
    for experiment, path in _EQUIL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"equil.py not found for experiment {experiment!r}: {path}"
            )
        py_compile.compile(str(path), doraise=True)
        result[experiment] = str(path)
    return result
