from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
INDENTATION_ROOT = REPO_ROOT / "emb" / "indentation"
for entry in (REPO_ROOT, SRC_ROOT, INDENTATION_ROOT):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from emb.indentation.evalkit.posterior_indentation import _validate_finite_position_array


def test_validate_finite_position_array_accepts_finite_coordinates() -> None:
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    validated = _validate_finite_position_array(
        positions,
        force_index=0,
        force_value=-0.0,
        path="particles/emb00000000.h5",
    )

    assert validated is positions


def test_validate_finite_position_array_rejects_nan_coordinates() -> None:
    positions = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0]])

    with pytest.raises(RuntimeError, match=r"force_index=0.*nan_count=1.*inf_count=0"):
        _validate_finite_position_array(
            positions,
            force_index=0,
            force_value=-0.0,
            path="particles/emb00000001.h5",
        )


def test_validate_finite_position_array_rejects_empty_coordinates() -> None:
    positions = np.empty((0, 3))

    with pytest.raises(RuntimeError, match=r"empty EMB position array.*force_index=1"):
        _validate_finite_position_array(
            positions,
            force_index=1,
            force_value=-714.2857142857143,
            path="particles/emb00000200.h5",
        )


def test_validate_finite_position_array_rejects_invalid_shape() -> None:
    positions = np.ones((3, 2))

    with pytest.raises(RuntimeError, match=r"invalid EMB position array.*shape=\(3, 2\)"):
        _validate_finite_position_array(
            positions,
            force_index=0,
            force_value=-0.0,
            path="particles/emb00000000.h5",
        )
