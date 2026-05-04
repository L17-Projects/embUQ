from __future__ import annotations

import pytest

import meso_uq.structures.gv as gv


def test_gv_package_lazy_numerical_generation_exports() -> None:
    assert gv.GVNumericalGenerationResult.__name__ == "GVNumericalGenerationResult"
    assert callable(gv.generate_gv_numerical_data)
    exported = dir(gv)
    assert "GVNumericalGenerationResult" in exported
    assert "generate_gv_numerical_data" in exported


def test_gv_package_rejects_unknown_lazy_export() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        gv.__getattr__("not_an_export")
