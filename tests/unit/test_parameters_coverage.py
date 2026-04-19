"""Extra coverage for meso_uq.parameters and meso_uq.diameters (previously uncovered paths)."""
from __future__ import annotations

import pytest

from meso_uq.parameters import create_custom_parameter_set, ParameterSet
from meso_uq.diameters import get_surrogate_path, validate_diameter


# ---------------------------------------------------------------------------
# meso_uq.parameters
# ---------------------------------------------------------------------------

def test_get_info_raises_key_error_for_unknown_param():
    pset = create_custom_parameter_set("test", ["Yt", "kb"])
    with pytest.raises(KeyError, match="sigma"):
        pset.get_info("sigma")


def test_create_custom_parameter_set_raises_for_unknown_name():
    with pytest.raises(ValueError, match="Unknown parameter"):
        create_custom_parameter_set("bad", ["Yt", "nonexistent"])


# ---------------------------------------------------------------------------
# meso_uq.diameters
# ---------------------------------------------------------------------------

def test_get_surrogate_path_uses_default_project_root():
    """get_surrogate_path(None) must resolve project_root from __file__."""
    path = get_surrogate_path(2.1)
    assert "surrogate" in str(path)
    assert "2.1um" in str(path)


def test_validate_diameter_known():
    assert validate_diameter(2.1) is True


def test_validate_diameter_unknown():
    assert validate_diameter(99.9) is False
