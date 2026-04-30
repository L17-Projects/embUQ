from __future__ import annotations

import pytest

from meso_uq.structures import get_structure, list_structures


def test_gv_calibrated_parameter_order_matches_contract() -> None:
    gv = get_structure("gv")
    assert gv.parameter_contract.calibrated_names == (
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
    )


def test_gv_controls_are_excluded_from_calibrated_vector() -> None:
    gv = get_structure("gv")
    calibrated_names = set(gv.parameter_contract.calibrated_names)
    control_names = {
        control_name
        for experiment in gv.experiments
        for control_name in experiment.control_names
    }
    assert calibrated_names.isdisjoint(control_names)


def test_gv_declares_optional_d0_and_multiplicative_sigma_noise() -> None:
    gv = get_structure("gv")
    d0 = gv.parameter_contract.get_parameter("d0")
    sigma = gv.parameter_contract.get_parameter("sigma")
    assert d0.is_nuisance is True
    assert d0.optional is True
    assert sigma.is_nuisance is True
    assert gv.parameter_contract.noise_model is not None
    assert gv.parameter_contract.noise_model.kind == "multiplicative"
    assert gv.parameter_contract.noise_model.parameter == "sigma"


def test_gv_nuisance_names_and_unknown_parameter_error() -> None:
    gv = get_structure("gv")

    assert gv.parameter_contract.nuisance_names == ("d0", "sigma")
    with pytest.raises(KeyError, match="Unknown parameter"):
        gv.parameter_contract.get_parameter("not_a_parameter")


def test_structure_registry_errors_and_listing() -> None:
    assert [structure.name for structure in list_structures()] == ["gv"]
    with pytest.raises(KeyError, match="Unknown structure"):
        get_structure("emb")
