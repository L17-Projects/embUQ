import pytest

from meso_uq.parameters import PARAM_6, PARAM_8, create_custom_parameter_set, get_parameter_set


def test_get_parameter_set_returns_expected_sizes():
    assert get_parameter_set(6).n_params == 6
    assert get_parameter_set(8).n_params == 8
    assert PARAM_6.param_names == ["Yt", "kb", "b1", "b2", "a3", "a4"]
    assert PARAM_8.param_names[-2:] == ["d0", "sigma"]


def test_validate_values_reports_missing_and_out_of_bounds():
    params = get_parameter_set(8)
    valid, errors = params.validate_values({
        "Yt": 2e7,
        "kb": 500,
        "b1": 1.0,
        "b2": 4.0,
        "a3": 0.0,
        "a4": 2.0,
        "d0": 2.0,
    })
    assert not valid
    assert any("d0" in e for e in errors)
    assert any("sigma" in e for e in errors)


def test_create_custom_parameter_set_applies_custom_bounds():
    custom = create_custom_parameter_set("tight", ["Yt", "kb"], bounds={"Yt": (2e7, 6e7)})
    assert custom.get_bounds("Yt") == (2e7, 6e7)
    assert custom.get_bounds("kb") == get_parameter_set(8).get_bounds("kb")


def test_unknown_parameter_set_raises():
    with pytest.raises(ValueError):
        get_parameter_set(7)
