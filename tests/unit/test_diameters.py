"""Unit tests for src/meso_uq/diameters.py."""

from pathlib import Path

import pytest

from meso_uq.diameters import (
    DIAMETER_REGISTRY,
    PARAM_NAMES_6,
    PARAM_NAMES_8,
    DiameterConfig,
    get_available_diameters,
    get_data_file_path,
    get_diameter_config,
    get_plotting_style,
    get_surrogate_path,
    validate_diameter,
)


def test_get_available_diameters_returns_sorted_list() -> None:
    diameters = get_available_diameters()
    assert diameters == [2.1, 2.9, 3.0]
    assert diameters == sorted(diameters)


def test_diameter_registry_has_three_entries() -> None:
    assert len(DIAMETER_REGISTRY) == 3
    assert 2.1 in DIAMETER_REGISTRY
    assert 2.9 in DIAMETER_REGISTRY
    assert 3.0 in DIAMETER_REGISTRY


def test_get_diameter_config_valid_diameters() -> None:
    for diameter in [2.1, 2.9, 3.0]:
        cfg = get_diameter_config(diameter)
        assert isinstance(cfg, DiameterConfig)
        assert cfg.value_um == diameter


def test_get_diameter_config_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not registered"):
        get_diameter_config(1.5)


def test_get_diameter_config_invalid_lists_available() -> None:
    with pytest.raises(ValueError, match="2.1"):
        get_diameter_config(99.0)


def test_validate_diameter_returns_true_for_registered() -> None:
    assert validate_diameter(2.1) is True
    assert validate_diameter(2.9) is True
    assert validate_diameter(3.0) is True


def test_validate_diameter_returns_false_for_unregistered() -> None:
    assert validate_diameter(0.0) is False
    assert validate_diameter(5.0) is False


def test_get_plotting_style_returns_required_keys() -> None:
    style = get_plotting_style(2.1)
    assert "color" in style
    assert "marker" in style
    assert "label" in style


def test_get_plotting_style_distinct_colors() -> None:
    colors = {get_plotting_style(d)["color"] for d in [2.1, 2.9, 3.0]}
    assert len(colors) == 3


def test_get_plotting_style_distinct_markers() -> None:
    markers = {get_plotting_style(d)["marker"] for d in [2.1, 2.9, 3.0]}
    assert len(markers) == 3


def test_get_plotting_style_invalid_raises() -> None:
    with pytest.raises(ValueError):
        get_plotting_style(99.0)


def test_get_data_file_path_with_explicit_root(tmp_path: Path) -> None:
    p = get_data_file_path(2.1, project_root=tmp_path)
    assert p == tmp_path / "compression" / "evalkit" / "data" / "data_1.csv"


def test_get_data_file_path_uses_expected_filenames() -> None:
    root = Path("/fake/root")
    assert get_data_file_path(2.1, project_root=root).name == "data_1.csv"
    assert get_data_file_path(2.9, project_root=root).name == "data_2.csv"
    assert get_data_file_path(3.0, project_root=root).name == "data_3.csv"


def test_get_data_file_path_invalid_diameter_raises() -> None:
    with pytest.raises(ValueError):
        get_data_file_path(99.0)


def test_get_surrogate_path_with_explicit_root(tmp_path: Path) -> None:
    p = get_surrogate_path(2.1, project_root=tmp_path)
    assert p == tmp_path / "compression" / "surrogate" / "diameters" / "2.1um" / "trained"


def test_get_surrogate_path_distinct_per_diameter(tmp_path: Path) -> None:
    paths = [get_surrogate_path(d, project_root=tmp_path) for d in [2.1, 2.9, 3.0]]
    assert len(set(str(p) for p in paths)) == 3


def test_get_surrogate_path_invalid_diameter_raises() -> None:
    with pytest.raises(ValueError):
        get_surrogate_path(0.5)


def test_diameter_config_is_frozen() -> None:
    cfg = get_diameter_config(2.1)
    with pytest.raises((AttributeError, TypeError)):
        cfg.value_um = 99.0  # type: ignore[misc]


def test_param_names_constants() -> None:
    assert PARAM_NAMES_6 == ["Yt", "kb", "b1", "b2", "a3", "a4"]
    assert PARAM_NAMES_8 == ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]
    assert len(PARAM_NAMES_6) == 6
    assert len(PARAM_NAMES_8) == 8


def test_diameter_config_label_contains_um() -> None:
    for diameter in [2.1, 2.9, 3.0]:
        cfg = get_diameter_config(diameter)
        assert "μm" in cfg.label


def test_diameter_config_default_marker() -> None:
    cfg = DiameterConfig(
        value_um=5.0,
        data_file="data_x.csv",
        surrogate_dir="5.0um",
        color="#000000",
        label="5.0 μm",
    )
    assert cfg.marker == "o"


def test_get_data_file_path_default_root_uses_package_parents() -> None:
    p = get_data_file_path(2.1)
    assert p.name == "data_1.csv"
    assert "evalkit" in str(p)
