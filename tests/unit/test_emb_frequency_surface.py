from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from meso_uq.inference.emb_frequency_surface import (
    DpdFrequencySurface,
    FREQUENCY_SURFACE_SCHEMA,
    fit_tensor_cubic_frequency_surface,
    leave_one_ka_out_metrics,
    leave_one_radius_out_metrics,
)


def _surface() -> DpdFrequencySurface:
    coefficients = np.zeros((4, 4), dtype=np.float64)
    coefficients[0, 0] = 4.0
    coefficients[1, 0] = 1.5
    coefficients[0, 1] = 0.8
    coefficients[1, 1] = 0.25
    return DpdFrequencySurface(
        agent="definity",
        ka_bounds_dpd=(0.0, 30000.0),
        radius_bounds_um=(1.05, 5.6),
        coefficients=coefficients,
        conditions={"fixed_kb_dpd": 389.2325564501215},
        provenance={"git_commit": "synthetic"},
    )


def test_artifact_round_trip_and_diameter_conversion(tmp_path: Path) -> None:
    surface = _surface()
    path = surface.write(tmp_path / "definity_surface.json")
    loaded = DpdFrequencySurface.load(path)

    predicted = loaded.predict_mhz(np.asarray([0.0, 30000.0]), np.asarray([1.05, 5.6]))
    via_diameter = loaded.predict_for_diameters_mhz(15000.0, np.asarray([2.1, 11.2]))

    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == FREQUENCY_SURFACE_SCHEMA
    assert loaded.agent == "definity"
    assert loaded.conditions["fixed_kb_dpd"] == pytest.approx(389.2325564501215)
    assert predicted == pytest.approx(np.sqrt([4.0, 6.55]))
    assert via_diameter == pytest.approx(np.sqrt([4.75, 5.675]))


@pytest.mark.parametrize(
    ("ka", "radius", "message"),
    [
        (-1.0, 1.05, "ka_dpd is outside"),
        (30000.1, 1.05, "ka_dpd is outside"),
        (0.0, 1.049, "radius_um is outside"),
        (0.0, 5.601, "radius_um is outside"),
    ],
)
def test_surface_rejects_support_extrapolation(ka: float, radius: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _surface().predict_mhz(ka, radius)


def test_surface_rejects_negative_frequency_squared() -> None:
    coefficients = np.zeros((4, 4), dtype=np.float64)
    coefficients[0, 0] = -1.0
    surface = DpdFrequencySurface(
        agent="sonovue",
        ka_bounds_dpd=(0.0, 30000.0),
        radius_bounds_um=(1.3, 2.9),
        coefficients=coefficients,
    )
    with pytest.raises(ValueError, match="negative"):
        surface.predict_mhz(0.0, 1.3)


def test_surface_schema_and_unit_contracts_are_strict() -> None:
    payload = _surface().to_mapping()
    payload["schema"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported frequency surface schema"):
        DpdFrequencySurface.from_mapping(payload)

    payload = _surface().to_mapping()
    payload["coordinates"]["radius_coordinate"] = "diameter"
    with pytest.raises(ValueError, match="physical_radius"):
        DpdFrequencySurface.from_mapping(payload)

    payload = _surface().to_mapping()
    payload["conditions"] = ["not", "a", "mapping"]
    with pytest.raises(ValueError, match="conditions must be a mapping"):
        DpdFrequencySurface.from_mapping(payload)


def test_tensor_fit_recovers_known_frequency_squared_surface() -> None:
    reference = _surface()
    ka_values = np.linspace(0.0, 30000.0, 8)
    radius_values = np.linspace(1.05, 5.6, 7)
    ka_grid, radius_grid = np.meshgrid(ka_values, radius_values, indexing="ij")
    frequency = reference.predict_mhz(ka_grid, radius_grid)

    fitted = fit_tensor_cubic_frequency_surface(
        agent="definity",
        ka_dpd=ka_grid,
        radius_um=radius_grid,
        frequency_mhz=frequency,
        ka_bounds_dpd=(0.0, 30000.0),
        radius_bounds_um=(1.05, 5.6),
    )

    assert fitted.coefficients == pytest.approx(reference.coefficients, abs=1.0e-11)
    assert fitted.predict_mhz(ka_grid, radius_grid) == pytest.approx(frequency, abs=1.0e-11)


def test_leave_one_radius_out_returns_one_metric_per_radius() -> None:
    reference = _surface()
    ka_values = np.linspace(0.0, 30000.0, 7)
    radius_values = np.linspace(1.05, 5.6, 10)
    ka_grid, radius_grid = np.meshgrid(ka_values, radius_values, indexing="ij")
    frequency = reference.predict_mhz(ka_grid, radius_grid)

    metrics = leave_one_radius_out_metrics(
        agent="definity",
        ka_dpd=ka_grid,
        radius_um=radius_grid,
        frequency_mhz=frequency,
        ka_bounds_dpd=(0.0, 30000.0),
        radius_bounds_um=(1.05, 5.6),
    )

    assert [metric.held_out_radius_um for metric in metrics] == pytest.approx(radius_values)
    assert all(metric.point_count == ka_values.size for metric in metrics)
    assert max(metric.rmse_mhz for metric in metrics) < 1.0e-10


def test_leave_one_ka_out_returns_one_metric_per_ka_node() -> None:
    reference = _surface()
    ka_values = np.linspace(0.0, 30000.0, 9)
    radius_values = np.linspace(1.05, 5.6, 7)
    ka_grid, radius_grid = np.meshgrid(ka_values, radius_values, indexing="ij")
    frequency = reference.predict_mhz(ka_grid, radius_grid)

    metrics = leave_one_ka_out_metrics(
        agent="definity",
        ka_dpd=ka_grid,
        radius_um=radius_grid,
        frequency_mhz=frequency,
        ka_bounds_dpd=(0.0, 30000.0),
        radius_bounds_um=(1.05, 5.6),
    )

    assert [metric.held_out_ka_dpd for metric in metrics] == pytest.approx(ka_values)
    assert all(metric.point_count == radius_values.size for metric in metrics)
    assert max(metric.rmse_mhz for metric in metrics) < 1.0e-10
