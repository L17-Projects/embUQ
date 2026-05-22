import numpy as np
import pytest

from meso_uq.noise import (
    GeometryParameterUncertainty,
    GeometrySensitivityInputs,
    GeometryUncertaintyConfig,
    build_geometry_uncertainty_covariance,
)


def test_geometry_uncertainty_builds_jacobian_covariance_components():
    config = GeometryUncertaintyConfig(
        parameters=(
            GeometryParameterUncertainty("radius", sigma=0.02, units="micrometer", nominal=2.0),
            GeometryParameterUncertainty("height", sigma=0.05, units="micrometer", nominal=14.28),
        ),
    )
    inputs = GeometrySensitivityInputs(
        predictions=(1.0, 1.5, 2.0),
        sensitivities={
            "radius": (2.0, 2.0, 2.0),
            "height": (0.1, 0.2, 0.3),
        },
        curve_id="gv/stretching",
    )

    result = build_geometry_uncertainty_covariance(inputs, config)

    radius = (0.02**2) * np.outer([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
    height = (0.05**2) * np.outer([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert np.allclose(result.covariance_components["geometry:radius"], radius)
    assert np.allclose(result.covariance_components["geometry:height"], height)
    assert np.allclose(result.covariance.covariance, radius + height)
    assert result.variance_components["geometry_jacobian"] == pytest.approx(tuple(np.diag(radius + height)))
    assert result.summary["model"] == "geometry_jacobian_covariance"
    assert result.summary["parameter_names"] == ["radius", "height"]


def test_geometry_uncertainty_accepts_explicit_parameter_covariance():
    config = GeometryUncertaintyConfig(
        parameters=(
            GeometryParameterUncertainty("diameter_um", sigma=0.0, units="micrometer", nominal=2.1),
            GeometryParameterUncertainty("thickness_um", sigma=0.0, units="micrometer", nominal=0.2),
        ),
        covariance=((0.04, 0.01), (0.01, 0.09)),
    )
    inputs = GeometrySensitivityInputs(
        predictions=(1.0, 2.0),
        sensitivities={"diameter_um": (1.0, 0.0), "thickness_um": (0.0, 2.0)},
        curve_id="emb/compression",
    )

    result = build_geometry_uncertainty_covariance(inputs, config)

    jacobian = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    expected = jacobian @ np.asarray([[0.04, 0.01], [0.01, 0.09]]) @ jacobian.T
    assert np.allclose(result.covariance.covariance, expected)
    assert np.allclose(
        result.covariance_components["geometry_cross:diameter_um:thickness_um"],
        np.asarray([[0.0, 0.02], [0.02, 0.0]]),
    )
    reconstructed = sum(result.covariance_components[name] for name in result.summary["component_names"] if name != "geometry_jacobian")
    assert np.allclose(reconstructed, result.covariance.covariance)
    assert result.summary["active"] is True


def test_geometry_uncertainty_disabled_has_no_numerical_effect():
    config = GeometryUncertaintyConfig(enabled=False)
    inputs = GeometrySensitivityInputs(predictions=(1.0, 2.0), sensitivities={})

    result = build_geometry_uncertainty_covariance(inputs, config)

    assert result.summary["enabled"] is False
    assert result.summary["active"] is False
    assert np.allclose(result.covariance.covariance, np.zeros((2, 2)))
    assert result.standard_deviation == pytest.approx((0.0, 0.0))


def test_geometry_uncertainty_nominal_accepts_signed_finite_metadata():
    offset = GeometryParameterUncertainty("offset_um", sigma=0.01, units="micrometer", nominal=0.0)
    signed_shape = GeometryParameterUncertainty("shape_delta", sigma=0.02, units="dimensionless", nominal=-1.5)

    assert offset.nominal == pytest.approx(0.0)
    assert signed_shape.nominal == pytest.approx(-1.5)

    with pytest.raises(ValueError, match="must be finite"):
        GeometryParameterUncertainty("bad", sigma=0.01, units="micrometer", nominal=np.nan)


def test_geometry_uncertainty_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="requires at least one parameter"):
        GeometryUncertaintyConfig(enabled=True)
    with pytest.raises(ValueError, match=">= 0.0"):
        GeometryParameterUncertainty("radius", sigma=-0.1, units="micrometer")
    with pytest.raises(ValueError, match="requires units"):
        GeometryParameterUncertainty("radius", sigma=0.1, units="")
    with pytest.raises(ValueError, match="unique"):
        GeometryUncertaintyConfig(
            parameters=(
                GeometryParameterUncertainty("radius", 0.1, "micrometer"),
                GeometryParameterUncertainty("radius", 0.1, "micrometer"),
            )
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        GeometryUncertaintyConfig(
            parameters=(
                GeometryParameterUncertainty("a", 0.0, "u"),
                GeometryParameterUncertainty("b", 0.0, "u"),
            ),
            covariance=((1.0, 2.0), (2.0, 1.0)),
        )
    with pytest.raises(ValueError, match="Missing geometry sensitivity"):
        build_geometry_uncertainty_covariance(
            GeometrySensitivityInputs(predictions=(1.0,), sensitivities={}),
            GeometryUncertaintyConfig(parameters=(GeometryParameterUncertainty("radius", 0.1, "micrometer"),)),
        )
    with pytest.raises(ValueError, match="Unknown geometry sensitivity"):
        build_geometry_uncertainty_covariance(
            GeometrySensitivityInputs(predictions=(1.0,), sensitivities={"radius": (1.0,), "height": (1.0,)}),
            GeometryUncertaintyConfig(parameters=(GeometryParameterUncertainty("radius", 0.1, "micrometer"),)),
        )
    with pytest.raises(ValueError, match="does not match prediction count"):
        GeometrySensitivityInputs(predictions=(1.0, 2.0), sensitivities={"radius": (1.0,)})
