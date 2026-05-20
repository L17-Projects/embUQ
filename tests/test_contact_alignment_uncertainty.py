import numpy as np
import pytest

from meso_uq.noise import (
    ContactAlignmentInputs,
    ContactAlignmentUncertaintyConfig,
    build_contact_alignment_covariance,
    legacy_indentation_surrogate_likelihood,
)


def test_contact_alignment_covariance_exposes_named_components():
    inputs = ContactAlignmentInputs(
        controls=(0.0, 1.0, 2.0),
        predictions=(1.0, 2.0, 4.0),
        force_sensitivity=(0.5, 0.5, 0.5),
        curve_id="indentation",
    )
    config = ContactAlignmentUncertaintyConfig(
        contact_offset_sigma=0.2,
        alignment_slope_sigma=0.1,
        displacement_scale_sigma=0.05,
        force_scale_sigma=0.1,
        minimum_variance=0.01,
    )

    result = build_contact_alignment_covariance(inputs, config)

    assert result.summary["enabled"] is True
    assert result.summary["active"] is True
    assert set(result.variance_components) == {
        "contact_offset",
        "alignment_tilt",
        "displacement_scale_calibration",
        "force_scale_calibration",
        "minimum_variance_floor",
        "contact_alignment_total",
    }
    assert np.allclose(result.covariance.covariance, result.covariance.covariance.T)
    assert result.variance_components["contact_offset"] == pytest.approx((0.04, 0.04, 0.04))
    assert result.variance_components["alignment_tilt"] == pytest.approx((0.01, 0.0, 0.01))
    assert result.variance_components["displacement_scale_calibration"] == pytest.approx((0.0025, 0.01, 0.04))
    assert result.variance_components["force_scale_calibration"] == pytest.approx((0.0, 0.0025, 0.01))
    assert result.variance_components["minimum_variance_floor"] == pytest.approx((0.01, 0.01, 0.01))


def test_contact_alignment_disabled_has_no_numerical_effect_even_with_sigmas():
    inputs = ContactAlignmentInputs(controls=(0.0, 1.0), predictions=(2.0, 3.0))
    config = ContactAlignmentUncertaintyConfig(enabled=False, contact_offset_sigma=1.0, minimum_variance=1.0)

    result = build_contact_alignment_covariance(inputs, config)

    assert result.summary["enabled"] is False
    assert result.summary["active"] is False
    assert np.allclose(result.covariance.covariance, np.zeros((2, 2)))
    assert result.standard_deviation == pytest.approx((0.0, 0.0))


def test_contact_alignment_zero_enabled_config_is_named_but_inactive():
    inputs = ContactAlignmentInputs(controls=(0.0, 1.0), predictions=(2.0, 3.0))
    config = ContactAlignmentUncertaintyConfig(enabled=True)

    result = build_contact_alignment_covariance(inputs, config)

    assert result.summary["enabled"] is True
    assert result.summary["active"] is False
    assert "contact_alignment_total" in result.variance_components
    assert result.variance_components["contact_alignment_total"] == pytest.approx((0.0, 0.0))


def test_contact_alignment_preserves_legacy_indentation_d0_adjustment_boundary():
    legacy = legacy_indentation_surrogate_likelihood([1.0, 3.0], 0.1, d0=-2.0)
    inputs = ContactAlignmentInputs(
        controls=(0.0, 1.0),
        predictions=legacy.reference_evaluations,
    )

    result = build_contact_alignment_covariance(
        inputs,
        ContactAlignmentUncertaintyConfig(contact_offset_sigma=0.2),
    )

    assert legacy.reference_evaluations == pytest.approx((0.0, 1.0))
    assert result.variance_components["contact_offset"] == pytest.approx((0.04, 0.04))


def test_contact_alignment_validation_errors_are_explicit():
    with pytest.raises(ValueError, match=">= 0.0"):
        ContactAlignmentUncertaintyConfig(contact_offset_sigma=-0.1)
    with pytest.raises(ValueError, match="max_jitter"):
        ContactAlignmentUncertaintyConfig(jitter=1e-4, max_jitter=1e-6)
    with pytest.raises(ValueError, match="controls count"):
        ContactAlignmentInputs(controls=(0.0,), predictions=(1.0, 2.0))
    with pytest.raises(ValueError, match="force_sensitivity count"):
        ContactAlignmentInputs(controls=(0.0, 1.0), predictions=(1.0, 2.0), force_sensitivity=(1.0,))
    with pytest.raises(ValueError, match="requires force_sensitivity"):
        build_contact_alignment_covariance(
            ContactAlignmentInputs(controls=(0.0, 1.0), predictions=(1.0, 2.0)),
            ContactAlignmentUncertaintyConfig(force_scale_sigma=0.1),
        )
