import warnings

import numpy as np
import pytest

from meso_uq.noise import (
    CovarianceTerm,
    TotalCovarianceConfig,
    assemble_total_covariance,
    covariance_term_from_diagonal,
)


def test_total_covariance_sums_canonical_terms_once_and_preserves_child_diagnostics():
    observation = covariance_term_from_diagonal(
        "observation:additive_relative",
        (0.50, 0.60, 0.70),
        children={
            "observation:additive": np.diag([0.10, 0.10, 0.10]),
            "observation:relative": np.diag([0.40, 0.50, 0.60]),
        },
    )
    correlated = CovarianceTerm(
        "observation:correlated_curve",
        ((0.03, 0.01, 0.00), (0.01, 0.03, 0.01), (0.00, 0.01, 0.03)),
    )
    contact = CovarianceTerm(
        "measurement:contact_alignment",
        0.02 * np.ones((3, 3)),
        children={"contact_offset": 0.02 * np.ones((3, 3))},
    )
    geometry_child = np.asarray(((0.0, 0.015, 0.0), (0.015, 0.0, -0.01), (0.0, -0.01, 0.0)))
    geometry = CovarianceTerm(
        "measurement:geometry",
        ((0.04, 0.01, 0.00), (0.01, 0.05, 0.01), (0.00, 0.01, 0.06)),
        children={"geometry_cross:radius:height": geometry_child},
    )
    surrogate = CovarianceTerm("surrogate:predictive", np.diag([0.09, 0.16, 0.25]))
    discrepancy = CovarianceTerm("discrepancy:low_rank", 0.01 * np.outer([1.0, 0.5, -0.25], [1.0, 0.5, -0.25]))

    result = assemble_total_covariance(
        (observation, correlated, contact, geometry, surrogate, discrepancy),
        TotalCovarianceConfig(jitter=1e-12, max_jitter=1e-8),
    )

    expected = sum(result.covariance_components[name] for name in result.included_term_names)
    assert np.allclose(result.covariance.covariance, expected)
    assert "geometry_cross:radius:height" in result.child_covariance_components
    assert "geometry_cross:radius:height" not in result.covariance_components
    assert not np.allclose(result.covariance.covariance, expected + geometry_child)
    assert result.included_term_names == (
        "observation:additive_relative",
        "observation:correlated_curve",
        "measurement:contact_alignment",
        "measurement:geometry",
        "surrogate:predictive",
        "discrepancy:low_rank",
    )
    assert result.summary["cholesky_success"] is True
    assert result.standard_deviation == pytest.approx(tuple(np.sqrt(np.diag(expected))))


def test_total_covariance_excluded_zero_term_is_diagnostic_only():
    observation = covariance_term_from_diagonal("observation:additive_relative", (0.2, 0.3))
    disabled_discrepancy = CovarianceTerm("discrepancy:low_rank", np.zeros((2, 2)), included=False)

    result = assemble_total_covariance((observation, disabled_discrepancy))

    assert result.included_term_names == ("observation:additive_relative",)
    assert result.excluded_term_names == ("discrepancy:low_rank",)
    assert np.allclose(result.covariance.covariance, np.diag([0.2, 0.3]))
    assert result.summary["term_summaries"]["discrepancy:low_rank"]["included"] is False


def test_total_covariance_duplicate_names_aliases_and_children_fail():
    eye = np.eye(2)
    with pytest.raises(ValueError, match="Duplicate covariance term identifier"):
        assemble_total_covariance((CovarianceTerm("a", eye), CovarianceTerm("a", eye)))
    with pytest.raises(ValueError, match="Duplicate covariance term identifier"):
        assemble_total_covariance((CovarianceTerm("a", eye, aliases=("shared",)), CovarianceTerm("b", eye, aliases=("shared",))))
    with pytest.raises(ValueError, match="duplicates child component"):
        assemble_total_covariance((CovarianceTerm("a", eye, children={"b": eye}), CovarianceTerm("b", eye)))


def test_total_covariance_rejects_invalid_component_matrices():
    with pytest.raises(ValueError, match="shape"):
        assemble_total_covariance((CovarianceTerm("a", np.eye(2)), CovarianceTerm("b", np.eye(3))))
    with pytest.raises(ValueError, match="symmetric"):
        assemble_total_covariance((CovarianceTerm("a", ((1.0, 0.2), (0.0, 1.0))),))
    with pytest.raises(ValueError, match="positive semidefinite"):
        assemble_total_covariance((CovarianceTerm("a", ((1.0, 2.0), (2.0, 1.0))),))
    with pytest.raises(ValueError, match=">= 0.0"):
        covariance_term_from_diagonal("observation:additive_relative", (0.1, -0.1))


def test_total_covariance_accepts_tiny_positive_definite_scale():
    tiny = CovarianceTerm("tiny_observation", np.diag([1.0e-12, 2.0e-12]))

    result = assemble_total_covariance((tiny,), TotalCovarianceConfig(jitter=0.0, max_jitter=0.0))

    assert result.summary["active"] is True
    assert result.summary["cholesky_success"] is True
    assert result.covariance.cholesky is not None
    assert result.standard_deviation == pytest.approx((1.0e-6, np.sqrt(2.0e-12)))


def test_total_covariance_requires_final_positive_definiteness_or_jitter():
    singular = CovarianceTerm("discrepancy:low_rank", np.ones((2, 2)))

    with pytest.raises(ValueError, match="not positive definite"):
        assemble_total_covariance((singular,))

    result = assemble_total_covariance(
        (singular,),
        TotalCovarianceConfig(jitter=1e-8, max_jitter=1e-8),
    )

    assert result.summary["cholesky_success"] is True
    assert result.summary["jitter_added"] == pytest.approx(1e-8)
    assert result.summary["min_eigenvalue"] == pytest.approx(1e-8)


def test_total_covariance_rejects_derived_nonfinite_total_without_warnings():
    large_terms = tuple(CovarianceTerm(f"large_{index}", np.diag([1.0e307])) for index in range(200))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="total_covariance values must be finite"):
            assemble_total_covariance(large_terms)

    assert caught == []


def test_total_covariance_accepts_large_finite_psd_terms_without_symmetry_overflow():
    result = assemble_total_covariance(
        (CovarianceTerm("large", np.diag([1.0e308])),),
        TotalCovarianceConfig(jitter=0.0, max_jitter=0.0),
    )

    assert result.summary["cholesky_success"] is True
    assert result.summary["max_eigenvalue"] == pytest.approx(1.0e308)
