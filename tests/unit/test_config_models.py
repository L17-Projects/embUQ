import pytest

from meso_uq.config.models import InferenceConfig, PriorBounds, create_default_inference_config


def test_prior_bounds_from_list_and_contains():
    bounds = PriorBounds.from_list([1.0, 3.0])
    assert bounds.contains(2.0)
    assert not bounds.contains(4.0)
    assert bounds.as_tuple() == (1.0, 3.0)


def test_inference_config_sorts_diameters_and_exposes_bounds():
    config = create_default_inference_config()
    assert config.emb_diameters == [2.1, 2.9, 3.0]
    assert config.phase1_burn_in == 1
    assert config.get_prior_bounds("Yt").as_list() == [10000000.0, 50000000.0]


def test_inference_config_rejects_invalid_prior_order():
    with pytest.raises(Exception):
        InferenceConfig(
            emb_diameters=[2.9],
            prior_Yt=[5.0, 1.0],
            prior_kb=[1.0, 2.0],
            prior_b1=[0.0, 1.0],
            prior_b2=[0.0, 1.0],
            prior_a3=[-1.0, 1.0],
            prior_a4=[0.0, 1.0],
            prior_d0=[0.0, 0.5],
            prior_sigma=[0.0, 1.0],
        )


def test_inference_config_defaults_phase1_burn_in_from_hbi_burn_in():
    config = InferenceConfig(
        emb_diameters=[2.9],
        phase1_burn_in=None,
        hbi_burn_in=2,
        prior_Yt=[1.0, 2.0],
        prior_kb=[1.0, 2.0],
        prior_b1=[0.0, 1.0],
        prior_b2=[0.0, 1.0],
        prior_a3=[-1.0, 1.0],
        prior_a4=[0.0, 1.0],
        prior_d0=[0.0, 0.5],
        prior_sigma=[0.0, 1.0],
    )
    assert config.phase1_burn_in == 2
