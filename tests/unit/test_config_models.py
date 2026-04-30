import pytest

from meso_uq.config.models import (
    ExperimentSelection,
    InferenceConfig,
    PriorBounds,
    create_default_inference_config,
    emb_geometry_id,
    infer_structure,
)


def _gv_prior_kwargs() -> dict[str, list[float]]:
    return {
        "prior_ka": [0.1, 1.1],
        "prior_kb": [0.2, 1.2],
        "prior_mu": [0.3, 1.3],
        "prior_b1": [0.0, 1.0],
        "prior_b2": [0.0, 1.0],
        "prior_a3": [-1.0, 1.0],
        "prior_a4": [0.0, 1.0],
        "prior_mu_l": [0.8, 1.8],
        "prior_c": [0.9, 1.9],
        "prior_sigma": [0.0, 1.0],
    }


def test_prior_bounds_from_list_and_contains():
    bounds = PriorBounds.from_list([1.0, 3.0])
    assert bounds.contains(2.0)
    assert not bounds.contains(4.0)
    assert bounds.as_tuple() == (1.0, 3.0)


def test_inference_config_sorts_diameters_and_exposes_bounds():
    config = create_default_inference_config()
    assert config.emb_diameters == [2.1, 2.9, 3.0]
    assert config.geometries == [emb_geometry_id(2.1), emb_geometry_id(2.9), emb_geometry_id(3.0)]
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


def test_inference_config_rejects_invalid_prior_shape():
    with pytest.raises(Exception):
        InferenceConfig(
            emb_diameters=[2.9],
            prior_Yt=[5.0],
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


def test_infer_structure_rejects_unknown_explicit_structure():
    with pytest.raises(ValueError, match="Unsupported structure"):
        infer_structure("compression", "vesicle")


def test_experiment_selection_normalizes_controls_and_geometries():
    selection = ExperimentSelection(
        structure="emb",
        name="compression",
        geometries=[emb_geometry_id(2.9)],
        controls=["default", "default", "alternate"],
        diameters=[2.1],
    )

    assert selection.geometries == [emb_geometry_id(2.1), emb_geometry_id(2.9)]
    assert selection.controls == ["alternate", "default"]


def test_experiment_selection_rejects_invalid_structure_and_diameters():
    with pytest.raises(ValueError, match="Unsupported structure"):
        ExperimentSelection(structure="vesicle", name="stretching")
    with pytest.raises(ValueError, match="All diameters must be positive"):
        ExperimentSelection(structure="emb", name="compression", diameters=[-2.1])


def test_experiment_selection_rejects_structureless_non_emb_name():
    with pytest.raises(ValueError, match="requires an explicit structure"):
        ExperimentSelection(name="stretching")


def test_inference_config_normalizes_legacy_emb_experiment_selection():
    config = InferenceConfig(
        emb_diameters=[2.9, 2.1],
        experiment="compression",
        prior_Yt=[1.0, 2.0],
        prior_kb=[1.0, 2.0],
        prior_b1=[0.0, 1.0],
        prior_b2=[0.0, 1.0],
        prior_a3=[-1.0, 1.0],
        prior_a4=[0.0, 1.0],
        prior_d0=[0.0, 0.5],
        prior_sigma=[0.0, 1.0],
    )
    assert config.structure == "emb"
    assert config.structures == ["emb"]
    assert config.experiments is not None
    assert config.experiments[0].structure == "emb"
    assert config.experiments[0].name == "compression"
    assert config.experiments[0].geometries == [emb_geometry_id(2.1), emb_geometry_id(2.9)]


def test_inference_config_supports_gv_without_emb_diameters():
    config = InferenceConfig(
        structure="gv",
        structures=["gv", "gv"],
        experiments=[
            {
                "structure": "gv",
                "name": "stretching",
                "geometries": ["gv_rad2_height14_28"],
            }
        ],
        **_gv_prior_kwargs(),
    )

    assert config.emb_diameters is None
    assert config.geometries is None
    assert config.structures == ["gv"]
    assert config.prior_Yt is None
    assert config.prior_d0 is None
    assert config.get_prior_bounds("ka").as_list() == [0.1, 1.1]


def test_inference_config_merges_scalar_structure_into_structures():
    config = InferenceConfig(
        structure="emb",
        structures=["gv"],
        experiment="compression",
        emb_diameters=[2.1],
        prior_Yt=[1.0, 2.0],
        prior_kb=[1.0, 2.0],
        prior_b1=[0.0, 1.0],
        prior_b2=[0.0, 1.0],
        prior_a3=[-1.0, 1.0],
        prior_a4=[0.0, 1.0],
        prior_d0=[0.0, 0.5],
        prior_sigma=[0.0, 1.0],
        prior_ka=[0.1, 1.1],
        prior_mu=[0.3, 1.3],
        prior_mu_l=[0.8, 1.8],
        prior_c=[0.9, 1.9],
    )

    assert config.structures == ["emb", "gv"]


def test_inference_config_rejects_missing_gv_prior_bounds():
    with pytest.raises(ValueError, match="Missing required GV Phase 1 prior bounds"):
        InferenceConfig(
            structure="gv",
            structures=["gv"],
            experiments=[
                {
                    "structure": "gv",
                    "name": "torsion",
                    "geometries": ["gv_rad2_height14_28"],
                }
            ],
            prior_kb=[0.2, 1.2],
            prior_b1=[0.0, 1.0],
            prior_b2=[0.0, 1.0],
            prior_a3=[-1.0, 1.0],
            prior_a4=[0.0, 1.0],
            prior_sigma=[0.0, 1.0],
        )


def test_inference_config_rejects_unknown_structures_entry():
    with pytest.raises(ValueError, match="Unsupported structures"):
        InferenceConfig(
            structures=["emb", "vesicle"],
            prior_Yt=[1.0, 2.0],
            prior_kb=[1.0, 2.0],
            prior_b1=[0.0, 1.0],
            prior_b2=[0.0, 1.0],
            prior_a3=[-1.0, 1.0],
            prior_a4=[0.0, 1.0],
            prior_d0=[0.0, 0.5],
            prior_sigma=[0.0, 1.0],
        )


def test_inference_config_rejects_duplicate_experiment_selection():
    with pytest.raises(ValueError, match="Duplicate experiment selection"):
        InferenceConfig(
            experiments=[
                {"structure": "emb", "name": "compression", "diameters": [2.1]},
                {"structure": "emb", "name": "compression", "diameters": [2.9]},
            ],
            prior_Yt=[1.0, 2.0],
            prior_kb=[1.0, 2.0],
            prior_b1=[0.0, 1.0],
            prior_b2=[0.0, 1.0],
            prior_a3=[-1.0, 1.0],
            prior_a4=[0.0, 1.0],
            prior_d0=[0.0, 0.5],
            prior_sigma=[0.0, 1.0],
        )


def test_inference_config_rejects_ambiguous_experiment_without_structure():
    with pytest.raises(ValueError, match="requires an explicit structure"):
        InferenceConfig(
            experiment="stretching",
            prior_Yt=[1.0, 2.0],
            prior_kb=[1.0, 2.0],
            prior_b1=[0.0, 1.0],
            prior_b2=[0.0, 1.0],
            prior_a3=[-1.0, 1.0],
            prior_a4=[0.0, 1.0],
            prior_d0=[0.0, 0.5],
            prior_sigma=[0.0, 1.0],
        )
