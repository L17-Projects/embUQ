from __future__ import annotations

from pathlib import Path

import pytest

from meso_uq.config.loader import load_inference_config
from meso_uq.config.models import InferenceConfig
from meso_uq.experiments import canonical_dataset_id, load_experiments
from meso_uq.structures import get_structure


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "inference" / "configs" / "fixtures"
GV_CALIBRATED_PARAMETERS = ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"]
GV_FIXTURE_NAMES = ("gv_hierarchical_subset.yaml", "gv_hierarchical_single_lane.yaml")


def _top_level_yaml_keys(path: Path) -> list[str]:
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith(" ") or line.startswith("-"):
            continue
        key, separator, _value = line.partition(":")
        if separator:
            keys.append(key)
    return keys


@pytest.mark.parametrize(
    ("fixture_name", "expected_names"),
    (
        ("gv_hierarchical_subset.yaml", ["stretching", "torsion"]),
        ("gv_hierarchical_single_lane.yaml", ["eigenmodes"]),
    ),
)
def test_gv_inference_fixture_selects_expected_subset_of_experiments(
    fixture_name: str,
    expected_names: list[str],
) -> None:
    config = load_inference_config(FIXTURE_ROOT / fixture_name)

    assert config.structure == "gv"
    assert config.structures == ["gv"]
    assert config.surrogate is not None
    assert config.surrogate["backend"] == "dnn"
    assert "predictive_mc_samples" in config.surrogate
    assert "predictive_mc_chunk_size" in config.surrogate
    assert config.experimental_gv_hbi is True
    assert config.calibrated_parameters == GV_CALIBRATED_PARAMETERS
    assert config.prior_Yt is None
    assert config.prior_d0 is None
    for parameter_name in [*GV_CALIBRATED_PARAMETERS, "sigma"]:
        assert getattr(config, f"prior_{parameter_name}") is not None
    for parameter_name in GV_CALIBRATED_PARAMETERS:
        assert getattr(config, f"hyperprior_mu_{parameter_name}") is not None
        assert getattr(config, f"hyperprior_sigma_{parameter_name}") is not None
    assert config.experiments is not None
    assert [selection.name for selection in config.experiments] == expected_names
    assert all(selection.structure == "gv" for selection in config.experiments)


@pytest.mark.parametrize("fixture_name", GV_FIXTURE_NAMES)
def test_gv_inference_fixtures_do_not_duplicate_top_level_keys(fixture_name: str) -> None:
    keys = _top_level_yaml_keys(FIXTURE_ROOT / fixture_name)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})

    assert duplicates == []


def test_gv_subset_fixture_supports_multiple_geometries_and_fixed_controls(tmp_path: Path) -> None:
    config = load_inference_config(FIXTURE_ROOT / "gv_hierarchical_subset.yaml")
    experiments = load_experiments(config.model_dump(exclude_none=True), tmp_path)

    assert [spec.name for spec in experiments] == ["stretching", "torsion"]
    for spec, expected_control in zip(
        experiments,
        ["tot_force_500_50000__bpress_-91", "theta_0_01_0_1"],
        strict=True,
    ):
        assert spec.structure == "gv"
        assert spec.geometries == ["gv_rad2_5_height12", "gv_rad2_height14_28"]
        assert spec.controls == [expected_control]
        for geometry in spec.geometries:
            assert spec.dataset_name(geometry) == canonical_dataset_id(
                "gv",
                spec.name,
                geometry,
                expected_control,
            )


def test_gv_fixture_calibrated_parameters_match_structure_contract() -> None:
    config = load_inference_config(FIXTURE_ROOT / "gv_hierarchical_single_lane.yaml")
    gv = get_structure("gv")

    assert config.calibrated_parameters == GV_CALIBRATED_PARAMETERS
    assert tuple(config.calibrated_parameters) == gv.parameter_contract.calibrated_names


@pytest.mark.parametrize(
    "bad_config",
    (
        {
            "experiments": [{"structure": "emb", "name": "stretching", "diameters": [2.1]}],
            "prior_Yt": [1.0, 2.0],
            "prior_kb": [1.0, 2.0],
            "prior_b1": [0.0, 1.0],
            "prior_b2": [0.0, 1.0],
            "prior_a3": [-1.0, 1.0],
            "prior_a4": [0.0, 1.0],
            "prior_d0": [0.0, 0.5],
            "prior_sigma": [0.0, 1.0],
        },
        {
            "experiments": [{"structure": "gv", "name": "compression", "geometries": ["gv_rad2_height14_28"]}],
            "prior_Yt": [1.0, 2.0],
            "prior_kb": [1.0, 2.0],
            "prior_b1": [0.0, 1.0],
            "prior_b2": [0.0, 1.0],
            "prior_a3": [-1.0, 1.0],
            "prior_a4": [0.0, 1.0],
            "prior_d0": [0.0, 0.5],
            "prior_sigma": [0.0, 1.0],
        },
    ),
)
def test_structure_aware_config_rejects_emb_gv_experiment_name_collisions(bad_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="does not belong to structure"):
        InferenceConfig(**bad_config)

    with pytest.raises(ValueError, match="does not belong to structure"):
        load_experiments(bad_config, REPO_ROOT)
