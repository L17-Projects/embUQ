from pathlib import Path

import json

from meso_uq.production_sanity import (
    PRODUCTION_SMOKE_OVERRIDES,
    build_production_smoke_config,
    load_korali_build_state,
    resolve_production_sanity_selections,
)


def test_production_sanity_defaults_to_single_compression_full_lane() -> None:
    selections = resolve_production_sanity_selections([])

    assert [selection.experiment for selection in selections] == ["compression"]
    assert [selection.model_family for selection in selections] == ["full-model"]
    assert [selection.profile for selection in selections] == ["production"]


def test_production_sanity_all_lanes_expands_both_experiments_and_model_families() -> None:
    selections = resolve_production_sanity_selections([], all_lanes=True)

    assert len(selections) == 4
    assert {selection.experiment for selection in selections} == {"compression", "indentation"}
    assert {selection.model_family for selection in selections} == {"full-model", "reduced-model"}
    assert {selection.profile for selection in selections} == {"production"}


def test_build_production_smoke_config_only_lowers_cost_knobs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    selection = resolve_production_sanity_selections(["indentation:reduced-model:production"])[0]

    base_config, smoke_config = build_production_smoke_config(repo_root, selection)

    assert base_config.name == "reduced_config_indentation.yaml"
    for key, value in PRODUCTION_SMOKE_OVERRIDES.items():
        assert smoke_config[key] == value
    assert smoke_config["fixed_params"] == {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}
    assert smoke_config["experiment"] == "indentation"


def test_load_korali_build_state_reports_known_build_flags(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    build_info = repo_root / "_vega" / "korali" / "build" / "meson-info"
    build_info.mkdir(parents=True)
    (build_info / "intro-buildoptions.json").write_text(
        json.dumps(
            [
                {"name": "buildtype", "value": "release"},
                {"name": "mpi", "value": True},
                {"name": "mpi4py", "value": True},
                {"name": "openmp", "value": False},
                {"name": "native_cuda_batch", "value": False},
                {"name": "other", "value": "ignored"},
            ]
        ),
        encoding="utf-8",
    )

    state = load_korali_build_state(repo_root)

    assert state["status"] == "detected"
    assert state["build_options"] == {
        "buildtype": "release",
        "mpi": True,
        "mpi4py": True,
        "openmp": False,
        "native_cuda_batch": False,
    }
