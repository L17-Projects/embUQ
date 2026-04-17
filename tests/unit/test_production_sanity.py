import json
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

from meso_uq.production_sanity import (
    PRODUCTION_SMOKE_OVERRIDES,
    _inference_config_environment,
    _parameter_columns,
    _selection_output_root,
    build_production_smoke_config,
    load_korali_build_state,
    render_production_sanity_plots,
    resolve_production_sanity_selections,
    selection_key,
    write_production_smoke_config,
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


def test_write_production_smoke_config_writes_yaml(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "out"
    selection = resolve_production_sanity_selections(["compression:full-model:production"])[0]

    base_config, config_path = write_production_smoke_config(repo_root, selection, output_root)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert base_config.exists()
    assert config_path.exists()
    for key, value in PRODUCTION_SMOKE_OVERRIDES.items():
        assert payload[key] == value


def test_load_korali_build_state_missing_reports_unknown(tmp_path: Path) -> None:
    state = load_korali_build_state(tmp_path / "missing_repo")

    assert state["status"] == "unknown"
    assert "build_options_path" in state
    assert "reason" in state


def test_inference_config_environment_restores_previous_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = "old-value"
    monkeypatch.setenv("HUQ_INFERENCE_CONFIG", previous)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    with _inference_config_environment(config_path):
        assert os.environ["HUQ_INFERENCE_CONFIG"] == str(config_path)

    assert os.environ["HUQ_INFERENCE_CONFIG"] == previous


def test_parameter_columns_excludes_log_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "Yt": 1.0,
                "kb": 2.0,
                "logLikelihood": -1.0,
                "logPrior": -2.0,
                "logPosterior": -3.0,
            }
        ]
    )
    columns = _parameter_columns(frame)
    assert columns == ["Yt", "kb"]


def test_selection_output_root_layout() -> None:
    matrix_root = Path("/tmp/matrix")
    selection = resolve_production_sanity_selections(["compression:full-model:production"])[0]
    output = _selection_output_root(matrix_root, selection)
    assert output == matrix_root / "runs" / "compression" / "full-model" / "production"


def test_render_production_sanity_plots_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import meso_uq.production_sanity as ps

    selection = resolve_production_sanity_selections(["compression:full-model:production"])[0]
    selection_name = selection_key(selection)
    matrix_root = tmp_path / "matrix"
    output_root = tmp_path / "out"
    selection_output_root = _selection_output_root(matrix_root, selection)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiments: []\n", encoding="utf-8")

    dataset = "compression_2.1um"
    map_csv = selection_output_root / "map_phase3b" / f"{dataset}.csv"
    map_csv.parent.mkdir(parents=True, exist_ok=True)
    map_csv.write_text("Yt,kb,d0,sigma\n1.0,2.0,0.1,0.01\n", encoding="utf-8")

    ref_csv = selection_output_root / "references" / f"{dataset}.csv"
    ref_csv.parent.mkdir(parents=True, exist_ok=True)
    ref_csv.write_text("x,y\n0.0,0.0\n1.0,1.0\n", encoding="utf-8")

    prop_csv = selection_output_root / "propagation_phase3b" / dataset / "summary.csv"
    prop_csv.parent.mkdir(parents=True, exist_ok=True)
    prop_csv.write_text("x,mean\n0.0,0.0\n1.0,1.0\n", encoding="utf-8")

    class _FakeExperiment:
        name = "compression"
        enabled = True
        diameters = [2.1]

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"compression_{diameter_um}um"

        @staticmethod
        def get_reference_points(diameter_um: float) -> list[float]:
            return [0.0, 1.0]

    monkeypatch.setattr(ps, "load_experiments", lambda cfg, root: [_FakeExperiment()])
    monkeypatch.setattr(
        ps,
        "_render_korali_plot_bundle",
        lambda run_dir, out: {
            "samples_csv": str(out / "samples.csv"),
            "korali_plot": str(out / "plot.png"),
        },
    )
    monkeypatch.setattr(
        ps,
        "_evaluate_map_surrogate_prediction",
        lambda *args, **kwargs: pd.DataFrame({"x": [0.0, 1.0], "map_surrogate": [0.0, 1.0]}),
    )

    def _write_plot(*args, **kwargs) -> None:
        plot_path = Path(args[-1])
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_text("plot", encoding="utf-8")

    monkeypatch.setattr(ps, "plot_propagation_summary", _write_plot)
    monkeypatch.setattr(ps, "plot_validation_overlay", _write_plot)

    rendered = render_production_sanity_plots(
        tmp_path,
        selections=[selection],
        sanity_configs={selection_name: str(config_path)},
        matrix_root=matrix_root,
        output_root=output_root,
    )

    assert selection_name in rendered
    assert dataset in rendered[selection_name]["datasets"]


def test_evaluate_map_surrogate_prediction_for_both_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import meso_uq.production_sanity as ps

    monkeypatch.setattr(ps, "_ensure_evalkit_paths", lambda repo_root: None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    map_csv = tmp_path / "map.csv"
    map_csv.write_text("Yt,kb,d0,sigma\n1.0,2.0,0.1,0.01\n", encoding="utf-8")

    comp_mod = types.ModuleType("compression.evalkit.posterior_compression")
    comp_mod.compute_compression_surrogate = lambda sample, points, diameter: sample.__setitem__(
        "Reference Evaluations", [diameter + x for x in points]
    )
    ind_mod = types.ModuleType("indentation.evalkit.posterior_indentation")
    ind_mod.compute_indentation_surrogate = lambda sample, points, diameter: sample.__setitem__(
        "Reference Evaluations", [diameter - x for x in points]
    )
    monkeypatch.setitem(sys.modules, "compression.evalkit.posterior_compression", comp_mod)
    monkeypatch.setitem(sys.modules, "indentation.evalkit.posterior_indentation", ind_mod)

    comp = ps._evaluate_map_surrogate_prediction(
        tmp_path,
        experiment="compression",
        config_path=config_path,
        diameter_um=2.1,
        map_csv=map_csv,
        reference_points=[0.0, 1.0],
    )
    ind = ps._evaluate_map_surrogate_prediction(
        tmp_path,
        experiment="indentation",
        config_path=config_path,
        diameter_um=2.1,
        map_csv=map_csv,
        reference_points=[0.0, 1.0],
    )

    assert list(comp["map_surrogate"]) == [2.1, 3.1]
    assert list(ind["map_surrogate"]) == [2.1, 1.1]
