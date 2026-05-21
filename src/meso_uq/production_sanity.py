from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from meso_uq.experiments import load_experiments
from meso_uq.postprocess import load_posterior_samples
from meso_uq.site_runtime import get_site_runtime_paths
from meso_uq.postprocess.plots import (
    plot_posterior_marginals,
    plot_propagation_summary,
    plot_validation_overlay,
)
from meso_uq.vega_workflows import (
    VALID_EXPERIMENTS,
    VALID_MODEL_FAMILIES,
    VegaWorkflowSelection,
    expand_selection_matrix,
    parse_selection,
    resolve_workflow_config_path,
    selection_key,
    selection_slug,
)

DEFAULT_PRODUCTION_SANITY_SELECTION = VegaWorkflowSelection("compression", "full-model", "production")
PRODUCTION_SMOKE_OVERRIDES = {
    "pop_size": 10,
    "max_gen": 1,
    "hbi_pop_size": 10,
    "phase3b_pop_size": 10,
    "phase3b_max_gen": 1,
    "map_n_displacements": 1,
}


def _deduplicate(selections: Iterable[VegaWorkflowSelection]) -> list[VegaWorkflowSelection]:
    ordered: list[VegaWorkflowSelection] = []
    seen: set[str] = set()
    for selection in selections:
        key = selection_key(selection)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(selection)
    return ordered


def resolve_production_sanity_selections(
    values: Iterable[str],
    *,
    all_lanes: bool = False,
) -> list[VegaWorkflowSelection]:
    explicit = [parse_selection(value) for value in values]
    if explicit:
        selections = explicit
    elif all_lanes:
        selections = expand_selection_matrix(VALID_EXPERIMENTS, VALID_MODEL_FAMILIES, ("production",))
    else:
        selections = [DEFAULT_PRODUCTION_SANITY_SELECTION]

    for selection in selections:
        if selection.profile != "production":
            raise ValueError(
                "Production sanity only supports production selections. "
                f"Got: {selection_key(selection)}"
            )
    return _deduplicate(selections)


def build_production_smoke_config(
    repo_root: Path | str,
    selection: VegaWorkflowSelection,
) -> tuple[Path, dict[str, object]]:
    repo_root = Path(repo_root).resolve()
    base_config = resolve_workflow_config_path(repo_root, selection)
    with base_config.open("rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {base_config}, got {type(config).__name__}")

    smoke_config = dict(config)
    for key, value in PRODUCTION_SMOKE_OVERRIDES.items():
        if key in smoke_config:
            smoke_config[key] = value
    smoke_config["description"] = (
        f"Production sanity smoke derived from shipped production config {base_config.name}"
    )
    return base_config, smoke_config


def write_production_smoke_config(
    repo_root: Path | str,
    selection: VegaWorkflowSelection,
    output_root: Path | str,
) -> tuple[Path, Path]:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    base_config, smoke_config = build_production_smoke_config(repo_root, selection)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{selection_slug(selection)}.yaml"
    config_path.write_text(yaml.safe_dump(smoke_config, sort_keys=False), encoding="utf-8")
    return base_config, config_path


def load_korali_build_state(
    repo_root: Path | str,
    *,
    site: str | None = None,
    runtime_root: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    repo_root = Path(repo_root).resolve()
    try:
        paths = get_site_runtime_paths(repo_root, site=site, runtime_root=runtime_root, env=env)
    except Exception as exc:
        return {
            "status": "unknown",
            "build_options_path": None,
            "reason": f"site runtime root unavailable: {exc}",
        }
    build_options_path = paths.korali_build_dir / "meson-info" / "intro-buildoptions.json"
    if not build_options_path.exists():
        return {
            "status": "unknown",
            "build_options_path": str(build_options_path),
            "reason": "canonical Korali Meson build metadata not found",
        }

    data = json.loads(build_options_path.read_text(encoding="utf-8"))
    selected = {}
    for item in data:
        name = item.get("name")
        if name in {"buildtype", "mpi", "mpi4py", "openmp", "native_cuda_batch"}:
            selected[name] = item.get("value")

    return {
        "status": "detected",
        "build_options_path": str(build_options_path),
        "build_options": selected,
    }


def _selection_output_root(matrix_root: Path, selection: VegaWorkflowSelection) -> Path:
    return matrix_root / "runs" / selection.experiment / selection.model_family / selection.profile


def _ensure_evalkit_paths(repo_root: Path) -> None:
    for path in (
        repo_root / "emb" / "compression",
        repo_root / "emb" / "compression" / "evalkit",
        repo_root / "emb" / "indentation",
        repo_root / "emb" / "indentation" / "evalkit",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


@contextmanager
def _inference_config_environment(config_path: Path):
    previous = os.environ.get("HUQ_INFERENCE_CONFIG")
    os.environ["HUQ_INFERENCE_CONFIG"] = str(config_path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HUQ_INFERENCE_CONFIG", None)
        else:
            os.environ["HUQ_INFERENCE_CONFIG"] = previous


def _parameter_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name not in {"logLikelihood", "logPrior", "logPosterior"}]


def _evaluate_map_surrogate_prediction(
    repo_root: Path,
    *,
    experiment: str,
    config_path: Path,
    diameter_um: float,
    map_csv: Path,
    reference_points: list[float],
) -> pd.DataFrame:
    _ensure_evalkit_paths(repo_root)
    map_df = pd.read_csv(map_csv)
    if len(map_df) != 1:
        raise ValueError(f"Expected a single MAP row in {map_csv}, got {len(map_df)}")
    params = [float(map_df.iloc[0][column]) for column in _parameter_columns(map_df)]
    sample = {"Parameters": params}

    with _inference_config_environment(config_path):
        if experiment == "compression":
            from emb.compression.evalkit.posterior_compression import compute_compression_surrogate

            compute_compression_surrogate(sample, reference_points, diameter_um)
        elif experiment == "indentation":
            from emb.indentation.evalkit.posterior_indentation import compute_indentation_surrogate

            compute_indentation_surrogate(sample, reference_points, diameter_um)
        else:
            raise ValueError(f"Unsupported experiment '{experiment}'")

    return pd.DataFrame(
        {
            "x": reference_points,
            "map_surrogate": sample["Reference Evaluations"],
        }
    )


def _render_korali_plot_bundle(run_dir: Path, output_root: Path) -> dict[str, str]:
    samples_df = load_posterior_samples(run_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    samples_csv = output_root / "samples.csv"
    plot_path = output_root / "korali_posterior_marginals.png"
    samples_df.to_csv(samples_csv, index=False)
    plot_posterior_marginals(str(samples_csv), str(plot_path))
    return {
        "samples_csv": str(samples_csv),
        "korali_plot": str(plot_path),
    }


def render_production_sanity_plots(
    repo_root: Path | str,
    *,
    selections: Iterable[VegaWorkflowSelection],
    sanity_configs: dict[str, str],
    matrix_root: Path | str,
    output_root: Path | str,
) -> dict[str, object]:
    repo_root = Path(repo_root).resolve()
    matrix_root = Path(matrix_root).resolve()
    output_root = Path(output_root).resolve()
    plots_root = output_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, object] = {}
    for selection in selections:
        selection_name = selection_key(selection)
        config_path = Path(sanity_configs[selection_name]).resolve()
        selection_output_root = _selection_output_root(matrix_root, selection)
        selection_plot_root = plots_root / selection_slug(selection)

        with config_path.open("rb") as handle:
            config = yaml.load(handle, Loader=yaml.CLoader)
        experiments = [exp for exp in load_experiments(config, repo_root) if exp.enabled and exp.name == selection.experiment]

        selection_payload: dict[str, object] = {
            "phase2": _render_korali_plot_bundle(
                selection_output_root / "results_phase_2",
                selection_plot_root / "phase2",
            ),
            "datasets": {},
        }

        for exp in experiments:
            for diameter_um in exp.diameters:
                dataset = exp.dataset_name(diameter_um)
                dataset_payload: dict[str, str] = {}
                dataset_payload["phase1"] = _render_korali_plot_bundle(
                    selection_output_root / "results_phase_1" / dataset,
                    selection_plot_root / "phase1" / dataset,
                )
                dataset_payload["phase3b"] = _render_korali_plot_bundle(
                    selection_output_root / "results_phase_3b" / dataset,
                    selection_plot_root / "phase3b" / dataset,
                )

                reference_csv = selection_output_root / "references" / f"{dataset}.csv"
                propagation_csv = selection_output_root / "propagation_phase3b" / dataset / "summary.csv"
                propagation_plot = selection_plot_root / "propagation_phase3b" / f"{dataset}.png"
                plot_propagation_summary(
                    str(propagation_csv),
                    str(propagation_plot),
                    reference_csv=str(reference_csv),
                )
                dataset_payload["propagation_phase3b"] = {
                    "summary_csv": str(propagation_csv),
                    "reference_csv": str(reference_csv),
                    "plot": str(propagation_plot),
                }

                map_csv = selection_output_root / "map_phase3b" / f"{dataset}.csv"
                map_prediction_csv = selection_plot_root / "map_phase3b" / "predictions" / f"{dataset}.csv"
                map_prediction_csv.parent.mkdir(parents=True, exist_ok=True)
                map_prediction = _evaluate_map_surrogate_prediction(
                    repo_root,
                    experiment=selection.experiment,
                    config_path=config_path,
                    diameter_um=diameter_um,
                    map_csv=map_csv,
                    reference_points=exp.get_reference_points(diameter_um),
                )
                map_prediction.to_csv(map_prediction_csv, index=False)
                map_plot = selection_plot_root / "map_phase3b" / f"{dataset}.png"
                plot_validation_overlay(
                    str(reference_csv),
                    str(map_prediction_csv),
                    str(map_plot),
                    x_col="x",
                    y_pred_col="map_surrogate",
                    label_pred="map surrogate",
                )
                dataset_payload["map_phase3b"] = {
                    "map_csv": str(map_csv),
                    "prediction_csv": str(map_prediction_csv),
                    "reference_csv": str(reference_csv),
                    "plot": str(map_plot),
                }

                selection_payload["datasets"][dataset] = dataset_payload

        rendered[selection_name] = selection_payload

    return rendered
