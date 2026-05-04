from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_group_holdout(root: Path, spec) -> None:  # noqa: ANN001
    leaf = root / spec.experiment / f"{spec.diameter}um" / "dnn"
    leaf.mkdir(parents=True, exist_ok=True)
    if spec.experiment == "compression":
        df = pd.DataFrame(
            {
                "disp": [0.0, 0.1, 0.2],
                "F_true": [1.0, 1.4, 1.8],
                "F_pred": [0.95, 1.45, 1.75],
                "pred_std": [0.03, 0.03, 0.04],
            }
        )
    else:
        df = pd.DataFrame(
            {
                "F": [0.0, 0.1, 0.2],
                "disp_true": [1.0, 1.3, 1.6],
                "disp_pred": [1.05, 1.25, 1.62],
                "pred_std": [0.02, 0.03, 0.03],
            }
        )
    df.to_csv(leaf / "representative_curve.csv", index=False)
    (leaf / "summary.json").write_text(
        json.dumps(
            {
                "modality": spec.experiment,
                "diameter_um": spec.diameter,
                "representative_curve_rel_l2_pct": 2.5,
                "best_median_curve_rel_l2_pct": 2.1,
            }
        ),
        encoding="utf-8",
    )


def _write_sobol(root: Path, spec) -> None:  # noqa: ANN001
    leaf = root / spec.experiment / "dnn"
    leaf.mkdir(parents=True, exist_ok=True)
    name = (
        f"sobol_vs_disp_{spec.diameter}um.csv"
        if spec.experiment == "compression"
        else f"sobol_vs_force_{spec.diameter}um.csv"
    )
    rows = []
    for parameter, base in zip(("Yt", "kb", "b1", "b2", "a3", "a4"), np.linspace(0.2, 0.7, 6)):
        for axis in (0.0, 0.1, 0.2):
            rows.append(
                {
                    "axis": axis,
                    "parameter": parameter,
                    "index_type": "ST",
                    "value": base + 0.05 * axis,
                    "confidence": 0.01,
                }
            )
    pd.DataFrame(rows).to_csv(leaf / name, index=False)


def _write_phase1(root: Path, spec, model_family: str) -> None:  # noqa: ANN001
    leaf = root / "results_phase_1" / spec.dataset
    leaf.mkdir(parents=True, exist_ok=True)
    names = ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "[Sigma]"]
    if model_family == "reduced-model":
        names = ["Yt", "kb", "d0", "[Sigma]"]
    samples = [[float(i + j + 1) for j in range(len(names))] for i in range(80)]
    payload = {
        "Variables": [{"Name": name} for name in names],
        "Results": {"Posterior Sample Database": samples},
    }
    (leaf / "genLatest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_map(root: Path, spec, model_family: str) -> None:  # noqa: ANN001
    leaf = root / "map_mirheo"
    result_dir = leaf / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{spec.dataset}_result.json"
    result_path.write_text(
        json.dumps(
            {
                "diameter_um": float(spec.diameter),
                "displacement_points": [0.0, 0.1, 0.2],
                "forces": [1.0, 1.2, 1.5],
                "grid_config": {"d0_offset": 0.01},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = leaf / "map_mirheo_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"model_family": model_family, "diameters": []}
    manifest["diameters"].append(
        {
            "dataset_name": spec.dataset,
            "result_json": str(result_path),
            "status": "passed",
        }
    )
    (leaf / "map_mirheo_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_generate_out_of_scope_figures_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "papers" / "huq_emb" / "generate_out_of_scope_figures.py",
        "generate_out_of_scope_figures_test",
    )

    paper_root = tmp_path / "paper_data"
    campaign_root = paper_root / "runs" / "camp1"
    workflow_runs_root = campaign_root / "workflow_matrix" / "runs"
    group_holdout_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging" / "group_holdout"
    sobol_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging" / "sobol"

    for spec in module.DIAMETERS:
        _write_group_holdout(group_holdout_root, spec)
        _write_sobol(sobol_root, spec)
        for model_family in module.MODEL_FAMILIES:
            lane = workflow_runs_root / spec.experiment / model_family / "production"
            _write_phase1(lane, spec, model_family)
            _write_map(lane, spec, model_family)

    (campaign_root / "postprocess_graph" / "dnn_figure_input_staging").mkdir(parents=True, exist_ok=True)
    (campaign_root / "postprocess_graph" / "dnn_figure_input_staging" / "dnn_figure_input_staging_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "group_holdout_root": str(group_holdout_root),
                "sobol_root": str(sobol_root),
            }
        ),
        encoding="utf-8",
    )

    rc = module.main(
        [
            "--paper-data-root",
            str(paper_root),
            "--campaign-id",
            "camp1",
            "--max-posterior-samples",
            "30",
        ]
    )
    assert rc == 0

    output_root = paper_root / "figures" / "out_of_paper_scope"
    manifest = json.loads((output_root / "out_of_scope_figures_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert {item["symbol"] for item in manifest["diameter_mapping"]} == {"d1", "d2", "d3", "d4", "d5", "d6"}
    assert (output_root / "grouped_holdout_validation" / "grouped_holdout_validation_all_diameters.png").exists()
    assert (output_root / "sobol_sensitivity" / "sobol_sensitivity_ST_all_diameters.png").exists()
    assert (
        output_root
        / "posterior_marginals_phase1"
        / "posterior_marginals_phase1_full-model_d1_compression_2.1um.png"
    ).exists()
    assert (output_root / "map_vs_simulation" / "map_vs_simulation_all_diameters.png").exists()
