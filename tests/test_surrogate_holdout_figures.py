from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_surrogate_holdout_l2_figure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "postprocess" / "generate_surrogate_holdout_l2_figure.py",
        "surrogate_holdout_l2_figure_test",
    )

    for modality in ("compression", "indentation"):
        for family in ("dnn", "bnn"):
            out_dir = tmp_path / modality / "2.1um" / family
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "modality": modality,
                "surrogate_family": family,
                "diameter_um": "2.1",
                "best_mean_curve_rel_l2_pct": 10.0,
                "best_median_curve_rel_l2_pct": 8.0,
                "best_max_curve_rel_l2_pct": 20.0,
            }
            (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    summary_df = module.collect_holdout_summary_rows(tmp_path, metric="median")
    assert set(summary_df["surrogate_family"]) == {"dnn", "bnn"}
    assert set(summary_df["modality"]) == {"compression", "indentation"}

    output_dir = tmp_path / "figures"
    output_png = output_dir / "holdout.png"
    output_pdf = output_dir / "holdout.pdf"
    module.plot_holdout_l2(summary_df, metric="median", output_png=output_png, output_pdf=output_pdf)
    assert output_png.exists()
    assert output_pdf.exists()


def test_generate_surrogate_holdout_l2_figure_main_and_error_paths(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "postprocess" / "generate_surrogate_holdout_l2_figure.py",
        "surrogate_holdout_l2_figure_test_main",
    )

    # Invalid files should be ignored during collection.
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "summary.json").write_text("{not-json}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        module.collect_holdout_summary_rows(tmp_path, metric="mean")

    for modality in ("compression", "indentation"):
        for family in ("dnn", "bnn"):
            out_dir = tmp_path / modality / "3.2um" / family
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "modality": modality,
                "surrogate_family": family,
                "diameter_um": "3.2",
                "best_mean_curve_rel_l2_pct": 11.0,
                "best_median_curve_rel_l2_pct": 9.0,
                "best_max_curve_rel_l2_pct": 22.0,
            }
            (out_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    output_dir = tmp_path / "figures_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_surrogate_holdout_l2_figure.py",
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--metric",
            "mean",
        ],
    )
    module.main()
    assert (output_dir / "surrogate_holdout_l2_summary.csv").exists()
    assert (output_dir / "surrogate_holdout_l2_comparison.png").exists()
    assert (output_dir / "surrogate_holdout_l2_comparison.pdf").exists()


def test_generate_surrogate_sensitivity_figure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "postprocess" / "generate_surrogate_sensitivity_figure.py",
        "surrogate_sensitivity_figure_test",
    )

    rows = []
    for axis in (0.1, 0.2, 0.3):
        for parameter in ("Yt", "kb", "b1", "b2", "a3", "a4"):
            rows.append(
                {
                    "axis": axis,
                    "parameter": parameter,
                    "index_type": "ST",
                    "value": 0.1,
                    "confidence": 0.01,
                }
            )
    df = pd.DataFrame(rows)
    for modality in ("compression", "indentation"):
        for family in ("dnn", "bnn"):
            out_dir = tmp_path / modality / family / "csv"
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / "sobol_vs_disp_2.1um.csv", index=False)

    agg = module.collect_sobol_rows(
        tmp_path,
        index_type="ST",
        parameters=["Yt", "kb", "b1", "b2", "a3", "a4"],
    )
    assert set(agg["modality"]) == {"compression", "indentation"}
    assert set(agg["surrogate_family"]) == {"dnn", "bnn"}

    output_dir = tmp_path / "figures"
    output_png = output_dir / "sensitivity.png"
    output_pdf = output_dir / "sensitivity.pdf"
    module.plot_sensitivity_panels(
        agg,
        index_type="ST",
        parameters=["Yt", "kb", "b1", "b2", "a3", "a4"],
        output_png=output_png,
        output_pdf=output_pdf,
    )
    assert output_png.exists()
    assert output_pdf.exists()


def test_generate_surrogate_sensitivity_figure_main_and_no_data_panel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "postprocess" / "generate_surrogate_sensitivity_figure.py",
        "surrogate_sensitivity_figure_test_main",
    )

    # Build one valid file and several invalid candidates to cover collection filters.
    valid_dir = tmp_path / "compression" / "bnn" / "csv"
    valid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"axis": 0.1, "parameter": "Yt", "index_type": "ST", "value": 0.2},
            {"axis": 0.2, "parameter": "Yt", "index_type": "ST", "value": 0.25},
        ]
    ).to_csv(valid_dir / "sobol_vs_disp_2.1um.csv", index=False)
    (tmp_path / "compression" / "dnn").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"foo": 1}]).to_csv(tmp_path / "compression" / "dnn" / "sobol_bad.csv", index=False)
    (tmp_path / "indentation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "indentation" / "notes.txt").write_text("skip", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        module.collect_sobol_rows(tmp_path / "nope", index_type="ST", parameters=["Yt"])

    rows = module.collect_sobol_rows(tmp_path, index_type="ST", parameters=["Yt"])
    assert set(rows["modality"]) == {"compression"}
    assert set(rows["surrogate_family"]) == {"bnn"}

    out_png = tmp_path / "partial.png"
    out_pdf = tmp_path / "partial.pdf"
    module.plot_sensitivity_panels(
        rows,
        index_type="ST",
        parameters=["Yt"],
        output_png=out_png,
        output_pdf=out_pdf,
    )
    assert out_png.exists()
    assert out_pdf.exists()

    output_dir = tmp_path / "figures_main"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_surrogate_sensitivity_figure.py",
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--index-type",
            "ST",
            "--parameters",
            "Yt",
        ],
    )
    module.main()
    assert (output_dir / "surrogate_sensitivity_summary.csv").exists()
    assert (output_dir / "surrogate_sensitivity_comparison.png").exists()
    assert (output_dir / "surrogate_sensitivity_comparison.pdf").exists()
