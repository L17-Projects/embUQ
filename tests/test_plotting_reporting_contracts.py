from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from meso_uq.plotting import FigureManifest, PlotSeries, XYPlotRequest, configure_headless_environment
from meso_uq.reporting import ReportSection, build_report_manifest


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_plotting_contracts_are_manifestable() -> None:
    request = XYPlotRequest(
        figure_id="holdout",
        title="Holdout",
        x_label="x",
        y_label="force",
        series=(PlotSeries("prediction", x=(0.0, 1.0), y=(0.1, 0.2), units="nN"),),
        output_path="reports/figures/holdout.png",
    )
    manifest = FigureManifest.from_request(request)

    assert manifest.as_dict()["artifact_class"] == "figure"
    assert manifest.as_dict()["source_request"]["series"][0]["units"] == "nN"


def test_plotting_and_reporting_imports_do_not_load_matplotlib() -> None:
    code = """
import meso_uq.plotting
import meso_uq.reporting
import sys
assert "matplotlib" not in sys.modules
"""
    result = _run_import_probe(code)
    assert result.returncode == 0, result.stderr


def test_plot_request_rejects_mismatched_series_and_absolute_output() -> None:
    with pytest.raises(ValueError, match="mismatched"):
        PlotSeries("bad", x=(0.0,), y=(1.0, 2.0))

    with pytest.raises(ValueError, match="relative"):
        XYPlotRequest(
            figure_id="bad",
            title="Bad",
            x_label="x",
            y_label="y",
            series=(PlotSeries("ok", x=(0.0,), y=(1.0,)),),
            output_path="/tmp/bad.png",
        )


def test_report_manifest_carries_run_provenance_and_sections() -> None:
    report = build_report_manifest(
        report_id="validation-summary",
        path="reports/validation-summary.md",
        run_id="run-123",
        config_digest="abc123",
        sections=(
            ReportSection(
                section_id="gpu",
                title="GPU Validation",
                artifact_ids=("gpu-matrix",),
                text="passed",
            ),
        ),
    )

    payload = report.as_dict()

    assert payload["artifact_class"] == "report"
    assert payload["run_id"] == "run-123"
    assert payload["sections"][0]["artifact_ids"] == ["gpu-matrix"]


def test_headless_environment_sets_agg_without_importing_renderer() -> None:
    env: dict[str, str] = {}

    assert configure_headless_environment(env) == "Agg"
    assert env["MPLBACKEND"] == "Agg"
