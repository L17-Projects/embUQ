from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_report import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
    EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME,
    build_emb_34um_dnn_causal_validation_report,
    write_emb_34um_dnn_causal_validation_artifacts,
)


def _row(
    *,
    replicate: int,
    branch: str,
    cycle: int,
    relative_l2: float,
    train_seconds: float,
    score_seconds: float,
    replacement: bool = False,
    quarantine: bool = False,
) -> dict[str, object]:
    return {
        "replicate": replicate,
        "branch": branch,
        "cycle": cycle,
        "relative_l2": relative_l2,
        "train_seconds": train_seconds,
        "score_seconds": score_seconds,
        "replacement": replacement,
        "quarantine": quarantine,
    }


def _fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replicate in range(1, 6):
        for cycle in range(1, 6):
            baseline = 1.85 + 0.10 * cycle + 0.04 * replicate
            rows.append(
                _row(
                    replicate=replicate,
                    branch="al",
                    cycle=cycle,
                    relative_l2=baseline - 0.24 - 0.01 * replicate,
                    train_seconds=11.0 + replicate + cycle * 0.5,
                    score_seconds=2.0 + cycle * 0.2,
                    replacement=(replicate % 2 == 0 and cycle == 3),
                    quarantine=(replicate == 5 and cycle == 4),
                )
            )
            rows.append(
                _row(
                    replicate=replicate,
                    branch="lhs",
                    cycle=cycle,
                    relative_l2=baseline + 0.05 + 0.01 * replicate,
                    train_seconds=12.0 + replicate + cycle * 0.4,
                    score_seconds=2.5 + cycle * 0.15,
                    replacement=False,
                    quarantine=(replicate == 4 and cycle == 2),
                )
            )
    return rows


def test_build_report_passes_for_synthetic_fixture() -> None:
    report, cycle_rows = build_emb_34um_dnn_causal_validation_report(
        rows=_fixture_rows(),
        metadata={"ensemble_size": 10, "scenario": "synthetic_fixture"},
        bootstrap_resamples=1500,
        bootstrap_seed=17,
        confidence=0.95,
    )

    assert report["schema_version"] == EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["decision"]["passed"] is True
    assert report["final_cycle"]["paired_delta"] < 0.0
    assert report["final_cycle"]["paired_delta_ci_upper"] < 0.0
    assert report["auc_comparison"]["point_estimate"] < 0.0
    assert report["metadata"]["ensemble_size"] == 10
    assert len(report["summary_rows"]) == 50
    assert len(cycle_rows) == 5
    assert report["runtime_replacement_summary"]["overall"]["replacement_count"] > 0
    assert report["runtime_replacement_summary"]["overall"]["quarantine_count"] > 0


def test_write_artifacts_emits_non_empty_report_csv_plots_and_sidecars(tmp_path: Path) -> None:
    artifacts = write_emb_34um_dnn_causal_validation_artifacts(
        rows=_fixture_rows(),
        output_root=tmp_path / "artifacts",
        metadata={"ensemble_size": 10, "scenario": "synthetic_fixture"},
        bootstrap_resamples=1000,
        bootstrap_seed=23,
        confidence=0.95,
        include_plot=False,
        generation_command="analyze_emb_34um_dnn_causal_validation.py --rows fixture.json",
    )

    assert artifacts.report_path == tmp_path / "artifacts" / EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME
    assert artifacts.summary_csv_path == tmp_path / "artifacts" / EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME
    assert artifacts.report_path.is_file()
    assert artifacts.summary_csv_path.is_file()

    report_payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert report_payload["schema_version"] == EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION
    assert report_payload["metadata"]["ensemble_size"] == 10
    assert report_payload["status"] == "passed"

    with artifacts.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 5
    assert float(csv_rows[-1]["paired_delta"]) < 0.0

    for plot_path in artifacts.plot_paths.values():
        path = Path(plot_path)
        assert path.is_file()
        assert path.stat().st_size > 0

    for sidecar_path in artifacts.plot_sidecar_paths.values():
        path = Path(sidecar_path)
        assert path.is_file()
        assert path.stat().st_size > 0
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["bootstrap_resamples"] == 1000
        assert payload["metadata"]["ensemble_size"] == 10
        assert payload["source_inputs"]


def test_build_report_rejects_missing_ensemble_size() -> None:
    with pytest.raises(ValueError, match="ensemble_size=10"):
        build_emb_34um_dnn_causal_validation_report(
            rows=_fixture_rows(),
            metadata={},
        )
