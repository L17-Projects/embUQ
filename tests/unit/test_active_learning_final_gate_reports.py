from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import (
    ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_LHS_COMPARATOR,
    ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_SCHEMA_VERSION,
    build_active_learning_final_gate_report,
    validate_plot_path,
    write_active_learning_final_gate_artifacts,
)


def _build_round_payload(
    round_index: int,
    *,
    al_median: float,
    lhs_median: float,
    al_curve_count: int = 30,
    lhs_curve_count: int = 90,
    model_selection_notes: str = "rerun",
) -> dict[str, object]:
    base_curve = [0.0, 1.0, 2.0, 3.0]
    return {
        "round": round_index,
        "al_curve_count": al_curve_count,
        "lhs_curve_count": lhs_curve_count,
        "acquisition_scores": [0.25 * i for i in range(4)],
        "selected_candidate_scores": [0.3 * i for i in range(4)],
        "yt_kb_coverage": [
            [0.1 * round_index, 3.4],
            [0.2 * round_index, 3.1],
        ],
        "model_selection": {
            "rerun": True,
            "architecture": "microbubble_displacement_BEST",
            "backend": "dnn",
            "notes": model_selection_notes,
        },
        "failure_counts": {"runtime": 0},
        "quarantine_counts": {"quota": 0},
        "al": {
            "median_curve_rel_l2_pct": al_median,
            "mean_curve_rel_l2_pct": al_median + 1.0,
            "max_curve_rel_l2_pct": al_median + 2.5,
            "residuals": [0.1, 0.2, 0.3],
            "predicted_curves": [base_curve, [1.0, 2.0, 3.0, 4.0]],
            "reference_curves": [base_curve, [0.0, 0.8, 1.9, 2.7]],
        },
        "lhs": {
            "median_curve_rel_l2_pct": lhs_median,
            "mean_curve_rel_l2_pct": lhs_median + 1.0,
            "max_curve_rel_l2_pct": lhs_median + 2.5,
            "residuals": [0.2, 0.3, 0.4],
            "predicted_curves": [base_curve, [1.0, 2.0, 3.0, 4.0]],
            "reference_curves": [base_curve, [0.0, 0.8, 1.9, 2.7]],
        },
    }


def _synthetic_rounds_for_success() -> tuple[dict[str, object], ...]:
    return (
        _build_round_payload(1, al_median=2.0, lhs_median=3.0),
        _build_round_payload(2, al_median=1.5, lhs_median=2.6),
        _build_round_payload(3, al_median=1.7, lhs_median=2.1),
    )


def test_build_active_learning_final_gate_report_prefers_al_over_lhs_per_round(tmp_path: Path) -> None:
    report, round_rows, copy_targets = build_active_learning_final_gate_report(
        run_id="al-vs-lhs-pass",
        iteration=1,
        rounds=_synthetic_rounds_for_success(),
        scratch_copy_destination=tmp_path / "scratch-session",
        vault_copy_destination=tmp_path / "vault-session",
        metadata={"creator": "unit-test"},
    )

    assert report["schema_version"] == ACTIVE_LEARNING_FINAL_GATE_SCHEMA_VERSION
    assert report["experiment"] == "indentation"
    assert report["diameter_um"] == "3.4"
    assert report["round_count"] == 3
    assert report["al_total_curves"] == 90
    assert report["lhs_total_curves"] == 90
    assert report["lhs_comparator"] == ACTIVE_LEARNING_FINAL_GATE_LHS_COMPARATOR
    assert report["pass_fail_criteria"]["lhs_comparator_count_is_90"]["passed"] is True
    assert report["starting_from_no_initial_data"] is True
    assert report["passed"] is True
    assert report["status"] == "passed"
    assert all(row["median_improved"] for row in round_rows)
    assert copy_targets["scratch"] == str(tmp_path / "scratch-session")
    assert copy_targets["vault"] == str(tmp_path / "vault-session")


def test_build_active_learning_final_gate_report_accepts_reused_90_curve_lhs_batch(tmp_path: Path) -> None:
    report, round_rows, _ = build_active_learning_final_gate_report(
        run_id="al-vs-lhs-one-batch",
        iteration=7,
        rounds=(
            _build_round_payload(1, al_median=2.0, lhs_median=3.0, lhs_curve_count=90),
            _build_round_payload(2, al_median=1.8, lhs_median=2.9, lhs_curve_count=90),
            _build_round_payload(3, al_median=1.5, lhs_median=2.4, lhs_curve_count=90),
        ),
        scratch_copy_destination=tmp_path / "scratch-session",
    )

    assert report["lhs_total_curves"] == 90
    assert report["pass_fail_criteria"]["lhs_comparator_count_is_90"]["passed"] is True
    assert all(row["lhs_curve_count"] == 90 for row in round_rows)


def test_build_active_learning_final_gate_report_flags_rounds_where_al_worse_than_lhs() -> None:
    report, round_rows, _ = build_active_learning_final_gate_report(
        run_id="al-vs-lhs-fail",
        iteration="0003",
        rounds=(
            _build_round_payload(1, al_median=4.0, lhs_median=3.0),
            _build_round_payload(2, al_median=1.5, lhs_median=2.2),
            _build_round_payload(3, al_median=1.8, lhs_median=1.9),
        ),
    )

    assert report["passed"] is False
    assert report["status"] == "failed"
    assert any(not row["median_improved"] for row in round_rows)
    assert "al_improves_over_lhs_each_round" in report["failure_reasons"]
    assert round_rows[0]["median_improved"] is False
    assert round_rows[0]["median_delta"] > 0.0


def test_build_active_learning_final_gate_report_rejects_lhs_batch_30_per_round() -> None:
    report, _, _ = build_active_learning_final_gate_report(
        run_id="al-vs-lhs-wrong-lhs-count",
        iteration=8,
        rounds=(
            _build_round_payload(1, al_median=2.0, lhs_median=3.0, lhs_curve_count=30),
            _build_round_payload(2, al_median=1.5, lhs_median=2.6, lhs_curve_count=30),
            _build_round_payload(3, al_median=1.3, lhs_median=2.1, lhs_curve_count=30),
        ),
    )

    assert report["passed"] is False
    assert "lhs_comparator_count_is_90" in report["failure_reasons"]


def test_write_active_learning_final_gate_artifacts_generates_report_payload_and_plots(tmp_path: Path) -> None:
    artifacts = write_active_learning_final_gate_artifacts(
        output_root=tmp_path / "active_learning_final_gate",
        run_id="run-artifacts",
        iteration=2,
        rounds=_synthetic_rounds_for_success(),
        include_plots=True,
        scratch_copy_destination=tmp_path / "scratch-session",
        vault_copy_destination=tmp_path / "vault-session",
    )

    assert artifacts.report_path == (
        tmp_path
        / "active_learning_final_gate"
        / "run-artifacts"
        / "iterations"
        / "iter_0002"
        / ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME
    )
    assert artifacts.markdown_path == (
        tmp_path
        / "active_learning_final_gate"
        / "run-artifacts"
        / "iterations"
        / "iter_0002"
        / ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME
    )
    assert artifacts.summary_csv_path == (
        tmp_path
        / "active_learning_final_gate"
        / "run-artifacts"
        / "iterations"
        / "iter_0002"
        / ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME
    )

    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-artifacts"
    assert payload["iteration"] == "iter_0002"
    assert payload["pass_fail_criteria"]["al_improves_over_lhs_each_round"]["passed"] is True
    assert payload["copy_targets"]["scratch"] == str(tmp_path / "scratch-session")
    assert payload["copy_targets"]["vault"] == str(tmp_path / "vault-session")

    assert artifacts.summary_rows and len(artifacts.summary_rows) == 3

    assert artifacts.summary_rows[0]["round"] == 1
    assert artifacts.summary_rows[0]["al_curve_count"] == 30
    assert artifacts.summary_rows[1]["model_selection_rerun"] is True

    # plot files are written as non-empty PNGs in fallback mode
    for path in payload["plot_paths"].values():
        validate_plot_path(path)
        assert Path(path).is_file()

    with artifacts.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert "al_median_curve_rel_l2_pct" in rows[0]
    assert rows[0]["round"] == "1"
    assert float(rows[0]["al_curve_count"]) == 30.0
    assert rows[0]["lhs_curve_count"] == "90"
