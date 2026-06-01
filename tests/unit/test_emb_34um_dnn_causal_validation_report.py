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
    for replicate in range(1, 4):
        cycle0_value = 2.2 + 0.03 * replicate
        rows.append(
            _row(
                replicate=replicate,
                branch="al",
                cycle=0,
                relative_l2=cycle0_value,
                train_seconds=10.0 + replicate,
                score_seconds=2.0,
                replacement=False,
                quarantine=False,
            )
        )
        rows[-1]["finite_test_count"] = 300
        rows.append(
            _row(
                replicate=replicate,
                branch="lhs",
                cycle=0,
                relative_l2=cycle0_value,
                train_seconds=10.0 + replicate,
                score_seconds=2.0,
                replacement=False,
                quarantine=False,
            )
        )
        rows[-1]["finite_test_count"] = 300
        for cycle in range(1, 6):
            baseline = 1.85 + 0.10 * cycle + 0.04 * replicate
            rows.append(
                _row(
                    replicate=replicate,
                    branch="al",
                    cycle=cycle,
                    relative_l2=baseline if cycle == 1 else baseline - 0.24 - 0.01 * replicate,
                    train_seconds=11.0 + replicate + cycle * 0.5,
                    score_seconds=2.0 + cycle * 0.2,
                    replacement=(replicate % 2 == 0 and cycle == 3),
                    quarantine=(replicate == 3 and cycle == 4),
                )
            )
            rows[-1]["finite_test_count"] = 300
            rows[-1]["added_candidate_count"] = 100
            rows[-1]["train_curve_count"] = 200 + 100 * cycle
            rows.append(
                _row(
                    replicate=replicate,
                    branch="lhs",
                    cycle=cycle,
                    relative_l2=baseline + 0.05 + 0.01 * replicate,
                    train_seconds=12.0 + replicate + cycle * 0.4,
                    score_seconds=2.5 + cycle * 0.15,
                    replacement=False,
                    quarantine=(replicate == 2 and cycle == 2),
                )
            )
            rows[-1]["finite_test_count"] = 300
            rows[-1]["added_candidate_count"] = 100
            rows[-1]["train_curve_count"] = 200 + 100 * cycle
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
    assert report["cycle_rows"][0]["cycle"] == 0
    assert report["cycle_rows"][0]["paired_delta"] == pytest.approx(0.0)
    assert "negative values favor AL" in report["interpretation"]["paired_delta"]
    assert report["final_cycle"]["paired_delta"] < 0.0
    assert report["final_cycle"]["paired_delta_ci_upper"] < 0.0
    assert report["decision"]["criteria"]["minimum_final_mean_relative_improvement"] == pytest.approx(0.10)
    assert report["decision"]["final_cycle_mean_relative_improvement_ci"]["point_estimate"] >= 0.10
    assert report["auc_comparison"]["point_estimate"] < 0.0
    assert report["area_over_cycles_delta"]["point_estimate"] < 0.0
    assert report["active_replicate_count"] == 3
    assert report["metadata"]["ensemble_size"] == 10
    assert len(report["summary_rows"]) == 36
    assert len(cycle_rows) == 6
    assert report["runtime_replacement_summary"]["overall"]["replacement_count"] > 0
    assert report["runtime_replacement_summary"]["overall"]["quarantine_count"] > 0


def test_build_report_fails_when_final_mean_improvement_is_below_ten_percent() -> None:
    rows = _fixture_rows()
    for row in rows:
        if int(row["cycle"]) == 5 and row["branch"] == "al":
            row["relative_l2"] = float(row["relative_l2"]) + 0.22

    report, _ = build_emb_34um_dnn_causal_validation_report(
        rows=rows,
        metadata={"ensemble_size": 10},
        bootstrap_resamples=800,
        bootstrap_seed=29,
        confidence=0.95,
    )

    assert report["status"] == "failed"
    assert report["decision"]["passed"] is False
    assert report["decision"]["final_cycle_mean_relative_improvement_ci"]["point_estimate"] < 0.10
    assert any("below 10%" in blocker for blocker in report["decision"]["blockers"])


def test_build_report_cycle_delta_sign_is_al_minus_lhs() -> None:
    rows: list[dict[str, object]] = []
    for replicate in (1, 2):
        rows.append(_row(replicate=replicate, branch="al", cycle=0, relative_l2=1.0, train_seconds=1.0, score_seconds=0.5))
        rows.append(_row(replicate=replicate, branch="lhs", cycle=0, relative_l2=1.0, train_seconds=1.0, score_seconds=0.5))
        rows.append(
            _row(
                replicate=replicate,
                branch="al",
                cycle=1,
                relative_l2=0.8 - 0.05 * replicate,
                train_seconds=1.0,
                score_seconds=0.5,
            )
        )
        rows.append(
            _row(
                replicate=replicate,
                branch="lhs",
                cycle=1,
                relative_l2=1.2,
                train_seconds=1.0,
                score_seconds=0.5,
            )
        )

    report, cycle_rows = build_emb_34um_dnn_causal_validation_report(
        rows=rows,
        metadata={"ensemble_size": 10},
    )

    assert cycle_rows[0]["cycle"] == 0
    assert cycle_rows[-1]["paired_delta"] < 0.0
    assert report["final_cycle"]["paired_delta"] < 0.0
    assert report["interpretation"]["paired_delta"].startswith("paired_delta is defined as AL median")


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
    assert len(csv_rows) == 6
    assert float(csv_rows[0]["paired_delta"]) == pytest.approx(0.0)
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

    assert artifacts.artifact_sidecar_json_path.is_file()
    assert artifacts.artifact_sidecar_csv_path.is_file()
    artifact_sidecar = json.loads(artifacts.artifact_sidecar_json_path.read_text(encoding="utf-8"))
    assert artifact_sidecar["active_replicate_count"] == 3
    assert artifact_sidecar["final_cycle_delta"]["point_estimate"] < 0.0
    assert set(artifact_sidecar["plot_paths"]) == set(artifacts.plot_paths)
    with artifacts.artifact_sidecar_csv_path.open(newline="", encoding="utf-8") as handle:
        artifact_rows = list(csv.DictReader(handle))
    assert len(artifact_rows) == len(artifacts.plot_paths)
    assert {row["plot_group"] for row in artifact_rows} == set(artifacts.plot_paths)
    assert artifact_rows[0]["paired_delta_interpretation"] == "paired_delta is defined as AL median relative L2 minus LHS median relative L2; negative values favor AL"


def test_report_sidecars_capture_d4_fields_and_selection_summary(tmp_path: Path) -> None:
    rows = _fixture_rows()
    rows[2]["added_candidate_count"] = 1
    rows[2]["added_candidate_ids"] = ["c-a1"]
    rows[2]["added_ka"] = [3.0e3]
    rows[2]["added_kb"] = [4.5e3]
    rows[2]["added_radp"] = [6.68]
    rows[2]["added_shell_th"] = [3.9e-9]
    rows[2]["added_selection_sources"] = ["acquisition"]
    rows[2]["added_candidate_pool_ids"] = ["pool-1"]
    rows[2]["added_selection_status"] = ["accepted"]
    rows[2]["added_selection_payloads"] = [{"confidence": 0.98}]

    artifacts = write_emb_34um_dnn_causal_validation_artifacts(
        rows=rows,
        output_root=tmp_path / "artifacts",
        metadata={
            "ensemble_size": 10,
            "parameter_names": ["ka", "kb"],
            "test_set_source": "unseen_merged",
        },
        bootstrap_resamples=200,
        bootstrap_seed=11,
        confidence=0.95,
        include_plot=False,
    )

    report_payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert report_payload["selection_summary"]["count"] == 1
    assert report_payload["selection_projection_modes"][0]["mode"] == "log10(ka)-vs-log10(kb)"
    assert report_payload["selection_projection_modes"][1]["mode"] == "log10(ka)-vs-log10(radp)"
    assert report_payload["selection_projection_modes"][2]["mode"] == "log10(ka)-vs-log10(shell_th)"
    assert report_payload["selection_projection_modes"][3]["mode"] == "log10(kb)-vs-log10(radp)"
    assert report_payload["selection_projection_modes"][4]["mode"] == "log10(kb)-vs-log10(shell_th)"
    assert report_payload["selection_projection_modes"][5]["mode"] == "log10(radp)-vs-log10(shell_th)"
    assert report_payload["added_selection_points"][0]["ka"] == pytest.approx(3.0e3)
    assert report_payload["added_selection_points"][0]["kb"] == pytest.approx(4.5e3)
    assert report_payload["added_selection_points"][0]["added_radp"] == pytest.approx(6.68)
    assert report_payload["added_selection_points"][0]["added_shell_th"] == pytest.approx(3.9e-9)
    assert report_payload["added_selection_points"][0]["radp"] == pytest.approx(6.68)
    assert report_payload["added_selection_points"][0]["shell_th"] == pytest.approx(3.9e-9)

    artifact_sidecar = json.loads(artifacts.artifact_sidecar_json_path.read_text(encoding="utf-8"))
    assert artifact_sidecar["parameter_names"] == ["ka", "kb"]
    assert artifact_sidecar["test_set_source"] == "unseen_merged"
    assert artifact_sidecar["replicate_count"] == 3
    assert artifact_sidecar["cycle_count"] == 6
    assert artifact_sidecar["selection_summary"]["point_count"] == 1
    assert artifact_sidecar["selection_summary"]["projection_count"] == 6
    assert artifact_sidecar["selection_summary"]["source_counts"] == {"acquisition": 1}
    assert artifact_sidecar["selection_projection_modes"][1]["x_label"] == "log10(ka)"
    assert artifact_sidecar["selection_projection_modes"][1]["y_label"] == "log10(radp)"
    assert artifact_sidecar["selection_projection_modes"][2]["x_label"] == "log10(ka)"
    assert artifact_sidecar["selection_projection_modes"][2]["y_label"] == "log10(shell_th)"
    assert artifact_sidecar["selection_projection_modes"][3]["x_label"] == "log10(kb)"
    assert artifact_sidecar["selection_projection_modes"][3]["y_label"] == "log10(radp)"
    assert artifact_sidecar["selection_projection_modes"][5]["x_label"] == "log10(radp)"
    assert artifact_sidecar["selection_projection_modes"][5]["y_label"] == "log10(shell_th)"
    assert artifact_sidecar["added_selection_points"][0]["added_radp"] == pytest.approx(6.68)
    assert artifact_sidecar["added_selection_points"][0]["added_shell_th"] == pytest.approx(3.9e-9)
    assert artifact_sidecar["added_selection_points"][0]["radp"] == pytest.approx(6.68)
    assert artifact_sidecar["added_selection_points"][0]["shell_th"] == pytest.approx(3.9e-9)

    with artifacts.artifact_sidecar_csv_path.open(newline="", encoding="utf-8") as handle:
        sidecar_csv_rows = list(csv.DictReader(handle))
    assert sidecar_csv_rows
    assert any(
        row["al_replacement_count"] is not None and row["lhs_replacement_count"] is not None
        for row in sidecar_csv_rows
    )
    assert any(row["paired_delta_interpretation"] for row in sidecar_csv_rows)

    parameter_coverage_sidecar = json.loads(
        Path(artifacts.plot_sidecar_paths["parameter_coverage"]).read_text(encoding="utf-8")
    )
    assert parameter_coverage_sidecar["projection_count"] == 6
    assert [entry["mode"] for entry in parameter_coverage_sidecar["projection_modes"]] == [
        "log10(ka)-vs-log10(kb)",
        "log10(ka)-vs-log10(radp)",
        "log10(ka)-vs-log10(shell_th)",
        "log10(kb)-vs-log10(radp)",
        "log10(kb)-vs-log10(shell_th)",
        "log10(radp)-vs-log10(shell_th)",
    ]
    assert parameter_coverage_sidecar["projection_modes"][1]["x_label"] == "log10(ka)"
    assert parameter_coverage_sidecar["projection_modes"][1]["y_label"] == "log10(radp)"
    assert parameter_coverage_sidecar["projection_modes"][2]["x_label"] == "log10(ka)"
    assert parameter_coverage_sidecar["projection_modes"][2]["y_label"] == "log10(shell_th)"
    assert parameter_coverage_sidecar["projection_modes"][3]["x_label"] == "log10(kb)"
    assert parameter_coverage_sidecar["projection_modes"][3]["y_label"] == "log10(radp)"
    assert parameter_coverage_sidecar["projection_modes"][5]["x_label"] == "log10(radp)"
    assert parameter_coverage_sidecar["projection_modes"][5]["y_label"] == "log10(shell_th)"
    assert parameter_coverage_sidecar["added_selection_points"][0]["added_radp"] == pytest.approx(6.68)
    assert parameter_coverage_sidecar["added_selection_points"][0]["added_shell_th"] == pytest.approx(3.9e-9)
    assert parameter_coverage_sidecar["added_selection_points"][0]["ka"] == pytest.approx(3.0e3)
    assert parameter_coverage_sidecar["added_selection_points"][0]["kb"] == pytest.approx(4.5e3)
    assert parameter_coverage_sidecar["added_selection_points"][0]["radp"] == pytest.approx(6.68)
    assert parameter_coverage_sidecar["added_selection_points"][0]["shell_th"] == pytest.approx(3.9e-9)

    selection_summary_sidecar = json.loads(
        Path(artifacts.plot_sidecar_paths["selection_summary"]).read_text(encoding="utf-8")
    )
    assert selection_summary_sidecar["added_selection_points"][0]["added_radp"] == pytest.approx(6.68)
    assert selection_summary_sidecar["added_selection_points"][0]["added_shell_th"] == pytest.approx(3.9e-9)
    assert selection_summary_sidecar["added_selection_points"][0]["radp"] == pytest.approx(6.68)
    assert selection_summary_sidecar["added_selection_points"][0]["shell_th"] == pytest.approx(3.9e-9)


def test_build_report_rejects_cycle_zero_mismatch() -> None:
    rows = _fixture_rows()
    for row in rows:
        if row["replicate"] == 1 and row["cycle"] == 0 and row["branch"] == "lhs":
            row["relative_l2"] = float(row["relative_l2"]) + 0.15
            break
    with pytest.raises(ValueError, match="cycle_0_paired_delta_must_be_zero"):
        build_emb_34um_dnn_causal_validation_report(
            rows=rows,
            metadata={"ensemble_size": 10},
        )


def test_build_report_preserves_replacement_and_quarantine_counts() -> None:
    rows = _fixture_rows()
    for row in rows:
        if row["replicate"] == 1 and row["cycle"] == 3 and row["branch"] == "al":
            row["replacement"] = True
            row["replacement_count"] = 3
            row["quarantine"] = True
            row["quarantine_count"] = 2
            break

    report, _ = build_emb_34um_dnn_causal_validation_report(
        rows=rows,
        metadata={"ensemble_size": 10},
    )

    summary_row = next(
        row
        for row in report["summary_rows"]
        if row["replicate"] == 1 and row["cycle"] == 3 and row["branch"] == "al"
    )
    assert summary_row["replacement_count"] == 3
    assert summary_row["quarantine_count"] == 2
    assert report["runtime_replacement_summary"]["by_branch"]["al"]["replacement_count"] >= 3
    assert report["runtime_replacement_summary"]["by_branch"]["al"]["quarantine_count"] >= 2


def test_build_report_synthesizes_cycle_zero_from_initial_metric_fields() -> None:
    rows = [
        {
            **row,
            "initial_relative_l2": 2.5 + 0.01 * int(row["replicate"]),
            "initial_train_seconds": 1.2,
            "initial_score_seconds": 0.3,
            "finite_test_count": 100,
        }
        for row in _fixture_rows()
        if int(row["cycle"]) > 0
    ]

    report, cycle_rows = build_emb_34um_dnn_causal_validation_report(
        rows=rows,
        metadata={"ensemble_size": 10},
    )

    assert cycle_rows[0]["cycle"] == 0
    assert cycle_rows[0]["paired_delta"] == pytest.approx(0.0)
    assert report["paired_cycle_rows"][0]["synthetic_cycle_0"] is True


def test_build_report_blocks_when_protocol_counts_are_incomplete() -> None:
    rows = _fixture_rows()
    rows = [
        row for row in rows
        if not (row["branch"] == "lhs" and int(row["replicate"]) == 3 and int(row["cycle"]) == 5)
    ]

    report, _ = build_emb_34um_dnn_causal_validation_report(
        rows=rows,
        metadata={"ensemble_size": 10},
    )

    assert report["status"] == "blocked"
    assert report["decision"]["passed"] is False
    assert report["protocol_completeness"]["passed"] is False
    assert any("lhs missing replicate=3, cycle=5" in blocker for blocker in report["decision"]["blockers"])


def test_build_report_rejects_missing_ensemble_size() -> None:
    with pytest.raises(ValueError, match="ensemble_size=10"):
        build_emb_34um_dnn_causal_validation_report(
            rows=_fixture_rows(),
            metadata={},
        )
