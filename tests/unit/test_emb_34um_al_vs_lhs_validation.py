from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_al_vs_lhs_validation import (  # noqa: E402
    EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_SCHEMA_VERSION,
    build_emb_34um_al_vs_lhs_validation_report,
    write_emb_34um_al_vs_lhs_validation_artifacts,
)
from meso_uq.active_learning.final_gate_reports import validate_plot_path  # noqa: E402


def _curve_row(
    *,
    strategy: str,
    curve_id: str,
    rel_l2_pct: float,
    round_index: int | None = None,
    order: int = 0,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "curve_id": curve_id,
        "round": round_index,
        "order": order,
        "predicted_curve": [1.0 + rel_l2_pct / 100.0, 0.0],
        "reference_curve": [1.0, 0.0],
    }


def _validation_rows() -> list[dict[str, object]]:
    return [
        _curve_row(strategy="al", curve_id="al-r01-c001", round_index=1, order=1, rel_l2_pct=1.0),
        _curve_row(strategy="al", curve_id="al-r01-c002", round_index=1, order=2, rel_l2_pct=2.0),
        _curve_row(strategy="al", curve_id="al-r02-c001", round_index=2, order=3, rel_l2_pct=1.5),
        _curve_row(strategy="al", curve_id="al-r02-c002", round_index=2, order=4, rel_l2_pct=1.0),
        _curve_row(strategy="lhs", curve_id="lhs-c001", order=1, rel_l2_pct=4.0),
        _curve_row(strategy="lhs", curve_id="lhs-c002", order=2, rel_l2_pct=5.0),
        _curve_row(strategy="lhs", curve_id="lhs-c003", order=3, rel_l2_pct=6.0),
        _curve_row(strategy="lhs", curve_id="lhs-c004", order=4, rel_l2_pct=7.0),
    ]


def test_emb_34um_al_vs_lhs_validation_computes_round_prefix_medians() -> None:
    manifest, rows = build_emb_34um_al_vs_lhs_validation_report(curve_rows=_validation_rows())

    assert manifest["schema_version"] == EMB_34UM_AL_VS_LHS_VALIDATION_SCHEMA_VERSION
    assert manifest["primary_metric"] == "median_curve_rel_l2_pct"
    assert manifest["status"] == "ready"
    assert manifest["adaptive_acquisition"]["available"] is False
    assert "Adaptive acquisition engine was not supplied" in " ".join(manifest["limitations"])
    assert manifest["prefix_curve_counts"] == [2, 4]

    assert len(rows) == 2
    assert rows[0]["al_round_prefix"] == 1
    assert rows[0]["prefix_curve_count"] == 2
    assert math.isclose(rows[0]["al_median_curve_rel_l2_pct"], 1.5)
    assert math.isclose(rows[0]["lhs_median_curve_rel_l2_pct"], 4.5)
    assert rows[0]["median_improved"] is True

    assert rows[1]["al_round_prefix"] == 2
    assert rows[1]["prefix_curve_count"] == 4
    assert math.isclose(rows[1]["al_median_curve_rel_l2_pct"], 1.25)
    assert math.isclose(rows[1]["lhs_median_curve_rel_l2_pct"], 5.5)


def test_emb_34um_al_vs_lhs_validation_writes_png_json_and_csv_sidecars(tmp_path: Path) -> None:
    artifacts = write_emb_34um_al_vs_lhs_validation_artifacts(
        curve_rows=_validation_rows(),
        output_root=tmp_path / "validation",
        include_plot=False,
    )

    assert artifacts.manifest_path == tmp_path / "validation" / EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME
    assert artifacts.manifest_path.is_file()
    assert artifacts.summary_csv_path.is_file()
    assert artifacts.plot_sidecar_path.is_file()
    validate_plot_path(artifacts.plot_path)

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["plot_paths"]["al_vs_lhs_validation"] == str(artifacts.plot_path)
    assert manifest["summary_rows"][0]["al_curve_count"] == 2

    with artifacts.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert csv_rows[0]["al_round_prefix"] == "1"
    assert math.isclose(float(csv_rows[1]["lhs_median_curve_rel_l2_pct"]), 5.5)


def test_emb_34um_al_vs_lhs_validation_marks_ingestion_only_rows_blocked(tmp_path: Path) -> None:
    ingestion_only = {
        "records": [
            {
                "candidate_id": "emb-34um-final-gate-full-r01-c001",
                "gate": "full_gate",
                "round": 1,
                "status": "completed",
                "reduced_observables": {"mean_output": 1.0},
            }
        ]
    }
    artifacts = write_emb_34um_al_vs_lhs_validation_artifacts(
        curve_rows=ingestion_only,
        output_root=tmp_path / "blocked",
        include_plot=False,
    )

    assert artifacts.manifest["status"] == "blocked"
    assert artifacts.manifest["usable_curve_count"] == 0
    assert artifacts.manifest["skipped_row_reasons"] == {"missing_curve_metric_inputs": 1}
    assert "No usable curve-level validation metrics were available." in artifacts.manifest["blockers"]
    validate_plot_path(artifacts.plot_path)


def test_emb_34um_al_vs_lhs_validation_stops_before_unpaired_lhs_prefixes() -> None:
    rows = [
        _curve_row(strategy="al", curve_id=f"al-r01-c{index:03d}", round_index=1, order=index, rel_l2_pct=1.0)
        for index in range(1, 3)
    ]
    rows.extend(
        _curve_row(strategy="al", curve_id=f"al-r02-c{index:03d}", round_index=2, order=index + 2, rel_l2_pct=1.0)
        for index in range(1, 3)
    )
    rows.extend(
        _curve_row(strategy="al", curve_id=f"al-r03-c{index:03d}", round_index=3, order=index + 4, rel_l2_pct=1.0)
        for index in range(1, 3)
    )
    rows.extend(
        _curve_row(strategy="lhs", curve_id=f"lhs-c{index:03d}", order=index, rel_l2_pct=3.0)
        for index in range(1, 4)
    )

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(curve_rows=rows)

    assert manifest["prefix_curve_counts"] == [2]
    assert len(summary_rows) == 1
    assert summary_rows[0]["al_round_prefix"] == 1


def test_emb_34um_al_vs_lhs_validation_skips_bad_point_axis_without_aborting() -> None:
    rows = [
        {
            "strategy": "al",
            "curve_id": "al-bad-axis",
            "round": 1,
            "force": "not-numeric",
            "predicted": 1.0,
            "reference": 1.0,
        },
        _curve_row(strategy="lhs", curve_id="lhs-c001", order=1, rel_l2_pct=3.0),
    ]

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(curve_rows=rows)

    assert summary_rows == ()
    assert manifest["status"] == "blocked"
    assert manifest["skipped_row_reasons"] == {"invalid_point_axis": 1}


def test_emb_34um_al_vs_lhs_validation_skips_bad_metric_without_aborting() -> None:
    rows = [
        {
            "strategy": "al",
            "curve_id": "al-bad-metric",
            "round": 1,
            "order": 1,
            "curve_rel_l2_pct": "nan",
        },
        _curve_row(strategy="al", curve_id="al-good", round_index=1, order=2, rel_l2_pct=2.0),
        _curve_row(strategy="lhs", curve_id="lhs-good", order=1, rel_l2_pct=5.0),
    ]

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(curve_rows=rows)

    assert manifest["status"] == "ready"
    assert manifest["skipped_row_reasons"] == {"invalid_curve_metric": 1}
    assert len(summary_rows) == 1
    assert summary_rows[0]["prefix_curve_count"] == 1
    assert math.isclose(summary_rows[0]["al_median_curve_rel_l2_pct"], 2.0)


def test_emb_34um_al_vs_lhs_validation_skips_malformed_bracketed_arrays() -> None:
    rows = [
        {
            "strategy": "al",
            "curve_id": "al-bad-array",
            "round": 1,
            "order": 1,
            "predicted_curve": "[1.0, 2.0",
            "reference_curve": "[1.0, 2.0]",
        },
        _curve_row(strategy="lhs", curve_id="lhs-good", order=1, rel_l2_pct=5.0),
    ]

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(curve_rows=rows)

    assert summary_rows == ()
    assert manifest["status"] == "blocked"
    assert manifest["skipped_row_reasons"] == {"invalid_curve_arrays": 1}


def test_emb_34um_al_vs_lhs_validation_requires_paired_explicit_prefix_counts() -> None:
    rows = [
        _curve_row(strategy="al", curve_id=f"al-c{index:03d}", round_index=1, order=index, rel_l2_pct=2.0)
        for index in range(1, 4)
    ]
    rows.append(_curve_row(strategy="lhs", curve_id="lhs-c001", order=1, rel_l2_pct=5.0))

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(
        curve_rows=rows,
        prefix_curve_counts=[1, 2, 3],
    )

    assert manifest["prefix_curve_counts"] == [1]
    assert len(summary_rows) == 1
    assert summary_rows[0]["al_curve_count"] == 1
    assert summary_rows[0]["lhs_curve_count"] == 1


def test_emb_34um_al_vs_lhs_validation_preserves_round_zero_order() -> None:
    rows = [
        _curve_row(strategy="al", curve_id="al-r00", round_index=0, order=1, rel_l2_pct=10.0),
        _curve_row(strategy="al", curve_id="al-r01", round_index=1, order=2, rel_l2_pct=1.0),
        _curve_row(strategy="lhs", curve_id="lhs-c001", order=1, rel_l2_pct=5.0),
        _curve_row(strategy="lhs", curve_id="lhs-c002", order=2, rel_l2_pct=5.0),
    ]

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(curve_rows=rows)

    assert manifest["prefix_curve_counts"] == [1, 2]
    assert summary_rows[0]["al_round_prefix"] == 0
    assert math.isclose(summary_rows[0]["al_median_curve_rel_l2_pct"], 10.0)
    assert summary_rows[0]["median_improved"] is False


def test_emb_34um_al_vs_lhs_validation_groups_round_local_point_curve_ids() -> None:
    rows: list[dict[str, object]] = []
    for round_index, predicted in ((1, 1.01), (2, 1.02)):
        rows.extend(
            [
                {
                    "strategy": "al",
                    "curve_id": "round-local-c001",
                    "round": round_index,
                    "force": force,
                    "predicted": predicted,
                    "reference": 1.0,
                }
                for force in (0.0, 1.0)
            ]
        )
    rows.extend(
        [
            _curve_row(strategy="lhs", curve_id="lhs-c001", order=1, rel_l2_pct=5.0),
            _curve_row(strategy="lhs", curve_id="lhs-c002", order=2, rel_l2_pct=5.0),
        ]
    )

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(curve_rows=rows)

    assert manifest["prefix_curve_counts"] == [1, 2]
    assert len(summary_rows) == 2
    assert summary_rows[1]["al_curve_count"] == 2
