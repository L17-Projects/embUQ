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

import meso_uq.active_learning.emb_34um_causal_validation_report as causal_report_module  # noqa: E402
from meso_uq.active_learning.emb_34um_causal_validation_report import (  # noqa: E402
    EMB_34UM_CAUSAL_VALIDATION_REPORT_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
    EMB_34UM_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME,
    build_emb_34um_causal_validation_report,
    write_emb_34um_causal_validation_artifacts,
)
from meso_uq.active_learning.final_gate_reports import validate_plot_path  # noqa: E402


def _metric_row(
    *,
    strategy: str,
    seed: int,
    step: int,
    validation_set_id: str,
    metric: float,
    ka: float = 100.0,
    kb: float = 200.0,
    runtime_seconds: float | None = None,
    replacement: bool = False,
    source_id: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "strategy": strategy,
        "seed": seed,
        "step": step,
        "validation_set_id": validation_set_id,
        "candidate_id": source_id,
        "ka": ka,
        "kb": kb,
        "median_curve_rel_l2_pct": metric,
        "replaced": replacement,
    }
    if runtime_seconds is not None:
        row["runtime_seconds"] = runtime_seconds
    return row


def _paired_rows_for_seed_step(
    *,
    seed: int,
    step: int,
    validation_set_id: str,
    lhs: float,
    al: float,
    seed_rows_per_strategy: int = 1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(seed_rows_per_strategy):
        rows.append(
            _metric_row(
                strategy="al",
                seed=seed,
                step=step,
                validation_set_id=validation_set_id,
                metric=al + index * 0.2,
                source_id=f"seed-{seed:03d}-step-{step:02d}-set-{validation_set_id}-al-{index}",
            )
        )
    for index in range(seed_rows_per_strategy):
        rows.append(
            _metric_row(
                strategy="lhs",
                seed=seed,
                step=step,
                validation_set_id=validation_set_id,
                metric=lhs + index * 0.2,
                source_id=f"seed-{seed:03d}-step-{step:02d}-set-{validation_set_id}-lhs-{index}",
            )
        )
    return rows


def _seed_set_rows(
    *,
    seed: int,
    deltas: tuple[float, float, float],
    step: int = 3,
) -> list[dict[str, object]]:
    lhs_base = 10.0
    rows: list[dict[str, object]] = []
    rows.extend(_paired_rows_for_seed_step(seed=seed, step=step, validation_set_id="set-a", lhs=lhs_base, al=lhs_base + deltas[0]))
    rows.extend(_paired_rows_for_seed_step(seed=seed, step=step, validation_set_id="set-b", lhs=lhs_base + 1.0, al=lhs_base + 1.0 + deltas[1]))
    rows.extend(_paired_rows_for_seed_step(seed=seed, step=step, validation_set_id="set-c", lhs=lhs_base + 2.0, al=lhs_base + 2.0 + deltas[2]))
    return rows


def _single_set_rows(
    *,
    seed: int,
    step: int,
    validation_set_id: str,
    lhs: float,
    al: float,
) -> list[dict[str, object]]:
    return [
        _metric_row(
            strategy="al",
            seed=seed,
            step=step,
            validation_set_id=validation_set_id,
            metric=al,
            source_id=f"seed-{seed:03d}-step-{step:02d}-{validation_set_id}-al",
        ),
        _metric_row(
            strategy="lhs",
            seed=seed,
            step=step,
            validation_set_id=validation_set_id,
            metric=lhs,
            source_id=f"seed-{seed:03d}-step-{step:02d}-{validation_set_id}-lhs",
        ),
    ]


def _ingestion_curve_record(
    *,
    seed: int,
    method: str,
    step: int,
    curve_ix: int,
    source: str = "candidate",
) -> dict[str, object]:
    ka = float(100.0 + seed * 10.0 + step * 2.0 + curve_ix)
    kb = float(300.0 + seed * 20.0 + step * 3.0 + curve_ix)
    return {
        "candidate_id": f"seed-{seed:03d}-{method}-step-{step:02d}-c{curve_ix:03d}",
        "seed": seed,
        "step": step,
        "method": method,
        "status": "completed",
        "sample_source": source,
        "ka": ka,
        "kb": kb,
        "force_grid": [1.0, 2.0, 3.0],
        "reference_curve": [ka * 0.001 + kb * 0.0001 + 0.1 * f for f in (1.0, 2.0, 3.0)],
    }


def _ingestion_full_scale_rows(
    *,
    seed: int,
    step_count: int = 5,
    step_size: int = 100,
    shared_size: int = 100,
    validation_size: int = 100,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for curve_index in range(shared_size):
        rows.append(
            _ingestion_curve_record(
                seed=seed,
                method="shared_initial",
                step=0,
                curve_ix=curve_index + 1,
                source="candidate",
            )
        )
    for curve_index in range(validation_size):
        rows.append(
            _ingestion_curve_record(
                seed=seed,
                method="validation",
                step=0,
                curve_ix=curve_index + 1,
                source="candidate",
            )
        )
    for step in range(1, step_count + 1):
        for curve_index in range(step_size):
            rows.append(
                _ingestion_curve_record(
                    seed=seed,
                    method="al",
                    step=step,
                    curve_ix=curve_index + 1,
                    source="candidate",
                )
            )
            rows.append(
                _ingestion_curve_record(
                    seed=seed,
                    method="lhs",
                    step=step,
                    curve_ix=curve_index + 1,
                    source="candidate",
                )
            )
    return rows


def _fake_train_validation_metric(*, training_rows: list[dict[str, object]] | tuple[dict[str, object], ...], validation_rows: list[dict[str, object]] | tuple[dict[str, object], ...]) -> dict[str, float]:
    del validation_rows
    max_step = 0
    has_al = False
    has_lhs = False
    for row in training_rows:
        method = str(row.get("method") or "")
        step = int(row.get("step") or 0)
        if method == "al":
            has_al = True
            max_step = max(max_step, step)
        if method == "lhs":
            has_lhs = True
            max_step = max(max_step, step)
    base = 10.0
    if has_al:
        if max_step <= 1:
            median = base - 0.1
        else:
            median = base - (0.1 + 0.45 * (max_step - 1))
    elif has_lhs:
        median = base - 0.1 - 0.02 * max(0, max_step - 1)
    else:
        median = base
    return {
        "median_curve_rel_l2_pct": float(median),
        "mean_curve_rel_l2_pct": float(median + 0.1),
        "max_curve_rel_l2_pct": float(median + 0.5),
        "runtime_seconds": 0.01,
    }


def test_build_report_passes_when_final_ci_upper_is_negative() -> None:
    rows = _seed_set_rows(seed=1, deltas=(-1.4, -2.0, -1.1), step=1)
    rows.extend(_seed_set_rows(seed=2, deltas=(-0.9, -1.6, -1.0), step=1))
    rows.extend(_seed_set_rows(seed=3, deltas=(-1.2, -1.5, -1.3), step=1))

    report, _ = build_emb_34um_causal_validation_report(
        rows=rows,
        bootstrap_resamples=2500,
        bootstrap_seed=10_01,
        confidence=0.95,
    )

    assert report["schema_version"] == EMB_34UM_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["decision"]["passed"] is True
    assert report["step_rows"][0]["paired_delta_ci_upper"] < 0.0
    assert report["final_step"]["paired_delta_ci_upper"] < 0.0


def test_build_report_fails_when_final_ci_upper_is_positive() -> None:
    rows = _seed_set_rows(seed=1, deltas=(1.2, 0.8, 0.4), step=2)
    rows.extend(_seed_set_rows(seed=2, deltas=(0.6, 1.1, 0.9), step=2))
    rows.extend(_seed_set_rows(seed=3, deltas=(0.7, 1.0, 0.8), step=2))

    report, _ = build_emb_34um_causal_validation_report(rows=rows, bootstrap_resamples=2000, bootstrap_seed=10_02)

    assert report["status"] == "failed"
    assert report["decision"]["passed"] is False
    assert report["decision"]["final_step"] == 2
    assert report["final_step"]["paired_delta_ci_upper"] > 0.0


def test_build_report_marks_inconclusive_for_three_seed_zero_straddling_ci() -> None:
    rows: list[dict[str, object]] = []
    rows.extend(_single_set_rows(seed=1, step=3, validation_set_id="set-a", lhs=10.0, al=9.0))
    rows.extend(_single_set_rows(seed=2, step=3, validation_set_id="set-a", lhs=10.0, al=10.5))
    rows.extend(_single_set_rows(seed=3, step=3, validation_set_id="set-a", lhs=10.0, al=11.0))

    report, _ = build_emb_34um_causal_validation_report(rows=rows, bootstrap_resamples=5000, bootstrap_seed=10_03)

    assert report["status"] == "inconclusive"
    assert report["decision"]["passed"] is False
    assert report["final_step"]["paired_delta_ci_lower"] <= 0.0 <= report["final_step"]["paired_delta_ci_upper"]


def test_build_report_pairs_validation_sets_by_median_delta() -> None:
    rows = [
        *_single_set_rows(seed=1, step=1, validation_set_id="A", lhs=4.0, al=2.0),
        _metric_row(
            strategy="al",
            seed=1,
            step=1,
            validation_set_id="A",
            metric=1.0,
            source_id="seed-001-step-01-A-al-extra",
        ),
        _metric_row(
            strategy="lhs",
            seed=1,
            step=1,
            validation_set_id="A",
            metric=5.0,
            source_id="seed-001-step-01-A-lhs-extra",
        ),
        *[
            _metric_row(
                strategy="al",
                seed=1,
                step=1,
                validation_set_id="B",
                metric=value,
                source_id=f"seed-001-step-01-B-al-{value}",
            )
            for value in (7.0, 9.0)
        ],
        *[
            _metric_row(
                strategy="lhs",
                seed=1,
                step=1,
                validation_set_id="B",
                metric=value,
                source_id=f"seed-001-step-01-B-lhs-{value}",
            )
            for value in (1.0, 3.0)
        ],
    ]

    report, _ = build_emb_34um_causal_validation_report(rows=rows, bootstrap_seed=10_04)

    step_row = report["step_rows"][0]
    assert len(report["seed_step_rows"]) == 1
    assert step_row["paired_delta"] == pytest.approx(1.5)
    assert step_row["paired_seed_count"] == 1
    # Expected medians:
    # Set A: AL median=1.5, LHS median=4.5, delta=-3.0
    # Set B: AL median=8.0, LHS median=2.0, delta=6.0
    # Seed-level median = 1.5


def test_build_report_enforces_same_validation_set_for_pairing() -> None:
    rows = [
        *_single_set_rows(seed=1, step=1, validation_set_id="A", lhs=3.0, al=2.0),
        _single_set_rows(seed=1, step=1, validation_set_id="B", lhs=1.5, al=1.0)[0],  # missing B-LHS sample
        *[
            _metric_row(
                strategy="lhs",
                seed=1,
                step=1,
                validation_set_id="C",
                metric=2.0,
                source_id="seed-001-step-01-C-lhs",
            )
        ],
    ]
    rows.append(
        _metric_row(
            strategy="al",
            seed=1,
            step=2,
            validation_set_id="A",
            metric=1.0,
            source_id="seed-001-step-02-a-al",
        )
    )
    rows.append(
        _metric_row(
            strategy="lhs",
            seed=1,
            step=2,
            validation_set_id="A",
            metric=2.0,
            source_id="seed-001-step-02-a-lhs",
        )
    )

    report, _ = build_emb_34um_causal_validation_report(rows=rows, bootstrap_seed=10_05)

    assert report["blocked_reasons"]["unpaired_validation_set_rows"] >= 2
    assert report["step_count"] == 2
    assert len(report["seed_step_rows"]) == 2
    assert report["seed_step_rows"][0]["step"] == 1
    assert report["seed_step_rows"][0]["matched_validation_set_count"] == 1


def test_build_report_from_ingestion_manifest_requires_shared_initial_and_wins_after_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(causal_report_module, "_train_validation_metric", _fake_train_validation_metric)
    records: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        records.extend(
            [
                _ingestion_curve_record(seed=seed, method="shared_initial", step=0, curve_ix=1),
                _ingestion_curve_record(seed=seed, method="shared_initial", step=0, curve_ix=2),
                _ingestion_curve_record(seed=seed, method="validation", step=0, curve_ix=1),
                _ingestion_curve_record(seed=seed, method="validation", step=0, curve_ix=2),
            ]
        )
        for step in range(1, 6):
            source = "ensemble_disagreement_diversity" if step % 2 else "exploration"
            records.append(_ingestion_curve_record(seed=seed, method="al", step=step, curve_ix=1, source=source))
            records.append(_ingestion_curve_record(seed=seed, method="lhs", step=step, curve_ix=1, source="lhs_grid"))

    payload = {
        "schema_version": "meso_uq.active_learning.emb_34um_causal_validation.v1",
        "records": records,
    }
    report, _ = build_emb_34um_causal_validation_report(
        rows=payload,
        bootstrap_resamples=2000,
        bootstrap_seed=1234,
    )

    assert report["decision"]["criteria"]["minimum_step_count"] == 5
    assert report["step_count"] == 5
    assert report["status"] == "passed"
    assert report["step_rows"][0]["paired_delta"] == pytest.approx(0.0)
    assert report["final_step"]["paired_delta"] < 0.0
    assert report["selection_rows"]


def test_build_report_from_full_scale_ingestion_manifest_records_expected_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        rows.extend(_ingestion_full_scale_rows(seed=seed))

    monkeypatch.setattr(causal_report_module, "_train_validation_metric", _fake_train_validation_metric)
    report, _ = build_emb_34um_causal_validation_report(
        rows={"schema_version": "meso_uq.active_learning.emb_34um_causal_validation.v1", "records": rows},
        bootstrap_seed=202,
        bootstrap_resamples=400,
    )

    assert report["status"] == "passed"
    assert report["step_count"] == 5
    assert report["decision"]["final_step"] == 5
    assert report["decision"]["criteria"]["minimum_step_count"] == 5

    pairing_profiles = report["metadata"]["ingestion"]["pairing_profiles"]
    assert len(pairing_profiles) == 3
    assert all(profile["is_full_scale_candidate"] for profile in pairing_profiles)
    assert all(profile["pairing_violation_count"] == 0 for profile in pairing_profiles)


def test_build_report_from_full_scale_ingestion_uses_campaign_step_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        rows.extend(_ingestion_full_scale_rows(seed=seed, step_count=6))
    campaign_manifest_path = tmp_path / "campaign_manifest.json"
    campaign_manifest_path.write_text(
        json.dumps(
            {
                "policy": {
                    "step_count": 6,
                    "step_size": 100,
                    "shared_size": 100,
                    "validation_size": 100,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(causal_report_module, "_train_validation_metric", _fake_train_validation_metric)
    report, _ = build_emb_34um_causal_validation_report(
        rows={
            "schema_version": "meso_uq.active_learning.emb_34um_causal_validation.v1",
            "campaign_manifest_path": str(campaign_manifest_path),
            "records": rows,
        },
        bootstrap_seed=206,
        bootstrap_resamples=400,
    )

    assert report["status"] == "passed"
    assert report["step_count"] == 6
    assert report["decision"]["final_step"] == 6
    assert report["decision"]["criteria"]["minimum_step_count"] == 6
    assert report["metadata"]["ingestion"]["policy"]["step_count"] == 6
    assert all(profile["pairing_violation_count"] == 0 for profile in report["metadata"]["ingestion"]["pairing_profiles"])


def test_build_report_from_full_scale_ingestion_manifest_blocks_step_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        seed_rows = _ingestion_full_scale_rows(seed=seed)
        seed_rows = [
            row
            for row in seed_rows
            if not (
                row.get("method") == "lhs"
                and row.get("step") == 3
                and str(row.get("candidate_id", "")).endswith("c001")
            )
        ]
        seed_rows.append(_ingestion_curve_record(seed=seed, method="lhs", step=6, curve_ix=201))
        rows.extend(seed_rows)

    monkeypatch.setattr(causal_report_module, "_train_validation_metric", _fake_train_validation_metric)
    report, _ = build_emb_34um_causal_validation_report(
        rows={"schema_version": "meso_uq.active_learning.emb_34um_causal_validation.v1", "records": rows},
        bootstrap_seed=303,
        bootstrap_resamples=200,
    )

    assert report["status"] == "blocked"
    assert report["step_count"] == 0
    assert report["blocked_reasons"].get("seed_pairing_coverage_violation", 0) == 3
    assert report["blocked_reasons"].get("minimum_step_count", 0) == 5
    profiles = report["metadata"]["ingestion"]["pairing_profiles"]
    assert len(profiles) == 3
    assert all(profile["pairing_violation_count"] > 0 for profile in profiles)


def test_build_report_includes_replacement_curve_rows_from_ingestion_manifest() -> None:
    failed_candidate_id = "seed-001-step-01-al-failed"
    replacement_candidate_id = "seed-001-step-01-al-replacement"
    rows: list[dict[str, object]] = [
        _ingestion_curve_record(seed=1, method="shared_initial", step=0, curve_ix=1),
        _ingestion_curve_record(seed=1, method="validation", step=0, curve_ix=1),
        {
            "candidate_id": failed_candidate_id,
            "seed": 1,
            "step": 1,
            "method": "al",
            "status": "failed",
            "sample_source": "candidate",
            "ka": 100.0,
            "kb": 200.0,
        },
        _ingestion_curve_record(seed=1, method="lhs", step=1, curve_ix=1),
    ]
    replacement_row = _ingestion_curve_record(seed=1, method="al", step=1, curve_ix=2)
    replacement_row["candidate_id"] = replacement_candidate_id
    replacement_row["replaced"] = True
    replacement_row["replacement_for"] = failed_candidate_id
    replacement_row["sample_source"] = "replacement"
    rows.append(replacement_row)

    report, _ = build_emb_34um_causal_validation_report(rows={"schema_version": "meso_uq.active_learning.emb_34um_causal_validation.v1", "records": rows})

    selection_rows = {item["candidate_id"]: item for item in report["selection_rows"]}
    assert selection_rows[replacement_candidate_id]["replacement"] is True
    assert selection_rows[replacement_candidate_id]["replacement_for"] == failed_candidate_id
    assert report["status"] == "blocked"
    assert selection_rows[replacement_candidate_id]["status"] == "completed"
    assert report["blocked_reasons"]["minimum_step_count"] == 4
    assert len(report["seed_step_rows"]) == 1
    assert report["seed_step_rows"][0]["selection_replacement_count"] == 1
    assert report["seed_step_rows"][0]["selection_failed_count"] == 1
    assert report["seed_step_rows"][0]["replacement_count"] == 1
    assert report["step_rows"][0]["step_selection_replacement_count"] == 1
    assert report["step_rows"][0]["step_selection_failed_count"] == 1
    assert report["step_rows"][0]["step_replacement_count"] == 1


def test_write_artifacts_generates_report_csv_pngs_and_sidecars(tmp_path: Path) -> None:
    rows = [
        *_single_set_rows(seed=1, step=1, validation_set_id="A", lhs=5.0, al=4.2),
        *_single_set_rows(seed=2, step=1, validation_set_id="A", lhs=5.0, al=4.0),
        *_single_set_rows(seed=3, step=1, validation_set_id="A", lhs=5.0, al=4.5),
    ]
    runtime_rows = [
        {
            "candidate_id": "seed-001-step-01-a-lhs",
            "seed": 1,
            "step": 1,
            "strategy": "lhs",
            "runtime_seconds": 1.3,
        },
        {
            "candidate_id": "seed-001-step-01-a-al",
            "seed": 1,
            "step": 1,
            "strategy": "al",
            "runtime_seconds": 2.1,
            "replaced": True,
        },
    ]

    artifacts = write_emb_34um_causal_validation_artifacts(
        rows=rows,
        runtime_rows=runtime_rows,
        output_root=tmp_path / "artifacts",
        include_plot=False,
        generation_command="analysis --rows rows.json",
    )

    assert artifacts.report_path == tmp_path / "artifacts" / EMB_34UM_CAUSAL_VALIDATION_REPORT_FILENAME
    assert artifacts.summary_csv_path == tmp_path / "artifacts" / EMB_34UM_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME
    assert artifacts.report_path.is_file()
    assert artifacts.summary_csv_path.is_file()

    assert artifacts.plot_paths["learning_curves"] == str(tmp_path / "artifacts" / "emb_34um_causal_validation_learning_curves.png")
    assert artifacts.plot_paths["final_delta_ci"] == str(tmp_path / "artifacts" / "emb_34um_causal_validation_final_paired_delta_ci.png")
    assert "added_samples_by_step" in artifacts.plot_paths
    assert "acquisition_selection_diagnostics" in artifacts.plot_paths

    for plot_path in artifacts.plot_paths.values():
        validate_plot_path(plot_path)

    fallback_png = causal_report_module._fallback_png()
    for plot_name in (
        "added_samples_by_step",
        "runtime_diagnostics",
        "acquisition_selection_diagnostics",
    ):
        assert Path(artifacts.plot_paths[plot_name]).read_bytes() == fallback_png

    for sidecar_path in artifacts.plot_sidecar_paths.values():
        assert Path(sidecar_path).is_file()
        payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        assert payload["generation_command"] == "analysis --rows rows.json"
        assert payload["bootstrap_resamples"] > 0
        assert payload["source_inputs"]

    with artifacts.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 1
    assert csv_rows[0]["step"] == "1"
    assert float(csv_rows[0]["paired_delta"]) < 0.0
