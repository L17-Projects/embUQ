from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics import (
    build_emb_34um_dnn_causal_validation_metric_rows,
    write_emb_34um_dnn_causal_validation_metric_artifacts,
)
from meso_uq.active_learning.emb_34um_dnn_surrogate import Emb34umDnnSurrogateFit


FORCE_GRID = tuple(5000.0 * index / 7.0 for index in range(8))


def _curve(ka: float, kb: float) -> list[float]:
    ka_log = math.log10(float(ka))
    kb_log = math.log10(float(kb))
    return [float(0.2 + 0.03 * step + 0.01 * ka_log - 0.005 * kb_log) for step in range(len(FORCE_GRID))]


def _row(
    *,
    candidate_id: str,
    branch: str,
    replicate: int,
    cycle: int,
    ka: float,
    kb: float,
    candidate_space: str = "d2",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "branch": branch,
        "replicate": replicate,
        "cycle": cycle,
        "candidate_space": candidate_space,
        "ka": ka,
        "kb": kb,
        "parameters": {"ka": ka, "kb": kb},
        "force_grid": list(FORCE_GRID),
        "reference_curve": _curve(ka, kb),
        "status": "completed",
        "replacement": False,
        "quarantine": False,
    }


def _d4_row(
    *,
    candidate_id: str,
    branch: str,
    replicate: int,
    cycle: int,
    ka: float,
    kb: float,
    radp: float,
    shell_th: float,
) -> dict[str, object]:
    payload = _row(
        candidate_id=candidate_id,
        branch=branch,
        replicate=replicate,
        cycle=cycle,
        ka=ka,
        kb=kb,
        candidate_space="d4",
    )
    payload["parameters"] = {"ka": ka, "kb": kb, "radp": radp, "shell_th": shell_th}
    payload["radp"] = radp
    payload["shell_th"] = shell_th
    return payload


def _completed_rows() -> list[dict[str, object]]:
    rows = [
        _row(candidate_id="u-001", branch="unseen_test", replicate=0, cycle=0, ka=2.0e3, kb=2.5e3),
        _row(candidate_id="u-002", branch="unseen_test", replicate=0, cycle=0, ka=8.0e3, kb=3.5e3),
        _row(candidate_id="s-001", branch="shared_initial", replicate=1, cycle=0, ka=1.5e3, kb=2.0e3),
        _row(candidate_id="s-002", branch="shared_initial", replicate=1, cycle=0, ka=4.0e3, kb=4.0e3),
        _row(candidate_id="a-101", branch="al", replicate=1, cycle=1, ka=2.0e4, kb=7.0e3),
        _row(candidate_id="l-101", branch="lhs", replicate=1, cycle=1, ka=1.0e3, kb=1.5e4),
        _row(candidate_id="a-201", branch="al", replicate=1, cycle=2, ka=6.0e4, kb=2.0e4),
        _row(candidate_id="l-201", branch="lhs", replicate=1, cycle=2, ka=3.0e3, kb=1.8e4),
    ]
    return rows


def test_dry_run_builds_paired_al_lhs_rows_and_divergent_training_sets() -> None:
    rows, report = build_emb_34um_dnn_causal_validation_metric_rows(
        completed_rows_source=_completed_rows(),
        dry_run=True,
    )

    assert report["passed"] is False
    assert report["status"] == "blocked"
    assert any("unseen_test count mismatch" in blocker for blocker in report["blockers"])
    assert len(rows) == 4
    assert {(int(item["cycle"]), str(item["branch"])) for item in rows} == {
        (1, "al"),
        (1, "lhs"),
        (2, "al"),
        (2, "lhs"),
    }

    by_cycle_branch = {(int(item["cycle"]), str(item["branch"])): item for item in rows}
    assert int(by_cycle_branch[(1, "al")]["train_curve_count"]) == 3
    assert int(by_cycle_branch[(1, "lhs")]["train_curve_count"]) == 3
    assert int(by_cycle_branch[(2, "al")]["train_curve_count"]) == 4
    assert int(by_cycle_branch[(2, "lhs")]["train_curve_count"]) == 4

    assert by_cycle_branch[(1, "al")]["train_set_signature"] != by_cycle_branch[(1, "lhs")]["train_set_signature"]
    assert by_cycle_branch[(2, "al")]["train_set_signature"] != by_cycle_branch[(2, "lhs")]["train_set_signature"]

    assert float(by_cycle_branch[(2, "al")]["relative_l2"]) != float(by_cycle_branch[(2, "lhs")]["relative_l2"])
    assert float(by_cycle_branch[(2, "al")]["relative_l2_std"]) >= 0.0
    assert by_cycle_branch[(2, "al")]["per_force_point_error_summary"]
    assert by_cycle_branch[(2, "al")]["ensemble_mean_prediction_error"]["relative_l2"]["mean"] >= 0.0
    assert by_cycle_branch[(2, "al")]["ensemble_spread_summary"]["member_count"] == 10
    assert by_cycle_branch[(2, "al")]["model_checkpoint_index"]["architecture"] == "dnn_mlp_64_64"
    assert float(by_cycle_branch[(1, "al")]["initial_relative_l2"]) == float(
        by_cycle_branch[(1, "lhs")]["initial_relative_l2"]
    )
    assert float(by_cycle_branch[(2, "al")]["initial_relative_l2"]) == float(
        by_cycle_branch[(2, "lhs")]["initial_relative_l2"]
    )
    assert float(by_cycle_branch[(1, "al")]["initial_relative_l2"]) == float(
        by_cycle_branch[(2, "al")]["initial_relative_l2"]
    )
    assert int(by_cycle_branch[(1, "al")]["finite_test_count"]) == 2


def test_dry_run_scores_are_computed_on_unseen_test_rows(monkeypatch) -> None:
    rows = _completed_rows()
    rows.append(_row(candidate_id="u-003", branch="unseen_test", replicate=0, cycle=0, ka=1.3e4, kb=2.8e3))
    rows.append(_row(candidate_id="s-extra", branch="shared_initial", replicate=1, cycle=0, ka=1.8e3, kb=1.9e3))

    observed_unseen: list[tuple[str, int, list[str]]] = []

    def _fake_dry_run_metrics(
        *,
        dry_run,
        train_rows,
        unseen_rows,
        branch,
        cycle,
        output_root,
        ensemble_size,
        device,
        force_grid,
        train_kwargs,
    ):
        del train_rows, output_root, ensemble_size, device, force_grid, train_kwargs
        del dry_run
        observed_unseen.append((str(branch), int(cycle), sorted(str(item["candidate_id"]) for item in unseen_rows)))
        return {
            "relative_l2_mean": 1.0 + 0.001 * int(cycle),
            "relative_l2_median": 1.0 + 0.001 * int(cycle),
            "train_seconds": float(len(unseen_rows)),
            "score_seconds": float(len(unseen_rows)),
        }

    monkeypatch.setattr(
        "meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics._evaluate_relative_l2_metrics",
        _fake_dry_run_metrics,
    )

    build_emb_34um_dnn_causal_validation_metric_rows(
        completed_rows_source=rows,
        dry_run=True,
    )

    expected_unseen = {"u-001", "u-002", "u-003"}
    assert observed_unseen
    for _, _, unseen_ids in observed_unseen:
        assert set(unseen_ids) == expected_unseen


def test_production_mode_uses_dnn_helpers(monkeypatch, tmp_path: Path) -> None:
    train_calls: list[int] = []
    predict_calls: list[tuple[float, float]] = []

    def _fake_train(
        completed_curve_rows,
        *,
        architecture="dnn_mlp_64_64",
        ensemble_size=10,
        ensemble_seeds=None,
        epochs=200,
        batch_size=128,
        learning_rate=1.0e-3,
        validation_fraction=0.1,
        force_grid=None,
        output_root=None,
        device=None,
        seed_offset=0,
        timing_canary=False,
    ):
        del architecture, ensemble_seeds, epochs, batch_size, learning_rate, validation_fraction, device, seed_offset, timing_canary
        train_calls.append(len(completed_curve_rows))
        fit = Emb34umDnnSurrogateFit(
            architecture="dnn_mlp_64_64",
            backend="fixed_architecture_dnn",
            ensemble_seeds=tuple(range(1, int(ensemble_size) + 1)),
            ensemble_checkpoints=(),
            ensemble_member_count=int(ensemble_size),
            train_runtime_seconds=0.01,
            score_runtime_seconds=0.0,
            feature_mean=(0.0, 0.0, 0.0),
            feature_scale=(1.0, 1.0, 1.0),
            target_mean=0.0,
            target_scale=1.0,
            force_min=float(force_grid[0]),
            force_max=float(force_grid[-1]),
            force_grid=tuple(float(v) for v in force_grid),
            device="cpu",
            train_row_count=len(completed_curve_rows),
            validation_row_count=1,
            model_path=Path(output_root or ".") / "fake.pt",
            training_history=(),
        )
        return fit, {"ok": True}

    def _fake_predict(fit, force_values, *, ka, kb, radp=None, shell_th=None):
        del fit
        predict_calls.append((float(ka), float(kb)))
        ka_log = math.log10(float(ka))
        kb_log = math.log10(float(kb))
        curve = tuple(float(0.21 + 0.03 * idx + 0.01 * ka_log - 0.005 * kb_log) for idx, _ in enumerate(force_values))
        return curve, ()

    monkeypatch.setattr(
        "meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics.train_emb_34um_dnn_surrogate_ensemble",
        _fake_train,
    )
    monkeypatch.setattr(
        "meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics.load_ensemble_member_predictions",
        _fake_predict,
    )

    rows, report = build_emb_34um_dnn_causal_validation_metric_rows(
        completed_rows_source=_completed_rows(),
        dry_run=False,
        output_root=tmp_path / "metrics",
    )

    assert report["passed"] is False
    assert report["status"] == "blocked"
    assert len(rows) == 4
    assert sorted(train_calls) == [2, 3, 3, 4, 4]
    assert len(predict_calls) == 10


def _completed_rows_with_d4_geometry() -> list[dict[str, object]]:
    return [
        _d4_row(
            candidate_id="u-001",
            branch="unseen_test",
            replicate=0,
            cycle=0,
            ka=2.0e3,
            kb=2.5e3,
            radp=6.5,
            shell_th=3.75e-9,
        ),
        _d4_row(
            candidate_id="u-002",
            branch="unseen_test",
            replicate=0,
            cycle=0,
            ka=8.0e3,
            kb=3.5e3,
            radp=6.6,
            shell_th=3.75e-9,
        ),
        _d4_row(
            candidate_id="s-001",
            branch="shared_initial",
            replicate=1,
            cycle=0,
            ka=1.5e3,
            kb=2.0e3,
            radp=6.6,
            shell_th=3.8e-9,
        ),
        _d4_row(
            candidate_id="s-002",
            branch="shared_initial",
            replicate=1,
            cycle=0,
            ka=4.0e3,
            kb=4.0e3,
            radp=6.6,
            shell_th=3.85e-9,
        ),
        _d4_row(
            candidate_id="a-101",
            branch="al",
            replicate=1,
            cycle=1,
            ka=2.0e4,
            kb=7.0e3,
            radp=6.45,
            shell_th=3.75e-9,
        ),
        _d4_row(
            candidate_id="l-101",
            branch="lhs",
            replicate=1,
            cycle=1,
            ka=1.0e3,
            kb=1.5e4,
            radp=6.52,
            shell_th=3.8e-9,
        ),
        _d4_row(
            candidate_id="a-201",
            branch="al",
            replicate=1,
            cycle=2,
            ka=6.0e4,
            kb=2.0e4,
            radp=6.55,
            shell_th=3.9e-9,
        ),
        _d4_row(
            candidate_id="l-201",
            branch="lhs",
            replicate=1,
            cycle=2,
            ka=3.0e3,
            kb=1.8e4,
            radp=6.57,
            shell_th=4.0e-9,
        ),
    ]


def test_production_mode_uses_d4_geometry_and_records_added_points(monkeypatch, tmp_path: Path) -> None:
    train_calls: list[int] = []
    predict_calls: list[tuple[float, float, float | None, float | None]] = []

    def _fake_train(
        completed_curve_rows,
        *,
        architecture="dnn_mlp_64_64",
        ensemble_size=10,
        ensemble_seeds=None,
        epochs=200,
        batch_size=128,
        learning_rate=1.0e-3,
        validation_fraction=0.1,
        force_grid=None,
        output_root=None,
        device=None,
        seed_offset=0,
        timing_canary=False,
    ):
        del architecture, ensemble_seeds, epochs, batch_size, learning_rate, validation_fraction, device, seed_offset, timing_canary
        train_calls.append(len(completed_curve_rows))
        fit = Emb34umDnnSurrogateFit(
            architecture="dnn_mlp_64_64",
            backend="fixed_architecture_dnn",
            ensemble_seeds=tuple(range(1, int(ensemble_size) + 1)),
            ensemble_checkpoints=(),
            ensemble_member_count=int(ensemble_size),
            train_runtime_seconds=0.01,
            score_runtime_seconds=0.0,
            feature_mean=(3.0, 2.7, 6.5, 3.75e-9, 0.0),
            feature_scale=(1.0, 1.0, 1.0, 1.0, 1.0),
            target_mean=0.0,
            target_scale=1.0,
            force_min=float(force_grid[0]),
            force_max=float(force_grid[-1]),
            force_grid=tuple(float(v) for v in force_grid),
            device="cpu",
            train_row_count=len(completed_curve_rows),
            validation_row_count=1,
            model_path=Path(output_root or ".") / "fake.pt",
            training_history=(),
        )
        return fit, {"ok": True}

    def _fake_predict(fit, force_values, *, ka: float, kb: float, radp=None, shell_th=None):
        del fit
        predict_calls.append((float(ka), float(kb), None if radp is None else float(radp), None if shell_th is None else float(shell_th)))
        ka_log = math.log10(float(ka))
        kb_log = math.log10(float(kb))
        curve = tuple(float(0.21 + 0.03 * idx + 0.01 * ka_log - 0.005 * kb_log) for idx, _ in enumerate(force_values))
        return curve, ()

    monkeypatch.setattr(
        "meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics.train_emb_34um_dnn_surrogate_ensemble",
        _fake_train,
    )
    monkeypatch.setattr(
        "meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics.load_ensemble_member_predictions",
        _fake_predict,
    )

    rows, report = build_emb_34um_dnn_causal_validation_metric_rows(
        completed_rows_source=_completed_rows_with_d4_geometry(),
        dry_run=False,
        output_root=tmp_path / "metrics",
    )

    assert report["passed"] is False
    assert report["status"] == "blocked"
    assert len(rows) == 4
    assert sorted(train_calls) == [2, 3, 3, 4, 4]
    assert len(predict_calls) == 10
    assert {
        (radp, shell_th)
        for _, _, radp, shell_th in predict_calls
        if radp is not None and shell_th is not None
    }.issuperset({(6.5, 3.75e-9), (6.6, 3.75e-9)})

    for row in rows:
        if row["branch"] in {"al", "lhs"}:
            assert row["added_radp"] != []
            assert row["added_shell_th"] != []
            assert all(item is not None for item in row["added_radp"])
            assert all(item is not None for item in row["added_shell_th"])


def test_metrics_require_d4_geometry_for_unseen_rows() -> None:
    rows = _completed_rows_with_d4_geometry()
    rows[0].pop("radp")
    rows[0].pop("shell_th")
    rows[0]["parameters"] = {"ka": rows[0]["ka"], "kb": rows[0]["kb"]}

    with pytest.raises(ValueError, match="completed_rows\\[1\\] must include radp and shell_th for D4 causal validation"):
        build_emb_34um_dnn_causal_validation_metric_rows(
            completed_rows_source=rows,
            dry_run=False,
            output_root=Path("unused"),
        )


def test_metrics_require_d4_geometry_for_training_rows() -> None:
    rows = _completed_rows_with_d4_geometry()
    rows[2].pop("radp")
    rows[2].pop("shell_th")
    rows[2]["parameters"] = {"ka": rows[2]["ka"], "kb": rows[2]["kb"]}

    with pytest.raises(ValueError, match="completed_rows\\[3\\] must include radp and shell_th for D4 causal validation"):
        build_emb_34um_dnn_causal_validation_metric_rows(
            completed_rows_source=rows,
            dry_run=False,
            output_root=Path("unused"),
        )


def test_metrics_report_blocks_incomplete_protocol_counts_even_with_metric_rows() -> None:
    rows, report = build_emb_34um_dnn_causal_validation_metric_rows(
        completed_rows_source=_completed_rows_with_d4_geometry(),
        dry_run=True,
    )

    assert rows
    assert report["status"] == "blocked"
    assert report["passed"] is False
    assert report["protocol_completeness"]["passed"] is False
    assert any("shared_initial count mismatch" in blocker for blocker in report["blockers"])


def test_metric_artifact_metadata_counts_source_rows_before_metric_reduction(tmp_path: Path) -> None:
    artifacts = write_emb_34um_dnn_causal_validation_metric_artifacts(
        completed_rows_source=_completed_rows_with_d4_geometry(),
        output_root=tmp_path / "metrics",
        rows_filename="metric_rows.json",
        report_filename="metric_report.json",
        dry_run=True,
    )

    payload = json.loads(artifacts.rows_path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]

    assert metadata["shared_initial_count"] == 2
    assert metadata["unseen_test_count"] == 2
    assert metadata["parameter_names"] == ["ka", "kb", "radp", "shell_th"]
    assert {row["branch"] for row in payload["rows"]} == {"al", "lhs"}
