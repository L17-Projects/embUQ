from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_final_gate_surrogate import (
    train_emb_34um_surrogate_ensemble,
    score_emb_34um_candidate_pool,
    select_scored_candidates,
)


def _synthetic_records() -> tuple[dict[str, object], ...]:
    force_grid = (0.0, 0.5, 1.0, 1.5)
    return (
        {
            "curve_id": "c1",
            "ka": 1000.0,
            "kb": 1000.0,
            "force_grid": force_grid,
            "reference_curve": [1.0 + 0.2 * force for force in force_grid],
        },
        {
            "curve_id": "c2",
            "ka": 2000.0,
            "kb": 1100.0,
            "force_grid": force_grid,
            "reference_curve": [1.4 + 0.18 * force for force in force_grid],
        },
        {
            "curve_id": "c3",
            "ka": 3000.0,
            "kb": 1200.0,
            "force_grid": force_grid,
            "reference_curve": [2.0 + 0.13 * force for force in force_grid],
        },
    )


def test_train_returns_round_metric_payload_fields() -> None:
    records = _synthetic_records()
    report = train_emb_34um_surrogate_ensemble(
        records=records,
        validation_records=records,
        seeds=(11, 17),
        architecture_names=("linear_rf4", "poly2_rf4"),
    )

    assert report["fit"].backend == "numpy"
    assert set(
        (
            "median_curve_rel_l2_pct",
            "mean_curve_rel_l2_pct",
            "max_curve_rel_l2_pct",
            "residuals",
            "predicted_curves",
            "reference_curves",
            "model_selection",
        )
    ) <= report.keys()

    model_selection = report["model_selection"]
    assert model_selection["rerun"] is True
    assert model_selection["backend"] == "numpy"
    assert "notes" in model_selection

    assert len(report["residuals"]) == 3
    assert len(report["predicted_curves"]) == 3
    assert len(report["reference_curves"]) == 3

    for value in (
        report["median_curve_rel_l2_pct"],
        report["mean_curve_rel_l2_pct"],
        report["max_curve_rel_l2_pct"],
    ):
        assert math.isfinite(float(value))

    for curve in report["predicted_curves"]:
        assert len(curve) == 4
        assert np.all(np.isfinite(curve))


def test_score_pool_high_diversity_moves_candidate_up() -> None:
    records = _synthetic_records()
    report = train_emb_34um_surrogate_ensemble(
        records=records,
        validation_records=records,
        seeds=(11, 17),
        architecture_names=("linear_rf4", "poly2_rf4"),
    )

    candidates = (
        {
            "ka": 1050.0,
            "kb": 1030.0,
        },
        {
            "ka": 100_000.0,
            "kb": 20_000.0,
        },
    )

    scored = score_emb_34um_candidate_pool(
        report["fit"],
        candidates,
        force_grid=(0.0, 0.5, 1.0, 1.5),
        existing_points=((1000.0, 1000.0),),
        diversity_weight=50.0,
    )

    assert len(scored) == 2
    assert scored[0]["ka"] == 100_000.0
    assert scored[0]["acquisition_score"] >= scored[1]["acquisition_score"]
    assert "predicted_curve" in scored[0]
    assert len(scored[0]["predicted_curve"]) == 4


def test_select_scored_candidates_deterministic_order_for_tie_and_exploration() -> None:
    scored = (
        {
            "ka": 30.0,
            "kb": 40.0,
            "ensemble_disagreement": 0.5,
            "acquisition_score": 1.0,
        },
        {
            "ka": 10.0,
            "kb": 20.0,
            "ensemble_disagreement": 0.5,
            "acquisition_score": 1.0,
        },
        {
            "ka": 1000.0,
            "kb": 1000.0,
            "ensemble_disagreement": 0.5,
            "acquisition_score": 0.9,
        },
    )

    selected = select_scored_candidates(scored, count=1, exploration_count=1)

    assert len(selected) == 2
    assert selected[0]["ka"] == 10.0
    assert selected[0]["kb"] == 20.0
    assert selected[1]["ka"] == 1000.0
    assert selected[1]["kb"] == 1000.0


def test_train_and_score_empty_inputs_are_handled() -> None:
    records = _synthetic_records()
    report = train_emb_34um_surrogate_ensemble(
        records=records,
        validation_records=(),
        seeds=(11, 17),
        architecture_names=("linear_rf4",),
    )
    assert len(report["residuals"]) == 3

    no_candidates = score_emb_34um_candidate_pool(
        report["fit"],
        candidate_records=(),
        force_grid=(0.0, 0.5, 1.0, 1.5),
        existing_points=(),
        diversity_weight=0.2,
    )
    assert no_candidates == ()
