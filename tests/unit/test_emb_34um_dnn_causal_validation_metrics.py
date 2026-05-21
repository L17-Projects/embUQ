from __future__ import annotations

import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics import (
    build_emb_34um_dnn_causal_validation_metric_rows,
)
from meso_uq.active_learning.emb_34um_dnn_surrogate import Emb34umDnnSurrogateFit


FORCE_GRID = tuple(5000.0 * index / 7.0 for index in range(8))


def _curve(ka: float, kb: float) -> list[float]:
    ka_log = math.log10(float(ka))
    kb_log = math.log10(float(kb))
    return [float(0.2 + 0.03 * step + 0.01 * ka_log - 0.005 * kb_log) for step in range(len(FORCE_GRID))]


def _row(*, candidate_id: str, branch: str, replicate: int, cycle: int, ka: float, kb: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "branch": branch,
        "replicate": replicate,
        "cycle": cycle,
        "ka": ka,
        "kb": kb,
        "parameters": {"ka": ka, "kb": kb},
        "force_grid": list(FORCE_GRID),
        "reference_curve": _curve(ka, kb),
        "status": "completed",
        "replacement": False,
        "quarantine": False,
    }


def _completed_rows() -> list[dict[str, object]]:
    rows = [
        _row(candidate_id="u-001", branch="unseen_test", replicate=0, cycle=0, ka=2.0e3, kb=2.5e3),
        _row(candidate_id="u-002", branch="unseen_test", replicate=0, cycle=0, ka=8.0e3, kb=3.5e3),
        _row(candidate_id="s-001", branch="shared_initial", replicate=1, cycle=0, ka=1.5e3, kb=2.0e3),
        _row(candidate_id="s-002", branch="shared_initial", replicate=1, cycle=0, ka=4.0e3, kb=4.0e3),
        _row(candidate_id="a-101", branch="al", replicate=1, cycle=1, ka=2.0e4, kb=7.0e3),
        _row(candidate_id="l-101", branch="lhs", replicate=1, cycle=1, ka=1.0e3, kb=6.0e4),
        _row(candidate_id="a-201", branch="al", replicate=1, cycle=2, ka=6.0e4, kb=2.0e4),
        _row(candidate_id="l-201", branch="lhs", replicate=1, cycle=2, ka=3.0e3, kb=5.5e4),
    ]
    return rows


def test_dry_run_builds_paired_al_lhs_rows_and_divergent_training_sets() -> None:
    rows, report = build_emb_34um_dnn_causal_validation_metric_rows(
        completed_rows_source=_completed_rows(),
        dry_run=True,
    )

    assert report["passed"] is True
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

    def _fake_predict(fit, force_values, *, ka, kb):
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

    assert report["passed"] is True
    assert len(rows) == 4
    assert sorted(train_calls) == [3, 3, 4, 4]
    assert len(predict_calls) == 8
