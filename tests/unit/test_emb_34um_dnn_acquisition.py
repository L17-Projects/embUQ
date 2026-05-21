from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import meso_uq.active_learning.emb_34um_dnn_acquisition as module
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import EMB_34UM_DNN_CAUSAL_FORCE_GRID
from meso_uq.active_learning.emb_34um_dnn_surrogate import Emb34umDnnSurrogateFit


def _fit() -> Emb34umDnnSurrogateFit:
    return Emb34umDnnSurrogateFit(
        architecture="dnn_mlp_64_64",
        backend="fixed_architecture_dnn",
        ensemble_seeds=tuple(range(1, 11)),
        ensemble_checkpoints=tuple(Path(f"/tmp/member_{index:02d}.pt") for index in range(1, 11)),
        ensemble_member_count=10,
        train_runtime_seconds=1.0,
        score_runtime_seconds=0.0,
        feature_mean=(0.0, 0.0, 0.0),
        feature_scale=(1.0, 1.0, 1.0),
        target_mean=0.0,
        target_scale=1.0,
        force_min=0.0,
        force_max=5000.0,
        force_grid=tuple(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
        device="cpu",
        train_row_count=16,
        validation_row_count=2,
        model_path=Path("/tmp/model.pt"),
        training_history=(),
    )


def test_score_candidate_pool_supports_tuple_existing_points(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_predict(_fit, force_grid, *, ka: float, kb: float):
        disagreement = abs(math.log10(ka) - 3.0) + abs(math.log10(kb) - 3.0)
        member_curves = tuple(
            tuple(disagreement + 0.01 * member + 0.001 * step for step, _ in enumerate(force_grid))
            for member in range(10)
        )
        mean_curve = tuple(sum(values) / len(values) for values in zip(*member_curves))
        return mean_curve, member_curves

    monkeypatch.setattr(module, "load_ensemble_member_predictions", _fake_predict)

    scored = module.score_candidate_pool(
        _fit(),
        candidate_records=[
            {"candidate_id": "a", "ka": 1.0e3, "kb": 1.0e3},
            {"candidate_id": "b", "ka": 1.0e5, "kb": 5.0e4},
        ],
        force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
        existing_points=[(1.0e2, 4.0e2)],
    )

    assert len(scored) == 2
    assert scored[0]["candidate_index"] == 2
    assert scored[0]["ensemble_disagreement"] > scored[1]["ensemble_disagreement"]
    assert scored[0]["diversity_term"] > 0.0
    assert len(scored[0]["predicted_curve"]) == len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)


def test_select_greedy_diversity_prefers_spatial_spread() -> None:
    scored = (
        {
            "candidate_id": "near-1",
            "candidate_log10_point": (3.00, 3.00),
            "ensemble_disagreement": 1.0,
            "diversity_term": 0.0,
            "acquisition_score": 1.0,
        },
        {
            "candidate_id": "near-2",
            "candidate_log10_point": (3.02, 3.02),
            "ensemble_disagreement": 0.99,
            "diversity_term": 0.0,
            "acquisition_score": 0.99,
        },
        {
            "candidate_id": "far",
            "candidate_log10_point": (5.0, 4.8),
            "ensemble_disagreement": 0.4,
            "diversity_term": 0.0,
            "acquisition_score": 0.4,
        },
    )

    selected = module.select_greedy_diversity(scored, count=2, diversity_weight=0.5)

    assert [item["candidate_id"] for item in selected] == ["near-1", "far"]
