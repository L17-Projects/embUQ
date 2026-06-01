from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import meso_uq.active_learning.emb_34um_dnn_acquisition as module
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import EMB_34UM_DNN_CAUSAL_FORCE_GRID
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import EMB_34UM_DNN_CAUSAL_BOUNDS
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


def _fit_d4() -> Emb34umDnnSurrogateFit:
    return Emb34umDnnSurrogateFit(
        architecture="dnn_mlp_64_64",
        backend="fixed_architecture_dnn",
        ensemble_seeds=tuple(range(1, 11)),
        ensemble_checkpoints=tuple(Path(f"/tmp/member_{index:02d}.pt") for index in range(1, 11)),
        ensemble_member_count=10,
        train_runtime_seconds=1.0,
        score_runtime_seconds=0.0,
        feature_mean=(3.0, 2.7, 6.5, 3.75e-9, 0.0),
        feature_scale=(1.0, 1.0, 1.0, 1.0, 1.0),
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
    def _fake_predict(_fit, force_grid, *, ka: float, kb: float, radp=None, shell_th=None):
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
            {"candidate_id": "a", "ka": 1.0e3, "kb": 2.0e3},
            {"candidate_id": "b", "ka": 1.0e5, "kb": 2.0e4},
        ],
        force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
        existing_points=[(1.0e3, 2.0e3)],
    )

    assert len(scored) == 2
    assert scored[0]["candidate_index"] == 2
    assert scored[0]["ensemble_disagreement"] > scored[1]["ensemble_disagreement"]
    assert scored[0]["diversity_term"] > 0.0
    assert len(scored[0]["predicted_curve"]) == len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)


def test_score_candidate_pool_d4_passes_geometry_to_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[float, float, float | None, float | None]] = []

    def _fake_predict(_fit, force_grid, *, ka: float, kb: float, radp: float | None, shell_th: float | None):
        observed.append((float(ka), float(kb), None if radp is None else float(radp), None if shell_th is None else float(shell_th)))
        member_curves = tuple(
            tuple(0.0 + 0.001 * member + 0.001 * step for step, _ in enumerate(force_grid))
            for member in range(10)
        )
        mean_curve = tuple(sum(values) / len(values) for values in zip(*member_curves))
        return mean_curve, member_curves

    monkeypatch.setattr(module, "load_ensemble_member_predictions", _fake_predict)

    scored = module.score_candidate_pool(
        _fit_d4(),
        candidate_records=[
            {
                "candidate_id": "d4",
                "ka": 1.0e4,
                "kb": 2.0e3,
                "radp": 6.45,
                "shell_th": 3.75e-9,
            },
        ],
        force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
        candidate_space="d4",
    )

    assert len(scored) == 1
    assert observed == [(1.0e4, 2.0e3, 6.45, 3.75e-9)]
    assert len(scored[0]["candidate_log10_point"]) == 4
    assert scored[0]["candidate_log10_point"][2] == pytest.approx(
        (6.45 - EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][0])
        / (EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][1] - EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][0])
    )
    assert scored[0]["candidate_log10_point"][3] == pytest.approx(
        (3.75e-9 - EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][0])
        / (EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][1] - EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][0])
    )


def test_score_candidate_pool_d4_rejects_legacy_two_value_existing_points() -> None:
    with pytest.raises(ValueError, match=r"existing_points\[1\].*4-item numeric sequence"):
        module.score_candidate_pool(
            _fit_d4(),
            candidate_records=[
                {
                    "candidate_id": "d4",
                    "ka": 1.0e4,
                    "kb": 2.0e3,
                    "radp": 6.45,
                    "shell_th": 3.75e-9,
                }
            ],
            force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
            candidate_space="d4",
            existing_points=[(1.0e3, 2.0e3)],
        )


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


def _unit_projection(record: Mapping[str, Any], dimension: str) -> float:
    value = float(record[dimension])
    if dimension in {"ka", "kb"}:
        return (math.log10(value) - math.log10(EMB_34UM_DNN_CAUSAL_BOUNDS[dimension][0])) / (
            math.log10(EMB_34UM_DNN_CAUSAL_BOUNDS[dimension][1]) - math.log10(EMB_34UM_DNN_CAUSAL_BOUNDS[dimension][0])
        )
    return (value - EMB_34UM_DNN_CAUSAL_BOUNDS[dimension][0]) / (
        EMB_34UM_DNN_CAUSAL_BOUNDS[dimension][1] - EMB_34UM_DNN_CAUSAL_BOUNDS[dimension][0]
    )


def _max_ks_stat(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    best = 0.0
    for index, value in enumerate(sorted_values, start=1):
        empirical = index / count
        lower = (index - 1) / count
        best = max(best, abs(empirical - value), abs(lower - value))
    return best


def _candidate_pool_coverage_score(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        return 1.0
    return sum(
        _max_ks_stat([_unit_projection(record, dimension) for record in records])
        for dimension in ("ka", "kb", "radp", "shell_th")
    ) / 4.0


def test_build_d4_candidate_pool_is_structured_and_deterministic() -> None:
    structured_a, structured_meta = module.build_d4_candidate_pool(
        requested_size=400,
        seed=123,
        generator="stratified",
    )
    structured_b, structured_meta_b = module.build_d4_candidate_pool(
        requested_size=400,
        seed=123,
        generator="stratified",
    )

    assert structured_meta["generator_type"] == "stratified"
    assert structured_meta["requested_size"] == 400
    assert structured_meta["actual_size"] == 400
    assert structured_meta["seed"] == 123
    assert structured_meta["dimension_names"] == ["ka", "kb", "radp", "shell_th"]
    assert structured_meta["bounds"] == {
        "ka": [EMB_34UM_DNN_CAUSAL_BOUNDS["ka"][0], EMB_34UM_DNN_CAUSAL_BOUNDS["ka"][1]],
        "kb": [EMB_34UM_DNN_CAUSAL_BOUNDS["kb"][0], EMB_34UM_DNN_CAUSAL_BOUNDS["kb"][1]],
        "radp": [EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][0], EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][1]],
        "shell_th": [EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][0], EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][1]],
    }
    assert structured_meta_b == structured_meta
    assert structured_a == structured_b


def test_build_d4_candidate_pool_can_exceed_500_without_dpd_exclusion_retries() -> None:
    pool, metadata = module.build_d4_candidate_pool(
        requested_size=750,
        seed=99,
        generator="stratified",
        max_attempts_multiplier=25,
    )

    assert len(pool) == 750
    assert metadata["actual_size"] == 750
    assert metadata["requested_size"] == 750
    assert metadata["generator_type"] == "stratified"


def test_structured_candidate_pool_better_marginal_coverage_than_small_random_pool() -> None:
    structured, _ = module.build_d4_candidate_pool(
        requested_size=500,
        seed=7,
        generator="stratified",
    )
    random_legacy, _ = module.build_d4_candidate_pool(
        requested_size=500,
        seed=7,
        generator="random",
    )
    structured_score = _candidate_pool_coverage_score(structured)
    random_score = _candidate_pool_coverage_score(random_legacy)

    assert structured_score < random_score


def test_build_d4_candidate_pool_allows_same_ka_kb_with_distinct_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    def _scripted_unit_points(*, count: int, dimensions: int, seed: int):
        assert count == 2
        assert dimensions == 4
        assert seed == 101
        return (
            (0.5, 0.5, 0.10, 0.10),
            (0.5, 0.5, 0.90, 0.90),
        )

    monkeypatch.setattr(module, "_build_stratified_unit_points", _scripted_unit_points)
    first_pool, _ = module.build_d4_candidate_pool(
        requested_size=1,
        seed=101,
        generator="stratified",
        max_attempts_multiplier=2,
    )
    first = first_pool[0]

    second_pool, _ = module.build_d4_candidate_pool(
        requested_size=1,
        seed=101,
        generator="stratified",
        existing_points=[(first["ka"], first["kb"], first["radp"], first["shell_th"])],
        max_attempts_multiplier=2,
    )
    second = second_pool[0]

    assert second["ka"] == first["ka"]
    assert second["kb"] == first["kb"]
    assert second["radp"] != first["radp"]
    assert second["shell_th"] != first["shell_th"]
