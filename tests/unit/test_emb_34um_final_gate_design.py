from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import (  # noqa: E402
    EMB_34UM_FINAL_GATE_ACTIVE_VARIABLES,
    EMB_34UM_FINAL_GATE_ACQUISITION_COUNT,
    EMB_34UM_FINAL_GATE_BATCH_SIZE,
    EMB_34UM_FINAL_GATE_EXPLORATION_COUNT,
    EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES,
    EMB_34UM_FINAL_GATE_SOURCE_ACQUISITION,
    EMB_34UM_FINAL_GATE_SOURCE_EXPLORATION,
    EMB_34UM_FINAL_GATE_SOURCE_INITIAL,
    EMB_34UM_FINAL_GATE_SOURCE_VALIDATION,
    EMB_34UM_FINAL_GATE_VALIDATION_SEED_OFFSET,
    EMB_34UM_FINAL_GATE_BOUNDS,
    build_emb_34um_final_gate_design_round,
    build_emb_34um_final_gate_validation_design,
    Emb34umFinalGateDesignResult,
    Emb34umFinalGateValidationResult,
)
from meso_uq.active_learning.emb_34um_final_gate_design import _select_acquisition


def _as_points(candidates: tuple) -> list[tuple[float, float]]:
    return [(float(item.parameters["ka"]), float(item.parameters["kb"])) for item in candidates]


def _assert_bounds(points: list[tuple[float, float]]) -> None:
    ka_min, ka_max = EMB_34UM_FINAL_GATE_BOUNDS["ka"]
    kb_min, kb_max = EMB_34UM_FINAL_GATE_BOUNDS["kb"]
    for ka, kb in points:
        assert ka_min <= ka <= ka_max
        assert kb_min <= kb <= kb_max


def _coerce_source_counts(round_result: Emb34umFinalGateDesignResult) -> Mapping[str, int]:
    return {source: round_result.selected_sources().count(source) for source in set(round_result.selected_sources())}


def test_round_one_design_emits_30_initial_maximin_points_and_is_deterministic() -> None:
    first = build_emb_34um_final_gate_design_round(
        run_id="emb-34um-final-gate-test",
        round_index=1,
        seed=1234,
    )
    second = build_emb_34um_final_gate_design_round(
        run_id="emb-34um-final-gate-test",
        round_index=1,
        seed=1234,
    )

    assert isinstance(first, Emb34umFinalGateDesignResult)
    assert first.manifest["active_variables"] == list(EMB_34UM_FINAL_GATE_ACTIVE_VARIABLES)
    assert first.manifest["lhs_comparator_prefixes"] == list(EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES)
    assert first.manifest["candidate_pool_size"] == 100
    assert first.manifest["batch_size"] == EMB_34UM_FINAL_GATE_BATCH_SIZE
    assert first.manifest["selected_source_distribution"] == {EMB_34UM_FINAL_GATE_SOURCE_INITIAL: 30}

    assert len(first.candidates) == EMB_34UM_FINAL_GATE_BATCH_SIZE == 30
    assert len(first.round_pool) == 100
    assert first.selected_sources() == (EMB_34UM_FINAL_GATE_SOURCE_INITIAL,) * 30
    assert [candidate.candidate_id for candidate in first.candidates] == [
        candidate.candidate_id for candidate in second.candidates
    ]
    assert _as_points(first.round_pool) == _as_points(second.round_pool)
    _assert_bounds(_as_points(first.candidates))
    assert set(first.selected_sources()) == {EMB_34UM_FINAL_GATE_SOURCE_INITIAL}

    for candidate in first.candidates:
        params = set(candidate.parameters)
        assert params == {"family", "experiment", "ka", "kb"}
        assert "Yt" not in params
        assert "d0" not in params
        assert "sigma" not in params


def test_round_two_design_honors_6explore_24acquisition_split() -> None:
    round_one = build_emb_34um_final_gate_design_round(
        run_id="emb-34um-final-gate-test",
        round_index=1,
        seed=42,
        batch_size=30,
    )
    round_two = build_emb_34um_final_gate_design_round(
        run_id="emb-34um-final-gate-test",
        round_index=2,
        seed=43,
        existing_points=[(candidate.parameters["ka"], candidate.parameters["kb"]) for candidate in round_one.candidates],
        exploration_count=EMB_34UM_FINAL_GATE_EXPLORATION_COUNT,
        acquisition_count=EMB_34UM_FINAL_GATE_ACQUISITION_COUNT,
    )

    assert len(round_two.candidates) == 30
    source_counts = _coerce_source_counts(round_two)
    assert source_counts.get(EMB_34UM_FINAL_GATE_SOURCE_EXPLORATION) == 6
    assert source_counts.get(EMB_34UM_FINAL_GATE_SOURCE_ACQUISITION) == 24
    assert round_two.selected_sources().count(EMB_34UM_FINAL_GATE_SOURCE_INITIAL) == 0
    _assert_bounds(_as_points(round_two.candidates))
    assert len(set(_as_points(round_two.candidates))) == len(round_two.candidates)
    assert round_two.manifest["candidate_pool_size"] == 100


def test_validation_design_is_independent_sobol_maximin_design() -> None:
    round_design = build_emb_34um_final_gate_design_round(
        run_id="emb-34um-final-gate-test",
        round_index=1,
        seed=11,
    )
    validation = build_emb_34um_final_gate_validation_design(
        run_id="emb-34um-final-gate-test",
        seed=11,
    )
    validation_repeat = build_emb_34um_final_gate_validation_design(
        run_id="emb-34um-final-gate-test",
        seed=11,
    )

    assert isinstance(validation, Emb34umFinalGateValidationResult)
    assert validation.manifest["source"] == EMB_34UM_FINAL_GATE_SOURCE_VALIDATION
    assert validation.manifest["design_type"] == "validation"
    assert validation.manifest["lhs_comparator_prefixes"] == list(EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES)
    assert len(validation.candidates) == 90
    assert validation.manifest["selection_seed"] == 11 + EMB_34UM_FINAL_GATE_VALIDATION_SEED_OFFSET
    assert validation_repeat.manifest["selection_seed"] == 11 + EMB_34UM_FINAL_GATE_VALIDATION_SEED_OFFSET
    assert validation.manifest["selection_seed"] != round_design.manifest["selection_seed"]
    assert [candidate.candidate_id for candidate in validation.candidates] == [
        candidate.candidate_id for candidate in validation_repeat.candidates
    ]
    _assert_bounds(_as_points(validation.candidates))
    assert set(_as_points(validation.candidates)).isdisjoint(set(_as_points(round_design.candidates)))

    for candidate in validation.candidates:
        params = set(candidate.parameters)
        assert params == {"family", "experiment", "ka", "kb"}


def test_disagreement_acquisition_applies_diversity_penalty_after_top_disagreement() -> None:
    # Two top disagreement points that are near each other should yield to a farther lower-disagreement
    # point once diversity is scored during greedy selection.
    picked = _select_acquisition(
        unit_candidates=((0.5, 0.0), (0.51, 0.0), (0.0, 1.0)),
        disagreement=(1.0, 0.9, 0.89),
        count=2,
        reference=[(0.0, 0.0)],
        seed=0,
    )

    assert picked == (0, 2)
    assert picked[1] != 1


def test_validation_and_round_manifests_expose_expected_controls_metadata() -> None:
    round_one = build_emb_34um_final_gate_design_round(
        run_id="emb-34um-final-gate-test",
        round_index=1,
        seed=101,
    )
    validation = build_emb_34um_final_gate_validation_design(
        run_id="emb-34um-final-gate-test",
        seed=101,
    )

    assert round_one.manifest["use_log_space"] is True
    assert validation.manifest["use_log_space"] is True
    assert round_one.manifest["active_variables"] == ["ka", "kb"]
    assert set(round_one.manifest["parameter_bounds"]) == {"ka", "kb"}
    assert set(validation.manifest["parameter_bounds"]) == {"ka", "kb"}


def test_rejected_previous_points_and_seed_must_be_valid() -> None:
    round_one = build_emb_34um_final_gate_design_round(
        run_id="emb-34um-final-gate-test",
        round_index=1,
        seed=12,
    )
    bad_previous = ({"wrong": 1.0, "also_wrong": 2.0},)

    with pytest.raises(ValueError, match="existing_points"):
        build_emb_34um_final_gate_design_round(
            run_id="emb-34um-final-gate-test",
            round_index=2,
            seed=12,
            existing_points=bad_previous,
        )
    with pytest.raises(ValueError, match="seed must be"):
        build_emb_34um_final_gate_design_round(
            run_id="emb-34um-final-gate-test",
            round_index=2,
            seed=-1,
            existing_points=[(candidate.parameters["ka"], candidate.parameters["kb"]) for candidate in round_one.candidates],
        )
