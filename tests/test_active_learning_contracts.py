from __future__ import annotations

import json
from dataclasses import replace

from meso_uq.active_learning import (
    ActiveLearningDryRunEngine,
    AcquisitionScore,
    Candidate,
    LoopState,
    RetrainingResult,
    SimulationResult,
    StoppingCriteria,
)
from meso_uq.active_learning.engine import Retrainer
from meso_uq.active_learning.contracts import RetrainingRequest


def _build_fake_retrainer() -> tuple[Retrainer, list[tuple[RetrainingRequest, int]]]:
    calls: list[tuple[RetrainingRequest, int]] = []

    def retrain(request: RetrainingRequest, state) -> RetrainingResult:
        calls.append((request, state.iteration))
        return RetrainingResult(
            iteration=request.iteration,
            status="completed",
            model_token=f"model-{request.iteration}",
            metrics={"count": len(request.successful_results)},
            metadata={"batch": len(request.candidate_ids)},
        )

    return retrain, calls


def test_loop_initialization_uses_fresh_state() -> None:
    def no_op(_: LoopState) -> list[Candidate]:
        return []

    def zero_score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [
            AcquisitionScore(candidate_id=candidate.candidate_id, score=0.0) for candidate in candidates
        ]

    def empty_simulator(requests, _state):
        return []

    retrain, _ = _build_fake_retrainer()
    engine = ActiveLearningDryRunEngine(
        candidate_generator=no_op,
        acquisition_policy=zero_score,
        simulator=empty_simulator,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=1),
        batch_size=1,
        loop_id="init-loop",
    )
    assert engine.state.loop_id == "init-loop"
    assert engine.state.iteration == 0
    assert engine.state.candidates == ()
    assert engine.state.is_complete is False


def test_candidate_validation_is_enforced_before_simulation() -> None:
    def candidates(_: LoopState) -> list[dict]:
        return [
            {"candidate_id": "good", "parameters": {"lr": 0.1}},
            {"candidate_id": "bad", "parameters": "not-a-mapping"},
            {"candidate_id": "also-bad", "parameters": {}},
        ]

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        assert tuple(candidate.candidate_id for candidate in candidates) == ("good",)
        return [AcquisitionScore(candidate_id="good", score=1.0)]

    simulation_calls: list[list[str]] = []

    def simulate(requests: tuple, _state):
        simulation_calls.append([request.candidate.candidate_id for request in requests])
        return [
            SimulationResult(
                request_id=request.request_id,
                candidate_id=request.candidate.candidate_id,
                status="success",
                metrics={"score": 1.0},
            )
            for request in requests
        ]

    retrain, _ = _build_fake_retrainer()
    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=1),
        batch_size=2,
    )
    state = engine.run()
    assert simulation_calls == [["good"]]
    assert state.simulation_results[0].candidate_id == "good"
    assert len(state.failures) == 2
    assert {record.stage for record in state.failures} == {"validation"}
    assert any(record.candidate_id == "bad" for record in state.failures)
    assert any(record.candidate_id == "also-bad" for record in state.failures)


def test_acquisition_selects_top_scored_batch_for_simulation() -> None:
    def candidates(_: LoopState) -> list[Candidate]:
        return [
            Candidate(candidate_id="c-1", parameters={"x": 1}),
            Candidate(candidate_id="c-2", parameters={"x": 2}),
            Candidate(candidate_id="c-3", parameters={"x": 3}),
        ]

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [
            AcquisitionScore(candidate_id="c-1", score=1.0),
            AcquisitionScore(candidate_id="c-2", score=0.1),
            AcquisitionScore(candidate_id="c-3", score=0.9),
        ]

    order: list[list[str]] = []

    def simulate(requests: tuple, _state):
        order.append([request.candidate.candidate_id for request in requests])
        return [
            SimulationResult(
                request_id=request.request_id,
                candidate_id=request.candidate.candidate_id,
                status="success",
                metrics={"request": idx},
            )
            for idx, request in enumerate(requests)
        ]

    retrain, retrain_calls = _build_fake_retrainer()
    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=1),
        batch_size=2,
    )
    state = engine.run()
    assert order == [["c-1", "c-3"]]
    assert state.selected_candidate_ids == ("c-1", "c-3")
    assert len(state.simulation_requests) == 2
    assert len(retrain_calls) == 1


def test_batch_selection_is_capped_by_remaining_evaluation_budget() -> None:
    def candidates(_: LoopState) -> list[Candidate]:
        return [
            Candidate(candidate_id=f"budget-{idx}", parameters={"x": idx})
            for idx in range(5)
        ]

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [
            AcquisitionScore(candidate_id=candidate.candidate_id, score=float(10 - idx))
            for idx, candidate in enumerate(candidates)
        ]

    simulated_batches: list[list[str]] = []

    def simulate(requests: tuple, _state):
        simulated_batches.append([request.candidate.candidate_id for request in requests])
        return [
            SimulationResult(
                request_id=request.request_id,
                candidate_id=request.candidate.candidate_id,
                status="success",
                metrics={},
            )
            for request in requests
        ]

    retrain, _ = _build_fake_retrainer()
    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=3, max_evaluations=2),
        batch_size=5,
    )

    state = engine.run()

    assert simulated_batches == [["budget-0", "budget-1"]]
    assert len(state.simulation_requests) == 2
    assert state.completion_reason == "max_evaluations_reached"


def test_simulation_results_and_retraining_results_are_honored() -> None:
    def candidates(_: LoopState) -> list[Candidate]:
        return [Candidate(candidate_id="r-0", parameters={"x": 1}), Candidate(candidate_id="r-1", parameters={"x": 2})]

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [
            AcquisitionScore(candidate_id="r-0", score=0.5),
            AcquisitionScore(candidate_id="r-1", score=0.7),
        ]

    def simulate(requests: tuple, _state):
        return [
            SimulationResult(
                request_id=request.request_id,
                candidate_id=request.candidate.candidate_id,
                status="success" if request.candidate.candidate_id == "r-1" else "failed",
                metrics={"loss": 1.0 if request.candidate.candidate_id == "r-1" else 2.0},
                error="failed on purpose" if request.candidate.candidate_id == "r-0" else None,
            )
            for request in requests
        ]

    retrain, retrain_calls = _build_fake_retrainer()
    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=1),
        batch_size=2,
    )
    state = engine.run()
    assert len(state.simulation_results) == 2
    assert {result.status for result in state.simulation_results} == {"success", "failed"}
    assert state.retraining_results[0].status == "completed"
    assert state.retraining_results[0].metrics["count"] == 1
    assert retrain_calls[0][0].iteration == 0


def test_resume_from_json_state_continues_loop() -> None:
    call_order: list[int] = []

    def candidates(state: LoopState) -> list[Candidate]:
        call_order.append(state.iteration)
        return [Candidate(candidate_id=f"candidate-{state.iteration}", parameters={"step": state.iteration})]

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [AcquisitionScore(candidate_id=candidate.candidate_id, score=1.0 / (idx + 1)) for idx, candidate in enumerate(candidates)]

    sim_calls: list[str] = []

    def simulate(requests: tuple, _state):
        sim_calls.append(_state.loop_id)
        return [
            SimulationResult(
                request_id=request.request_id,
                candidate_id=request.candidate.candidate_id,
                status="success",
                metrics={"ok": True},
            )
            for request in requests
        ]

    retrain, _ = _build_fake_retrainer()
    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=1),
        batch_size=1,
        loop_id="resume-loop",
    )
    first = engine.run()
    payload = first.to_json()
    loaded = LoopState.from_json(payload)
    assert loaded == first

    resuming = replace(loaded, completion_reason=None)
    resumed = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=2),
        batch_size=1,
        initial_state=resuming,
    )
    final = resumed.run()
    assert final.iteration == 2
    assert len(final.simulation_results) == 2
    assert call_order == [0, 1]


def test_failure_records_capture_retrainer_errors() -> None:
    def candidates(_: LoopState) -> list[Candidate]:
        return [Candidate(candidate_id="bad-retrain", parameters={"x": 1})]

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [AcquisitionScore(candidate_id="bad-retrain", score=0.0)]

    def simulate(requests: tuple, _state):
        return [
            SimulationResult(
                request_id=requests[0].request_id,
                candidate_id="bad-retrain",
                status="success",
                metrics={},
            )
        ]

    def retrain(_request: RetrainingRequest, _state) -> RetrainingResult:
        raise RuntimeError("retraining is intentionally broken")

    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=1),
        batch_size=1,
    )
    state = engine.run()
    assert any(record.stage == "retraining" for record in state.failures)
    assert state.retraining_results[0].status == "failed"
    assert "retrainer exception" in state.retraining_results[0].metrics["error"]
    assert isinstance(json.loads(state.to_json()), dict)


def test_failure_records_capture_candidate_generator_errors() -> None:
    def candidates(_: LoopState) -> list[Candidate]:
        raise RuntimeError("candidate generation is intentionally broken")

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [AcquisitionScore(candidate_id=candidate.candidate_id, score=0.0) for candidate in candidates]

    def simulate(requests: tuple, _state):
        return []

    retrain, _ = _build_fake_retrainer()
    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=retrain,
        stopping_criteria=StoppingCriteria(max_iterations=1),
    )

    state = engine.run()

    assert len(state.failures) == 1
    assert state.failures[0].stage == "candidate_generation"
    assert "candidate generation is intentionally broken" in state.failures[0].message
