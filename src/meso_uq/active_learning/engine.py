from __future__ import annotations

"""Pure-python dry-run active-learning engine."""

from dataclasses import replace
from typing import Callable, Mapping, Sequence

from meso_uq.active_learning.contracts import (
    AcquisitionScore,
    Candidate,
    FailureRecord,
    LoopState,
    RetrainingRequest,
    RetrainingResult,
    SimulationRequest,
    SimulationResult,
    StoppingCriteria,
    as_candidate,
    as_scores,
    as_simulation_results,
    as_retraining_result,
)

CandidateGenerator = Callable[[LoopState], Sequence[object]]
AcquisitionPolicy = Callable[[tuple[Candidate, ...], LoopState], Sequence[object]]
Simulator = Callable[[tuple[SimulationRequest, ...], LoopState], Sequence[object]]
Retrainer = Callable[[RetrainingRequest, LoopState], object]


class ActiveLearningDryRunEngine:
    """Dry-run orchestrator that never launches platform execution."""

    def __init__(
        self,
        *,
        candidate_generator: CandidateGenerator,
        acquisition_policy: AcquisitionPolicy,
        simulator: Simulator,
        retrainer: Retrainer,
        stopping_criteria: StoppingCriteria,
        batch_size: int = 1,
        loop_id: str = "active-learning-loop",
        initial_state: LoopState | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self._candidate_generator = candidate_generator
        self._acquisition_policy = acquisition_policy
        self._simulator = simulator
        self._retrainer = retrainer
        self._stopping_criteria = stopping_criteria
        self._batch_size = batch_size
        self._state = (
            initial_state
            if initial_state is not None
            else LoopState(loop_id=loop_id, iteration=0)
        )

    @property
    def state(self) -> LoopState:
        return self._state

    def run(self) -> LoopState:
        if self._state.completion_reason is not None:
            return self._state
        while True:
            stop_reason = self._stopping_criteria.should_stop(self._state)
            if stop_reason is not None:
                self._state = replace(self._state, completion_reason=stop_reason)
                break
            self._state = self._run_single_iteration()
            if self._state.completion_reason is not None:
                break
        return self._state

    def _run_single_iteration(self) -> LoopState:
        state = self._state
        generated, candidate_generation_failures = self._generate_candidates(state)
        state = self._append_failures(state, candidate_generation_failures)
        valid, generation_failures = self._validate_candidates(generated, state.iteration)
        state = self._append_failures(state, generation_failures)
        scores, acquisition_failures = self._score_candidates(valid, state)
        state = self._append_failures(state, acquisition_failures)

        selected = self._select_batch(scores, valid, set(state.evaluated_candidate_ids))
        requests = tuple(
            SimulationRequest(
                request_id=f"{state.loop_id}-iter-{state.iteration}-cand-{idx+1}",
                candidate=candidate,
                iteration=state.iteration,
            )
            for idx, candidate in enumerate(selected)
        )

        results, simulation_failures = self._simulate(requests, state)
        state = self._append_failures(state, simulation_failures)

        state = replace(
            state,
            candidates=state.candidates + valid,
            scored_candidates=state.scored_candidates + scores,
            selected_candidate_ids=state.selected_candidate_ids
            + tuple(candidate.candidate_id for candidate in selected),
            simulation_requests=state.simulation_requests + requests,
            simulation_results=state.simulation_results + results,
        )
        state = self._retrain_if_possible(state, requests)
        state = replace(state, iteration=state.iteration + 1)
        return state

    def _generate_candidates(
        self, state: LoopState
    ) -> tuple[tuple[object, ...], tuple[FailureRecord, ...]]:
        try:
            payload = self._candidate_generator(state)
        except Exception as exc:
            return (), (
                FailureRecord(
                    stage="candidate_generation",
                    message=f"candidate generator raised: {exc}",
                    iteration=state.iteration,
                ),
            )
        if not isinstance(payload, Sequence):
            return (), (
                FailureRecord(
                    stage="candidate_generation",
                    message="candidate_generator must return a sequence.",
                    iteration=state.iteration,
                ),
            )
        return tuple(payload), ()

    def _validate_candidates(
        self, payload: Sequence[object], iteration: int
    ) -> tuple[tuple[Candidate, ...], tuple[FailureRecord, ...]]:
        valid: list[Candidate] = []
        failures: list[FailureRecord] = []
        for item in payload:
            try:
                candidate = as_candidate(item)
            except Exception as exc:
                candidate_id = item["candidate_id"] if isinstance(item, Mapping) else None
                failures.append(
                    FailureRecord(
                        stage="validation",
                        message=str(exc),
                        iteration=iteration,
                        candidate_id=str(candidate_id) if candidate_id is not None else None,
                    )
                )
                continue
            valid.append(candidate)
        return tuple(valid), tuple(failures)

    def _score_candidates(
        self, candidates: tuple[Candidate, ...], state: LoopState
    ) -> tuple[tuple[AcquisitionScore, ...], tuple[FailureRecord, ...]]:
        if not candidates:
            return (), ()
        try:
            payload = self._acquisition_policy(candidates, state)
            scores = as_scores(payload)
        except Exception as exc:
            return (), (
                FailureRecord(
                    stage="acquisition",
                    message=str(exc),
                    iteration=state.iteration,
                    details={"candidate_count": len(candidates)},
                ),
            )
        return tuple(scores), ()

    def _select_batch(
        self,
        scores: tuple[AcquisitionScore, ...],
        candidates: tuple[Candidate, ...],
        already_evaluated: set[str],
    ) -> tuple[Candidate, ...]:
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in candidates if candidate.candidate_id not in already_evaluated
        }
        ordered = sorted(scores, key=lambda item: item.score, reverse=True)
        selected: list[Candidate] = []
        seen: set[str] = set()
        for score in ordered:
            if score.candidate_id in seen or score.candidate_id in already_evaluated:
                continue
            candidate = candidates_by_id.get(score.candidate_id)
            if candidate is None:
                continue
            selected.append(candidate)
            seen.add(candidate.candidate_id)
            if len(selected) >= self._batch_size:
                break
        return tuple(selected)

    def _simulate(
        self, requests: tuple[SimulationRequest, ...], state: LoopState
    ) -> tuple[tuple[SimulationResult, ...], tuple[FailureRecord, ...]]:
        if not requests:
            return (), ()

        requests_by_id = {request.request_id: request for request in requests}
        try:
            payload = self._simulator(requests, state)
            results = as_simulation_results(payload)
        except Exception as exc:
            failure_results = tuple(
                SimulationResult(
                    request_id=request.request_id,
                    candidate_id=request.candidate.candidate_id,
                    status="failed",
                    metrics={},
                    error=f"simulation raised: {exc}",
                    metadata={"stage": "simulator_exception"},
                )
                for request in requests
            )
            return failure_results, (
                FailureRecord(
                    stage="simulation",
                    message=f"simulator raised: {exc}",
                    iteration=state.iteration,
                    details={"request_count": len(requests)},
                ),
            )

        resolved: list[SimulationResult] = []
        for result in results:
            if result.request_id not in requests_by_id:
                continue
            resolved.append(result)
            requests_by_id.pop(result.request_id, None)

        failures: list[FailureRecord] = []
        for missing in requests_by_id.values():
            failures.append(
                FailureRecord(
                    stage="simulation",
                    message=f"missing simulation result for {missing.request_id}",
                    iteration=state.iteration,
                    candidate_id=missing.candidate.candidate_id,
                )
            )
            resolved.append(
                SimulationResult(
                    request_id=missing.request_id,
                    candidate_id=missing.candidate.candidate_id,
                    status="failed",
                    metrics={},
                    error=f"missing simulation result for {missing.request_id}",
                )
            )
        return tuple(resolved), tuple(failures)

    def _retrain_if_possible(self, state: LoopState, requests: tuple[SimulationRequest, ...]) -> LoopState:
        request = RetrainingRequest(
            iteration=state.iteration,
            successful_results=tuple(
                result
                for result in state.simulation_results
                if result.request_id.startswith(f"{state.loop_id}-iter-{state.iteration}-") and result.is_success
            ),
            candidate_ids=tuple(item.candidate.candidate_id for item in requests),
        )
        try:
            retrain_results = self._retrainer(request, state)
            retrain_result = as_retraining_result(retrain_results)
        except Exception as exc:
            failure = FailureRecord(
                stage="retraining",
                message=f"retrainer raised: {exc}",
                iteration=state.iteration,
            )
            state = self._append_failures(
                state,
                (
                    failure,
                ),
            )
            retrain_result = RetrainingResult(
                iteration=state.iteration,
                status="failed",
                model_token="dry-run-retrain-failed",
                metrics={"error": "retrainer exception"},
                metadata={"error": str(exc)},
            )
        return replace(state, retraining_results=state.retraining_results + (retrain_result,))

    def _append_failures(self, state: LoopState, failures: tuple[FailureRecord, ...]) -> LoopState:
        if not failures:
            return state
        return replace(state, failures=state.failures + failures)
