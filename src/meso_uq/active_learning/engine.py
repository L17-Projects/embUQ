from __future__ import annotations

"""Pure-python dry-run active-learning engine."""

import hashlib
import json
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from meso_uq.active_learning.contracts import (
    AcquisitionScore,
    Candidate,
    ActiveLearningConfig,
    ActiveLearningIterationLineage,
    FailureRecord,
    StoppingCriteria,
    LoopState,
    ActiveLearningRuntimeConfig,
    RetrainingRequest,
    RetrainingResult,
    SimulationRequest,
    SimulationResult,
    ActiveLearningBudgetConfig,
    ActiveLearningOutputConfig,
    ActiveLearningPlatformConfig,
    as_candidate,
    as_scores,
    as_simulation_results,
    as_retraining_result,
    candidate_hash,
)
from meso_uq.active_learning.artifacts import write_iteration_artifacts

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
        active_learning_config: ActiveLearningConfig | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self._candidate_generator = candidate_generator
        self._acquisition_policy = acquisition_policy
        self._simulator = simulator
        self._retrainer = retrainer
        preserve_config_loop_id = (
            active_learning_config is not None
            and initial_state is None
            and loop_id == "active-learning-loop"
        )
        effective_loop_id = (
            initial_state.loop_id
            if initial_state is not None
            else (
                active_learning_config.loop_id
                if preserve_config_loop_id
                else loop_id
            )
        )
        self._active_learning_config = (
            active_learning_config
            if active_learning_config is not None
            else ActiveLearningConfig(
                budget=ActiveLearningBudgetConfig(
                    max_iterations=stopping_criteria.max_iterations,
                    max_evaluations=stopping_criteria.max_evaluations,
                    max_failures=stopping_criteria.max_failures,
                ),
                runtime=dict(batch_size=batch_size),
                platform=ActiveLearningPlatformConfig(platform="workstation", walltime="00:30:00", gpu_count=1),
                output=ActiveLearningOutputConfig(output_root="_runs/active_learning"),
                loop_id=effective_loop_id,
            )
        )
        legacy_batch_size_was_set = batch_size != 1
        should_override_batch_size = active_learning_config is None or legacy_batch_size_was_set
        if should_override_batch_size and batch_size != self._active_learning_config.runtime.batch_size:
            self._active_learning_config = replace(
                self._active_learning_config,
                runtime=ActiveLearningRuntimeConfig(batch_size=batch_size),
            )
        self._active_learning_config = replace(
            self._active_learning_config,
            run_id=(
                initial_state.run_id
                if initial_state and initial_state.run_id is not None
                else self._active_learning_config.run_id
            ),
        )
        self._stopping_criteria = StoppingCriteria(
            max_iterations=self._active_learning_config.budget.max_iterations,
            max_evaluations=self._active_learning_config.budget.max_evaluations,
            max_failures=self._active_learning_config.budget.max_failures,
        )
        self._batch_size = self._active_learning_config.runtime.batch_size
        base_loop_state = (
            initial_state
            if initial_state is not None
            else LoopState(loop_id=effective_loop_id, iteration=0, run_id=self._active_learning_config.run_id)
        )
        self._state = (
            replace(base_loop_state, run_id=self._active_learning_config.run_id)
            if base_loop_state.run_id != self._active_learning_config.run_id
            else base_loop_state
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
        iteration_failures: list[FailureRecord] = []

        generated, candidate_generation_failures = self._generate_candidates(state)
        iteration_failures.extend(candidate_generation_failures)
        state = self._append_failures(state, candidate_generation_failures)

        valid, generation_failures = self._validate_candidates(generated, state.iteration)
        iteration_failures.extend(generation_failures)
        state = self._append_failures(state, generation_failures)

        scores, acquisition_failures = self._score_candidates(valid, state)
        iteration_failures.extend(acquisition_failures)
        state = self._append_failures(state, acquisition_failures)

        selected = self._select_batch(
            scores,
            valid,
            set(state.evaluated_candidate_ids),
            limit=self._remaining_batch_budget(state),
        )
        requests = tuple(
            SimulationRequest(
                request_id=f"{state.loop_id}-iter-{state.iteration}-cand-{idx+1}",
                candidate=candidate,
                iteration=state.iteration,
            )
            for idx, candidate in enumerate(selected)
        )

        results, simulation_failures = self._simulate(requests, state)
        iteration_failures.extend(simulation_failures)
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
        retrain_state, retrain_failures = self._retrain_if_possible(state, requests)
        iteration_failures.extend(retrain_failures)
        state = self._append_failures(retrain_state, retrain_failures)
        iteration_scores = {
            score.candidate_id: score.score
            for score in scores
            if score.candidate_id in {candidate.candidate_id for candidate in selected}
        }
        state = self._append_iteration_lineage(
            state=state,
            selected=selected,
            results=results,
            failure_records=tuple(iteration_failures),
            stage_scores=iteration_scores,
        )
        state = self._emit_iteration_artifacts(
            state,
            stage_scores=iteration_scores,
            selected_candidate_ids=tuple(candidate.candidate_id for candidate in selected),
        )
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

    def _append_iteration_lineage(
        self,
        *,
        state: LoopState,
        selected: tuple[Candidate, ...],
        results: tuple[SimulationResult, ...],
        failure_records: tuple[FailureRecord, ...],
        stage_scores: Mapping[str, float],
    ) -> LoopState:
        selected_hashes = tuple(candidate_hash(candidate) for candidate in selected)
        success_count = sum(1 for result in results if result.is_success)
        request_count = len(results)
        parent = state.latest_iteration_lineage
        payload = {
            "run_id": state.run_id,
            "iteration": state.iteration,
            "parent_iteration_id": parent.iteration_id if parent else None,
            "selected_hashes": list(selected_hashes),
            "request_count": request_count,
            "success_count": success_count,
            "failure_count": len(failure_records),
            "stage_scores": sorted(stage_scores.items()),
        }
        lineage_id = self._stable_id(payload)
        lineage = ActiveLearningIterationLineage(
            iteration=state.iteration,
            iteration_id=f"iter-{state.iteration:04d}-{lineage_id}",
            run_id=state.run_id or self._active_learning_config.run_id or "active-learning",
            selected_candidate_hashes=selected_hashes,
            request_count=request_count,
            success_count=success_count,
            failure_count=len(failure_records),
            parent_iteration_id=parent.iteration_id if parent else None,
        )
        return replace(state, iteration_lineage=state.iteration_lineage + (lineage,))

    def _stable_id(self, payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

    def _iteration_failures(self, state: LoopState, *, iteration: int) -> tuple[FailureRecord, ...]:
        return tuple(
            failure
            for failure in state.failures
            if failure.iteration == iteration
        )

    def _emit_iteration_artifacts(
        self,
        state: LoopState,
        *,
        stage_scores: Mapping[str, float],
        selected_candidate_ids: tuple[str, ...],
    ) -> LoopState:
        if not self._active_learning_config.output.emit_artifacts:
            return state
        output_root = self._active_learning_config.output.output_root
        if state.latest_iteration_lineage is None:
            return state
        write_iteration_artifacts(
            output_root=output_root,
            run_id=str(state.run_id or self._active_learning_config.run_id),
            iteration=state.iteration,
            lineage=state.latest_iteration_lineage,
            stage_failures=self._iteration_failures(state, iteration=state.iteration),
            stage_scores=stage_scores,
            selected_candidate_ids=selected_candidate_ids,
            include_plot=self._active_learning_config.output.emit_plots,
        )
        return state

    def _select_batch(
        self,
        scores: tuple[AcquisitionScore, ...],
        candidates: tuple[Candidate, ...],
        already_evaluated: set[str],
        *,
        limit: int,
    ) -> tuple[Candidate, ...]:
        if limit < 1:
            return ()
        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in candidates
            if candidate.candidate_id not in already_evaluated
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
            if len(selected) >= limit:
                break
        return tuple(selected)

    def _remaining_batch_budget(self, state: LoopState) -> int:
        if self._stopping_criteria.max_evaluations is None:
            return self._batch_size
        remaining = self._stopping_criteria.max_evaluations - state.evaluated_candidate_count
        return max(0, min(self._batch_size, remaining))

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

    def _retrain_if_possible(
        self,
        state: LoopState,
        requests: tuple[SimulationRequest, ...],
    ) -> tuple[LoopState, tuple[FailureRecord, ...]]:
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
            retrain_result = RetrainingResult(
                iteration=state.iteration,
                status="failed",
                model_token="dry-run-retrain-failed",
                metrics={"error": "retrainer exception"},
                metadata={"error": str(exc)},
            )
            return replace(state, retraining_results=state.retraining_results + (retrain_result,)), (failure,)
        return replace(state, retraining_results=state.retraining_results + (retrain_result,)), ()

    def _append_failures(self, state: LoopState, failures: tuple[FailureRecord, ...]) -> LoopState:
        if not failures:
            return state
        return replace(state, failures=state.failures + failures)
