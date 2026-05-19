from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import ActiveLearningDryRunEngine
from meso_uq.active_learning.artifacts import (
    ACTIVE_LEARNING_ITERATION_MANIFEST_FILENAME,
    ACTIVE_LEARNING_PLOT_FILENAME,
    ACTIVE_LEARNING_STAGE_REPORT_FILENAME,
    write_iteration_artifacts,
)
from meso_uq.active_learning.contracts import (
    ActiveLearningBudgetConfig,
    ActiveLearningConfig,
    ActiveLearningIterationLineage,
    ActiveLearningOutputConfig,
    ActiveLearningPlatformConfig,
    ActiveLearningRuntimeConfig,
    AcquisitionScore,
    Candidate,
    FailureRecord,
    LoopState,
    StoppingCriteria,
    RetrainingRequest,
    RetrainingResult,
    SimulationResult,
)


def _fake_retrainer() -> tuple:
    def retrain(request: RetrainingRequest, state) -> RetrainingResult:
        return RetrainingResult(
            iteration=request.iteration,
            status="completed",
            model_token=f"model-{request.iteration}",
            metrics={"batch": len(request.candidate_ids)},
            metadata={},
        )

    return retrain


def _png_is_loadable(data: bytes) -> bool:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    idx = 8
    idat: list[bytes] = []
    has_iend = False
    width: int | None = None
    height: int | None = None
    bit_depth: int | None = None
    color_type: int | None = None
    while idx < len(data):
        if idx + 8 > len(data):
            return False
        length = struct.unpack(">I", data[idx : idx + 4])[0]
        idx += 4
        chunk_type = data[idx : idx + 4]
        idx += 4
        chunk_data = data[idx : idx + length]
        if len(chunk_data) != length:
            return False
        idx += length
        chunk_crc = struct.unpack(">I", data[idx : idx + 4])[0]
        idx += 4
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != chunk_crc:
            return False
        if chunk_type == b"IHDR":
            if length != 13:
                return False
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            if bit_depth not in (1, 2, 4, 8, 16):
                return False
            if color_type not in (0, 2, 3, 4, 6):
                return False
        elif chunk_type == b"IDAT":
            idat.append(chunk_data)
        if chunk_type == b"IEND":
            has_iend = True
            break
    if not has_iend:
        return False
    if idx != len(data):
        return False
    if not idat:
        return False
    if width is None or height is None or bit_depth is None or color_type is None:
        return False
    decompressed = zlib.decompress(b"".join(idat))
    bytes_per_pixel = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    expected_row = 1 + width * bytes_per_pixel
    expected_unfiltered = expected_row * height
    return len(decompressed) == expected_unfiltered


def test_active_learning_config_roundtrip_and_validation(tmp_path: Path) -> None:
    config = ActiveLearningConfig(
        loop_id="foundation-loop",
        budget=ActiveLearningBudgetConfig(max_iterations=2, max_evaluations=4, max_failures=1),
        platform=ActiveLearningPlatformConfig(platform="workstation", walltime="01:02:03", gpu_count=1),
        runtime=ActiveLearningRuntimeConfig(batch_size=2),
        output=ActiveLearningOutputConfig(
            output_root=str(tmp_path / "_runs" / "active_learning"),
            emit_artifacts=True,
            emit_plots=False,
        ),
        metadata={"seed": 7, "flags": ("alpha", "beta")},
    )
    payload = config.to_json()
    loaded = ActiveLearningConfig.from_json(payload)
    assert loaded == config
    assert loaded.loop_id == "foundation-loop"
    assert loaded.run_id == config.run_id

    with pytest.raises(ValueError, match="max_iterations"):
        ActiveLearningConfig(budget=ActiveLearningBudgetConfig(max_iterations=0))

    with pytest.raises(ValueError, match="walltime"):
        ActiveLearningConfig(platform=ActiveLearningPlatformConfig(platform="workstation", walltime="01:80:00"))

    repeat = ActiveLearningConfig(
        loop_id="foundation-loop",
        budget=ActiveLearningBudgetConfig(max_iterations=2, max_evaluations=4, max_failures=1),
        platform=ActiveLearningPlatformConfig(platform="workstation", walltime="01:02:03", gpu_count=1),
        runtime=ActiveLearningRuntimeConfig(batch_size=2),
        output=ActiveLearningOutputConfig(
            output_root=str(tmp_path / "_runs" / "active_learning"),
            emit_artifacts=True,
            emit_plots=False,
        ),
        metadata={"seed": 7, "flags": ("alpha", "beta")},
    )
    assert repeat.run_id == config.run_id

    defaults = ActiveLearningConfig.from_dict({"output": {"output_root": str(tmp_path)}})
    assert defaults.budget.max_iterations == 1
    assert defaults.runtime.batch_size == 1
    assert defaults.output.output_root == str(tmp_path)


def test_candidate_hash_and_state_roundtrip_with_lineage() -> None:
    candidate_a = Candidate(candidate_id="a", parameters={"x": 1, "y": 2}, metadata={"m": {"values": [1, 2]}})
    candidate_b = Candidate(candidate_id="a", parameters={"y": 2, "x": 1}, metadata={"m": {"values": [1, 2]}})
    assert candidate_a.stable_hash() == candidate_b.stable_hash()
    assert candidate_a.stable_hash(length=12) == candidate_b.stable_hash(length=12)

    lineage = ActiveLearningIterationLineage(
        iteration=0,
        iteration_id="iter-0000-abcdef123456",
        run_id="run-001",
        selected_candidate_hashes=(candidate_a.stable_hash(),),
        request_count=1,
        success_count=1,
        failure_count=0,
    )

    state = LoopState(
        loop_id="foundation-loop",
        iteration=1,
        run_id="run-001",
        candidates=(candidate_a,),
        iteration_lineage=(lineage,),
        completion_reason="max_iterations_reached",
    )
    payload = state.to_json()
    assert json.loads(payload)["run_id"] == "run-001"
    assert LoopState.from_json(payload) == state


def test_write_iteration_artifacts_generates_manifest_and_sidecar(tmp_path: Path) -> None:
    lineage = ActiveLearningIterationLineage(
        iteration=0,
        iteration_id="iter-0000-abcdef123456",
        run_id="run-001",
        selected_candidate_hashes=("h1", "h2"),
        request_count=2,
        success_count=1,
        failure_count=1,
        parent_iteration_id=None,
    )
    result = write_iteration_artifacts(
        output_root=str(tmp_path / "_runs" / "active_learning"),
        run_id="run-001",
        iteration=0,
        lineage=lineage,
        stage_failures=(
            FailureRecord(stage="validation", message="bad candidate", iteration=0),
        ),
        stage_scores={"candidate-a": 0.75, "candidate-b": 0.25},
        selected_candidate_ids=("candidate-a", "candidate-b"),
        include_plot=True,
    )

    assert result.iteration_dir == tmp_path / "_runs" / "active_learning" / "run-001" / "iterations" / "iter_0000"
    assert result.iteration_dir.exists()
    assert result.manifest_path.name == ACTIVE_LEARNING_ITERATION_MANIFEST_FILENAME
    assert result.sidecar_path.name == ACTIVE_LEARNING_STAGE_REPORT_FILENAME
    assert result.plot_path.name == ACTIVE_LEARNING_PLOT_FILENAME
    assert result.manifest_path.is_file() and result.manifest_path.stat().st_size > 0
    assert result.sidecar_path.is_file() and result.sidecar_path.stat().st_size > 0
    assert result.plot_path.is_file() and result.plot_path.stat().st_size > 0

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert manifest["iteration_id"] == lineage.iteration_id
    assert manifest["candidate_count"] == 2
    assert sidecar["selected_candidates"] == ["candidate-a", "candidate-b"]
    assert sidecar["selected_scores"] == [0.75, 0.25]


def test_write_iteration_artifacts_emits_valid_fallback_png_without_matplotlib(tmp_path: Path) -> None:
    lineage = ActiveLearningIterationLineage(
        iteration=0,
        iteration_id="iter-0000-abcdef123456",
        run_id="run-001",
        selected_candidate_hashes=("h1",),
        request_count=1,
        success_count=1,
        failure_count=0,
    )
    result = write_iteration_artifacts(
        output_root=str(tmp_path / "_runs" / "active_learning"),
        run_id="run-001",
        iteration=0,
        lineage=lineage,
        stage_failures=(),
        stage_scores={"candidate-a": 0.75},
        selected_candidate_ids=("candidate-a",),
        include_plot=False,
    )

    assert result.plot_path.is_file()
    assert _png_is_loadable(result.plot_path.read_bytes())


def test_active_learning_engine_emits_deterministic_lineage_artifacts(tmp_path: Path) -> None:
    def candidates(state: LoopState) -> list[Candidate]:
        return [
            Candidate(candidate_id=f"candidate-{state.iteration}", parameters={"x": state.iteration}),
            Candidate(candidate_id=f"candidate-alt-{state.iteration}", parameters={"x": state.iteration + 1}),
        ]

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [AcquisitionScore(candidate_id=item.candidate_id, score=float(100 - idx)) for idx, item in enumerate(candidates)]

    def simulate(requests: tuple, _state: LoopState):
        return [
            SimulationResult(
                request_id=request.request_id,
                candidate_id=request.candidate.candidate_id,
                status="success",
                metrics={"step": request.iteration},
            )
            for request in requests
        ]

    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=_fake_retrainer(),
        stopping_criteria=StoppingCriteria(max_iterations=2),
        batch_size=1,
        loop_id="foundation-loop",
        active_learning_config=ActiveLearningConfig(
            loop_id="foundation-loop",
            budget=ActiveLearningBudgetConfig(max_iterations=2, max_evaluations=2),
            runtime=ActiveLearningRuntimeConfig(batch_size=1),
            output=ActiveLearningOutputConfig(
                output_root=str(tmp_path / "_runs" / "active_learning"),
                emit_artifacts=True,
                emit_plots=False,
            ),
            run_id="run-002",
        ),
    )
    state = engine.run()
    assert state.iteration == 2
    assert len(state.iteration_lineage) == 2
    assert state.iteration_lineage[0].parent_iteration_id is None
    assert state.iteration_lineage[1].parent_iteration_id == state.iteration_lineage[0].iteration_id

    for index in range(2):
        iteration_dir = (
            tmp_path / "_runs" / "active_learning" / "run-002" / "iterations" / f"iter_{index:04d}"
        )
        manifest = iteration_dir / ACTIVE_LEARNING_ITERATION_MANIFEST_FILENAME
        sidecar = iteration_dir / ACTIVE_LEARNING_STAGE_REPORT_FILENAME
        plot = iteration_dir / ACTIVE_LEARNING_PLOT_FILENAME

        assert manifest.is_file() and manifest.stat().st_size > 0
        assert sidecar.is_file() and sidecar.stat().st_size > 0
        assert plot.is_file() and plot.stat().st_size > 0
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest_payload["run_id"] == "run-002"
        assert manifest_payload["iteration"] == index
        assert manifest_payload["candidate_count"] == 1


def test_active_learning_engine_preserves_active_learning_config_loop_id_when_no_override(tmp_path: Path) -> None:
    engine = ActiveLearningDryRunEngine(
        candidate_generator=lambda _: [],
        acquisition_policy=lambda _, __: [],
        simulator=lambda _, __: [],
        retrainer=_fake_retrainer(),
        stopping_criteria=StoppingCriteria(max_iterations=1),
        active_learning_config=ActiveLearningConfig(
            loop_id="from-config",
            budget=ActiveLearningBudgetConfig(max_iterations=1, max_evaluations=2),
            runtime=ActiveLearningRuntimeConfig(batch_size=1),
            output=ActiveLearningOutputConfig(
                output_root=str(tmp_path / "_runs" / "active_learning"),
                emit_artifacts=False,
            ),
        ),
    )
    assert engine.state.loop_id == "from-config"


def test_active_learning_engine_preserves_active_learning_config_batch_size_with_default_legacy_value(tmp_path: Path) -> None:
    def candidates(_: LoopState) -> list[Candidate]:
        return [
            Candidate(candidate_id=f"candidate-{idx}", parameters={"x": idx})
            for idx in range(5)
        ]

    requests: list[int] = []

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [
            AcquisitionScore(candidate_id=candidate.candidate_id, score=float(idx))
            for idx, candidate in enumerate(candidates)
        ]

    def simulate(requests_in: tuple, _state: LoopState):
        requests.append(len(requests_in))
        return [
            SimulationResult(
                request_id=request.request_id,
                candidate_id=request.candidate.candidate_id,
                status="success",
                metrics={"candidate": request.candidate.candidate_id},
            )
            for request in requests_in
        ]

    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=_fake_retrainer(),
        stopping_criteria=StoppingCriteria(max_iterations=1),
        active_learning_config=ActiveLearningConfig(
            loop_id="batch-config",
            budget=ActiveLearningBudgetConfig(max_iterations=1, max_evaluations=10),
            runtime=ActiveLearningRuntimeConfig(batch_size=4),
            output=ActiveLearningOutputConfig(
                output_root=str(tmp_path / "_runs" / "active_learning"),
                emit_artifacts=False,
            ),
        ),
    )

    engine.run()
    assert requests == [4]


def test_active_learning_engine_uses_active_learning_config_budget(tmp_path: Path) -> None:
    def candidates(_: LoopState) -> list[Candidate]:
        return [
            Candidate(candidate_id=f"candidate-{idx}", parameters={"x": idx})
            for idx in range(5)
        ]

    simulated_batches: list[int] = []

    def score(candidates: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return [
            AcquisitionScore(candidate_id=candidate.candidate_id, score=float(100 - idx))
            for idx, candidate in enumerate(candidates)
        ]

    def simulate(requests_in: tuple, _state: LoopState):
        simulated_batches.append(len(requests_in))
        return [
            SimulationResult(
                request_id=request.request_id,
                candidate_id=request.candidate.candidate_id,
                status="success",
                metrics={"candidate": request.candidate.candidate_id},
            )
            for request in requests_in
        ]

    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=_fake_retrainer(),
        stopping_criteria=StoppingCriteria(max_iterations=3, max_evaluations=5),
        active_learning_config=ActiveLearningConfig(
            loop_id="budget-config",
            budget=ActiveLearningBudgetConfig(max_iterations=1, max_evaluations=1),
            runtime=ActiveLearningRuntimeConfig(batch_size=4),
            output=ActiveLearningOutputConfig(
                output_root=str(tmp_path / "_runs" / "active_learning"),
                emit_artifacts=False,
            ),
        ),
    )

    state = engine.run()
    assert state.iteration == 1
    assert state.evaluated_candidate_count == 1
    assert state.completion_reason == "max_iterations_reached"
    assert simulated_batches == [1]


def test_active_learning_engine_default_config_uses_resumed_loop_id() -> None:
    def candidates(_: LoopState) -> list[Candidate]:
        return []

    def score(_: tuple[Candidate, ...], _state: LoopState) -> list[AcquisitionScore]:
        return []

    def simulate(_: tuple, _state: LoopState):
        return []

    initial_state = LoopState(loop_id="resume-loop", iteration=1)
    engine = ActiveLearningDryRunEngine(
        candidate_generator=candidates,
        acquisition_policy=score,
        simulator=simulate,
        retrainer=_fake_retrainer(),
        stopping_criteria=StoppingCriteria(max_iterations=1),
        initial_state=initial_state,
    )

    assert engine.state.loop_id == "resume-loop"
    assert engine.state.run_id is not None
    assert engine.state.run_id.startswith("resume-loop-")


def test_active_learning_artifacts_module_import_does_not_load_matplotlib() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    code = """
import importlib
import sys

importlib.import_module("meso_uq.active_learning.artifacts")
print("matplotlib" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
