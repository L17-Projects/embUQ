from __future__ import annotations

import builtins
import json
import os
import subprocess
import struct
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import (
    ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_FILENAME,
    ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_SIDECAR_FILENAME,
    SyntheticBenchmarkConfig,
    generate_synthetic_benchmark,
    write_synthetic_benchmark_artifacts,
    compare_to_baselines,
)


def _make_config(*, family: str, seed: int = 19) -> SyntheticBenchmarkConfig:
    return SyntheticBenchmarkConfig(
        family=family,
        experiment="stretching" if family == "gv" else "indentation",
        count=4,
        seed=seed,
        sequence_name="random",
        completion_steps=6,
        run_id=f"synthetic-{family}-test",
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return struct.unpack(">II", data[16:24])


def _png_is_visual(path: Path) -> bool:
    width, height = _png_dimensions(path)
    return width >= 600 and height >= 400 and path.stat().st_size > 1024


def test_generate_synthetic_benchmark_is_deterministic() -> None:
    config = _make_config(family="gv")
    first = generate_synthetic_benchmark(config)
    second = generate_synthetic_benchmark(config)
    assert first.as_dict() == second.as_dict()
    assert len(first.records) == 4
    assert first.config.family == second.config.family == "gv"

    candidate_vectors = [tuple(record.candidate.parameters["gv_launch"]["sample"].values()) for record in first.records]
    per_candidate, summary = compare_to_baselines(
        candidate_vectors=candidate_vectors,
        seed=first.config.seed,
        baseline_names=first.config.baseline_names,
    )
    assert len(per_candidate) == len(first.records)
    assert summary


def test_generate_synthetic_benchmark_family_shapes_are_compatible() -> None:
    gv_result = generate_synthetic_benchmark(_make_config(family="gv"))
    emb_result = generate_synthetic_benchmark(_make_config(family="emb"))

    assert gv_result.config.family == "gv"
    assert emb_result.config.family == "emb"
    assert set(gv_result.records[0].as_dict()) == set(emb_result.records[0].as_dict())
    assert gv_result.records[0].candidate.parameters.keys() == {"family", "gv_launch"}
    assert emb_result.records[0].candidate.parameters.keys() == {"family", "emb_launch"}

    gv_launch = gv_result.records[0].candidate.parameters["gv_launch"]
    emb_launch = emb_result.records[0].candidate.parameters["emb_launch"]
    assert gv_launch["experiment"] == "stretching"
    assert emb_launch["experiment"] == "indentation"
    assert set(gv_launch["material_parameters"]) == {"ka", "kb"}
    assert "physical_parameters" in emb_launch
    assert len(gv_result.records[0].response_curve) == len(gv_result.records[0].response_points)
    assert len(emb_result.records[0].response_curve) == len(emb_result.records[0].response_points)
    assert "peak" in dict(gv_result.records[0].reduced_observables)
    assert "peak" in dict(emb_result.records[0].reduced_observables)


def test_generate_synthetic_benchmark_partial_and_async_completion_states() -> None:
    result = generate_synthetic_benchmark(_make_config(family="gv", seed=33))
    statuses = Counter(record.completion_status for record in result.records)
    assert statuses["queued"] > 0
    assert statuses["partial"] > 0
    assert statuses["completed"] > 0

    for record in result.records:
        if record.completion_status == "partial":
            assert record.async_token is not None
            assert 0.0 < record.completion_ratio < 1.0
            assert record.completed_steps < record.total_steps
        elif record.completion_status == "queued":
            assert record.async_token is None
            assert record.completion_ratio == 0.0
            assert record.completed_steps == 0
        else:
            assert record.async_token is None
            assert record.completion_ratio == 1.0
            assert record.completed_steps == record.total_steps


def test_write_synthetic_benchmark_artifacts_creates_manifest_report_and_plot(tmp_path: Path) -> None:
    output_root = tmp_path / "_runs" / "active_learning"
    result = generate_synthetic_benchmark(_make_config(family="emb", seed=23))
    artifacts = write_synthetic_benchmark_artifacts(
        output_root=output_root,
        run_id="run-synthetic-artifacts",
        iteration=2,
        result=result,
        include_plot=True,
    )

    assert artifacts.manifest_path.is_file()
    assert artifacts.report_path.is_file()
    assert artifacts.plot_path.is_file()
    assert artifacts.plot_sidecar_path.is_file()
    assert _png_is_visual(artifacts.plot_path)
    assert artifacts.plot_path.name == ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_FILENAME
    assert artifacts.plot_sidecar_path.name == ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_SIDECAR_FILENAME

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    sidecar = json.loads(artifacts.plot_sidecar_path.read_text(encoding="utf-8"))

    assert manifest["candidate_count"] == 4
    assert manifest["candidate_count"] == len(result.records)
    assert report["record_count"] == 4
    assert "baseline_comparison" in report
    assert "completion_status" in report
    assert "reduced_observables" in report
    assert "response_curves" in report
    assert len(report["response_curves"]) == len(result.records)
    assert sidecar == report


def test_write_synthetic_benchmark_artifacts_fallback_plot_is_visual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "_runs" / "active_learning"
    result = generate_synthetic_benchmark(_make_config(family="gv", seed=29))
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("blocked matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    artifacts = write_synthetic_benchmark_artifacts(
        output_root=output_root,
        run_id="run-synthetic-fallback-plot",
        iteration=1,
        result=result,
        include_plot=True,
    )

    assert _png_is_visual(artifacts.plot_path)


def test_synthetic_benchmark_import_is_lightweight() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    code = """
import importlib
import sys
import meso_uq.active_learning.synthetic_benchmark
assert "matplotlib" not in sys.modules
importlib.import_module("meso_uq.active_learning")
assert "matplotlib" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
