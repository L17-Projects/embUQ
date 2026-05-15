from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from meso_uq.inference.native_cuda_performance import (
    NativeCudaPerformanceThresholds,
    check_native_cuda_profile,
    load_native_cuda_profile_records,
    summarize_native_cuda_profile,
)


def _setup_record(**overrides):
    record = {
        "kind": "native_cuda_setup",
        "device_id": 0,
        "sub_problem_count": 2,
        "dynamic_prior_count": 3,
        "context_seconds": 0.1,
        "compile_seconds": 0.2,
        "module_load_seconds": 0.03,
        "alloc_seconds": 0.04,
        "h2d_seconds": 0.05,
        "total_seconds": 0.42,
    }
    record.update(overrides)
    return record


def _batch_record(**overrides):
    record = {
        "kind": "native_cuda_batch",
        "batch_size": 64,
        "parameter_count": 8,
        "sub_problem_count": 2,
        "dynamic_prior_count": 3,
        "log_prior_seconds": 0.01,
        "flatten_seconds": 0.02,
        "context_seconds": 0.01,
        "alloc_seconds": 0.03,
        "h2d_seconds": 0.04,
        "launch_seconds": 0.01,
        "compute_seconds": 0.2,
        "d2h_seconds": 0.03,
        "free_seconds": 0.02,
        "host_reduce_seconds": 0.04,
        "total_seconds": 0.4,
    }
    record.update(overrides)
    return record


def _write_profile(path: Path, *records: dict[str, object]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_native_cuda_profile_summary_reports_bucket_max_and_mean(tmp_path: Path) -> None:
    profile = tmp_path / "phase2_native_cuda.jsonl"
    _write_profile(
        profile,
        _setup_record(),
        _batch_record(total_seconds=0.4, compute_seconds=0.2, batch_size=64),
        _batch_record(total_seconds=0.6, compute_seconds=0.3, batch_size=128),
    )

    records = load_native_cuda_profile_records(profile)
    summary = summarize_native_cuda_profile(records, source=profile)

    assert summary.setup_count == 1
    assert summary.batch_count == 2
    assert summary.batch_size_max == 128
    assert summary.parameter_count_max == 8
    assert summary.sub_problem_count_max == 2
    assert summary.dynamic_prior_count_max == 3
    assert summary.batch_max_seconds["total_seconds"] == pytest.approx(0.6)
    assert summary.batch_mean_seconds["compute_seconds"] == pytest.approx(0.25)
    assert summary.to_manifest()["source"] == str(profile)


def test_native_cuda_profile_threshold_report_names_exceeded_bucket(tmp_path: Path) -> None:
    profile = tmp_path / "phase2_native_cuda.jsonl"
    _write_profile(profile, _setup_record(), _batch_record(alloc_seconds=0.3, total_seconds=0.7))

    report = check_native_cuda_profile(
        profile,
        thresholds=NativeCudaPerformanceThresholds(
            max_batch_alloc_seconds=0.1,
            max_batch_total_seconds=0.6,
        ),
    )

    assert not report.passed
    assert any("batch.alloc_seconds max 0.3s exceeds max_batch_alloc_seconds=0.1s" in item for item in report.mismatches)
    assert any("batch.total_seconds max 0.7s exceeds max_batch_total_seconds=0.6s" in item for item in report.mismatches)
    manifest = report.to_manifest()
    assert manifest["passed"] is False
    assert manifest["thresholds"]["max_batch_alloc_seconds"] == pytest.approx(0.1)


def test_native_cuda_profile_rejects_malformed_records(tmp_path: Path) -> None:
    profile = tmp_path / "phase2_native_cuda.jsonl"
    bad = _batch_record()
    bad.pop("h2d_seconds")
    _write_profile(profile, _setup_record(), bad)

    with pytest.raises(ValueError, match="missing NativeCuda profile field 'h2d_seconds'"):
        load_native_cuda_profile_records(profile)


def test_native_cuda_profile_cli_writes_report_and_uses_exit_code(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    profile = tmp_path / "phase2_native_cuda.jsonl"
    output = tmp_path / "report.json"
    _write_profile(profile, _setup_record(), _batch_record(total_seconds=0.7))

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "platforms" / "hpc" / "check_native_cuda_profile.py"),
            str(profile),
            "--output-json",
            str(output),
            "--max-batch-total-seconds",
            "0.6",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "batch.total_seconds" in completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["summary"]["batch_count"] == 1
