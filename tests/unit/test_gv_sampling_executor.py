from __future__ import annotations

import subprocess
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.structures.gv.runtime import load_runtime_descriptor
from meso_uq.structures.gv.sampling.executor import execute_sampling_plan
from meso_uq.structures.gv.sampling.failures import (
    GVCommandFailure,
    GVCwdValidationError,
    GVLogScanFailure,
    GVTimeoutFailure,
    GVSamplingFailure,
)
from meso_uq.structures.gv.sampling.planner import GVSamplingPlan, SamplingCommand, SamplingRun
from meso_uq.structures.gv.sampling.planner import build_sampling_plan
from meso_uq.structures.gv.sampling.logs import scan_log_file, scan_log_text


def _build_plan(tmp_path: Path) -> GVSamplingPlan:
    runtime = load_runtime_descriptor("stretching").plan(output_root=tmp_path / "runtime")
    return build_sampling_plan(runtime, control_axis="tot_force", values=(500.0, 1500.0))


def test_executor_runs_command_list_synchronously_with_timeout(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "ok"
    mock_result.stderr = ""

    with patch("meso_uq.structures.gv.sampling.executor.subprocess.run", return_value=mock_result) as run_mock:
        result = execute_sampling_plan(plan, timeout_seconds=123)

    assert result.return_codes == (0, 0)
    assert result.executed_commands == (
        "python3 generate.py -p tot_force 500 1500 2 -p bpress -91 -100 1 --object gv --forward --first",
        "bash commands.txt",
    )
    assert run_mock.call_count == 2
    assert run_mock.call_args_list[0].kwargs["timeout"] == 123
    log_dir = Path(plan.runtime_manifest["work_dir"]) / "sampling_command_logs"
    assert (log_dir / "001.stdout").read_text(encoding="utf-8") == "ok"


def test_executor_rejects_missing_cwd(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    broken_plan = GVSamplingPlan(
        runtime_manifest={"work_dir": str(missing_dir)},
        control_axis="tot_force",
        values=(1.0,),
        linear=True,
        timeout_seconds=72,
        runs=(
            SamplingRun(
                command_sequence=(
                    SamplingCommand(
                        argv=("bash", "-c", "echo hi"),
                        cwd=str(missing_dir),
                    ),
                ),
                controls={"tot_force": 1.0},
                provenance={"mode": "exact", "value": 1.0},
            ),
        ),
    )

    with pytest.raises(GVCwdValidationError):
        execute_sampling_plan(broken_plan)


def test_executor_rejects_sbatch_submission(tmp_path: Path) -> None:
    runtime = load_runtime_descriptor("stretching").plan(output_root=tmp_path / "runtime")
    base_plan = build_sampling_plan(runtime, control_axis="tot_force", values=(500.0,))
    first_run = base_plan.runs[0]
    first_run = SamplingRun(
        command_sequence=(
            SamplingCommand(argv=("sbatch", "run_HPC.sbatch"), cwd=str(tmp_path / "runtime" / "stretching" / runtime.geometry / runtime.control_id / "work")),
            *first_run.command_sequence[1:],
        ),
        controls=first_run.controls,
        provenance=first_run.provenance,
    )
    sbatch_plan = base_plan.__class__(
        runtime_manifest=base_plan.runtime_manifest,
        control_axis=base_plan.control_axis,
        values=base_plan.values,
        linear=base_plan.linear,
        timeout_seconds=base_plan.timeout_seconds,
        runs=(first_run,),
    )

    with pytest.raises(GVCommandFailure):
        execute_sampling_plan(sbatch_plan)


def test_executor_detects_nonzero_exit_code(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    failed = MagicMock(returncode=7, stdout="", stderr="")

    with patch("meso_uq.structures.gv.sampling.executor.subprocess.run", return_value=failed):
        with pytest.raises(GVCommandFailure):
            execute_sampling_plan(plan)
    assert (Path(plan.runtime_manifest["work_dir"]) / "sampling_command_logs" / "failed.stderr").is_file()


def test_executor_times_out_and_raises(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)

    with patch("meso_uq.structures.gv.sampling.executor.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["python3"], timeout=120)):
        with pytest.raises(GVTimeoutFailure):
            execute_sampling_plan(plan)


def test_executor_scans_log_files_and_stdout_for_nans(tmp_path: Path) -> None:
    runtime = load_runtime_descriptor("stretching").plan(output_root=tmp_path / "runtime")
    plan = build_sampling_plan(runtime, control_axis="tot_force", values=(500.0,))

    log_path = Path(plan.runtime_manifest["work_dir"]) / "output.out"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("value: NaN\n", encoding="utf-8")

    bad = MagicMock(returncode=0, stdout="", stderr="")
    with patch("meso_uq.structures.gv.sampling.executor.subprocess.run", return_value=bad):
        with pytest.raises(GVLogScanFailure):
            execute_sampling_plan(plan)


def test_scan_log_text_detects_nan_and_inf_tokens() -> None:
    issues = scan_log_text("value NaN\nsomething inf\n", source="fake.out")
    assert len(issues) == 2
    assert issues[0].token.lower() == "nan"
    assert issues[1].token.lower() == "inf"


def test_scan_log_text_respects_issue_limit() -> None:
    issues = scan_log_text("value NaN\nvalue Inf\nvalue nan\n", source="fake.out", max_issues=1)
    assert len(issues) == 1


def test_scan_log_file_detects_plus_inf_token(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text("load +Inf\n", encoding="utf-8")
    issues = scan_log_file(log_path)
    assert len(issues) == 1
    assert issues[0].token == "+Inf"


def test_executor_rejects_non_plan_input() -> None:
    with pytest.raises(TypeError, match="plan must be a GVSamplingPlan"):
        execute_sampling_plan("not-a-plan")


def test_executor_rejects_invalid_timeout_override() -> None:
    with pytest.raises(GVSamplingFailure, match="timeout_seconds must be an integer"):
        execute_sampling_plan(_build_plan(Path(".")), timeout_seconds=1.2)


def test_executor_rejects_work_dir_violation(tmp_path: Path) -> None:
    runtime = {"work_dir": str(tmp_path / "runtime")}
    bad_run = GVSamplingPlan(
        runtime_manifest=runtime,
        control_axis="tot_force",
        values=(1.0,),
        linear=True,
        timeout_seconds=60,
        runs=(
            SamplingRun(
                command_sequence=(
                    SamplingCommand(argv=("python3", "-c", "print(1)"), cwd=str(tmp_path / "outside")),
                ),
                controls={"tot_force": 1.0},
                provenance={"mode": "exact", "value": 1.0},
            ),
        ),
    )
    (tmp_path / "runtime").mkdir()
    (tmp_path / "outside").mkdir()

    with pytest.raises(GVCwdValidationError, match="outside runtime work_dir"):
        execute_sampling_plan(bad_run)


def test_executor_scans_log_file_contents_for_nans(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    output_path = Path(plan.runtime_manifest["work_dir"]) / "output.out"
    output_path.write_text("step: NaN\n", encoding="utf-8")

    good = MagicMock(returncode=0, stdout="", stderr="")
    with patch("meso_uq.structures.gv.sampling.executor.subprocess.run", return_value=good):
        with pytest.raises(GVLogScanFailure, match="Detected non-finite"):
            execute_sampling_plan(plan)
