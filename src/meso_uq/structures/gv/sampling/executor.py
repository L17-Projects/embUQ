from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import os
import subprocess

from .failures import (
    GVCommandFailure,
    GVCwdValidationError,
    GVLogScanFailure,
    GVSamplingFailure,
    GVTimeoutFailure,
)
from .logs import LogScanIssue, scan_log_file, scan_log_text
from .planner import GVSamplingPlan, SamplingCommand


@dataclass(frozen=True)
class SamplingExecutionResult:
    executed_commands: tuple[str, ...]
    return_codes: tuple[int, ...]
    plan: dict[str, Any]


_DEFAULT_LOG_PATHS = ("output.out", "output.log", "logs/*.log", "logs/*.txt")
_SCHEDULER_COMMANDS = {"sbatch", "srun", "qsub", "qstat", "qdel"}


def execute_sampling_plan(
    plan: GVSamplingPlan,
    *,
    timeout_seconds: int | None = None,
    scan_log_paths: bool = True,
    env: Mapping[str, str] | None = None,
) -> SamplingExecutionResult:
    """Execute a sampling plan synchronously without scheduler submission."""

    if not isinstance(plan, GVSamplingPlan):
        raise TypeError("plan must be a GVSamplingPlan.")
    selected_timeout = _default_timeout(timeout_seconds, plan)
    executed: list[str] = []
    return_codes: list[int] = []
    command_index = 0

    runtime_manifest = dict(plan.runtime_manifest)
    work_dir = runtime_manifest.get("work_dir")

    for run in plan.runs:
        for command in run.command_sequence:
            _guard_scheduler_submission(command)
            cwd = _normalize_cwd(command.cwd, plan)
            if work_dir is not None:
                _guard_command_within_workdir(cwd, str(work_dir))
            command_index += 1
            result = _run_command(command, cwd, timeout=selected_timeout, env=env)
            _write_command_capture(cwd, command_index, result)
            executed.append(" ".join(command.argv))
            return_codes.append(result.returncode)
            _scan_text_for_failures(
                result.stdout,
                source=f"{cwd / 'stdout'}",
            )
            _scan_text_for_failures(
                result.stderr,
                source=f"{cwd / 'stderr'}",
            )
            if scan_log_paths:
                for path in _iter_scan_paths(cwd):
                    if path.is_file():
                        _scan_file_for_failures(path)

    return SamplingExecutionResult(
        executed_commands=tuple(executed),
        return_codes=tuple(return_codes),
        plan=plan.to_manifest(),
    )


def _run_command(
    command: SamplingCommand,
    cwd: Path,
    *,
    timeout: int,
    env: Mapping[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    selected_env = None
    if env is not None:
        selected_env = dict(os.environ)
        selected_env.update({str(key): str(value) for key, value in env.items()})
    try:
        result = subprocess.run(
            list(command.argv),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=selected_env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GVTimeoutFailure(
            f"Sampling command timed out after {timeout}s: {' '.join(command.argv)}"
        ) from exc
    if result.returncode != 0:
        _write_command_capture(cwd, -1, result)
        raise GVCommandFailure(
            f"Sampling command failed with exit code {result.returncode}: {' '.join(command.argv)}"
            f"{_failure_excerpt(result)}"
        )
    return result


def _write_command_capture(cwd: Path, index: int, result: subprocess.CompletedProcess[str]) -> None:
    log_dir = cwd / "sampling_command_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = "failed" if index < 0 else f"{index:03d}"
    (log_dir / f"{prefix}.stdout").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / f"{prefix}.stderr").write_text(result.stderr or "", encoding="utf-8")


def _failure_excerpt(result: subprocess.CompletedProcess[str]) -> str:
    pieces = []
    if result.stdout:
        pieces.append("stdout: " + _tail(result.stdout))
    if result.stderr:
        pieces.append("stderr: " + _tail(result.stderr))
    if not pieces:
        return ""
    return "\n" + "\n".join(pieces)


def _tail(text: str, *, max_lines: int = 12) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


def _default_timeout(timeout_seconds: int | None, plan: GVSamplingPlan) -> int:
    if timeout_seconds is None:
        return plan.timeout_seconds
    if not isinstance(timeout_seconds, int):
        raise GVSamplingFailure("timeout_seconds must be an integer.")
    if timeout_seconds <= 0:
        raise GVSamplingFailure("timeout_seconds must be positive.")
    return timeout_seconds


def _guard_scheduler_submission(command: SamplingCommand) -> None:
    if not command.argv:
        return
    head = command.argv[0]
    if Path(head).name.lower() in _SCHEDULER_COMMANDS:
        raise GVCommandFailure(
            "Scheduler submission is not allowed for synchronous GV sampling execution."
        )


def _normalize_cwd(cwd: str, plan: GVSamplingPlan) -> Path:
    path = Path(cwd)
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    if not resolved.exists():
        raise GVCwdValidationError(f"Command cwd does not exist: {path}")
    if not resolved.is_dir():
        raise GVCwdValidationError(f"Command cwd is not a directory: {path}")
    return resolved


def _guard_command_within_workdir(cwd: Path, work_dir: str) -> None:
    allowed = Path(work_dir).resolve()
    if not cwd.is_relative_to(allowed):
        raise GVCwdValidationError(
            f"Command cwd {cwd} is outside runtime work_dir {allowed}."
        )


def _scan_text_for_failures(text: str, *, source: str) -> None:
    issues = scan_log_text(text, source=source)
    if issues:
        raise GVLogScanFailure(_format_scan_issues(source, issues))


def _scan_file_for_failures(path: Path) -> None:
    issues = scan_log_file(path)
    if issues:
        raise GVLogScanFailure(_format_scan_issues(str(path), issues))


def _iter_scan_paths(work_dir: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for candidate in _DEFAULT_LOG_PATHS:
        if "*" in candidate:
            paths.extend(sorted(work_dir.glob(candidate)))
        else:
            paths.append(work_dir / candidate)
    return tuple(dict.fromkeys(paths))


def _format_scan_issues(source: str, issues: tuple[LogScanIssue, ...]) -> str:
    preview = ", ".join(f"{issue.line_number}:{issue.token}" for issue in issues[:3])
    return f"Detected non-finite value(s) in {source}: {preview}"


__all__ = [
    "SamplingExecutionResult",
    "execute_sampling_plan",
]
