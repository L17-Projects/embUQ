from __future__ import annotations

"""Validation helpers for GV Mirheo canary logs and disk footprint."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence


GV_NUMERICAL_DATA_GENERATION_DISK_CAP_BYTES = 25 * 1024**3

_NONFINITE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<token>[+-]?(?:nan|inf(?:inity)?))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LogValidationIssue:
    log_path: Path | None
    line_number: int
    token: str
    excerpt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "log_path": None if self.log_path is None else str(self.log_path),
            "line_number": self.line_number,
            "token": self.token,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class LogValidationResult:
    log_path: Path | None
    checked_lines: int
    issues: tuple[LogValidationIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "log_path": None if self.log_path is None else str(self.log_path),
            "checked_lines": self.checked_lines,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DiskCapValidationResult:
    root: Path
    used_bytes: int
    cap_bytes: int

    @property
    def passed(self) -> bool:
        return self.used_bytes <= self.cap_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "used_bytes": self.used_bytes,
            "cap_bytes": self.cap_bytes,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RuntimeCanaryValidationResult:
    log_results: tuple[LogValidationResult, ...]
    disk_result: DiskCapValidationResult

    @property
    def passed(self) -> bool:
        return self.disk_result.passed and all(result.passed for result in self.log_results)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "logs": [result.to_dict() for result in self.log_results],
            "disk": self.disk_result.to_dict(),
        }


def _line_excerpt(line: str, *, max_chars: int = 240) -> str:
    stripped = line.rstrip("\n")
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3] + "..."


def validate_mirheo_log_text(
    text: str,
    *,
    log_path: str | Path | None = None,
    max_issues: int = 50,
) -> LogValidationResult:
    """Reject token-like NaN/Inf occurrences while avoiding ordinary words."""

    path = None if log_path is None else Path(log_path)
    issues: list[LogValidationIssue] = []
    checked_lines = 0
    for checked_lines, line in enumerate(text.splitlines(), start=1):
        for match in _NONFINITE_TOKEN.finditer(line):
            issues.append(
                LogValidationIssue(
                    log_path=path,
                    line_number=checked_lines,
                    token=match.group("token"),
                    excerpt=_line_excerpt(line),
                )
            )
            if len(issues) >= max_issues:
                return LogValidationResult(log_path=path, checked_lines=checked_lines, issues=tuple(issues))
    return LogValidationResult(log_path=path, checked_lines=checked_lines, issues=tuple(issues))


def validate_mirheo_log_file(path: str | Path, *, max_issues: int = 50) -> LogValidationResult:
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return validate_mirheo_log_text(text, log_path=log_path, max_issues=max_issues)


def validate_mirheo_logs(paths: Sequence[str | Path], *, max_issues_per_file: int = 50) -> tuple[LogValidationResult, ...]:
    return tuple(validate_mirheo_log_file(path, max_issues=max_issues_per_file) for path in paths)


def summarize_log_validation(results: Iterable[LogValidationResult]) -> str:
    rows: list[str] = []
    for result in results:
        label = "<inline>" if result.log_path is None else str(result.log_path)
        if result.passed:
            rows.append(f"PASS {label}: {result.checked_lines} lines checked")
            continue
        rows.append(f"FAIL {label}: {len(result.issues)} non-finite token(s)")
        for issue in result.issues[:5]:
            rows.append(f"  line {issue.line_number}: {issue.token} :: {issue.excerpt}")
    return "\n".join(rows)


def measure_runs_footprint_bytes(root: str | Path) -> int:
    root_path = Path(root)
    if not root_path.exists():
        return 0
    if root_path.is_file():
        return root_path.stat().st_size
    total = 0
    for path in root_path.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def evaluate_runs_disk_cap(
    root: str | Path,
    *,
    cap_bytes: int = GV_NUMERICAL_DATA_GENERATION_DISK_CAP_BYTES,
) -> DiskCapValidationResult:
    return DiskCapValidationResult(
        root=Path(root),
        used_bytes=measure_runs_footprint_bytes(root),
        cap_bytes=int(cap_bytes),
    )


def evaluate_runtime_canary(
    *,
    log_paths: Sequence[str | Path],
    runs_root: str | Path,
    disk_cap_bytes: int = GV_NUMERICAL_DATA_GENERATION_DISK_CAP_BYTES,
) -> RuntimeCanaryValidationResult:
    return RuntimeCanaryValidationResult(
        log_results=validate_mirheo_logs(log_paths),
        disk_result=evaluate_runs_disk_cap(runs_root, cap_bytes=disk_cap_bytes),
    )


__all__ = [
    "GV_NUMERICAL_DATA_GENERATION_DISK_CAP_BYTES",
    "DiskCapValidationResult",
    "LogValidationIssue",
    "LogValidationResult",
    "RuntimeCanaryValidationResult",
    "evaluate_runtime_canary",
    "evaluate_runs_disk_cap",
    "measure_runs_footprint_bytes",
    "summarize_log_validation",
    "validate_mirheo_log_file",
    "validate_mirheo_log_text",
    "validate_mirheo_logs",
]
