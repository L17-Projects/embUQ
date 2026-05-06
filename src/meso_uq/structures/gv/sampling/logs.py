from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_NONFINITE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<token>[+-]?(?:nan|inf(?:inity)?))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LogScanIssue:
    source: Path | str | None
    line_number: int
    token: str
    excerpt: str


def _excerpt(line: str, *, max_chars: int = 240) -> str:
    stripped = line.rstrip("\n")
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3] + "..."


def scan_log_text(
    text: str,
    *,
    source: Path | str | None = None,
    max_issues: int = 50,
) -> tuple[LogScanIssue, ...]:
    """Return non-finite tokens detected in the provided text."""

    issues: list[LogScanIssue] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in _NONFINITE_TOKEN.finditer(line):
            issues.append(
                LogScanIssue(
                    source=None if source is None else str(source),
                    line_number=line_no,
                    token=match.group("token"),
                    excerpt=_excerpt(line),
                )
            )
            if len(issues) >= max_issues:
                return tuple(issues)
    return tuple(issues)


def scan_log_file(
    path: str | Path,
    *,
    max_issues: int = 50,
) -> tuple[LogScanIssue, ...]:
    """Return non-finite tokens detected in a log file."""

    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return scan_log_text(text, source=log_path, max_issues=max_issues)


def log_scan_passed(result: tuple[LogScanIssue, ...]) -> bool:
    return not result


__all__ = ["LogScanIssue", "scan_log_text", "scan_log_file", "log_scan_passed"]
