#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

_OPTIONAL_BNN_COVERAGE_EXCLUDES = {
    "src/meso_uq/surrogate/bnn.py",
    "src/meso_uq/surrogate/bnn_training.py",
    "compression/surrogate/evaluate_bnn.py",
    "indentation/surrogate/evaluate_bnn.py",
    "scripts/platforms/karolina/train_bnn_surrogates.py",
}


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _is_excluded(path: str) -> bool:
    return _normalize_path(path) in _OPTIONAL_BNN_COVERAGE_EXCLUDES


def _extract_percent(payload: dict) -> float:
    files = payload.get("files")
    if isinstance(files, dict):
        statements = 0
        covered = 0
        for path, entry in files.items():
            if _is_excluded(str(path)):
                continue
            if not isinstance(entry, dict):
                continue
            summary = entry.get("summary", {})
            if not isinstance(summary, dict):
                continue
            file_statements = int(summary.get("num_statements", 0))
            file_missing = int(summary.get("missing_lines", 0))
            file_covered = max(0, file_statements - file_missing)
            statements += file_statements
            covered += file_covered
        if statements > 0:
            return (covered / statements) * 100.0

    totals = payload.get("totals", {})
    statements = int(totals.get("num_statements", 0))
    covered = int(totals.get("covered_lines", 0))

    if statements > 0:
        return (covered / statements) * 100.0

    return float(totals.get("percent_covered", 0.0))


def build_markdown_summary(base_percent: float, head_percent: float) -> str:
    delta = head_percent - base_percent
    result = "PASS" if delta > 0 else "FAIL"
    sign = "+" if delta >= 0 else ""

    return "\n".join(
        [
            "## Coverage Delta Gate",
            "",
            f"- Base coverage: {base_percent:.3f}%",
            f"- PR coverage: {head_percent:.3f}%",
            f"- Delta: {sign}{delta:.3f} percentage points",
            "- Rule: PR coverage must be strictly greater than base coverage.",
            "- Scope: excludes optional BNN runtime/training files that require the `pyro-ppl` extra.",
            f"- Result: {result}",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    description = "Fail if PR coverage is not strictly greater than base " "branch coverage."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--head-json", required=True)
    parser.add_argument("--markdown-path", required=False)
    args = parser.parse_args(argv)

    base_payload = _load_payload(Path(args.base_json))
    head_payload = _load_payload(Path(args.head_json))

    base_percent = _extract_percent(base_payload)
    head_percent = _extract_percent(head_payload)
    markdown = build_markdown_summary(base_percent, head_percent)

    if args.markdown_path:
        Path(args.markdown_path).write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0 if head_percent > base_percent else 1


if __name__ == "__main__":
    raise SystemExit(main())
