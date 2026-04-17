#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_percent(payload: dict) -> float:
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
