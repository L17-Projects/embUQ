#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_coverage(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_markdown_summary(payload: dict, max_files: int = 10) -> str:
    totals = payload["totals"]
    files = []
    for filename, file_payload in payload.get("files", {}).items():
        summary = file_payload.get("summary", {})
        num_statements = int(summary.get("num_statements", 0))
        if num_statements == 0:
            continue
        percent_covered = float(summary.get("percent_covered", 0.0))
        missing_lines = int(summary.get("missing_lines", 0))
        files.append((percent_covered, missing_lines, filename))

    files.sort(key=lambda item: (item[0], -item[1], item[2]))
    top_files = files[:max_files]

    lines = [
        "## Coverage Summary",
        "",
        f"- Total coverage: {float(totals['percent_covered']):.1f}%",
        f"- Covered lines: {int(totals['covered_lines'])} / {int(totals['num_statements'])}",
        f"- Measured files: {len(files)}",
        "",
    ]

    if top_files:
        lines.extend(
            [
                "### Lowest-Coverage Files",
                "",
                "| File | Coverage | Missing Lines |",
                "| --- | ---: | ---: |",
            ]
        )
        for percent_covered, missing_lines, filename in top_files:
            lines.append(f"| `{filename}` | {percent_covered:.1f}% | {missing_lines} |")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a markdown summary from coverage.py JSON output.")
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--markdown-path", required=True)
    parser.add_argument("--max-files", type=int, default=10)
    args = parser.parse_args(argv)

    payload = _load_coverage(Path(args.json_path))
    markdown = build_markdown_summary(payload, max_files=args.max_files)
    Path(args.markdown_path).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
