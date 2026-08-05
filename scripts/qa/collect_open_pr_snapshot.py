#!/usr/bin/env python3
"""Collect a read-only snapshot of open pull requests with changed paths."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PR_FIELDS = (
    "number,title,headRefName,headRefOid,baseRefName,isDraft,mergeable,"
    "mergeStateStatus,statusCheckRollup,updatedAt,url"
)


def _run(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        [*args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    conclusions = Counter()
    incomplete = 0
    for check in checks:
        status = str(check.get("status", "UNKNOWN")).upper()
        conclusion = str(check.get("conclusion") or "NONE").upper()
        if status != "COMPLETED":
            incomplete += 1
        conclusions[conclusion] += 1
    return {
        "count": len(checks),
        "incomplete": incomplete,
        "conclusions": dict(sorted(conclusions.items())),
    }


def collect_snapshot(repo_root: Path) -> dict[str, Any]:
    rows = json.loads(
        _run(
            repo_root,
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            PR_FIELDS,
        )
    )
    pull_requests = []
    for row in sorted(rows, key=lambda item: int(item["number"])):
        changed_files = sorted(
            {
                path
                for path in _run(
                    repo_root, "gh", "pr", "diff", str(row["number"]), "--name-only"
                ).splitlines()
                if path
            }
        )
        checks = row.pop("statusCheckRollup", [])
        row["changed_file_count"] = len(changed_files)
        row["changed_files"] = changed_files
        row["checks"] = summarize_checks(checks)
        pull_requests.append(row)

    repository = json.loads(_run(repo_root, "gh", "repo", "view", "--json", "nameWithOwner"))
    return {
        "schema_version": "1.0",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "read-only",
        "repository": repository["nameWithOwner"],
        "local_head": _run(repo_root, "git", "rev-parse", "HEAD").strip(),
        "open_pr_count": len(pull_requests),
        "pull_requests": pull_requests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force")
    snapshot = collect_snapshot(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"open_pr_count": snapshot["open_pr_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
