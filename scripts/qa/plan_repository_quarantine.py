#!/usr/bin/env python3
"""Build a non-destructive repository cleanup and quarantine plan."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PROTECTED_PATTERNS = (
    "scripts/platforms/karolina/**",
    "scripts/platforms/vega/**",
    "scripts/platforms/hpc/**",
    "scripts/workflows/emb/active_learning/**",
    "gv/*/src/**",
)
ALLOWED_CLASSIFICATIONS = {
    "keep",
    "document",
    "false_positive",
    "needs_owner_review",
}


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _candidate_rows(orphan_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subsystem, candidates in orphan_report.get("by_subsystem", {}).items():
        for candidate in candidates:
            row = dict(candidate)
            row.setdefault("subsystem", subsystem)
            rows.append(row)
    return sorted(rows, key=lambda row: row["path"])


def _tracked_text(repo_root: Path, paths: list[str]) -> dict[str, str]:
    corpus: dict[str, str] = {}
    for relative in paths:
        path = repo_root / relative
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            corpus[relative] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return corpus


def _reference_hits(
    candidate: dict[str, Any],
    corpus: dict[str, str],
) -> list[dict[str, str]]:
    candidate_path = candidate["path"]
    module = candidate.get("module", "")
    basename = Path(candidate_path).name
    tokens = [candidate_path]
    if module:
        tokens.append(module)
    tokens.append(basename)

    hits: list[dict[str, str]] = []
    for path, text in corpus.items():
        if path == candidate_path:
            continue
        for token in tokens:
            if token and token in text:
                hits.append({"path": path, "token": token})
                break
    return hits[:50]


def _manifest_hits(manifest_dir: Path | None, candidate_path: str) -> list[str]:
    if manifest_dir is None or not manifest_dir.exists():
        return []
    hits = []
    for path in sorted(manifest_dir.rglob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if candidate_path in text:
            hits.append(path.as_posix())
    return hits


def _open_pr_touches(snapshot: dict[str, Any], candidate_path: str) -> list[int]:
    touched = []
    for pr in snapshot.get("pull_requests", []):
        if candidate_path in pr.get("changed_files", []):
            touched.append(int(pr["number"]))
    return sorted(touched)


def _protected_reasons(path: str) -> list[str]:
    return [pattern for pattern in PROTECTED_PATTERNS if fnmatch.fnmatch(path, pattern)]


def _classification(
    *,
    protected: list[str],
    references: list[dict[str, str]],
    manifest_references: list[str],
    open_prs: list[int],
) -> tuple[str, str]:
    if open_prs or manifest_references:
        return "keep", "retain-preserved-reachability"
    if protected:
        return "false_positive", "retain-protected-entrypoint"
    if references:
        return "document", "retain-and-document-entrypoint"
    return "needs_owner_review", "review-before-any-quarantine"


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(path)
        current = current.parent
    return current


def build_plan(
    *,
    repo_root: Path,
    orphan_report_path: Path,
    open_pr_snapshot_path: Path,
    quarantine_root: Path,
    manifest_dir: Path | None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    orphan_report = _load_json(orphan_report_path)
    open_pr_snapshot = _load_json(open_pr_snapshot_path)
    candidates = _candidate_rows(orphan_report)
    tracked_paths = [path for path in _git(repo_root, "ls-files").splitlines() if path]
    tracked_set = set(tracked_paths)
    corpus = _tracked_text(repo_root, tracked_paths)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        relative = candidate["path"]
        source = repo_root / relative
        references = _reference_hits(candidate, corpus)
        manifest_references = _manifest_hits(manifest_dir, relative)
        open_prs = _open_pr_touches(open_pr_snapshot, relative)
        protected = _protected_reasons(relative)
        classification, action = _classification(
            protected=protected,
            references=references,
            manifest_references=manifest_references,
            open_prs=open_prs,
        )
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise AssertionError(classification)

        record = {
            "path": relative,
            "subsystem": candidate.get("subsystem"),
            "tracked": relative in tracked_set,
            "exists": source.is_file(),
            "bytes": source.stat().st_size if source.is_file() else None,
            "sha256": _sha256(source) if source.is_file() else None,
            "git_blob_oid": _git(repo_root, "rev-parse", f"HEAD:{relative}").strip()
            if relative in tracked_set
            else None,
            "protected_patterns": protected,
            "reference_hits": references,
            "accepted_manifest_hits": manifest_references,
            "open_pull_requests_touching_path": open_prs,
            "classification": classification,
            "recommended_action": action,
            "planned_destination": (quarantine_root / relative).as_posix(),
        }
        records.append(record)

    review_records = [row for row in records if row["classification"] == "needs_owner_review"]
    planned_bytes = sum(int(row["bytes"] or 0) for row in review_records)
    capacity_parent = _existing_parent(quarantine_root)
    usage = shutil.disk_usage(capacity_parent)
    counts = Counter(row["classification"] for row in records)
    return {
        "schema_version": "1.0",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "dry-run",
        "policy": {
            "moves_performed": False,
            "deletions_performed": False,
            "delete_classification_allowed": False,
            "quarantine_requires_owner_confirmation": True,
            "quarantine_requires_copy_hash_verification": True,
            "source_removal_requires_verified_destination": True,
        },
        "source": {
            "repo_root": repo_root.as_posix(),
            "commit": _git(repo_root, "rev-parse", "HEAD").strip(),
            "orphan_report": orphan_report_path.resolve().as_posix(),
            "open_pr_snapshot": open_pr_snapshot_path.resolve().as_posix(),
            "manifest_dir": manifest_dir.resolve().as_posix() if manifest_dir else None,
        },
        "quarantine": {
            "root": quarantine_root.as_posix(),
            "capacity_checked_at": capacity_parent.as_posix(),
            "planned_review_bytes": planned_bytes,
            "available_bytes": usage.free,
            "capacity_sufficient": usage.free > planned_bytes,
        },
        "summary": {
            "candidate_count": len(records),
            "classification_counts": dict(sorted(counts.items())),
            "review_count": len(review_records),
        },
        "records": records,
    }


def render_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    quarantine = plan["quarantine"]
    lines = [
        "# Repository cleanup dry-run",
        "",
        f"- Commit: `{plan['source']['commit']}`",
        f"- Candidates: {summary['candidate_count']}",
        f"- Classifications: `{json.dumps(summary['classification_counts'], sort_keys=True)}`",
        f"- Manual-review bytes: {quarantine['planned_review_bytes']}",
        f"- Scratch bytes available: {quarantine['available_bytes']}",
        f"- Capacity sufficient: {quarantine['capacity_sufficient']}",
        "- Files moved: no",
        "- Files deleted: no",
        "",
        "## Manual review",
        "",
        "| Path | Bytes | Open PRs | Action |",
        "|---|---:|---|---|",
    ]
    for row in plan["records"]:
        if row["classification"] != "needs_owner_review":
            continue
        prs = ", ".join(f"#{number}" for number in row["open_pull_requests_touching_path"]) or "-"
        lines.append(
            f"| `{row['path']}` | {row['bytes'] or 0} | {prs} | {row['recommended_action']} |"
        )
    lines.extend(["", "## Classification summary", ""])
    for name, count in summary["classification_counts"].items():
        lines.append(f"- `{name}`: {count}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--orphan-report", type=Path, required=True)
    parser.add_argument("--open-pr-snapshot", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for output in (args.output_json, args.output_markdown):
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
    plan = build_plan(
        repo_root=args.repo_root,
        orphan_report_path=args.orphan_report,
        open_pr_snapshot_path=args.open_pr_snapshot,
        quarantine_root=args.quarantine_root,
        manifest_dir=args.manifest_dir,
    )
    for output in (args.output_json, args.output_markdown):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(plan), encoding="utf-8")
    print(json.dumps(plan["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
