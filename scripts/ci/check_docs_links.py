#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_PATHS = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class BrokenLink:
    document: Path
    target: str
    resolved: Path | None


def _strip_code_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("", text)


def _normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if " " in target and not target.startswith("<"):
        target = target.split(" ", 1)[0]
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return target


def _is_external_link(target: str) -> bool:
    if target.startswith("#"):
        return True
    parsed = urlparse(target)
    return parsed.scheme in {"http", "https", "mailto"}


def _resolve_target(document: Path, target: str, repo_root: Path) -> Path | None:
    path_text = target.split("#", 1)[0]
    if not path_text:
        return None
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    if path_text.startswith("/"):
        return (repo_root / path_text.lstrip("/")).resolve()
    return (document.parent / path_text).resolve()


def find_broken_links(paths: list[Path], repo_root: Path) -> list[BrokenLink]:
    broken: list[BrokenLink] = []
    for document in paths:
        text = _strip_code_fences(document.read_text(encoding="utf-8"))
        for match in MARKDOWN_LINK_RE.finditer(text):
            target = _normalize_target(match.group(1))
            if not target or _is_external_link(target):
                continue
            resolved = _resolve_target(document, target, repo_root)
            if resolved is None:
                continue
            if not resolved.exists():
                broken.append(BrokenLink(document=document, target=target, resolved=resolved))
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local Markdown links in README.md and docs/*.md.")
    parser.add_argument("paths", nargs="*", help="Optional markdown paths to validate. Defaults to README.md and docs/*.md.")
    args = parser.parse_args(argv)

    if args.paths:
        paths = []
        for item in args.paths:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = REPO_ROOT / path
            paths.append(path.resolve())
    else:
        paths = [path.resolve() for path in DEFAULT_DOC_PATHS]

    broken = find_broken_links(paths, REPO_ROOT)
    if broken:
        for item in broken:
            resolved = item.resolved if item.resolved is not None else "<none>"
            print(f"{item.document}: {item.target} -> {resolved}", file=sys.stderr)
        print(f"Broken docs links: {len(broken)}", file=sys.stderr)
        return 1

    print(f"Checked docs links in {len(paths)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
