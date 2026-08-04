#!/usr/bin/env python3
"""Compile the frozen UQ_EMB editor bundle in a fresh scratch directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

from verify_frozen_submission import verify_snapshot  # noqa: E402
from replay_provenance import replay_receipt_provenance  # noqa: E402


SCHEMA_VERSION = "mesouq.uq_emb.manuscript_replay.v1"
TARGETS = ("main.tex", "main_marked_up.tex", "si.tex")
BASELINE_PDFS = ("main.pdf", "main_marked_up.pdf")
FAILED_LOG_PATTERNS = (
    re.compile(r"(?:LaTeX|Package natbib) Warning: Citation .* undefined"),
    re.compile(r"LaTeX Warning: Reference .* undefined"),
    re.compile(r"There were undefined (?:citations|references)"),
    re.compile(r"! LaTeX Error:"),
    re.compile(r"Emergency stop"),
    re.compile(r"Fatal error occurred"),
)
RERUN_LOG_PATTERNS = (
    re.compile(r"Label\(s\) may have changed"),
    re.compile(r"Rerun to get cross-references right"),
)
MAX_POST_BIBTEX_PASSES = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_fresh_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty manuscript build directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _copy_manifest_files(bundle_root: Path, build_root: Path, manifest: dict[str, Any]) -> None:
    for entry in manifest["files"]:
        relative = Path(str(entry["path"]))
        source = bundle_root / relative
        destination = build_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not determine page count for {path}")


def _text_sha256(path: Path, text_root: Path) -> str:
    text_root.mkdir(parents=True, exist_ok=True)
    output = text_root / f"{path.stem}.txt"
    subprocess.run(["pdftotext", "-layout", str(path), str(output)], check=True)
    return _sha256(output)


def _failed_log_lines(log_path: Path) -> list[str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return sorted(
        {
            line.strip()
            for line in text.splitlines()
            if any(pattern.search(line) for pattern in FAILED_LOG_PATTERNS)
        }
    )


def _log_requires_rerun(log_path: Path) -> bool:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return any(pattern.search(text) for pattern in RERUN_LOG_PATTERNS)


def compile_manuscript(
    *,
    bundle_root: Path,
    manifest_path: Path,
    build_root: Path,
    pdflatex: str,
    bibtex: str,
) -> dict[str, Any]:
    started = time.monotonic()
    bundle_root = bundle_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    build_root = build_root.expanduser().resolve()

    frozen_report = verify_snapshot(root=bundle_root, manifest_path=manifest_path)
    if frozen_report["status"] != "PASS":
        raise ValueError(f"Frozen editor bundle verification failed: {frozen_report}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _require_fresh_directory(build_root)
    _copy_manifest_files(bundle_root, build_root, manifest)
    baseline_root = build_root / "_frozen_baseline"
    baseline_root.mkdir()
    for name in BASELINE_PDFS:
        shutil.copy2(build_root / name, baseline_root / name)

    environment = os.environ.copy()
    environment.setdefault("SOURCE_DATE_EPOCH", "1785834096")
    environment.setdefault("FORCE_SOURCE_DATE", "1")
    command_results: list[dict[str, Any]] = []
    for target in TARGETS:
        target_started = time.monotonic()
        stem = Path(target).stem
        latex_command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            target,
        ]
        commands = [latex_command, [bibtex, stem]]
        for command in commands:
            subprocess.run(command, cwd=build_root, env=environment, check=True)
        post_bibtex_passes = 0
        for _ in range(MAX_POST_BIBTEX_PASSES):
            subprocess.run(latex_command, cwd=build_root, env=environment, check=True)
            commands.append(latex_command)
            post_bibtex_passes += 1
            if not _log_requires_rerun(build_root / f"{stem}.log"):
                break
        else:
            raise ValueError(
                f"Cross-references did not stabilize for {target} after "
                f"{MAX_POST_BIBTEX_PASSES} post-BibTeX passes"
            )
        command_results.append(
            {
                "target": target,
                "commands": commands,
                "post_bibtex_passes": post_bibtex_passes,
                "wall_seconds": time.monotonic() - target_started,
            }
        )

    text_root = build_root / "_text_audit"
    text_root.mkdir()
    outputs: dict[str, Any] = {}
    for target in TARGETS:
        stem = Path(target).stem
        pdf_path = build_root / f"{stem}.pdf"
        log_path = build_root / f"{stem}.log"
        if not pdf_path.is_file() or not log_path.is_file():
            raise FileNotFoundError(f"Missing compiled output for {target}")
        failed_lines = _failed_log_lines(log_path)
        if failed_lines:
            raise ValueError(f"Compilation warnings/errors remain for {target}: {failed_lines}")
        outputs[stem] = {
            "pdf_path": str(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "pdf_pages": _pdf_pages(pdf_path),
            "text_sha256": _text_sha256(pdf_path, text_root),
            "log_path": str(log_path),
            "log_sha256": _sha256(log_path),
        }

    baseline_comparison: dict[str, Any] = {}
    for name in BASELINE_PDFS:
        baseline = baseline_root / name
        replay = build_root / name
        baseline_pages = _pdf_pages(baseline)
        replay_pages = _pdf_pages(replay)
        baseline_text = _text_sha256(baseline, text_root / "baseline")
        replay_text = outputs[Path(name).stem]["text_sha256"]
        baseline_comparison[name] = {
            "baseline_sha256": _sha256(baseline),
            "replay_sha256": _sha256(replay),
            "page_count_matches": baseline_pages == replay_pages,
            "text_matches": baseline_text == replay_text,
            "baseline_pages": baseline_pages,
            "replay_pages": replay_pages,
        }
    if not all(
        row["page_count_matches"] and row["text_matches"]
        for row in baseline_comparison.values()
    ):
        raise ValueError(f"Recompiled PDFs differ from the frozen text/page baseline: {baseline_comparison}")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "execution_provenance": replay_receipt_provenance(
            repo_root=REPO_ROOT,
            runner=Path(__file__),
            consumed_paths=[bundle_root],
        ),
        "bundle_verification": frozen_report,
        "build_root": str(build_root),
        "pdflatex": shutil.which(pdflatex) or pdflatex,
        "bibtex": shutil.which(bibtex) or bibtex,
        "commands": command_results,
        "outputs": outputs,
        "baseline_comparison": baseline_comparison,
        "wall_seconds": time.monotonic() - started,
    }
    _write_json(build_root / "uq_emb_manuscript_replay_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=REPO_ROOT / "papers/UQ_EMB/editor_submission/review2_v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "papers/UQ_EMB/manifests/editor_submission_review2_v1.json",
    )
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--pdflatex", default="pdflatex")
    parser.add_argument("--bibtex", default="bibtex")
    args = parser.parse_args(argv)
    receipt = compile_manuscript(
        bundle_root=args.bundle_root,
        manifest_path=args.manifest,
        build_root=args.build_root,
        pdflatex=args.pdflatex,
        bibtex=args.bibtex,
    )
    print(
        f"UQ_EMB manuscript replay passed in {receipt['wall_seconds']:.2f} s: "
        f"{receipt['build_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
