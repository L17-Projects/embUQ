#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_REF = "origin/main"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_ci" / "merge_ready_coverage"


def _resolve_path(raw_path: str | Path, base: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base / path


def _run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )


def _run_git(*args: str) -> None:
    command = ["git", *args]
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def _parse_test_command(raw: str) -> list[str]:
    parts = shlex.split(raw)
    return parts or ["pytest"]


@contextlib.contextmanager
def _base_worktree(base_ref: str):
    with tempfile.TemporaryDirectory(prefix="mesouq-coverage-base-") as worktree_root:
        worktree_path = Path(worktree_root) / "repo"
        _run_git("fetch", "origin")
        _run_git("worktree", "add", "--detach", str(worktree_path), base_ref)
        try:
            yield worktree_path
        finally:
            # remove worktree reference and clean directory in normal and error paths
            _run_git("worktree", "remove", "--force", str(worktree_path))
            shutil.rmtree(worktree_path, ignore_errors=True)


def _collect_coverage_json(
    *,
    python_bin: str,
    repo_root: Path,
    test_command: str,
    output_root: Path,
    coverage_json_path: Path,
) -> None:
    commands = _parse_test_command(test_command)
    run_command = [python_bin, "-m", "coverage", "run", "-m", *commands]
    _run_logged_command(
        run_command,
        cwd=repo_root,
        stdout_log=output_root / "coverage-test.stdout.log",
        stderr_log=output_root / "coverage-test.stderr.log",
    )

    _run_logged_command(
        [python_bin, "-m", "coverage", "json", "-o", str(coverage_json_path)],
        cwd=repo_root,
        stdout_log=output_root / "coverage-json.stdout.log",
        stderr_log=output_root / "coverage-json.stderr.log",
    )


def _run_delta_gate(
    *,
    base_json: Path,
    head_json: Path,
    markdown_path: Path,
) -> int:
    checker_path = REPO_ROOT / "scripts" / "qa" / "ci" / "check_coverage_increase.py"
    spec = importlib.util.spec_from_file_location("check_coverage_increase", checker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load coverage checker from {checker_path}")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return int(
        checker.main(
            [
                "--base-json",
                str(base_json),
                "--head-json",
                str(head_json),
                "--markdown-path",
                str(markdown_path),
                "--strict",
            ]
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a strict local merge-ready coverage gate.")
    parser.add_argument("--base-ref", default=DEFAULT_BASE_REF)
    parser.add_argument("--base-json", default=None, help="Optional path to precomputed base coverage JSON.")
    parser.add_argument("--head-json", default=None, help="Optional path to precomputed head coverage JSON.")
    parser.add_argument(
        "--test-command",
        default="pytest",
        help="Command to execute under coverage; default: pytest",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for temporary logs and coverage artifacts.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    output_root = _resolve_path(args.output_root, REPO_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_json = _resolve_path(args.base_json, output_root).resolve() if args.base_json else None
    head_json = _resolve_path(args.head_json, output_root).resolve() if args.head_json else output_root / "coverage-head.json"
    markdown_path = output_root / "merge-ready-coverage-gate.md"

    if base_json is None:
        base_json = output_root / "coverage-base.json"
        with _base_worktree(args.base_ref) as base_repo:
            _collect_coverage_json(
                python_bin=args.python_bin,
                repo_root=base_repo,
                test_command=args.test_command,
                output_root=output_root / "base-worktree",
                coverage_json_path=base_json,
            )
    else:
        if not base_json.exists():
            raise FileNotFoundError(f"Base coverage file not found: {base_json}")

    if args.head_json is None:
        _collect_coverage_json(
            python_bin=args.python_bin,
            repo_root=REPO_ROOT,
            test_command=args.test_command,
            output_root=output_root / "head-worktree",
            coverage_json_path=head_json,
        )
    elif not head_json.exists():
        raise FileNotFoundError(f"Head coverage file not found: {head_json}")

    return _run_delta_gate(
        base_json=base_json,
        head_json=head_json,
        markdown_path=markdown_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
