from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOT_PREFIXES = (
    "_out",
    "_runs",
    "_ci",
    "out_hierarchical",
)
INIT_COMPRESSION_PREFIX = "_init_compression_"

ROOT_SRUN_PATTERN = re.compile(r"^slurm-.*\.(?:out|err)$")


def _tracked_git_files() -> list[Path]:
    payload = subprocess.check_output(
        ["git", "ls-files"], cwd=str(REPO_ROOT), universal_newlines=True
    )
    return [Path(line) for line in payload.splitlines() if line]


def test_forbidden_generated_roots_not_tracked_at_repo_root() -> None:
    offenders: list[str] = []
    for path in _tracked_git_files():
        if not path.parts:
            continue

        root = path.parts[0]
        if root.startswith(INIT_COMPRESSION_PREFIX):
            offenders.append(f"{path}")
            continue
        if root in FORBIDDEN_ROOT_PREFIXES:
            offenders.append(f"{path}")
            continue
        if len(path.parts) == 1 and ROOT_SRUN_PATTERN.fullmatch(path.name):
            offenders.append(f"{path}")

    assert not offenders, (
        "Tracked repository artifacts must not use forbidden generated-root names at the repo root: "
        f"{offenders}"
    )
