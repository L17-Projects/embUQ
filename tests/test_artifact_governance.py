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
    "logs",
    "runtime",
    ".coverage",
    ".pytest_cache",
    "build",
    "dist",
)
INIT_COMPRESSION_PREFIX = "_init_compression_"
ROOT_SLURM_LOG_PATTERN = re.compile(r"^(?:mesouq|slurm)-.*\.(?:out|err)$")
PYTHON_COMPILED_SUFFIXES = (".pyc", ".pyd", ".pyo")

REQUIRED_GITIGNORE_PATTERNS = (
    "_out/",
    "_runs/",
    "_init_compression_*/",
    "out_hierarchical/",
    "_ci/",
    "runtime/",
    "logs/",
    "build/",
    "dist/",
    ".pytest_cache/",
    "__pycache__/",
    "*.py[cod]",
    "*.egg-info/",
    "/mesouq-*.out",
    "/mesouq-*.err",
    "/slurm-*.out",
    "/slurm-*.err",
    ".coverage",
)

SOURCE_EXCEPTION_FILES = (
    "configs/artifacts/artifact_manifest.example.json",
    "configs/platforms/generic_slurm.example.yaml",
    "docs/ARTIFACT_POLICY.md",
    "src/meso_uq/__init__.py",
    "src/meso_uq/agents/__init__.py",
    "src/meso_uq/artifacts/__init__.py",
    "src/meso_uq/config/__init__.py",
    "src/meso_uq/configs/__init__.py",
    "src/meso_uq/core/__init__.py",
    "src/meso_uq/inference/__init__.py",
    "src/meso_uq/mirheo/__init__.py",
    "src/meso_uq/modalities/__init__.py",
    "src/meso_uq/postprocess/__init__.py",
    "src/meso_uq/references/__init__.py",
    "src/meso_uq/sensitivity/__init__.py",
    "src/meso_uq/structures/__init__.py",
    "src/meso_uq/structures/gv/__init__.py",
    "src/meso_uq/structures/gv/paper_replay/__init__.py",
    "src/meso_uq/structures/gv/paper_replay_lanes/__init__.py",
    "src/meso_uq/structures/gv/postprocessing/__init__.py",
    "src/meso_uq/structures/gv/runtime/__init__.py",
    "src/meso_uq/structures/gv/sampling/__init__.py",
    "src/meso_uq/surrogate/__init__.py",
)


def _tracked_git_files() -> list[Path]:
    payload = subprocess.check_output(
        ["git", "ls-files"], cwd=str(REPO_ROOT), universal_newlines=True
    )
    return [Path(line) for line in payload.splitlines() if line]


def _is_forbidden_artifact(path: Path) -> bool:
    if not path.parts:
        return False

    root = path.parts[0]
    if root.startswith(INIT_COMPRESSION_PREFIX):
        return True
    if root in FORBIDDEN_ROOT_PREFIXES:
        return True
    if len(path.parts) == 1 and ROOT_SLURM_LOG_PATTERN.fullmatch(path.name):
        return True
    if any(part == "__pycache__" for part in path.parts):
        return True
    if path.suffix in PYTHON_COMPILED_SUFFIXES:
        return True
    if any(part.endswith(".egg-info") for part in path.parts):
        return True
    return False


def test_forbidden_generated_roots_not_tracked_at_repo_root() -> None:
    offenders = [f"{path}" for path in _tracked_git_files() if _is_forbidden_artifact(path)]

    assert not offenders, (
        "Tracked repository artifacts must not use forbidden generated-root names at the repo root: "
        f"{offenders}"
    )


def test_gitignore_capture_governance_for_generated_roots() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    gitignore_set = set(gitignore)
    missing = [entry for entry in REQUIRED_GITIGNORE_PATTERNS if entry not in gitignore_set]

    assert not missing, (
        "Required artifact-governance ignore patterns are missing from .gitignore: "
        f"{missing}"
    )


def test_source_exceptions_remain_tracked() -> None:
    tracked = {str(path) for path in _tracked_git_files()}
    missing = [path for path in SOURCE_EXCEPTION_FILES if path not in tracked]
    assert not missing, f"Source exceptions expected to remain tracked: {missing}"
