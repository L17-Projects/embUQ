from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GV_GENERATED_PATTERNS = (
    "*/__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*/CMakeFiles/*",
    "*/CMakeCache.txt",
    "*/cmake_install.cmake",
    "*/CMakeOutput.log",
    "*/CMakeError.log",
    "*/Makefile2",
    "*/progress.marks",
    "*.o",
    "*.d",
    "*.bin",
    "*.log",
    "*.out",
    "*.off",
    "*.png",
    "*.xyz",
    "*/commands.txt",
    "*/posq.txt",
    "*/mesh/*",
    "*/cgal_scripts/afm2",
    "*/cgal_scripts/scale_space",
    "*/cgal_scripts/scale_space_manifold",
    "*/cgal_scripts/a.out",
)


def test_codeowners_covers_repo_root() -> None:
    codeowners = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert "* @BrieucB" in codeowners
    assert "/src/ @BrieucB" in codeowners
    assert "/tests/ @BrieucB" in codeowners


def test_dependabot_covers_actions_and_python() -> None:
    payload = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))

    assert payload["version"] == 2
    updates = payload["updates"]
    ecosystems = {(entry["package-ecosystem"], entry["directory"]) for entry in updates}
    assert ("github-actions", "/") in ecosystems
    assert ("pip", "/") in ecosystems
    for entry in updates:
        assert entry["schedule"]["interval"] == "weekly"
        assert entry["open-pull-requests-limit"] == 5
        assert "dependencies" in entry["labels"]


def test_security_policy_documents_private_reporting() -> None:
    policy = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "do **not** open a public GitHub issue" in policy
    assert "main" in policy
    assert "latest `v0.1.x` release line" in policy


def test_gitignore_covers_generated_gv_artifacts() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    required_entries = (
        "gv_simulation_files/**/CMakeFiles/",
        "gv_simulation_files/**/*.py[cod]",
        "gv_simulation_files/**/*.off",
        "gv_simulation_files/**/*.out",
        "gv_simulation_files/**/mesh/",
        "gv_simulation_files/**/cgal_scripts/scale_space",
    )

    for entry in required_entries:
        assert entry in gitignore


def test_tracked_gv_staging_files_exclude_generated_artifacts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "gv_simulation_files"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    tracked_paths = [Path(path) for path in tracked if path]
    offenders = []
    for path in tracked_paths:
        path_text = path.as_posix()
        if any(path.match(pattern) for pattern in GV_GENERATED_PATTERNS):
            offenders.append(path_text)

    assert not offenders, (
        "Generated GV staging artifacts must not be tracked. "
        f"Offending paths: {offenders}"
    )
