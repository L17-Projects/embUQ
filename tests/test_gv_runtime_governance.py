from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GV_GENERATED_SAMPLES = (
    "gv_simulation_files/stretching/gv/mesh/generated.off",
    "gv_simulation_files/stretching/gv/cgal_scripts/scale_space",
    "gv_simulation_files/stretching/gv/commands.txt",
    "gv_simulation_files/stretching/gv/solver.out",
)


def test_generated_gv_samples_are_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-v", *GV_GENERATED_SAMPLES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    for sample in GV_GENERATED_SAMPLES:
        assert sample in stdout


def test_gv_source_of_truth_excludes_generated_samples() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "gv_simulation_files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    tracked_set = set(tracked)
    for sample in GV_GENERATED_SAMPLES:
        assert sample not in tracked_set


def test_wrapper_tree_exists_without_staging_outputs() -> None:
    wrapper_root = REPO_ROOT / "scripts" / "workflows" / "gv"
    assert (wrapper_root / "run_gv_dry_run.py").is_file()
    assert not any(
        path.is_file() and "gv_simulation_files" in path.as_posix()
        for path in wrapper_root.rglob("*")
    )
