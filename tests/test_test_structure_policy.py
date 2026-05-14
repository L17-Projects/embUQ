from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = REPO_ROOT / "docs" / "TEST_STRUCTURE_POLICY.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

REQUIRED_MARKERS = {
    "integration",
    "operational",
    "slow",
    "gpu",
    "cuda",
    "hpc",
    "slurm",
    "mpi",
    "mirheo",
    "korali",
    "pyro",
}


def _declared_markers() -> set[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    marker_block = text.split("markers = [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    marker_names: set[str] = set()
    for raw_line in marker_block.splitlines():
        line = raw_line.strip().strip(",")
        if not line.startswith('"'):
            continue
        marker_names.add(line.strip('"').split(":", maxsplit=1)[0].strip())
    return marker_names


def test_test_structure_policy_documents_layout_and_runtime_boundaries() -> None:
    text = POLICY.read_text(encoding="utf-8")

    for required in (
        "tests/unit/",
        "tests/integration/",
        "tests/support/",
        "tests/conftest.py",
        "pytest",
        "tests/support/optional_dependencies.py",
        "Installed-package and import-boundary checks",
        "_out/",
        "_runs/",
        "_ci/",
        "out_hierarchical/",
        "runtime/",
        "Karolina GPU validation matrix",
    ):
        assert required in text


def test_test_structure_policy_lists_all_pytest_runtime_markers() -> None:
    text = POLICY.read_text(encoding="utf-8")
    declared = _declared_markers()

    assert REQUIRED_MARKERS <= declared
    for marker in REQUIRED_MARKERS:
        assert f"`{marker}`" in text
