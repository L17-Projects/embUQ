from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIRED_MARKERS = (
    "gpu",
    "hpc",
    "slurm",
    "cuda",
    "mpi",
    "korali",
    "pyro",
    "operational",
    "slow",
    "integration",
)


def _configured_pytest_markers() -> set[str]:
    lines = PYPROJECT.read_text(encoding="utf-8").splitlines()
    marker_lines: list[str] = []
    in_markers_block = False
    for raw_line in lines:
        line = raw_line.strip()

        if not in_markers_block and line == "markers = [":
            in_markers_block = True
            continue
        if in_markers_block and line == "]":
            break
        if in_markers_block:
            marker_lines.append(line.strip('", '))

    declared: set[str] = set()
    for marker in marker_lines:
        if not marker:
            continue
        declared.add(marker.split(":", 1)[0].strip())
    return declared


@pytest.mark.parametrize("marker_name", REQUIRED_MARKERS)
def test_pytest_marker_contract_declares_required_governance_tokens(marker_name: str) -> None:
    declared = _configured_pytest_markers()
    assert marker_name in declared, f"Missing pytest marker declaration: {marker_name}"
