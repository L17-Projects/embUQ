from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_CLUSTER_PATH = "/ceph/hpc/home/eubrieucb"
SURFACE_PATHS = (
    REPO_ROOT / "docs",
    REPO_ROOT / ".github" / "workflows",
    REPO_ROOT / "scripts" / "platforms",
    REPO_ROOT / "scripts" / "workflows",
    REPO_ROOT / "scripts" / "qa" / "ci",
)
SURFACE_GLOBS = ("*.md", "*.sh", "*.py", "*.yml", "*.yaml")

ALLOWED_PATH_REFERENCES = (
    # Explicitly documented default for historical Mirheo bootstrap paths.
    ("docs/WORKFLOWS.md", "- `/ceph/hpc/home/eubrieucb/software/Mirheo`"),
    (
        "docs/VEGA_BOOTSTRAP.md",
        "- `/ceph/hpc/home/eubrieucb/software/Mirheo`",
    ),
    (
        "docs/KAROLINA_FULL_PLATFORM.md",
        "Do not expect `/ceph/hpc/home/eubrieucb` to be mounted on Karolina.",
    ),
    (
        "docs/MES-125_KAROLINA_ACCEPTANCE_CLOSEOUT.md",
        "Karolina acceptance for MES-125 does not depend on mounting `/ceph/hpc/home/eubrieucb`.",
    ),
    (
        "docs/MES-125_KAROLINA_ACCEPTANCE_CLOSEOUT.md",
        "the `/ceph/hpc/home/eubrieucb` mount is not.",
    ),
    (
        "docs/PLATFORM_POLICY.md",
        "Karolina policy explicitly forbids `/ceph/hpc/home/eubrieucb` inside checked-in platform config examples.",
    ),
)


def _surface_files() -> list[Path]:
    files: list[Path] = []
    for root in SURFACE_PATHS:
        if not root.exists():
            continue
        for pattern in SURFACE_GLOBS:
            files.extend(root.rglob(pattern))
    return sorted(set(files))


def _is_allowed_reference(relative: Path, line: str) -> bool:
    for allow_root, allow_fragment in ALLOWED_PATH_REFERENCES:
        if str(relative) == allow_root and allow_fragment in line:
            return True
    return False


def _forbidden_karolina_references() -> list[str]:
    forbidden: list[str] = []
    for path in _surface_files():
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_CLUSTER_PATH not in text:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN_CLUSTER_PATH not in line:
                continue
            relative = path.relative_to(REPO_ROOT)
            if _is_allowed_reference(relative, line):
                continue
            forbidden.append(f"{relative}:{line_no}: {line.strip()}")
    return forbidden


def test_canonical_surfaces_do_not_reference_forbidden_cluster_path() -> None:
    forbidden = _forbidden_karolina_references()
    assert not forbidden, (
        "Unexpected forbidden Karolina path references outside allowlist:\n"
        + "\n".join(sorted(forbidden))
    )


@pytest.mark.operational
@pytest.mark.parametrize("site", ["karolina", "vega"])
def test_validation_matrix_runner_and_template_assets_exist(site: str) -> None:
    script = REPO_ROOT / "scripts" / "platforms" / site / "run_validation_matrix.py"
    template = (
        REPO_ROOT / "scripts" / "platforms" / site / "sbatch" / "validation_matrix.sbatch"
    )

    assert script.is_file(), f"Missing matrix runner for {site}: {script.relative_to(REPO_ROOT)}"
    assert template.is_file(), (
        f"Missing matrix sbatch template for {site}: {template.relative_to(REPO_ROOT)}"
    )
