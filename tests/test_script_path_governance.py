from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

WORKFLOW_FILES = tuple((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
PLATFORM_SLRUM_FILES = tuple((REPO_ROOT / "scripts" / "platforms").rglob("*.sbatch"))

SCRIPT_PATH_REFERENCE_RE = re.compile(
    r"(?:\${[^}]+}/|\$[A-Za-z_][A-Za-z0-9_]*/)?['\"]?((?:\.?/)?scripts/[A-Za-z0-9._/-]+\.(?:py|sh|sbatch|yaml|yml))['\"]?"
)

CANONICAL_ROOT_PREFIXES = (
    "scripts/ci/",
    "scripts/qa/",
    "scripts/shared/",
    "scripts/workflows/",
    "scripts/platforms/",
    "scripts/run_",
)

COMPAT_ALIAS_TO_CANONICAL = {
    "scripts/ci/": "scripts/qa/ci/",
    "scripts/vega/": "scripts/platforms/vega/",
    "scripts/karolina/": "scripts/platforms/karolina/",
    "scripts/hpc/": "scripts/platforms/hpc/",
}


def _find_script_refs(path: Path) -> list[tuple[int, str]]:
    refs: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in SCRIPT_PATH_REFERENCE_RE.finditer(line):
            refs.append((line_no, match.group(1).lstrip("./")))
    return refs


def _iter_references(
    files: tuple[Path, ...],
) -> list[tuple[Path, int, str]]:
    matches: list[tuple[Path, int, str]] = []
    for path in files:
        for line_no, reference in _find_script_refs(path):
            matches.append((path, line_no, reference))
    return matches


def _find_alias_root(reference: str) -> str | None:
    for alias_root in COMPAT_ALIAS_TO_CANONICAL:
        if reference.startswith(alias_root):
            return alias_root
    return None


def _resolved_reference(reference: str) -> Path:
    return (REPO_ROOT / reference).resolve()


def _is_known_reference_root(reference: str) -> bool:
    if _find_alias_root(reference) is not None:
        return True
    return any(reference.startswith(prefix) for prefix in CANONICAL_ROOT_PREFIXES)


def _validate_reference(
    source: Path,
    line_no: int,
    reference: str,
    *,
    issues: list[str],
) -> None:
    repo_relative = REPO_ROOT / reference
    if not repo_relative.exists():
        issues.append(
            f"{source.relative_to(REPO_ROOT)}:{line_no}: referenced script does not exist: {reference}"
        )
        return

    canonical_ref = _find_alias_root(reference)
    if canonical_ref is not None:
        target_root = _resolved_reference(canonical_ref)
        resolved = _resolved_reference(reference)
        if not str(resolved).startswith(str(target_root)):
            issues.append(
                f"{source.relative_to(REPO_ROOT)}:{line_no}: legacy path {reference} must resolve under "
                f"{canonical_ref}"
            )
        return

    if not _is_known_reference_root(reference):
        issues.append(
            f"{source.relative_to(REPO_ROOT)}:{line_no}: ungoverned script root for {reference} "
            f"(expected qa/platforms/workflow/shared/ci family)"
        )


def _collect_issues(sources: tuple[Path, ...]) -> list[str]:
    issues: list[str] = []
    for source, line_no, reference in _iter_references(sources):
        _validate_reference(source, line_no, reference, issues=issues)
    return sorted(set(issues))


def test_workflow_yaml_script_references_are_governed() -> None:
    assert WORKFLOW_FILES, "No workflow files found to inspect."

    issues = _collect_issues(WORKFLOW_FILES)
    assert not issues, (
        "Script-path governance failed for workflow YAML references. "
        "Only static repo-relative script paths can be validated; dynamic shell expansions are intentionally omitted.\n"
        + "\n".join(issues)
    )


def test_platform_slurm_script_references_are_governed() -> None:
    assert PLATFORM_SLRUM_FILES, "No platform Slurm files found to inspect."

    issues = _collect_issues(PLATFORM_SLRUM_FILES)
    assert not issues, (
        "Script-path governance failed for platform Slurm templates. "
        "Only static repo-relative script paths can be validated from static template text.\n"
        + "\n".join(issues)
    )


def _collect_platform_references() -> list[tuple[Path, int, str]]:
    return _iter_references(WORKFLOW_FILES + PLATFORM_SLRUM_FILES)


@pytest.mark.parametrize(
    "alias_root",
    tuple(COMPAT_ALIAS_TO_CANONICAL.keys()),
)
def test_compatibility_alias_roots_remain_symlinks_if_present_and_used(alias_root: str) -> None:
    canonical_root = COMPAT_ALIAS_TO_CANONICAL[alias_root]
    alias_path = REPO_ROOT / alias_root.rstrip("/")

    referenced = any(
        reference.startswith(alias_root) for _, _, reference in _collect_platform_references()
    )
    if not referenced and not alias_path.exists():
        return

    assert alias_path.exists(), f"Compatibility root {alias_root.rstrip('/')} is referenced but missing."
    assert alias_path.is_dir(), f"Compatibility root {alias_root.rstrip('/')} must be a directory alias."
    assert alias_path.is_symlink(), f"Compatibility root {alias_root.rstrip('/')} must remain a symlink during migration."
    assert alias_path.resolve() == (REPO_ROOT / canonical_root), (
        f"Compatibility root {alias_root.rstrip('/')} must resolve to {canonical_root}."
    )
