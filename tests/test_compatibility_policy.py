from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "COMPATIBILITY_SHIMS.md"

REQUIRED_ROOTS = (
    "compression",
    "indentation",
    "inference/scripts",
    "propagation/scripts",
    "reduced",
    "scripts/vega",
    "scripts/ci",
    "scripts/karolina",
    "scripts/hpc",
)

REQUIRED_FIELDS = (
    "legacy path/import",
    "canonical replacement",
    "migration window",
    "removal condition",
    "CI noise policy",
    "compatibility-test expectations",
)


def _section_for_root(text: str, root: str) -> str | None:
    marker = f"## {root}"
    try:
        start = text.index(marker)
    except ValueError:
        return None

    tail = text[start + len(marker) :]
    next_heading = tail.find("\n## ")
    return tail if next_heading == -1 else tail[:next_heading]


def test_compatibility_shims_policy_doc_exists() -> None:
    assert DOC_PATH.is_file(), f"Missing policy doc: {DOC_PATH.relative_to(REPO_ROOT)}"


def test_compatibility_shims_policy_has_required_roots_and_fields() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    missing_roots: list[str] = []
    missing_fields: list[str] = []

    for root in REQUIRED_ROOTS:
        section = _section_for_root(text, root)
        if section is None:
            missing_roots.append(root)
            continue

        for field in REQUIRED_FIELDS:
            needle = f"- {field}:"
            if needle.lower() not in section.lower():
                missing_fields.append(f"{root}::{field}")

    assert not missing_roots, (
        "Missing policy sections for legacy roots: " + ", ".join(sorted(missing_roots))
    )
    assert not missing_fields, (
        "Missing required warning fields in policy sections: " + ", ".join(sorted(missing_fields))
    )


def test_compatibility_shims_migration_window_and_ci_notice_fields_are_visible() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").lower()
    # Lightweight guard against accidental template drift.
    assert "one release cycle" in text, (
        "Policy should include a concrete migration window; expected 'one release cycle'."
    )
    assert "deprecated" in text or "deprecation" in text, (
        "Policy should explicitly frame these wrappers as deprecation-based migration guidance."
    )
