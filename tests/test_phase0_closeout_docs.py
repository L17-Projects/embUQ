from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_migration_inventory_records_reproducible_artifact_snapshot() -> None:
    text = _text("docs/MESO_UQ_MIGRATION_INVENTORY.md")

    required_fragments = (
        "## Reproducible inventory commands",
        "## Artifact inventory snapshot (2026-05-13)",
        "| Path or glob | Tracked status | Ignored status | Approximate size and count | Extension or class examples | Suspected artifact class | Proposed disposition |",
        "`runtime/` | Untracked | Not ignored in this snapshot",
        "`compression/surrogate/diameters/*/data`",
        "`indentation/surrogate/diameters/*/data`",
        "## Decisions still required",
    )
    for fragment in required_fragments:
        assert fragment in text


def test_compatibility_surface_matrix_has_required_closeout_columns() -> None:
    text = _text("docs/MESO_UQ_COMPATIBILITY_SURFACE.md")

    required_fragments = (
        "## Reproducible compatibility inventory commands",
        "## MES-129 compatibility matrix",
        "| Surface | Current reference count and evidence | Replacement path or API | Owner or status | Shim plan | Warning plan | Removal release | Risk |",
        "Serialized artifact aliases `learning`, `learning.model`",
        "Env and CLI path layer",
        "## High-risk compatibility surfaces",
    )
    for fragment in required_fragments:
        assert fragment in text


def test_risk_register_contains_phase0_acceptance_columns() -> None:
    text = _text("docs/MESO_UQ_MIGRATION_RISK_REGISTER.md")

    required_fragments = (
        "| Risk | Evidence | Likelihood | Impact | Mitigation | Rollback plan | Validation command | Phase blocker | Owner decision needed |",
        "Paper scripts and `papers/` reproduction paths drift",
        "Karolina, Vega, and workstation platform behavior mismatch",
        "`extern/korali` vendoring/provenance risk",
        "BNN/Pyro optional dependency behavior",
        "Artifact movement and owner-approved deletion",
        "## High-risk gaps",
    )
    for fragment in required_fragments:
        assert fragment in text
