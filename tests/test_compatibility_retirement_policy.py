from __future__ import annotations

from pathlib import Path

from meso_uq.config.aliases import list_legacy_config_path_aliases
from meso_uq.surrogate.compat import list_serialized_surrogate_aliases
from meso_uq.workflows.legacy import list_legacy_workflow_surfaces


REPO_ROOT = Path(__file__).resolve().parents[1]
RETIREMENT_RECORD = REPO_ROOT / "docs" / "COMPATIBILITY_SHIM_RETIREMENT_RECORD.md"


def _record_text() -> str:
    return RETIREMENT_RECORD.read_text(encoding="utf-8").replace("`", "")


def test_retirement_record_tracks_current_workflow_surface_inventory() -> None:
    text = _record_text()

    for surface in list_legacy_workflow_surfaces():
        assert surface.legacy_path in text
        assert surface.replacement_api in text


def test_retirement_record_tracks_serialized_surrogate_alias_inventory() -> None:
    text = _record_text()

    for alias in list_serialized_surrogate_aliases():
        assert f"{alias.legacy_module}.{alias.legacy_attribute}" in text
        assert f"{alias.replacement_module}.{alias.replacement_attribute}" in text


def test_retirement_record_tracks_legacy_config_alias_inventory() -> None:
    text = _record_text()

    for alias in list_legacy_config_path_aliases():
        assert alias.legacy_root in text
        assert alias.canonical_root in text


def test_retirement_record_preserves_one_release_cycle_retention_policy() -> None:
    text = _record_text()

    assert "No compatibility shim is approved for removal in this slice." in text
    assert "one full release cycle has passed" in text
    assert "Review baseline: 2026-05-14." in text
