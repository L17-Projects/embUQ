from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from meso_uq.config.examples import (
    EXAMPLE_CONFIG_ROOT,
    ExampleConfigInventoryEntry,
    ExampleConfigManifestEntry,
    ExampleConfigStatus,
    classify_example_config_record,
    current_example_config_inventory,
    load_example_config_manifest,
    validate_example_config_manifest,
    validate_example_config_manifest_entry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs" / "examples_manifest.example.json"


def test_manifest_example_parses() -> None:
    manifest = load_example_config_manifest(MANIFEST_PATH)

    assert manifest.schema_version == "1.0"
    assert manifest.manifest_id == "examples-config-policy-example"
    assert manifest.generated_at == "2026-05-13T00:00:00Z"
    assert [entry.path for entry in manifest.examples] == [
        "examples/configs/compression_full.yaml",
        "examples/configs/compression_reduced.yaml",
        "examples/configs/indentation_full.yaml",
        "examples/configs/indentation_reduced.yaml",
    ]
    assert all(entry.status is ExampleConfigStatus.NEEDS_REFRESH for entry in manifest.examples)
    assert validate_example_config_manifest(manifest, repo_root=REPO_ROOT) == []


def test_current_example_inventory_reports_the_four_bundle_files() -> None:
    inventory = current_example_config_inventory(REPO_ROOT)

    assert [entry.path for entry in inventory] == [
        "examples/configs/compression_full.yaml",
        "examples/configs/compression_reduced.yaml",
        "examples/configs/indentation_full.yaml",
        "examples/configs/indentation_reduced.yaml",
    ]
    assert all(entry.status is ExampleConfigStatus.NEEDS_REFRESH for entry in inventory)
    assert all(entry.validation_notes for entry in inventory)


def test_private_path_rejection(tmp_path: Path) -> None:
    repo_root = tmp_path
    example_path = repo_root / EXAMPLE_CONFIG_ROOT / "private_path.yaml"
    example_path.parent.mkdir(parents=True, exist_ok=True)
    example_path.write_text(
        "schema_version: '1.0'\n"
        "kind: dataset\n"
        "metadata:\n"
        "  id: private_path\n"
        "  name: Private Path\n"
        "spec:\n"
        "  path: /ceph/hpc/home/eubrieucb/secret.csv\n",
        encoding="utf-8",
    )

    entry = ExampleConfigManifestEntry(
        path="examples/configs/private_path.yaml",
        canonical_path=None,
        status=ExampleConfigStatus.OWNER_DECISION,
        reason="manual check",
        validation_notes=("manual check",),
    )

    errors = validate_example_config_manifest_entry(entry, repo_root=repo_root)

    assert any("forbidden private path" in error for error in errors)


def test_missing_file_detection(tmp_path: Path) -> None:
    repo_root = tmp_path
    entry = ExampleConfigManifestEntry(
        path="examples/configs/missing.yaml",
        canonical_path="configs/missing.yaml",
        status=ExampleConfigStatus.OWNER_DECISION,
        reason="missing",
        validation_notes=("missing",),
    )

    errors = validate_example_config_manifest_entry(entry, repo_root=repo_root)

    assert any("not found" in error for error in errors)


def test_maintained_example_loading(tmp_path: Path) -> None:
    repo_root = tmp_path
    example_path = repo_root / "examples" / "configs" / "maintained.yaml"
    canonical_path = repo_root / "configs" / "production" / "maintained.yaml"
    example_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "kind": "dataset",
        "metadata": {"id": "maintained", "name": "Maintained Example"},
        "spec": {"path": "relative/path.csv"},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    example_path.write_text(text, encoding="utf-8")
    canonical_path.write_text(text, encoding="utf-8")

    record = ExampleConfigInventoryEntry(
        path="examples/configs/maintained.yaml",
        canonical_path="configs/production/maintained.yaml",
    )
    entry = classify_example_config_record(record, repo_root=repo_root)
    errors = validate_example_config_manifest_entry(entry, repo_root=repo_root)

    assert entry.status is ExampleConfigStatus.MAINTAINED
    assert "matches the canonical workflow config" in entry.reason
    assert errors == []


def test_import_is_light(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "meso_uq.config.loader", raising=False)

    module = importlib.reload(importlib.import_module("meso_uq.config.examples"))

    assert "meso_uq.config.loader" not in sys.modules
    assert module.EXAMPLE_CONFIG_ROOT == EXAMPLE_CONFIG_ROOT
