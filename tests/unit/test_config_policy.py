from __future__ import annotations

from pathlib import Path

from meso_uq.configs import validate_config_document, validate_config_file


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


def test_config_skeleton_examples_validate() -> None:
    config_paths = sorted(CONFIG_ROOT.rglob("*.yaml"))

    assert config_paths, "Expected YAML config skeleton files under configs/."

    for path in config_paths:
        assert validate_config_file(path) == [], path.as_posix()


def test_config_validator_rejects_missing_identity_fields() -> None:
    errors = validate_config_document({"kind": "dataset", "metadata": {"id": "", "name": ""}})

    assert any("schema_version" in error for error in errors)
    assert any("metadata.id" in error for error in errors)
    assert any("metadata.name" in error for error in errors)


def test_config_validator_rejects_absolute_path_literals() -> None:
    errors = validate_config_document(
        {
            "schema_version": "1.0",
            "kind": "dataset",
            "metadata": {"id": "bad_dataset", "name": "Bad Dataset"},
            "spec": {"path_template": "/tmp/private/data.csv"},
        }
    )

    assert any("absolute path" in error for error in errors)
