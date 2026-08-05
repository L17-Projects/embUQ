from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "workflows" / "emb" / "uq_emb" / "stage_external_artifacts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("stage_external_artifacts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _spec(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "paper_id": "UQ_EMB",
                "artifact_set_id": "fixture",
                "artifact_set_dir": "fixture-v1",
                "locked": True,
                "destination_root": str(tmp_path / "artifacts"),
                "entries": [
                    {
                        "artifact_id": "input",
                        "source": str(source),
                        "destination": "inputs",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _snapshot(module, spec: Path, manifest: Path) -> bytes:
    report = module.snapshot_manifest(spec_path=spec, manifest_path=manifest)
    assert report["status"] == "PASS"
    return manifest.read_bytes()


def test_external_artifact_stage_and_verify(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"

    plan = module.plan_staging(spec)
    manifest_before = _snapshot(module, spec, manifest)
    report = module.stage_artifacts(spec_path=spec, manifest_path=manifest)

    assert plan["status"] == "PASS"
    assert plan["file_count"] == 1
    assert report["status"] == "PASS"
    assert manifest.read_bytes() == manifest_before
    assert (tmp_path / "artifacts" / "fixture-v1" / "inputs" / "input.csv").is_file()


def test_external_artifact_stage_preserves_internal_hardlinks(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    first = source / "latest"
    first.write_text("posterior\n", encoding="utf-8")
    second = source / "genLatest.json"
    second.hardlink_to(first)
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"

    plan = module.plan_staging(spec)
    _snapshot(module, spec, manifest)
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    staged = tmp_path / "artifacts" / "fixture-v1" / "inputs"

    assert plan["internal_hardlink_savings_bytes"] == first.stat().st_size
    assert (staged / "latest").stat().st_ino == (staged / "genLatest.json").stat().st_ino


def test_external_artifact_stage_rejects_symlinked_sources(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target.txt"
    target.write_text("content\n", encoding="utf-8")
    (source / "alias.txt").symlink_to(target)
    spec = _spec(tmp_path, source)

    try:
        module.plan_staging(spec)
    except ValueError as exc:
        assert "symlinks" in str(exc)
    else:
        raise AssertionError("Expected symlinked artifact source to be rejected")


def test_external_artifact_stage_rejects_symlinked_source_root(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("content\n", encoding="utf-8")
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    spec = _spec(tmp_path, alias)

    try:
        module.plan_staging(spec)
    except ValueError as exc:
        assert "symlinks" in str(exc)
    else:
        raise AssertionError("Expected a symlinked artifact root to be rejected")


def test_external_artifact_verify_detects_mutation(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    _snapshot(module, spec, manifest)
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    staged_file = tmp_path / "artifacts" / "fixture-v1" / "inputs" / "input.csv"
    staged_file.write_text("mutated\n", encoding="utf-8")

    report = module.verify_staged(root=staged_file.parents[1], manifest_path=manifest)

    assert report["status"] == "FAIL"
    assert report["mismatches"]


def test_external_artifact_verify_cli_rejects_symlinked_root(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    _snapshot(module, spec, manifest)
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    root = tmp_path / "artifacts" / "fixture-v1"
    alias = tmp_path / "artifact-alias"
    alias.symlink_to(root, target_is_directory=True)

    try:
        module.main(
            [
                "verify",
                "--root",
                str(alias),
                "--manifest",
                str(manifest),
            ]
        )
    except ValueError as exc:
        assert "roots cannot be symlinks" in str(exc)
    else:
        raise AssertionError("Expected a symlinked verification root to be rejected")


def test_external_artifact_verify_cli_rejects_symlinked_root_parent(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    artifact_parent = tmp_path / "artifacts"
    alias_parent = tmp_path / "artifact-parent-alias"
    alias_parent.symlink_to(artifact_parent, target_is_directory=True)

    try:
        module.main(
            [
                "verify",
                "--root",
                str(alias_parent / "fixture-v1"),
                "--manifest",
                str(manifest),
            ]
        )
    except ValueError as exc:
        assert "symlinked path components" in str(exc)
    else:
        raise AssertionError("Expected a symlinked verification-root parent to be rejected")


def test_external_artifact_verify_rejects_report_inside_root(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    _snapshot(module, spec, manifest)
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    root = tmp_path / "artifacts" / "fixture-v1"
    report = root / "verification.json"

    try:
        module.main(
            [
                "verify",
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--report",
                str(report),
            ]
        )
    except ValueError as exc:
        assert "must be outside" in str(exc)
    else:
        raise AssertionError("Expected an in-artifact report to be rejected")

    assert not report.exists()


def test_external_artifact_verify_rejects_aliased_report_inside_root(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    _snapshot(module, spec, manifest)
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    artifact_parent = tmp_path / "artifacts"
    root = artifact_parent / "fixture-v1"
    alias_parent = tmp_path / "artifact-parent-alias"
    alias_parent.symlink_to(artifact_parent, target_is_directory=True)
    report = alias_parent / "fixture-v1" / "verification.json"

    try:
        module.main(
            [
                "verify",
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--report",
                str(report),
            ]
        )
    except ValueError as exc:
        assert "must be outside" in str(exc)
    else:
        raise AssertionError("Expected an aliased in-artifact report to be rejected")

    assert not report.exists()


def test_external_artifact_verify_rejects_report_overwriting_manifest(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    _snapshot(module, spec, manifest)
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    original_manifest = manifest.read_bytes()
    root = tmp_path / "artifacts" / "fixture-v1"

    try:
        module.main(
            [
                "verify",
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--report",
                str(manifest),
            ]
        )
    except ValueError as exc:
        assert "must not overwrite" in str(exc)
    else:
        raise AssertionError("Expected a manifest-overwriting report to be rejected")

    assert manifest.read_bytes() == original_manifest


def test_external_artifact_verify_rejects_report_hardlinked_to_manifest(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    original_manifest = manifest.read_bytes()
    root = tmp_path / "artifacts" / "fixture-v1"
    report = tmp_path / "verification.json"
    report.hardlink_to(manifest)

    try:
        module.main(
            [
                "verify",
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--report",
                str(report),
            ]
        )
    except ValueError as exc:
        assert "must not overwrite" in str(exc)
    else:
        raise AssertionError("Expected a manifest-hardlinked report to be rejected")

    assert manifest.read_bytes() == original_manifest


def test_external_artifact_verify_rejects_report_hardlinked_to_artifact(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    root = tmp_path / "artifacts" / "fixture-v1"
    staged_file = root / "inputs" / "input.csv"
    original_artifact = staged_file.read_bytes()
    report = tmp_path / "verification.json"
    report.hardlink_to(staged_file)

    try:
        module.main(
            [
                "verify",
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--report",
                str(report),
            ]
        )
    except ValueError as exc:
        assert "hardlink to an artifact file" in str(exc)
    else:
        raise AssertionError("Expected an artifact-hardlinked report to be rejected")

    assert staged_file.read_bytes() == original_artifact


def test_external_artifact_stage_rejects_duplicate_artifact_ids(tmp_path: Path) -> None:
    module = _load_script()
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "paper_id": "UQ_EMB",
                "artifact_set_id": "fixture",
                "artifact_set_dir": "fixture-v1",
                "locked": True,
                "destination_root": str(tmp_path / "artifacts"),
                "entries": [
                    {"artifact_id": "duplicate", "source": str(first), "destination": "first"},
                    {"artifact_id": "duplicate", "source": str(second), "destination": "second"},
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        module.plan_staging(spec)
    except ValueError as exc:
        assert "artifact_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate artifact IDs to be rejected")


def test_external_artifact_stage_requires_existing_locked_manifest(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)

    with pytest.raises(FileNotFoundError, match="snapshot command"):
        module.stage_artifacts(spec_path=spec, manifest_path=tmp_path / "missing.json")


def test_external_artifact_snapshot_refuses_manifest_overwrite(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    original = _snapshot(module, spec, manifest)

    with pytest.raises(FileExistsError):
        module.snapshot_manifest(spec_path=spec, manifest_path=manifest)

    assert manifest.read_bytes() == original


def test_external_artifact_stage_rejects_source_mutation_without_manifest_churn(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "input.csv"
    source_file.write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    original = _snapshot(module, spec, manifest)
    original_sha = hashlib.sha256(original).hexdigest()
    source_file.write_text("x,y\n9,9\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="verification failed"):
        module.stage_artifacts(spec_path=spec, manifest_path=manifest)

    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == original_sha
    assert not (tmp_path / "artifacts" / "fixture-v1").exists()


def test_external_artifact_stage_accepts_existing_locked_manifest(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "input.csv"
    source_file.write_text("accepted\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"
    locked_bytes = _snapshot(module, spec, manifest)

    report = module.stage_artifacts(spec_path=spec, manifest_path=manifest)

    assert report["status"] == "PASS"
    assert manifest.read_bytes() == locked_bytes


def test_external_artifact_stage_rejects_manifest_inside_destination(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    final_root = tmp_path / "artifacts" / "fixture-v1"
    manifest = final_root / "manifest.json"

    try:
        module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    except ValueError as exc:
        assert "must be outside" in str(exc)
    else:
        raise AssertionError("Expected an in-artifact manifest to be rejected")

    assert not final_root.exists()


def test_external_artifact_stage_rejects_aliased_manifest_inside_destination(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    artifact_parent = tmp_path / "artifacts"
    artifact_parent.mkdir()
    alias_parent = tmp_path / "artifact-parent-alias"
    alias_parent.symlink_to(artifact_parent, target_is_directory=True)
    final_root = artifact_parent / "fixture-v1"
    manifest = alias_parent / "fixture-v1" / "manifest.json"

    try:
        module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    except ValueError as exc:
        assert "must be outside" in str(exc)
    else:
        raise AssertionError("Expected an aliased in-artifact manifest to be rejected")

    assert not final_root.exists()


def test_external_artifact_stage_rejects_overlapping_target_files(tmp_path: Path) -> None:
    module = _load_script()
    directory_source = tmp_path / "directory-source"
    directory_source.mkdir()
    (directory_source / "latest").write_text("directory selection\n", encoding="utf-8")
    file_source = tmp_path / "file-source"
    file_source.write_text("file selection\n", encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "paper_id": module.PAPER_ID,
                "artifact_set_id": "fixture",
                "artifact_set_dir": "fixture-v1",
                "locked": True,
                "destination_root": str(tmp_path / "artifacts"),
                "entries": [
                    {
                        "artifact_id": "directory",
                        "source": str(directory_source),
                        "destination": "run",
                    },
                    {
                        "artifact_id": "file",
                        "source": str(file_source),
                        "destination": "run/latest",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Overlapping artifact targets"):
        module.plan_staging(spec)


def test_external_artifact_stage_rejects_empty_environment_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("accepted\n", encoding="utf-8")
    monkeypatch.setenv("EMPTY_ARTIFACT_ROOT", "")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "paper_id": module.PAPER_ID,
                "artifact_set_id": "fixture",
                "artifact_set_dir": "fixture-v1",
                "locked": True,
                "destination_root": "$EMPTY_ARTIFACT_ROOT",
                "entries": [
                    {
                        "artifact_id": "input",
                        "source": str(source),
                        "destination": "inputs",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Empty environment variable"):
        module.plan_staging(spec)
