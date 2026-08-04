from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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


def test_external_artifact_stage_and_verify(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    manifest = tmp_path / "manifest.json"

    plan = module.plan_staging(spec)
    report = module.stage_artifacts(spec_path=spec, manifest_path=manifest)

    assert plan["status"] == "PASS"
    assert plan["file_count"] == 1
    assert report["status"] == "PASS"
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
    module.stage_artifacts(spec_path=spec, manifest_path=manifest)
    staged_file = tmp_path / "artifacts" / "fixture-v1" / "inputs" / "input.csv"
    staged_file.write_text("mutated\n", encoding="utf-8")

    report = module.verify_staged(root=staged_file.parents[1], manifest_path=manifest)

    assert report["status"] == "FAIL"
    assert report["mismatches"]


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
