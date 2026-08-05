from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "workflows" / "emb" / "uq_emb" / "verify_frozen_submission.py"
SNAPSHOT_ROOT = REPO_ROOT / "papers" / "UQ_EMB" / "editor_submission" / "review2_v1"
MANIFEST = REPO_ROOT / "papers" / "UQ_EMB" / "manifests" / "editor_submission_review2_v1.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_frozen_submission", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tracked_review2_snapshot_matches_locked_manifest() -> None:
    module = _load_script()
    report = module.verify_snapshot(root=SNAPSHOT_ROOT, manifest_path=MANIFEST)

    assert report["status"] == "PASS"
    assert report["file_count"] == 32
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["mismatches"] == []


def test_review2_snapshot_excludes_response_letter_and_markdown() -> None:
    tracked = {path.name for path in SNAPSHOT_ROOT.iterdir() if path.is_file()}

    assert "response_letter.pdf" not in tracked
    assert not any(name.lower().endswith(".md") for name in tracked)
    assert {"main.tex", "si.tex", "main.pdf", "main_marked_up.pdf"}.issubset(tracked)


def test_verifier_rejects_checksum_mutation(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "main.tex").write_text("original\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    entry = module._entry(root, root / "main.tex")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "paper_id": module.PAPER_ID,
                "snapshot_id": "fixture",
                "locked": True,
                "file_count": 1,
                "total_size_bytes": entry["size_bytes"],
                "files": [entry],
            }
        ),
        encoding="utf-8",
    )
    (root / "main.tex").write_text("mutated\n", encoding="utf-8")

    report = module.verify_snapshot(root=root, manifest_path=manifest_path)

    assert report["status"] == "FAIL"
    assert report["mismatches"]


def test_snapshot_refuses_symlinks(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    target = source / "main.tex"
    target.write_text("main\n", encoding="utf-8")
    (source / "alias.tex").symlink_to(target)

    try:
        module.create_snapshot(
            source=source,
            destination=tmp_path / "destination",
            manifest_path=tmp_path / "manifest.json",
            snapshot_id="fixture",
            source_hint="fixture",
        )
    except ValueError as exc:
        assert "symlinks" in str(exc)
    else:
        raise AssertionError("Expected symlinked snapshot source to be rejected")


def test_verifier_rejects_symlinked_snapshot_root(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "main.tex").write_text("main\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    entry = module._entry(root, root / "main.tex")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "paper_id": module.PAPER_ID,
                "snapshot_id": "fixture",
                "locked": True,
                "file_count": 1,
                "total_size_bytes": entry["size_bytes"],
                "files": [entry],
            }
        ),
        encoding="utf-8",
    )
    alias = tmp_path / "snapshot-alias"
    alias.symlink_to(root, target_is_directory=True)

    try:
        module.main(
            [
                "verify",
                "--root",
                str(alias),
                "--manifest",
                str(manifest_path),
            ]
        )
    except ValueError as exc:
        assert "roots cannot be symlinks" in str(exc)
    else:
        raise AssertionError("Expected a symlinked snapshot root to be rejected")


def test_snapshot_rejects_manifest_inside_destination(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.tex").write_text("main\n", encoding="utf-8")
    destination = tmp_path / "destination"

    try:
        module.create_snapshot(
            source=source,
            destination=destination,
            manifest_path=destination / "manifest.json",
            snapshot_id="fixture",
            source_hint="fixture",
        )
    except ValueError as exc:
        assert "must be outside" in str(exc)
    else:
        raise AssertionError("Expected an in-snapshot manifest to be rejected")

    assert not destination.exists()


def test_snapshot_rejects_manifest_inside_destination_through_parent_alias(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.tex").write_text("main\n", encoding="utf-8")
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    destination = real_parent / "destination"
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    try:
        module.create_snapshot(
            source=source,
            destination=destination,
            manifest_path=alias_parent / "destination" / "manifest.json",
            snapshot_id="fixture",
            source_hint="fixture",
        )
    except ValueError as exc:
        assert "must be outside" in str(exc)
    else:
        raise AssertionError("Expected aliased in-snapshot manifest to be rejected")

    assert not destination.exists()


def test_verifier_rejects_report_inside_snapshot(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "main.tex").write_text("main\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    entry = module._entry(root, root / "main.tex")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "paper_id": module.PAPER_ID,
                "snapshot_id": "fixture",
                "locked": True,
                "file_count": 1,
                "total_size_bytes": entry["size_bytes"],
                "files": [entry],
            }
        ),
        encoding="utf-8",
    )

    try:
        module.main(
            [
                "verify",
                "--root",
                str(root),
                "--manifest",
                str(manifest_path),
                "--report",
                str(root / "verification.json"),
            ]
        )
    except ValueError as exc:
        assert "must be outside" in str(exc)
    else:
        raise AssertionError("Expected an in-snapshot verification report to be rejected")

    assert not (root / "verification.json").exists()


def test_verifier_rejects_report_inside_snapshot_through_parent_alias(
    tmp_path: Path,
) -> None:
    module = _load_script()
    real_parent = tmp_path / "real"
    root = real_parent / "snapshot"
    root.mkdir(parents=True)
    (root / "main.tex").write_text("main\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    entry = module._entry(root, root / "main.tex")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "paper_id": module.PAPER_ID,
                "snapshot_id": "fixture",
                "locked": True,
                "file_count": 1,
                "total_size_bytes": entry["size_bytes"],
                "files": [entry],
            }
        ),
        encoding="utf-8",
    )
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    report_path = alias_parent / "snapshot" / "verification.json"

    try:
        module.main(
            [
                "verify",
                "--root",
                str(alias_parent / "snapshot"),
                "--manifest",
                str(manifest_path),
                "--report",
                str(report_path),
            ]
        )
    except ValueError as exc:
        assert "must be outside" in str(exc)
    else:
        raise AssertionError("Expected aliased in-snapshot report to be rejected")

    assert not report_path.exists()


def test_verifier_rejects_report_that_overwrites_manifest(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "main.tex").write_text("main\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    entry = module._entry(root, root / "main.tex")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "paper_id": module.PAPER_ID,
                "snapshot_id": "fixture",
                "locked": True,
                "file_count": 1,
                "total_size_bytes": entry["size_bytes"],
                "files": [entry],
            }
        ),
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()

    try:
        module.main(
            [
                "verify",
                "--root",
                str(root),
                "--manifest",
                str(manifest_path),
                "--report",
                str(manifest_path),
            ]
        )
    except ValueError as exc:
        assert "must not overwrite" in str(exc)
    else:
        raise AssertionError("Expected manifest-overwriting report to be rejected")

    assert manifest_path.read_bytes() == original_manifest


def test_verifier_writes_report_outside_snapshot(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "main.tex").write_text("main\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    entry = module._entry(root, root / "main.tex")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "paper_id": module.PAPER_ID,
                "snapshot_id": "fixture",
                "locked": True,
                "file_count": 1,
                "total_size_bytes": entry["size_bytes"],
                "files": [entry],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "verification.json"

    exit_code = module.main(
        [
            "verify",
            "--root",
            str(root),
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_snapshot_copies_within_size_limits(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.tex").write_text("main\n", encoding="utf-8")
    destination = tmp_path / "destination"
    manifest_path = tmp_path / "manifest.json"

    report = module.create_snapshot(
        source=source,
        destination=destination,
        manifest_path=manifest_path,
        snapshot_id="fixture",
        source_hint="fixture",
    )

    assert report["status"] == "PASS"
    assert (destination / "main.tex").read_text(encoding="utf-8") == "main\n"
    assert manifest_path.is_file()


def test_snapshot_rejects_symlinked_destination_before_copy(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.tex").write_text("main\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    destination = tmp_path / "destination"
    destination.symlink_to(external, target_is_directory=True)
    manifest_path = tmp_path / "manifest.json"

    try:
        module.create_snapshot(
            source=source,
            destination=destination,
            manifest_path=manifest_path,
            snapshot_id="fixture",
            source_hint="fixture",
        )
    except ValueError as exc:
        assert "destinations cannot be symlinks" in str(exc)
    else:
        raise AssertionError("Expected a symlinked snapshot destination to be rejected")

    assert list(external.iterdir()) == []
    assert not manifest_path.exists()


def test_snapshot_enforces_per_file_size_limit(tmp_path: Path) -> None:
    module = _load_script()
    module.MAX_SNAPSHOT_FILE_BYTES = 4
    source = tmp_path / "source"
    source.mkdir()
    (source / "large.pdf").write_bytes(b"12345")

    try:
        module.create_snapshot(
            source=source,
            destination=tmp_path / "destination",
            manifest_path=tmp_path / "manifest.json",
            snapshot_id="fixture",
            source_hint="fixture",
        )
    except ValueError as exc:
        assert "per-file limit" in str(exc)
    else:
        raise AssertionError("Expected an oversized snapshot file to be rejected")

    assert not (tmp_path / "destination").exists()


def test_snapshot_enforces_total_size_limit(tmp_path: Path) -> None:
    module = _load_script()
    module.MAX_SNAPSHOT_FILE_BYTES = 10
    module.MAX_SNAPSHOT_TOTAL_BYTES = 6
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.tex").write_bytes(b"1234")
    (source / "two.tex").write_bytes(b"5678")

    try:
        module.create_snapshot(
            source=source,
            destination=tmp_path / "destination",
            manifest_path=tmp_path / "manifest.json",
            snapshot_id="fixture",
            source_hint="fixture",
        )
    except ValueError as exc:
        assert "total size limit" in str(exc)
    else:
        raise AssertionError("Expected an oversized snapshot to be rejected")

    assert not (tmp_path / "destination").exists()


def test_frozen_submission_disables_git_text_conversion() -> None:
    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "--",
            "papers/UQ_EMB/editor_submission/review2_v1/main.tex",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().endswith("text: unset")
