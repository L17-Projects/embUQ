from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "workflows" / "emb" / "uq_emb"
SCRIPT = SCRIPT_DIR / "replay_provenance.py"


def _load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("uq_emb_replay_provenance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, *, artifact_set_dir: str, files: list[Path], root: Path) -> None:
    entries = [
        {
            "path": item.relative_to(root).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in files
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "paper_id": "UQ_EMB",
                "artifact_set_id": artifact_set_dir.replace("_", "-"),
                "artifact_set_dir": artifact_set_dir,
                "locked": True,
                "file_count": len(entries),
                "logical_size_bytes": sum(item["size_bytes"] for item in entries),
                "files": entries,
            }
        ),
        encoding="utf-8",
    )


def _binding(tmp_path: Path, materialize_module) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    manifest_root = repo_root / "papers" / "UQ_EMB" / "manifests"
    manifest_root.mkdir(parents=True)
    accepted_manifest = manifest_root / "accepted_production_outputs_202607.files.json"
    dependency_manifest = manifest_root / "frozen_runtime_dependencies_202607.files.json"
    artifact_root = tmp_path / "artifacts"
    accepted_root = artifact_root / materialize_module.ACCEPTED_SET
    dependency_root = artifact_root / materialize_module.DEPENDENCY_SET
    accepted_root.mkdir(parents=True)
    dependency_root.mkdir(parents=True)
    source_config = accepted_root / "source.yaml"
    source_config.write_text("source: true\n", encoding="utf-8")
    dependency_file = dependency_root / "dependency.bin"
    dependency_file.write_bytes(b"locked-dependency")
    _write_manifest(
        accepted_manifest,
        artifact_set_dir=materialize_module.ACCEPTED_SET,
        files=[source_config],
        root=accepted_root,
    )
    _write_manifest(
        dependency_manifest,
        artifact_set_dir=materialize_module.DEPENDENCY_SET,
        files=[dependency_file],
        root=dependency_root,
    )

    config_path = tmp_path / "definity_hbi_10000.yaml"
    config = {
        "out": str(tmp_path / "run"),
        "resonance": {"agent": "definity", "evaluator": {"provenance_path_overrides": {}}},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    receipt = {
        "paper_id": "UQ_EMB",
        "agent": "definity",
        "artifact_root": str(artifact_root),
        "source_config": str(source_config),
        "source_config_sha256": _sha256(source_config),
        "materialized_config": str(config_path),
        "materialized_config_sha256": _sha256(config_path),
        "semantic_config_sha256": materialize_module._semantic_config_sha256(
            config,
            agent="definity",
            dependency_root=dependency_root,
        ),
        "accepted_manifest": str(accepted_manifest),
        "accepted_manifest_sha256": _sha256(accepted_manifest),
        "dependency_manifest": str(dependency_manifest),
        "dependency_manifest_sha256": _sha256(dependency_manifest),
        "provenance": {"git_commit": "abc123", "git_status_clean": True},
    }
    config_path.with_suffix(".materialization.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return config_path, repo_root


def test_materialization_binding_enforces_semantic_manifest_and_git_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    import materialize_hbi_config

    config_path, repo_root = _binding(tmp_path, materialize_hbi_config)
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: "abc123" if args == ("rev-parse", "HEAD") else "",
    )

    binding = module.load_materialization_binding(config_path, repo_root=repo_root)

    assert binding["config_semantic_sha256"]
    assert binding["accepted_manifest_sha256"]
    assert binding["dependency_manifest_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("semantic", "semantic config hash mismatch"),
        ("manifest", "accepted_manifest_sha256 mismatch"),
        ("commit", "does not match runtime HEAD"),
    ),
)
def test_materialization_binding_rejects_tampered_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    import materialize_hbi_config

    config_path, repo_root = _binding(tmp_path, materialize_hbi_config)
    receipt_path = config_path.with_suffix(".materialization.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "semantic":
        receipt["semantic_config_sha256"] = "0" * 64
    elif mutation == "manifest":
        receipt["accepted_manifest_sha256"] = "0" * 64
    else:
        receipt["provenance"]["git_commit"] = "different"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: "abc123" if args == ("rev-parse", "HEAD") else "",
    )

    with pytest.raises(ValueError, match=message):
        module.load_materialization_binding(config_path, repo_root=repo_root)


def test_materialization_binding_rejects_dirty_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    import materialize_hbi_config

    config_path, repo_root = _binding(tmp_path, materialize_hbi_config)
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: "abc123" if args == ("rev-parse", "HEAD") else " M changed.py",
    )

    with pytest.raises(ValueError, match="clean runtime worktree"):
        module.load_materialization_binding(config_path, repo_root=repo_root)


def test_materialization_binding_rejects_mutated_dependency_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    import materialize_hbi_config

    config_path, repo_root = _binding(tmp_path, materialize_hbi_config)
    dependency = (
        tmp_path
        / "artifacts"
        / materialize_hbi_config.DEPENDENCY_SET
        / "dependency.bin"
    )
    dependency.write_bytes(b"mutated-dependency")
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: "abc123" if args == ("rev-parse", "HEAD") else "",
    )

    with pytest.raises(ValueError, match="artifact content mismatch"):
        module.load_materialization_binding(config_path, repo_root=repo_root)


def test_materialization_binding_rejects_missing_locked_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    import materialize_hbi_config

    config_path, repo_root = _binding(tmp_path, materialize_hbi_config)
    dependency_manifest = (
        repo_root
        / "papers"
        / "UQ_EMB"
        / "manifests"
        / f"{materialize_hbi_config.DEPENDENCY_SET}.files.json"
    )
    dependency_manifest.unlink()
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: "abc123" if args == ("rev-parse", "HEAD") else "",
    )

    with pytest.raises(ValueError, match="dependency_manifest_sha256 mismatch"):
        module.load_materialization_binding(config_path, repo_root=repo_root)


def test_replay_receipt_rejects_missing_default_manifest(tmp_path: Path) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    (repo_root / "papers" / "UQ_EMB" / "manifests").mkdir(parents=True)
    runner = repo_root / "runner.py"
    runner.write_text("pass\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Locked UQ_EMB manifest is missing"):
        module.replay_receipt_provenance(repo_root=repo_root, runner=runner)


def test_replay_receipt_verifies_consumed_artifact_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    manifest_root = repo_root / "papers" / "UQ_EMB" / "manifests"
    manifest_root.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts" / "frozen_runtime_dependencies_202607"
    artifact_root.mkdir(parents=True)
    dependency = artifact_root / "dependency.bin"
    dependency.write_bytes(b"locked")

    for name in module.DEFAULT_CLOSEOUT_MANIFESTS:
        if name == "frozen_runtime_dependencies_202607.files.json":
            _write_manifest(
                manifest_root / name,
                artifact_set_dir="frozen_runtime_dependencies_202607",
                files=[dependency],
                root=artifact_root,
            )
        else:
            (manifest_root / name).write_text(
                json.dumps(
                    {
                        "paper_id": "UQ_EMB",
                        "locked": True,
                        "artifact_set_dir": name.removesuffix(".files.json"),
                        "file_count": 0,
                        "logical_size_bytes": 0,
                        "files": [],
                    }
                ),
                encoding="utf-8",
            )
    runner = repo_root / "runner.py"
    runner.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: "abc123" if args == ("rev-parse", "HEAD") else "",
    )

    receipt = module.replay_receipt_provenance(
        repo_root=repo_root,
        runner=runner,
        consumed_paths=[artifact_root],
    )
    assert len(receipt["verified_artifact_sets"]) == 1

    dependency.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="artifact content mismatch"):
        module.replay_receipt_provenance(
            repo_root=repo_root,
            runner=runner,
            consumed_paths=[artifact_root],
        )
