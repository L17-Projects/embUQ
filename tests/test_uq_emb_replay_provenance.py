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
    seed_log = accepted_root / materialize_module.AGENT_PATHS["definity"]["stage_seed_log"]
    seed_log.parent.mkdir(parents=True)
    seed_log.write_text(
        "\n".join(
            (
                "[Korali] Random Seed: 1101",
                "[HBI] Random Seed: 2101",
                "[Phase 3b] Random Seed for compression_2.1um: 3104",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    dependency_file = dependency_root / "dependency.bin"
    dependency_file.write_bytes(b"locked-dependency")
    _write_manifest(
        accepted_manifest,
        artifact_set_dir=materialize_module.ACCEPTED_SET,
        files=[source_config, seed_log],
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
        "accepted_stage_seeds": {"phase1": 1101, "phase2": 2101, "phase3b": 3104},
        "accepted_phase3b_seed_mode": "repeat",
        "accepted_stage_seed_log": str(seed_log),
        "accepted_stage_seed_log_sha256": _sha256(seed_log),
    }
    config_path.with_suffix(".materialization.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return config_path, repo_root


def _alias_path(tmp_path: Path, target: Path, *, kind: str, name: str) -> Path:
    if kind == "direct":
        alias = tmp_path / name
        alias.symlink_to(target, target_is_directory=target.is_dir())
        return alias
    parent_alias = tmp_path / name
    parent_alias.symlink_to(target.parent, target_is_directory=True)
    return parent_alias / target.name


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
    assert binding["accepted_root_verification"]["status"] == "PASS"
    assert binding["accepted_phase3b_seed_mode"] == "repeat"


def test_materialization_binding_rejects_unconsumed_accepted_root_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    import materialize_hbi_config

    config_path, repo_root = _binding(tmp_path, materialize_hbi_config)
    receipt = json.loads(
        config_path.with_suffix(".materialization.json").read_text(encoding="utf-8")
    )
    accepted_root = Path(receipt["artifact_root"]) / materialize_hbi_config.ACCEPTED_SET
    (accepted_root / "unconsumed.txt").write_text("drift\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: "abc123" if args == ("rev-parse", "HEAD") else "",
    )

    with pytest.raises(ValueError, match="inventory mismatch"):
        module.load_materialization_binding(config_path, repo_root=repo_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("semantic", "semantic config hash mismatch"),
        ("manifest", "accepted_manifest_sha256 mismatch"),
        ("commit", "does not match runtime HEAD"),
        ("seed", "accepted stage seeds differ"),
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
    elif mutation == "seed":
        receipt["accepted_stage_seeds"]["phase1"] = 9999
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


@pytest.mark.parametrize("kind", ("direct", "ancestor"))
def test_materialization_binding_rejects_lexical_config_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    module = _load_module()
    import materialize_hbi_config

    config_path, repo_root = _binding(tmp_path, materialize_hbi_config)
    alias = _alias_path(tmp_path, config_path, kind=kind, name=f"config_{kind}_alias")
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: "abc123" if args == ("rev-parse", "HEAD") else "",
    )

    with pytest.raises(ValueError, match="symlinked path or ancestor"):
        module.load_materialization_binding(alias, repo_root=repo_root)


@pytest.mark.parametrize("kind", ("direct", "ancestor"))
def test_locked_artifact_and_manifest_reject_lexical_aliases(
    tmp_path: Path,
    kind: str,
) -> None:
    module = _load_module()
    import materialize_hbi_config

    config_path, repo_root = _binding(tmp_path, materialize_hbi_config)
    artifact_root = tmp_path / "artifacts" / materialize_hbi_config.ACCEPTED_SET
    manifest_path = (
        repo_root
        / "papers"
        / "UQ_EMB"
        / "manifests"
        / f"{materialize_hbi_config.ACCEPTED_SET}.files.json"
    )
    artifact_alias = _alias_path(
        tmp_path, artifact_root, kind=kind, name=f"artifact_{kind}_alias"
    )
    manifest_alias = _alias_path(
        tmp_path, manifest_path, kind=kind, name=f"manifest_{kind}_alias"
    )

    with pytest.raises(ValueError, match="symlinked path or ancestor"):
        module.verify_locked_artifact_root(root=artifact_alias, manifest_path=manifest_path)
    with pytest.raises(ValueError, match="symlinked path or ancestor"):
        module._locked_manifest(manifest_alias)


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

    direct_alias = _alias_path(tmp_path, artifact_root, kind="direct", name="consumed_direct")
    ancestor_alias = _alias_path(
        tmp_path, artifact_root, kind="ancestor", name="consumed_ancestor"
    )
    for alias in (direct_alias, ancestor_alias):
        with pytest.raises(ValueError, match="symlinked path or ancestor"):
            module.replay_receipt_provenance(
                repo_root=repo_root,
                runner=runner,
                consumed_paths=[alias],
            )

    dependency.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="artifact content mismatch"):
        module.replay_receipt_provenance(
            repo_root=repo_root,
            runner=runner,
            consumed_paths=[artifact_root],
        )


def test_replay_output_must_remain_outside_consumed_locked_roots(tmp_path: Path) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    manifest_root = repo_root / "papers" / "UQ_EMB" / "manifests"
    manifest_root.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts" / "frozen_runtime_dependencies_202607"
    artifact_root.mkdir(parents=True)
    dependency = artifact_root / "dependency.bin"
    dependency.write_bytes(b"locked")

    for name in module.DEFAULT_CLOSEOUT_MANIFESTS:
        artifact_set_dir = name.removesuffix(".files.json")
        files: list[Path] = []
        root = tmp_path / "artifacts" / artifact_set_dir
        root.mkdir(exist_ok=True)
        if name == "frozen_runtime_dependencies_202607.files.json":
            root = artifact_root
            files = [dependency]
        _write_manifest(
            manifest_root / name,
            artifact_set_dir=artifact_set_dir,
            files=files,
            root=root,
        )

    forbidden = artifact_root / "rendered"
    with pytest.raises(ValueError, match="outside immutable (?:consumed )?artifact roots"):
        module.require_output_outside_consumed_roots(
            output_path=forbidden,
            repo_root=repo_root,
            consumed_paths=[dependency],
        )
    assert not forbidden.exists()

    allowed = tmp_path / "rendered"
    assert module.require_output_outside_consumed_roots(
        output_path=allowed,
        repo_root=repo_root,
        consumed_paths=[dependency],
    ) == allowed.resolve()

    unconsumed_root = tmp_path / "artifacts" / "frozen_plotting_dependencies_202607"
    with pytest.raises(ValueError, match="outside immutable (?:consumed )?artifact roots"):
        module.require_output_outside_consumed_roots(
            output_path=unconsumed_root / "rendered",
            repo_root=repo_root,
            consumed_paths=[dependency],
        )

    hardlink = tmp_path / "dependency-hardlink.bin"
    hardlink.hardlink_to(dependency)
    with pytest.raises(
        ValueError, match="(?:hardlink to immutable artifact|existing multi-link file)"
    ):
        module.require_output_outside_consumed_roots(
            output_path=hardlink,
            repo_root=repo_root,
            consumed_paths=[dependency],
        )


def test_distinct_output_rejects_input_hardlink_and_symlink(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(source)
    with pytest.raises(ValueError, match="must not overwrite or hardlink input"):
        module.require_output_distinct_from_inputs(
            output_path=hardlink,
            input_paths=[source],
            label="Comparison output",
        )

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(tmp_path / "new-output.json")
    with pytest.raises(ValueError, match="symlinked path or ancestor"):
        module.require_output_distinct_from_inputs(
            output_path=symlink,
            input_paths=[source],
            label="Comparison output",
        )


def test_known_locked_root_guard_rejects_cross_root_and_unrelated_hardlink(
    tmp_path: Path,
) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    manifest_root = repo_root / "papers" / "UQ_EMB" / "manifests"
    manifest_root.mkdir(parents=True)
    for name in module.DEFAULT_CLOSEOUT_MANIFESTS:
        artifact_set_dir = name.removesuffix(".files.json")
        _write_manifest(
            manifest_root / name,
            artifact_set_dir=artifact_set_dir,
            files=[],
            root=tmp_path / artifact_set_dir,
        )

    cross_root = tmp_path / "frozen_plotting_dependencies_202607" / "receipt.json"
    with pytest.raises(ValueError, match="outside immutable artifact roots"):
        module.require_output_outside_known_locked_roots(
            output_path=cross_root,
            repo_root=repo_root,
            label="Replay receipt",
        )

    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("{}\n", encoding="utf-8")
    hardlink = tmp_path / "receipt.json"
    hardlink.hardlink_to(unrelated)
    with pytest.raises(ValueError, match="existing multi-link file"):
        module.require_output_outside_known_locked_roots(
            output_path=hardlink,
            repo_root=repo_root,
            label="Replay receipt",
        )
