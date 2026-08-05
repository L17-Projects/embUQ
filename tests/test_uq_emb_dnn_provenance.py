from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/workflows/emb/uq_emb/audit_dnn_surrogate_provenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_dnn_surrogate_provenance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, module):
    dependency_root = tmp_path / "dependencies"
    files = []
    patched_cases = []
    for index, source in enumerate(module.CASES):
        case = dict(source)
        case["rows"] = index + 1
        data = dependency_root / case["data"]
        model = dependency_root / case["model"]
        data.parent.mkdir(parents=True, exist_ok=True)
        model.parent.mkdir(parents=True, exist_ok=True)
        data.write_text("row\n" * case["rows"], encoding="utf-8")
        model.write_bytes(f"model-{index}".encode())
        files.extend(
            (
                {"path": case["data"], "sha256": _sha256(data)},
                {"path": case["model"], "sha256": _sha256(model)},
            )
        )
        patched_cases.append(case)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "paper_id": module.PAPER_ID,
                "artifact_set_id": module.DEPENDENCY_SET,
                "locked": True,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return dependency_root, manifest, tuple(patched_cases)


def test_audit_verifies_artifacts_and_labels_refresh_as_prospective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    dependency_root, manifest, cases = _fixture(tmp_path, module)
    module.CASES = cases

    payload = module.audit(
        dependency_root=dependency_root,
        manifest_path=manifest,
        repo_root=REPO_ROOT,
        output_root=tmp_path / "refresh",
        python_bin="/verified/env/bin/python",
        seed=17,
        max_epoch=80,
    )

    assert payload["status"] == "PASS"
    assert payload["accepted_artifacts"] == "immutable_verified"
    assert payload["exact_retraining"] is False
    assert len(payload["cases"]) == 6
    command = payload["cases"][0]["refresh_command"]
    assert command[0] == "/verified/env/bin/python"
    assert command[-4:] == ["--max-epoch", "80", "--seed", "17"]


def test_audit_rejects_mutated_frozen_model(tmp_path: Path) -> None:
    module = _load_module()
    dependency_root, manifest, cases = _fixture(tmp_path, module)
    module.CASES = cases
    (dependency_root / cases[0]["model"]).write_text("mutated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        module.audit(
            dependency_root=dependency_root,
            manifest_path=manifest,
            repo_root=REPO_ROOT,
            output_root=tmp_path / "refresh",
            python_bin="python",
            seed=17,
            max_epoch=80,
        )


def test_audit_rejects_unlocked_manifest(tmp_path: Path) -> None:
    module = _load_module()
    dependency_root, manifest, cases = _fixture(tmp_path, module)
    module.CASES = cases
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["locked"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not locked"):
        module.audit(
            dependency_root=dependency_root,
            manifest_path=manifest,
            repo_root=REPO_ROOT,
            output_root=tmp_path / "refresh",
            python_bin="python",
            seed=17,
            max_epoch=80,
        )


def test_audit_receipt_must_remain_outside_dependency_root(tmp_path: Path) -> None:
    module = _load_module()
    dependency_root = tmp_path / "dependencies"

    with pytest.raises(ValueError, match="outside the immutable dependency root"):
        module._require_output_outside_locked_root(
            output=dependency_root / "audit.json",
            locked_root=dependency_root,
        )


def test_audit_receipt_must_not_hardlink_dependency(tmp_path: Path) -> None:
    module = _load_module()
    dependency_root = tmp_path / "dependencies"
    dependency_root.mkdir()
    dependency = dependency_root / "model.pkl"
    dependency.write_bytes(b"locked")
    receipt = tmp_path / "audit.json"
    receipt.hardlink_to(dependency)

    with pytest.raises(ValueError, match="hardlink to immutable artifact"):
        module._require_output_outside_locked_root(
            output=receipt,
            locked_root=dependency_root,
        )


def test_audit_main_rejects_manifest_as_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    dependency_root = tmp_path / "dependencies"
    dependency_root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--dependency-root",
            str(dependency_root),
            "--manifest",
            str(manifest),
            "--repo-root",
            str(REPO_ROOT),
            "--output-root",
            str(tmp_path / "refresh"),
            "--python-bin",
            "/verified/python",
            "--receipt",
            str(manifest),
        ],
    )

    with pytest.raises(ValueError, match="must not overwrite or hardlink input"):
        module.main()


def test_audit_main_rejects_refresh_output_inside_dependency_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    dependency_root = tmp_path / "dependencies"
    dependency_root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--dependency-root",
            str(dependency_root),
            "--manifest",
            str(manifest),
            "--repo-root",
            str(REPO_ROOT),
            "--output-root",
            str(dependency_root / "refresh"),
            "--python-bin",
            "/verified/python",
            "--receipt",
            str(tmp_path / "receipt.json"),
        ],
    )

    with pytest.raises(ValueError, match="outside the immutable dependency root"):
        module.main()
