from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verifier_accepts_materialized_fixture(tmp_path: Path) -> None:
    fixture_module = _module(
        "tests/test_uq_emb_direct_dpd_replay.py", "uq_emb_direct_dpd_fixture"
    )
    materializer = _module(
        "scripts/workflows/emb/uq_emb/materialize_direct_dpd_replay.py",
        "uq_emb_direct_dpd_materializer_for_verification",
    )
    verifier = _module(
        "scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py",
        "uq_emb_direct_dpd_verifier",
    )
    output_root = tmp_path / "replay"
    accepted_root = fixture_module._accepted_root(tmp_path)
    accepted_manifest = fixture_module._accepted_manifest(tmp_path, accepted_root)
    materializer.materialize_direct_dpd_replay(
        accepted_root=accepted_root,
        accepted_manifest=accepted_manifest,
        output_root=output_root,
        site="karolina",
        python_bin="/usr/bin/python3.11",
    )
    report = verifier.verify(
        output_root / "direct_dpd_replay_plan.json",
        ROOT / "scripts/workflows/emb/run_emb_free_shell_breathing_protocol.py",
    )
    assert report["status"] == "PASS"
    assert report["bubble_count"] == 6
    assert report["planned_command_count"] == 12
    assert report["dpd_executed"] is False
    assert report["accepted_artifact_manifest"] == str(accepted_manifest.resolve())


def test_verifier_rejects_tampered_command(tmp_path: Path) -> None:
    fixture_module = _module(
        "tests/test_uq_emb_direct_dpd_replay.py", "uq_emb_direct_dpd_fixture_tamper"
    )
    materializer = _module(
        "scripts/workflows/emb/uq_emb/materialize_direct_dpd_replay.py",
        "uq_emb_direct_dpd_materializer_tamper",
    )
    verifier = _module(
        "scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py",
        "uq_emb_direct_dpd_verifier_tamper",
    )
    output_root = tmp_path / "replay"
    accepted_root = fixture_module._accepted_root(tmp_path)
    accepted_manifest = fixture_module._accepted_manifest(tmp_path, accepted_root)
    materializer.materialize_direct_dpd_replay(
        accepted_root=accepted_root,
        accepted_manifest=accepted_manifest,
        output_root=output_root,
        site="karolina",
        python_bin="/usr/bin/python3.11",
    )
    plan_path = output_root / "direct_dpd_replay_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["bubbles"][0]["acoustic"]["command"].extend(["--dt", "99"])
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        verifier.verify(
            plan_path,
            ROOT / "scripts/workflows/emb/run_emb_free_shell_breathing_protocol.py",
        )
    except ValueError as exc:
        assert "acoustic command differs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a tampered replay command to fail")


def test_verifier_rejects_incomplete_runtime_source_hashes(tmp_path: Path) -> None:
    fixture_module = _module(
        "tests/test_uq_emb_direct_dpd_replay.py", "uq_emb_direct_dpd_fixture_sources"
    )
    materializer = _module(
        "scripts/workflows/emb/uq_emb/materialize_direct_dpd_replay.py",
        "uq_emb_direct_dpd_materializer_sources",
    )
    verifier = _module(
        "scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py",
        "uq_emb_direct_dpd_verifier_sources",
    )
    output_root = tmp_path / "replay"
    accepted_root = fixture_module._accepted_root(tmp_path)
    accepted_manifest = fixture_module._accepted_manifest(tmp_path, accepted_root)
    materializer.materialize_direct_dpd_replay(
        accepted_root=accepted_root,
        accepted_manifest=accepted_manifest,
        output_root=output_root,
        site="karolina",
        python_bin="/usr/bin/python3.11",
    )
    plan_path = output_root / "direct_dpd_replay_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["runtime_source_hashes"].pop(next(iter(payload["runtime_source_hashes"])))
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="complete runtime source set"):
        verifier.verify(
            plan_path,
            ROOT / "scripts/workflows/emb/run_emb_free_shell_breathing_protocol.py",
        )


def test_verifier_rejects_alternate_acoustic_runner(tmp_path: Path) -> None:
    fixture_module = _module(
        "tests/test_uq_emb_direct_dpd_replay.py", "uq_emb_direct_dpd_fixture_runner"
    )
    materializer = _module(
        "scripts/workflows/emb/uq_emb/materialize_direct_dpd_replay.py",
        "uq_emb_direct_dpd_materializer_runner",
    )
    verifier = _module(
        "scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py",
        "uq_emb_direct_dpd_verifier_runner",
    )
    output_root = tmp_path / "replay"
    accepted_root = fixture_module._accepted_root(tmp_path)
    accepted_manifest = fixture_module._accepted_manifest(tmp_path, accepted_root)
    materializer.materialize_direct_dpd_replay(
        accepted_root=accepted_root,
        accepted_manifest=accepted_manifest,
        output_root=output_root,
        site="karolina",
        python_bin="/usr/bin/python3.11",
    )
    alternate_runner = tmp_path / "alternate_runner.py"
    alternate_runner.write_text("def load_bubbles(_path): return []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hashed acoustic runner"):
        verifier.verify(
            output_root / "direct_dpd_replay_plan.json",
            alternate_runner,
        )


def test_verifier_rejects_alternate_materializer_before_import(tmp_path: Path) -> None:
    fixture_module = _module(
        "tests/test_uq_emb_direct_dpd_replay.py",
        "uq_emb_direct_dpd_fixture_materializer",
    )
    materializer = _module(
        "scripts/workflows/emb/uq_emb/materialize_direct_dpd_replay.py",
        "uq_emb_direct_dpd_materializer_materializer",
    )
    verifier = _module(
        "scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py",
        "uq_emb_direct_dpd_verifier_materializer",
    )
    output_root = tmp_path / "replay"
    accepted_root = fixture_module._accepted_root(tmp_path)
    accepted_manifest = fixture_module._accepted_manifest(tmp_path, accepted_root)
    materializer.materialize_direct_dpd_replay(
        accepted_root=accepted_root,
        accepted_manifest=accepted_manifest,
        output_root=output_root,
        site="karolina",
        python_bin="/usr/bin/python3.11",
    )
    marker = tmp_path / "imported.txt"
    alternate_materializer = tmp_path / "alternate_materializer.py"
    alternate_materializer.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repository materializer"):
        verifier.verify(
            output_root / "direct_dpd_replay_plan.json",
            ROOT / "scripts/workflows/emb/run_emb_free_shell_breathing_protocol.py",
            alternate_materializer,
        )

    assert not marker.exists()


def test_verifier_rejects_unselected_accepted_artifact_drift(tmp_path: Path) -> None:
    fixture_module = _module(
        "tests/test_uq_emb_direct_dpd_replay.py", "uq_emb_direct_dpd_fixture_root_drift"
    )
    materializer = _module(
        "scripts/workflows/emb/uq_emb/materialize_direct_dpd_replay.py",
        "uq_emb_direct_dpd_materializer_root_drift",
    )
    verifier = _module(
        "scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py",
        "uq_emb_direct_dpd_verifier_root_drift",
    )
    output_root = tmp_path / "replay"
    accepted_root = fixture_module._accepted_root(tmp_path)
    accepted_manifest = fixture_module._accepted_manifest(tmp_path, accepted_root)
    materializer.materialize_direct_dpd_replay(
        accepted_root=accepted_root,
        accepted_manifest=accepted_manifest,
        output_root=output_root,
        site="karolina",
        python_bin="/usr/bin/python3.11",
    )
    (accepted_root / "unselected/audit_only.json").write_text(
        '{"purpose":"changed after materialization"}\n',
        encoding="utf-8",
    )

    try:
        verifier.verify(
            output_root / "direct_dpd_replay_plan.json",
            ROOT / "scripts/workflows/emb/run_emb_free_shell_breathing_protocol.py",
        )
    except ValueError as exc:
        assert "Locked UQ_EMB artifact content mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected whole-root accepted artifact drift to fail verification")


def test_verification_receipt_must_not_hardlink_accepted_artifact(tmp_path: Path) -> None:
    verifier = _module(
        "scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py",
        "uq_emb_direct_dpd_verifier_hardlink",
    )
    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    member = accepted_root / "result.json"
    member.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "verification.json"
    receipt.hardlink_to(member)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"accepted_artifact_root": str(accepted_root)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hardlink to immutable artifact"):
        verifier._require_receipt_outside_accepted_root(
            receipt=receipt,
            plan_path=plan,
        )


def test_verification_main_rejects_plan_as_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verifier = _module(
        "scripts/workflows/emb/uq_emb/verify_direct_dpd_replay_plan.py",
        "uq_emb_direct_dpd_verifier_input_collision",
    )
    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    accepted_manifest = tmp_path / "accepted.json"
    accepted_manifest.write_text("{}\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "accepted_artifact_root": str(accepted_root),
                "accepted_artifact_manifest": str(accepted_manifest),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_direct_dpd_replay_plan.py",
            "--plan",
            str(plan),
            "--receipt",
            str(plan),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()
    assert exc_info.value.code == 2
    assert "must not overwrite or hardlink input" in capsys.readouterr().err
