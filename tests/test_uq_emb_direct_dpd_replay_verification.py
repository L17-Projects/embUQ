from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
    materializer.materialize_direct_dpd_replay(
        accepted_root=fixture_module._accepted_root(tmp_path),
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
