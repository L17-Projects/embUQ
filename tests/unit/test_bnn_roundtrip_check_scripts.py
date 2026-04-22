from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_karolina_roundtrip_main_writes_pass_result(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_pass_test",
    )

    def _fake_check(args, run_dir):  # noqa: ANN001
        del args
        return {
            "status": "passed",
            "selection": "indentation_3.2um",
            "run_dir": str(run_dir),
            "updated_at": "2026-04-22T00:00:00+00:00",
        }

    monkeypatch.setattr(module, "run_roundtrip_check", _fake_check)

    rc = module.main(["--output-root", str(tmp_path), "--selection", "indentation_3.2um", "--site", "vega"])
    assert rc == 0

    result_path = tmp_path / "indentation_3.2um" / "roundtrip_strict_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"


def test_karolina_roundtrip_main_writes_failure_result(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_fail_test",
    )

    def _boom(args, run_dir):  # noqa: ANN001
        del args, run_dir
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(module, "run_roundtrip_check", _boom)

    rc = module.main(["--output-root", str(tmp_path), "--selection", "indentation_3.2um", "--site", "vega"])
    assert rc == 1

    result_path = tmp_path / "indentation_3.2um" / "roundtrip_strict_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "synthetic failure" in payload["error"]


def test_karolina_roundtrip_helpers_cover_selection_and_degradation() -> None:
    module = _load_module(
        Path("scripts/karolina/run_bnn_roundtrip_check.py"),
        "run_bnn_roundtrip_check_helper_test",
    )
    spec = module._resolve_selection_spec("compression_2.1um")
    assert spec["modality"] == "compression"
    assert spec["diameter_um"] == "2.1"
    assert spec["data"].endswith("compression/surrogate/diameters/2.1um/data/F_Delta.dat")

    abs_deg, rel_deg = module._compute_degradation(0.08, 0.10)
    assert abs_deg == pytest.approx(0.02)
    assert rel_deg == pytest.approx(0.25)


def test_hpc_roundtrip_wrapper_dispatches_to_selected_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/hpc/run_bnn_roundtrip_check.py"),
        "hpc_roundtrip_dispatch_test",
    )
    monkeypatch.setenv("HPC_SITE", "karolina")
    captured: list[list[str]] = []

    def _fake_call(cmd):  # noqa: ANN001
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    rc = module.main(["--selection", "indentation_3.2um"])
    assert rc == 0
    assert captured
    joined = " ".join(captured[0])
    assert sys.executable in captured[0][0]
    assert "scripts/karolina/run_bnn_roundtrip_check.py" in joined
    assert "--selection indentation_3.2um" in joined
