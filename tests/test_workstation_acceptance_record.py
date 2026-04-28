from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "platforms" / "workstation" / "validate_acceptance_record.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_workstation_acceptance_record", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_workstation_acceptance_report_is_structurally_valid() -> None:
    module = _load_module()
    example_path = REPO_ROOT / "examples" / "reports" / "workstation_acceptance_report.example.json"
    report = json.loads(example_path.read_text(encoding="utf-8"))

    assert module.validate_report(report) == []


def test_validator_requires_existing_paths_when_requested(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "target": "workstation",
        "status": "passed",
        "environment": {
            "timestamp_utc": "2026-04-09T09:30:00Z",
            "hostname": "gpu-workstation-01",
            "platform": "Linux",
            "python_bin": "/usr/bin/python3",
            "python_version": "3.11.9",
            "gpu_summary": "NVIDIA RTX A5000",
            "git_commit": "16cc54ec762339c0b4de44d5a4e0bb332de48130",
        },
        "steps": [
            {
                "name": "install",
                "status": "passed",
                "commands": ["python -m pip install -e '.[ci]'"],
                "artifacts": {"log": str(tmp_path / "install.log")},
            },
            {
                "name": "surrogate_retraining",
                "status": "passed",
                "commands": ["python scripts/qa/ci/run_retraining_canary.py"],
                "artifacts": {"output_root": str(tmp_path / "retraining")},
            },
            {
                "name": "phase1_gpu_batched",
                "status": "passed",
                "commands": ["python inference/scripts/run_phase_1.py --gpu-batch"],
                "artifacts": {"result_root": str(tmp_path / "phase1")},
            },
            {
                "name": "phase3b_gpu_batched",
                "status": "passed",
                "commands": ["python inference/scripts/run_phase_3b.py --gpu-batch"],
                "artifacts": {"result_root": str(tmp_path / "phase3b")},
            },
            {
                "name": "map_extraction_and_plotting",
                "status": "passed",
                "commands": ["python scripts/platforms/vega/extract_map.py"],
                "artifacts": {"map_root": str(tmp_path / "map")},
            },
        ],
        "artifacts": {
            "output_root": str(tmp_path),
            "report_path": str(tmp_path / "workstation_acceptance_report.json"),
        },
    }

    errors = module.validate_report(report, must_exist=True)
    assert errors
    assert any("missing step artifact path" in error for error in errors)


def test_validator_accepts_existing_paths(tmp_path: Path) -> None:
    module = _load_module()
    install_log = tmp_path / "install.log"
    retraining_root = tmp_path / "retraining"
    phase1_root = tmp_path / "phase1"
    phase3b_root = tmp_path / "phase3b"
    map_root = tmp_path / "map"
    report_path = tmp_path / "workstation_acceptance_report.json"

    install_log.write_text("ok", encoding="utf-8")
    retraining_root.mkdir()
    phase1_root.mkdir()
    phase3b_root.mkdir()
    map_root.mkdir()
    report_path.write_text("{}", encoding="utf-8")

    report = {
        "target": "workstation",
        "status": "passed",
        "environment": {
            "timestamp_utc": "2026-04-09T09:30:00Z",
            "hostname": "gpu-workstation-01",
            "platform": "Linux",
            "python_bin": "/usr/bin/python3",
            "python_version": "3.11.9",
            "gpu_summary": "NVIDIA RTX A5000",
            "git_commit": "16cc54ec762339c0b4de44d5a4e0bb332de48130",
        },
        "steps": [
            {
                "name": "install",
                "status": "passed",
                "commands": ["python -m pip install -e '.[ci]'"],
                "artifacts": {"log": str(install_log)},
            },
            {
                "name": "surrogate_retraining",
                "status": "passed",
                "commands": ["python scripts/qa/ci/run_retraining_canary.py"],
                "artifacts": {"output_root": str(retraining_root)},
            },
            {
                "name": "phase1_gpu_batched",
                "status": "passed",
                "commands": ["python inference/scripts/run_phase_1.py --gpu-batch"],
                "artifacts": {"result_root": str(phase1_root)},
            },
            {
                "name": "phase3b_gpu_batched",
                "status": "passed",
                "commands": ["python inference/scripts/run_phase_3b.py --gpu-batch"],
                "artifacts": {"result_root": str(phase3b_root)},
            },
            {
                "name": "map_extraction_and_plotting",
                "status": "passed",
                "commands": ["python scripts/platforms/vega/extract_map.py"],
                "artifacts": {"map_root": str(map_root)},
            },
        ],
        "artifacts": {
            "output_root": str(tmp_path),
            "report_path": str(report_path),
        },
    }

    assert module.validate_report(report, must_exist=True) == []
