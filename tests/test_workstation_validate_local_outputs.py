"""Tests for scripts/workstation/validate_local_outputs.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "workstation" / "validate_local_outputs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_local_outputs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_report(tmp_path: Path, *, create_files: bool = True) -> dict:
    """Build a minimal passing report; optionally create the overlay files."""
    sel = "compression:full-model:validation"
    prop_plot = tmp_path / "plots" / "prop.png"
    map_plot = tmp_path / "plots" / "map.png"
    if create_files:
        prop_plot.parent.mkdir(parents=True, exist_ok=True)
        prop_plot.write_text("plot", encoding="utf-8")
        map_plot.write_text("plot", encoding="utf-8")
    return {
        "status": "passed",
        "selections": [sel],
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": {
            sel: {
                "propagation_vs_reference_plots": [str(prop_plot)],
                "map_vs_reference_plots": [str(map_plot)],
            }
        },
    }


# ---------------------------------------------------------------------------
# validate_report: structure checks
# ---------------------------------------------------------------------------


def test_validate_report_passes_for_valid_minimal_report(tmp_path):
    module = _load_module()
    report = _minimal_report(tmp_path)
    errors = module.validate_report(report, check_files=True)
    assert errors == []


def test_validate_report_missing_required_key():
    module = _load_module()
    report = {"selections": ["x"], "overlays": {}, "phase2_backend_contract": "dual_backend"}
    errors = module.validate_report(report, check_files=False)
    assert any("status" in e for e in errors)


def test_validate_report_invalid_status():
    module = _load_module()
    report = {
        "status": "running",
        "selections": ["compression:full-model:validation"],
        "overlays": {},
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
    }
    errors = module.validate_report(report, check_files=False)
    assert any("status" in e for e in errors)


def test_validate_report_empty_selections():
    module = _load_module()
    report = {
        "status": "passed",
        "selections": [],
        "overlays": {},
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
    }
    errors = module.validate_report(report, check_files=False)
    assert any("selections" in e for e in errors)


def test_validate_report_selection_missing_from_overlays():
    module = _load_module()
    report = {
        "status": "passed",
        "selections": ["compression:full-model:validation"],
        "overlays": {},
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
    }
    errors = module.validate_report(report, check_files=False)
    assert any("missing from overlays" in e for e in errors)


def test_validate_report_non_string_selection_is_reported(tmp_path):
    module = _load_module()
    report = {
        "status": "passed",
        "selections": [{"bad": "value"}],
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": {},
    }
    errors = module.validate_report(report, check_files=False)
    assert any("selection entries must be strings" in error for error in errors)


def test_validate_report_no_propagation_plots():
    module = _load_module()
    sel = "compression:full-model:validation"
    report = {
        "status": "passed",
        "selections": [sel],
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": {
            sel: {
                "propagation_vs_reference_plots": [],
                "map_vs_reference_plots": ["/some/map.png"],
            }
        },
    }
    errors = module.validate_report(report, check_files=False)
    assert any("propagation" in e for e in errors)


def test_validate_report_no_map_plots():
    module = _load_module()
    sel = "indentation:reduced-model:validation"
    report = {
        "status": "passed",
        "selections": [sel],
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": {
            sel: {
                "propagation_vs_reference_plots": ["/some/prop.png"],
                "map_vs_reference_plots": [],
            }
        },
    }
    errors = module.validate_report(report, check_files=False)
    assert any("map" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# validate_report: file existence checks
# ---------------------------------------------------------------------------


def test_validate_report_missing_propagation_file(tmp_path):
    module = _load_module()
    sel = "compression:full-model:validation"
    map_plot = tmp_path / "map.png"
    map_plot.write_text("x", encoding="utf-8")
    report = {
        "status": "passed",
        "selections": [sel],
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": {
            sel: {
                "propagation_vs_reference_plots": [str(tmp_path / "nonexistent_prop.png")],
                "map_vs_reference_plots": [str(map_plot)],
            }
        },
    }
    errors = module.validate_report(report, check_files=True)
    assert any("propagation" in e and "nonexistent" in e for e in errors)


def test_validate_report_missing_map_file(tmp_path):
    module = _load_module()
    sel = "compression:full-model:validation"
    prop_plot = tmp_path / "prop.png"
    prop_plot.write_text("x", encoding="utf-8")
    report = {
        "status": "passed",
        "selections": [sel],
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": {
            sel: {
                "propagation_vs_reference_plots": [str(prop_plot)],
                "map_vs_reference_plots": [str(tmp_path / "nonexistent_map.png")],
            }
        },
    }
    errors = module.validate_report(report, check_files=True)
    assert any("MAP" in e and "nonexistent" in e for e in errors)


def test_validate_report_skip_file_check_ignores_missing(tmp_path):
    module = _load_module()
    sel = "indentation:full-model:validation"
    report = {
        "status": "passed",
        "selections": [sel],
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": {
            sel: {
                "propagation_vs_reference_plots": ["/does/not/exist/prop.png"],
                "map_vs_reference_plots": ["/does/not/exist/map.png"],
            }
        },
    }
    errors = module.validate_report(report, check_files=False)
    assert errors == []


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


def test_main_returns_zero_for_valid_report(tmp_path):
    module = _load_module()
    report = _minimal_report(tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    rc = module.main(["--report", str(report_path)])
    assert rc == 0


def test_main_returns_nonzero_for_invalid_report(tmp_path):
    module = _load_module()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"status": "bad"}), encoding="utf-8")
    rc = module.main(["--report", str(report_path)])
    assert rc == 1


def test_main_missing_report_file(tmp_path):
    module = _load_module()
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--report", str(tmp_path / "no_such_file.json")])
    assert exc_info.value.code != 0


def test_main_returns_zero_with_no_check_files_flag(tmp_path):
    module = _load_module()
    sel = "compression:reduced-model:validation"
    report = {
        "status": "passed",
        "selections": [sel],
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": {
            sel: {
                "propagation_vs_reference_plots": ["/ghost/prop.png"],
                "map_vs_reference_plots": ["/ghost/map.png"],
            }
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    rc = module.main(["--report", str(report_path), "--no-check-files"])
    assert rc == 0


def test_main_all_four_lanes_pass(tmp_path):
    module = _load_module()
    selections = [
        "compression:full-model:validation",
        "compression:reduced-model:validation",
        "indentation:full-model:validation",
        "indentation:reduced-model:validation",
    ]
    overlays = {}
    for sel in selections:
        prop = tmp_path / f"{sel}_prop.png"
        mp = tmp_path / f"{sel}_map.png"
        prop.write_text("p", encoding="utf-8")
        mp.write_text("m", encoding="utf-8")
        overlays[sel] = {
            "propagation_vs_reference_plots": [str(prop)],
            "map_vs_reference_plots": [str(mp)],
        }
    report = {
        "status": "passed",
        "selections": selections,
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_effective": "cpu-mpi",
        "overlays": overlays,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    rc = module.main(["--report", str(report_path)])
    assert rc == 0


def test_validate_report_rejects_unknown_phase2_contract(tmp_path):
    module = _load_module()
    report = _minimal_report(tmp_path)
    report["phase2_backend_contract"] = "cpu_mpi_only"
    errors = module.validate_report(report, check_files=False)
    assert any("phase2_backend_contract" in error for error in errors)


def test_validate_report_rejects_unknown_phase2_backend_effective(tmp_path):
    module = _load_module()
    report = _minimal_report(tmp_path)
    report["phase2_backend_effective"] = "bogus"
    errors = module.validate_report(report, check_files=False)
    assert any("phase2_backend_effective" in error for error in errors)
