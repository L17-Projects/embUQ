from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _clear_runtime_modules() -> None:
    for name in list(sys.modules):
        if name.startswith("meso_uq.structures.gv.runtime"):
            sys.modules.pop(name, None)


def _import_without_mirheo(monkeypatch: pytest.MonkeyPatch, module_name: str):
    original_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mirheo" or name.startswith("mirheo."):
            raise AssertionError("Mirheo must not be imported while loading dry-run descriptors.")
        return original_import(name, globals, locals, fromlist, level)

    _clear_runtime_modules()
    monkeypatch.setattr("builtins.__import__", guarded_import)
    return importlib.import_module(module_name)


def test_runtime_descriptor_imports_do_not_require_mirheo(monkeypatch: pytest.MonkeyPatch) -> None:
    stretching = _import_without_mirheo(monkeypatch, "meso_uq.structures.gv.runtime.stretching")
    buckling = _import_without_mirheo(monkeypatch, "meso_uq.structures.gv.runtime.buckling")

    assert stretching.EXPERIMENT_NAME == "stretching"
    assert buckling.EXPERIMENT_NAME == "buckling"


def test_stretching_uses_control_sweeps_from_run_all(tmp_path: Path) -> None:
    module = importlib.import_module("meso_uq.structures.gv.runtime.stretching")

    before = list(tmp_path.rglob("*"))
    dry_run = module.build_dry_run_descriptor(tmp_path / "planner")
    manifest = dry_run.to_manifest()
    after = list(tmp_path.rglob("*"))

    control_sweeps = {sweep.name: sweep for sweep in module.RUNTIME_DESCRIPTOR.control_sweeps}

    assert module.RUNTIME_DESCRIPTOR.control_names == ("tot_force", "bpress")
    assert module.get_runtime_descriptor() is module.RUNTIME_DESCRIPTOR
    assert module.RUNTIME_DESCRIPTOR.sweep_mode == "forward"
    assert module.RUNTIME_DESCRIPTOR.first_restart is True
    assert control_sweeps["tot_force"].values()[0] == pytest.approx(500.0)
    assert control_sweeps["tot_force"].values()[-1] == pytest.approx(50000.0)
    assert len(control_sweeps["tot_force"].values()) == 80
    assert control_sweeps["bpress"].values() == (-91.0,)
    assert control_sweeps["bpress"].stop == pytest.approx(-100.0)
    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "stretching"
    assert manifest["geometry"] == dry_run.geometry
    assert manifest["controls"] == {"tot_force": 500.0, "bpress": -91.0}
    assert manifest["work_dir"].startswith(str((tmp_path / "planner").resolve()))
    assert str(module.PROVENANCE_ROOT).endswith("gv_simulation_files/stretching/gv")
    assert set(module.RUNTIME_DESCRIPTOR.source_files) == {
        str(module.PROVENANCE_ROOT / "clean_all.sh"),
        str(module.PROVENANCE_ROOT / "run_all.sh"),
        str(module.PROVENANCE_ROOT / "generate.py"),
        str(module.PROVENANCE_ROOT / "parameters.py"),
        str(module.PROVENANCE_ROOT / "run.sh"),
        str(module.PROVENANCE_ROOT / "equil.py"),
        str(module.PROVENANCE_ROOT / "parameters-default.gv.yaml"),
    }
    assert before == after == []


def test_buckling_uses_control_sweeps_from_run_all(tmp_path: Path) -> None:
    module = importlib.import_module("meso_uq.structures.gv.runtime.buckling")

    before = list(tmp_path.rglob("*"))
    dry_run = module.build_dry_run_descriptor(tmp_path / "planner")
    manifest = dry_run.to_manifest()
    after = list(tmp_path.rglob("*"))

    control_sweeps = {sweep.name: sweep for sweep in module.RUNTIME_DESCRIPTOR.control_sweeps}

    assert module.RUNTIME_DESCRIPTOR.control_names == ("buck", "bpress")
    assert module.get_runtime_descriptor() is module.RUNTIME_DESCRIPTOR
    assert module.RUNTIME_DESCRIPTOR.sweep_mode == "forward"
    assert module.RUNTIME_DESCRIPTOR.first_restart is False
    assert control_sweeps["buck"].values()[0] == pytest.approx(0.0)
    assert control_sweeps["buck"].values()[-1] == pytest.approx(0.75)
    assert len(control_sweeps["buck"].values()) == 50
    assert control_sweeps["bpress"].values() == (-91.0,)
    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "buckling"
    assert manifest["geometry"] == dry_run.geometry
    assert manifest["controls"] == {"buck": 0.0, "bpress": -91.0}
    assert manifest["work_dir"].startswith(str((tmp_path / "planner").resolve()))
    assert str(module.PROVENANCE_ROOT).endswith("gv_simulation_files/buckling/gv")
    assert set(module.RUNTIME_DESCRIPTOR.source_files) == {
        str(module.PROVENANCE_ROOT / "clean_all.sh"),
        str(module.PROVENANCE_ROOT / "run_all.sh"),
        str(module.PROVENANCE_ROOT / "generate.py"),
        str(module.PROVENANCE_ROOT / "parameters.py"),
        str(module.PROVENANCE_ROOT / "run.sh"),
        str(module.PROVENANCE_ROOT / "equil.py"),
        str(module.PROVENANCE_ROOT / "parameters-default.gv.yaml"),
    }
    assert before == after == []


def test_dry_run_manifests_expose_identity_axes_and_reject_provenance_outputs(tmp_path: Path) -> None:
    stretching_module = importlib.import_module("meso_uq.structures.gv.runtime.stretching")
    buckling_module = importlib.import_module("meso_uq.structures.gv.runtime.buckling")

    for module in (stretching_module, buckling_module):
        dry_run = module.build_dry_run_descriptor(tmp_path / module.EXPERIMENT_NAME)
        manifest = dry_run.to_manifest()

        assert set(manifest) >= {"structure", "experiment", "geometry", "controls"}
        assert tuple(manifest["controls"]) == module.REQUIRED_CONTROLS
        assert dry_run.output_root == str((tmp_path / module.EXPERIMENT_NAME).resolve())
        assert Path(dry_run.work_dir).is_relative_to(Path(dry_run.output_root))
        assert "parameters.py" in " ".join(module.RUNTIME_DESCRIPTOR.notes)
        assert "equil.py" in " ".join(module.RUNTIME_DESCRIPTOR.notes)
        assert "run_all.sh" in " ".join(module.RUNTIME_DESCRIPTOR.notes)

        with pytest.raises(ValueError, match="must not be inside gv_simulation_files"):
            module.build_dry_run_descriptor(module.PROVENANCE_ROOT / "scratch")
