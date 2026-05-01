from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest


MODULE_NAME = "meso_uq.structures.gv.runtime.shear_flow"
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _import_module():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_importing_module_does_not_require_mirheoobmd(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mirheoOBMD":
            raise AssertionError("mirheoOBMD import attempted during module import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = _import_module()
    assert module.RUNTIME_MODULE_NAME == "mirheoOBMD"


def test_dry_run_without_experimental_opt_in_raises_clear_error() -> None:
    module = _import_module()
    with pytest.raises(ValueError, match="requires include_experimental=True"):
        module.build_dry_run_manifest()


def test_dry_run_with_opt_in_returns_canonical_manifest_with_staged_sources(tmp_path: Path) -> None:
    module = _import_module()
    assert module.build_descriptor() is module.SHEAR_FLOW_RUNTIME
    run_root = tmp_path / "runtime_runs"

    manifest = module.build_dry_run_manifest(run_root=run_root, include_experimental=True)

    work_dir = Path(manifest["paths"]["work_dir"])
    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "shear_flow"
    assert manifest["identity_axes"] == {
        "structure": "gv",
        "experiment": "shear_flow",
        "geometry": manifest["geometry"],
        "controls": ["ptan", "afsi", "bpress"],
    }
    assert manifest["geometry_spec"]["parameters"] == {"radius": 2.0, "height": 14.28}
    assert set(manifest["controls"]) == {"ptan", "afsi", "bpress"}
    assert all(item["role"] == "control" for item in manifest["control_metadata"].values())
    assert all(item["is_calibrated_parameter"] is False for item in manifest["control_metadata"].values())
    assert "ptan" not in manifest["parameter_contract"]["calibrated"]
    assert "afsi" not in manifest["parameter_contract"]["calibrated"]
    assert "bpress" not in manifest["parameter_contract"]["calibrated"]
    assert manifest["default_sweep"]["mode"] == "parallel"
    assert manifest["default_sweep"]["first_restart"] is True
    assert manifest["default_sweep"]["controls"]["ptan"]["values"] == [0.4]
    assert manifest["default_sweep"]["controls"]["afsi"]["value"] == 0.0
    assert manifest["default_sweep"]["controls"]["bpress"]["value"] == -91.0
    assert Path(manifest["paths"]["source_root"]).as_posix().endswith("gv/shear_flow/src")
    assert manifest["paths"]["source_root"] == manifest["paths"]["provenance_root"]

    runtime_req = manifest["runtime_requirements"]
    assert runtime_req == [
        {
            "module": "mirheoOBMD",
            "purpose": "OBMD-enabled GV shear-flow execution runtime",
            "required_at_execution": True,
            "imported_on_module_load": False,
        }
    ]

    known_issue = manifest["known_issues"][0]
    assert known_issue["id"] == "bouncer_collision_candidates_coarse"
    assert "triangle collision candidates" in known_issue["summary"]
    assert known_issue["evidence"] == "gv/shear_flow/src/fixtures/bouncer_collision_candidates_coarse_excerpt.txt"

    output_root = Path(manifest["paths"]["output_root"]).resolve()
    source_root = Path(manifest["paths"]["source_root"]).resolve()
    assert source_root != output_root
    assert not output_root.is_relative_to(source_root)
    assert manifest["paths"]["work_dir"].endswith(f"/{manifest['experiment']}/{manifest['geometry']}/{manifest['control_id']}/work")
    assert work_dir.is_dir()
    assert (work_dir / "source_manifest.json").is_file()
    assert manifest["generated_artifacts"] == []
    assert manifest["artifacts_produced"] is False
