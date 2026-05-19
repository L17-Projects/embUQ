from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _import_without_heavy_dependencies(module_name: str, blocked_roots: tuple[str, ...]) -> object:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in blocked_roots:
            raise AssertionError(f"blocked import attempted: {name}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    sys.modules.pop(module_name, None)
    try:
        return importlib.import_module(module_name)
    finally:
        builtins.__import__ = original_import


def test_torsion_import_is_lightweight() -> None:
    module = _import_without_heavy_dependencies(
        "meso_uq.structures.gv.runtime.torsion",
        ("mirheo", "trimesh", "MDAnalysis"),
    )
    assert module.get_runtime_descriptor().experiment == "torsion"


def test_eigenmodes_import_is_lightweight() -> None:
    module = _import_without_heavy_dependencies(
        "meso_uq.structures.gv.runtime.eigenmodes",
        ("mirheo", "trimesh", "MDAnalysis"),
    )
    assert module.get_runtime_descriptor().experiment == "eigenmodes"


def test_torsion_uses_theta_control_and_imported_default_sweep(tmp_path: Path) -> None:
    module = importlib.import_module("meso_uq.structures.gv.runtime.torsion")

    descriptor = module.get_runtime_descriptor()
    dry_run = descriptor.plan(output_root=tmp_path)
    manifest = dry_run.to_manifest()
    helper_manifest = module.build_dry_run_manifest(output_root=tmp_path)

    assert descriptor.control_names == ("theta",)
    assert descriptor.control_sweeps[0].values() == pytest.approx(
        (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)
    )
    assert manifest["controls"] == {"theta": 0.01}
    assert helper_manifest["experiment"] == "torsion"
    assert manifest["control_sweeps"] == [
        {"name": "theta", "start": 0.01, "stop": 0.1, "steps": 10}
    ]
    assert Path(manifest["source_manifest"]).is_file()


def test_eigenmodes_uses_bpress_control_and_tracks_analysis_provenance(tmp_path: Path) -> None:
    module = importlib.import_module("meso_uq.structures.gv.runtime.eigenmodes")

    descriptor = module.get_runtime_descriptor()
    dry_run = descriptor.plan(output_root=tmp_path)
    manifest = dry_run.to_manifest()
    helper_manifest = module.build_dry_run_manifest(output_root=tmp_path)

    assert descriptor.control_names == ("bpress",)
    assert descriptor.control_sweeps[0].values() == (-91.0,)
    assert manifest["controls"] == {"bpress": -91.0}
    assert helper_manifest["experiment"] == "eigenmodes"
    analysis_paths = set(manifest["source_files"])
    assert any(path.endswith("analysis/all_analysis.py") for path in analysis_paths)
    assert any(path.endswith("analysis/combine.py") for path in analysis_paths)
    assert any(path.endswith("analysis/trim_eigenmodes.py") for path in analysis_paths)
    assert any(path.endswith("analysis/trim_svd.sh") for path in analysis_paths)
    assert len(manifest["analysis_commands"]) == 4
    assert Path(manifest["source_manifest"]).is_file()


@pytest.mark.parametrize(
    ("module_name", "expected_analysis"),
    (
        ("meso_uq.structures.gv.runtime.torsion", False),
        ("meso_uq.structures.gv.runtime.eigenmodes", True),
    ),
)
def test_dry_run_manifests_include_identity_axes_and_external_output_roots(
    module_name: str,
    expected_analysis: bool,
    tmp_path: Path,
) -> None:
    module = importlib.import_module(module_name)
    dry_run = module.get_runtime_descriptor().plan(output_root=tmp_path / "planner-root")
    manifest = dry_run.to_manifest()

    assert manifest["structure"] == "gv"
    assert manifest["experiment"] in {"torsion", "eigenmodes"}
    assert manifest["geometry"].startswith("gv_rad")
    assert set(manifest["controls"]) in ({"theta"}, {"bpress"})
    assert manifest["dataset_id"]

    assert manifest["source_root"].startswith(REPO_ROOT.as_posix() + "/gv/")

    provenance_roots = (
        REPO_ROOT / "gv" / "torsion" / "src",
        REPO_ROOT / "gv" / "eigenmodes" / "src",
    )
    output_paths = [Path(manifest["output_root"]), Path(manifest["work_dir"])]
    for output_path in output_paths:
        assert REPO_ROOT not in output_path.parents
        assert all(provenance_root not in output_path.parents for provenance_root in provenance_roots)

    assert manifest["commands"]
    if expected_analysis:
        assert manifest["analysis_commands"]
    else:
        assert manifest["analysis_commands"] == []


def test_plan_rejects_output_root_inside_provenance_tree() -> None:
    torsion = importlib.import_module("meso_uq.structures.gv.runtime.torsion").get_runtime_descriptor()
    with pytest.raises(ValueError, match="gv_simulation_files"):
        torsion.plan(output_root="gv_simulation_files/torsion/gv")
