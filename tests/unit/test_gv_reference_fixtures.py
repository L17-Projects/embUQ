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


def _import_without_heavy_runtime_deps(module_name: str) -> object:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in {"mirheo", "MDAnalysis", "trimesh"}:
            raise AssertionError(f"blocked import attempted: {name}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    sys.modules.pop(module_name, None)
    try:
        return importlib.import_module(module_name)
    finally:
        builtins.__import__ = original_import


def test_reference_modules_import_without_heavy_runtime_dependencies() -> None:
    synthetic = _import_without_heavy_runtime_deps("meso_uq.references.synthetic")
    dpd_generated = _import_without_heavy_runtime_deps("meso_uq.references.dpd_generated")

    assert synthetic.DEFAULT_SURROGATE_BACKEND == "dnn"
    assert hasattr(dpd_generated, "dpd_generated_fixture_from_runtime_manifest")


def test_synthetic_fixture_from_runtime_manifest_preserves_gv_identity_axes() -> None:
    torsion = importlib.import_module("meso_uq.structures.gv.runtime.torsion")
    synthetic = importlib.import_module("meso_uq.references.synthetic")

    runtime_manifest = torsion.get_runtime_descriptor().plan().to_manifest()
    fixture = synthetic.synthetic_fixture_from_runtime_manifest(runtime_manifest, sigma=0.03)
    manifest = fixture.to_manifest()

    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "torsion"
    assert manifest["geometry"] == runtime_manifest["geometry"]
    assert manifest["controls"] == runtime_manifest["controls"]
    assert manifest["surrogate_backend"] == "dnn"
    assert manifest["reference_kind"] == "synthetic"
    assert manifest["dataset_id"] == runtime_manifest["dataset_id"]
    assert manifest["noise_model"] == {"kind": "multiplicative", "parameter": "sigma"}
    assert manifest["nuisance_parameters"] == {"sigma": 0.03}
    assert "theta" not in manifest["calibrated_parameter_names"]
    assert tuple(manifest["calibrated_parameter_names"]) == (
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
    )


def test_synthetic_fixture_adds_optional_d0_only_when_requested() -> None:
    synthetic = importlib.import_module("meso_uq.references.synthetic")

    without_d0 = synthetic.build_gv_synthetic_fixture(
        experiment="stretching",
        geometry="gv_rad2_height14_28",
        controls={"tot_force": 500.0, "bpress": -91.0},
        sigma=0.02,
    ).to_manifest()
    with_d0 = synthetic.build_gv_synthetic_fixture(
        experiment="stretching",
        geometry="gv_rad2_height14_28",
        controls={"tot_force": 500.0, "bpress": -91.0},
        sigma=0.02,
        d0=0.1,
    ).to_manifest()

    assert without_d0["nuisance_parameters"] == {"sigma": 0.02}
    assert with_d0["nuisance_parameters"] == {"sigma": 0.02, "d0": 0.1}


def test_dpd_generated_fixture_carries_runtime_and_analysis_metadata() -> None:
    eigenmodes = importlib.import_module("meso_uq.structures.gv.runtime.eigenmodes")
    dpd_generated = importlib.import_module("meso_uq.references.dpd_generated")

    runtime_manifest = eigenmodes.get_runtime_descriptor().plan().to_manifest()
    fixture = dpd_generated.dpd_generated_fixture_from_runtime_manifest(runtime_manifest, sigma=0.05, d0=0.0)
    manifest = fixture.to_manifest()

    assert manifest["reference_kind"] == "dpd_generated"
    assert manifest["surrogate_backend"] == "dnn"
    assert manifest["dataset_id"] == runtime_manifest["dataset_id"]
    assert manifest["controls"] == {"bpress": -91.0}
    assert manifest["nuisance_parameters"] == {"sigma": 0.05, "d0": 0.0}
    assert any(path.endswith("analysis/all_analysis.py") for path in manifest["metadata"]["source_files"])
    assert len(manifest["metadata"]["analysis_commands"]) == 4


def test_fixture_rejects_controls_that_overlap_calibrated_parameters() -> None:
    synthetic = importlib.import_module("meso_uq.references.synthetic")

    with pytest.raises(ValueError, match="controls must remain separate"):
        synthetic.build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"kb": 1.0},
            sigma=0.01,
        )


def test_fixture_rejects_unknown_or_missing_controls_and_non_dnn_backend() -> None:
    synthetic = importlib.import_module("meso_uq.references.synthetic")

    with pytest.raises(ValueError, match="Unknown GV controls"):
        synthetic.build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03, "extra": 1.0},
            sigma=0.01,
        )

    with pytest.raises(ValueError, match="Missing GV controls"):
        synthetic.build_gv_synthetic_fixture(
            experiment="stretching",
            geometry="gv_rad2_height14_28",
            controls={"tot_force": 500.0},
            sigma=0.01,
        )

    with pytest.raises(ValueError, match="only the 'dnn' surrogate backend"):
        synthetic.build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.01,
            surrogate_backend="bnn",
        )
