from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest
from meso_uq.references import build_gv_synthetic_reference_manifest
from meso_uq.references.gv_common import (
    resolve_gv_reference_context,
    validate_gv_dataset_id,
    validate_gv_reference_manifest,
)

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


def test_validate_gv_reference_manifest_rejects_missing_schema_fields() -> None:
    manifest = build_gv_synthetic_reference_manifest(
        seed=17,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        point_count=2,
    )
    for field in ("observable_schema", "generation_seed", "geometry_spec", "geometry_parameters"):
        incomplete = dict(manifest)
        incomplete.pop(field)
        with pytest.raises(ValueError, match="missing required schema fields"):
            validate_gv_reference_manifest(incomplete)


def test_validate_gv_reference_manifest_allows_missing_extended_fields_for_v1() -> None:
    manifest = build_gv_synthetic_reference_manifest(
        seed=17,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        point_count=2,
    )
    legacy_manifest = dict(manifest)
    legacy_manifest["manifest_schema_version"] = 1
    for field in ("geometry_parameters", "geometry_spec", "observable_schema", "generation_seed"):
        legacy_manifest.pop(field)
    validate_gv_reference_manifest(legacy_manifest)


def test_validate_gv_reference_manifest_rejects_schema_contract_mismatches() -> None:
    manifest = build_gv_synthetic_reference_manifest(
        seed=17,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        point_count=2,
    )
    invalid_payloads = [
        ({"manifest_schema_version": 99}, "Unsupported GV reference manifest schema version"),
        ({"structure": "emb"}, "structure='gv'"),
        ({"reference_kind": "real"}, "Unsupported GV reference kind"),
        ({"controls": []}, "controls must be a mapping"),
        ({"calibrated_parameter_names": 3}, "calibrated_parameter_names must be a sequence"),
        ({"calibrated_parameter_names": ["ka"]}, "calibrated_parameter_names must match"),
        ({"geometry_parameters": []}, "geometry_parameters must be a mapping"),
        ({"geometry_parameters": {"radius": 2.0}}, "geometry_parameters missing 'height'"),
        ({"observable_schema": 3}, "observable_schema must be a sequence"),
        ({"provenance": []}, "provenance must be a mapping"),
        ({"geometry_spec": []}, "geometry_spec must be a mapping"),
        (
            {"geometry_spec": {"id": "gv_rad2_height14_28", "parameters": []}},
            "geometry_spec.parameters must be a mapping",
        ),
        (
            {"geometry_spec": {"id": "gv_rad2_height14_28", "parameters": {"radius": 2.0}}},
            "geometry_spec.parameters missing 'height'",
        ),
        ({"noise_model": []}, "noise_model must be a mapping"),
        ({"noise_model": {"kind": "additive", "parameter": "sigma"}}, "noise_model must be multiplicative sigma"),
        ({"generation_seed": "seed"}, "generation_seed must be numeric"),
    ]
    for updates, match in invalid_payloads:
        invalid = dict(manifest)
        invalid.update(updates)
        with pytest.raises(ValueError, match=match):
            validate_gv_reference_manifest(invalid)

    context = resolve_gv_reference_context(
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
    )
    invalid_schema = dict(manifest)
    invalid_schema["observable_schema"] = [*manifest["observable_schema"], {"name": "extra"}]
    with pytest.raises(ValueError, match="observable_schema must match"):
        validate_gv_reference_manifest(invalid_schema, context=context)


def test_validate_gv_dataset_id_rejects_collisions() -> None:
    valid_gv_dataset = "gv:torsion:gv_rad2_height14_28:theta_0.03"
    invalid_structure = "emb:torsion:gv_rad2_height14_28:theta_0.03"
    malformed = "gv:torsion:theta_0.03"

    validate_gv_dataset_id(valid_gv_dataset, structure_name="gv")
    with pytest.raises(ValueError, match="must be strings"):
        validate_gv_dataset_id(3, structure_name="gv")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="structure-qualified"):
        validate_gv_dataset_id(invalid_structure, structure_name="gv")
    with pytest.raises(ValueError, match="4 components"):
        validate_gv_dataset_id(malformed, structure_name="gv")
