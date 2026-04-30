from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.references import build_gv_synthetic_fixture
from meso_uq.references import build_gv_synthetic_reference_manifest
from meso_uq.references import generate_gv_synthetic_reference
from meso_uq.references import gv_common as gv_common_module
from meso_uq.references import synthetic as synthetic_module
from meso_uq.references.gv_common import resolve_gv_reference_context
from meso_uq.structures.gv.parameters import GV_PARAMETER_CONTRACT
from meso_uq.structures.gv.runtime.catalog import load_runtime_descriptor


def test_synthetic_gv_reference_is_deterministic_for_seed(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("stretching").plan(output_root=tmp_path).to_manifest()

    reference_a = generate_gv_synthetic_reference(seed=17, runtime_manifest=runtime_manifest)
    reference_b = generate_gv_synthetic_reference(seed=17, runtime_manifest=runtime_manifest)
    reference_c = generate_gv_synthetic_reference(seed=18, runtime_manifest=runtime_manifest)

    assert reference_a.points == reference_b.points
    assert reference_a.values == reference_b.values
    assert reference_a.to_manifest()["values"] == reference_b.to_manifest()["values"]
    assert reference_a.values != reference_c.values


def test_synthetic_gv_reference_manifest_includes_contract_and_runtime_metadata(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("torsion").plan(output_root=tmp_path).to_manifest()

    manifest = generate_gv_synthetic_reference(seed=5, runtime_manifest=runtime_manifest).to_manifest()

    assert manifest["reference_kind"] == "synthetic"
    assert manifest["surrogate_backend"] == "dnn"
    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "torsion"
    assert manifest["geometry"].startswith("gv_rad")
    assert manifest["controls"] == {"theta": 0.01}
    assert manifest["calibrated_parameter_names"] == list(GV_PARAMETER_CONTRACT.calibrated_names)
    assert manifest["noise_model"]["kind"] == "multiplicative"
    assert manifest["provenance"]["runtime_provenance_root"].endswith("gv_simulation_files/torsion/gv")
    assert manifest["outputs"]["output_root"] == str(tmp_path.resolve())
    assert manifest["outputs"]["work_dir"].endswith("/gv/torsion/gv_rad2_height14_28/theta_0_01_0_1")
    assert len(manifest["points"]) == 64
    assert len(manifest["values"]) == 64


def test_synthetic_reference_can_resolve_explicit_axes_without_manifest() -> None:
    reference = generate_gv_synthetic_reference(
        seed=2,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        point_count=3,
    )
    manifest = reference.to_manifest()

    assert manifest["dataset_id"] == "gv:torsion:gv_rad2_height14_28:theta_0.03"
    assert manifest["control_id"] == "theta_0.03"
    assert manifest["points"] == [0.0, 0.5, 1.0]


def test_synthetic_reference_context_reports_missing_axes_and_controls() -> None:
    with pytest.raises(ValueError, match="require an experiment name"):
        resolve_gv_reference_context()

    with pytest.raises(ValueError, match="require a geometry id"):
        resolve_gv_reference_context(experiment="torsion")

    with pytest.raises(ValueError, match="require controls"):
        resolve_gv_reference_context(experiment="torsion", geometry="gv_rad2_height14_28")

    with pytest.raises(ValueError, match="Unknown GV controls"):
        resolve_gv_reference_context(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03, "tot_force": 1.0},
        )

    with pytest.raises(ValueError, match="Missing GV controls"):
        resolve_gv_reference_context(
            experiment="stretching",
            geometry="gv_rad2_height14_28",
            controls={"tot_force": 500.0},
        )

    with pytest.raises(ValueError, match="at least two sample points"):
        generate_gv_synthetic_reference(
            seed=1,
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            point_count=1,
        )


def test_synthetic_reference_context_rejects_non_gv_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    class NonGVStructure:
        name = "emb"

    monkeypatch.setattr(gv_common_module, "get_structure", lambda _name: NonGVStructure())

    with pytest.raises(ValueError, match="only support structure 'gv'"):
        gv_common_module.resolve_gv_reference_context(
            runtime_manifest={
                "structure": "emb",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
            }
        )


def test_synthetic_reference_context_merges_manifest_and_explicit_controls() -> None:
    context = resolve_gv_reference_context(
        runtime_manifest={
            "structure": "gv",
            "experiment": "stretching",
            "geometry": "gv_rad2_height14_28",
            "controls": {"tot_force": 500.0},
        },
        controls={"bpress": -91.0},
    )

    assert context.controls == {"tot_force": 500.0, "bpress": -91.0}
    assert context.control_id == "bpress_-91__tot_force_500"
    assert context.dataset_id == "gv:stretching:gv_rad2_height14_28:bpress_-91__tot_force_500"


def test_synthetic_gv_reference_rejects_bnn_backend() -> None:
    with pytest.raises(ValueError, match="only the 'dnn' surrogate backend"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            surrogate_backend="bnn",
        )


def test_synthetic_fixture_rejects_invalid_series_reference_kind_and_templates() -> None:
    with pytest.raises(ValueError, match="points must contain at least one value"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            points=[],
        )

    with pytest.raises(ValueError, match="No synthetic fixture template"):
        build_gv_synthetic_fixture(
            experiment="not_registered",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
        )

    with pytest.raises(ValueError, match="Unsupported GV reference kind"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            reference_kind="real",
        )

    with pytest.raises(ValueError, match="same length"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            points=[0.0, 1.0],
            values=[0.0],
        )


def test_synthetic_reference_wrappers_and_private_axis_helpers() -> None:
    manifest = build_gv_synthetic_reference_manifest(
        seed=9,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        point_count=2,
    )
    assert manifest["points"] == [0.0, 1.0]
    assert manifest["values"]

    identity = synthetic_module._identity_from_axes(
        structure="gv",
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        surrogate_backend="dnn",
        reference_kind="synthetic",
        dataset_id="explicit",
    )
    assert identity.dataset_id == "explicit"
    assert synthetic_module._control_id({}) == "default"
    assert synthetic_module._seeded_synthetic_value(3, "dataset", "not_registered", {}, 0.5) > 0.0

    with pytest.raises(ValueError, match="Unsupported GV reference kind"):
        synthetic_module._identity_from_axes(
            structure="gv",
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            surrogate_backend="dnn",
            reference_kind="real",
        )
