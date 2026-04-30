from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.references import generate_gv_synthetic_reference
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


def test_synthetic_gv_reference_rejects_bnn_backend() -> None:
    import pytest

    from meso_uq.references import build_gv_synthetic_fixture

    with pytest.raises(ValueError, match="only the 'dnn' surrogate backend"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            surrogate_backend="bnn",
        )
