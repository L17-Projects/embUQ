from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.references import load_gv_dpd_generated_reference
from meso_uq.structures.gv.runtime.catalog import load_runtime_descriptor


def test_dpd_generated_gv_reference_manifest_tracks_identity_and_paths(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("eigenmodes").plan(output_root=tmp_path / "runs").to_manifest()
    data_path = tmp_path / "references" / "eigenmodes_reference.dat"
    data_path.parent.mkdir()
    data_path.write_text("x y\n0.0 1.0\n")

    manifest = load_gv_dpd_generated_reference(data_path, runtime_manifest=runtime_manifest).to_manifest()

    assert manifest["reference_kind"] == "dpd_generated"
    assert manifest["surrogate_backend"] == "dnn"
    assert manifest["dataset_id"] == runtime_manifest["dataset_id"]
    assert manifest["controls"] == {"bpress": -91.0}
    assert manifest["provenance"]["runtime_provenance_root"].endswith("gv_simulation_files/eigenmodes/gv")
    assert manifest["outputs"]["output_root"] == str((tmp_path / "runs").resolve())
    assert manifest["data_reference"]["path"] == str(data_path.resolve())
    assert manifest["data_reference"]["format"] == "dat"


def test_dpd_generated_gv_reference_missing_data_raises_actionable_error(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("stretching").plan(output_root=tmp_path).to_manifest()
    missing_path = tmp_path / "missing" / "stretching_reference.dat"

    with pytest.raises(FileNotFoundError, match="Generate the DPD reference artifact first or pass a valid existing path"):
        load_gv_dpd_generated_reference(missing_path, runtime_manifest=runtime_manifest)

    with pytest.raises(FileNotFoundError, match=runtime_manifest["dataset_id"]):
        load_gv_dpd_generated_reference(missing_path, runtime_manifest=runtime_manifest)
