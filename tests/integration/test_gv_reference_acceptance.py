from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]

MATERIALIZE_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "materialize_gv_reference.py"
SMOKE_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dnn_smoke.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


NON_SHEAR_MATRIX = (
    ("stretching", ("tot_force=500", "bpress=-91")),
    ("buckling", ("buck=0.4", "bpress=-91")),
    ("torsion", ("theta=0.03",)),
    ("eigenmodes", ("bpress=-91",)),
)


def test_gv_reference_materialization_and_dnn_smoke_accept_non_shear_matrix(tmp_path: Path) -> None:
    materialize_module = _load_module(MATERIALIZE_SCRIPT_PATH, "mesouq_gv_reference_acceptance_materialize")
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_gv_reference_acceptance_smoke")
    repo_root = tmp_path / "materialization"
    output_root = tmp_path / "smoke"

    for experiment, control_args in NON_SHEAR_MATRIX:
        control_args_flat = [item for control_arg in control_args for item in ("--control", control_arg)]
        assert (
            materialize_module.main(
                [
                    "--repo-root",
                    str(repo_root),
                    "--selection",
                    f"gv:{experiment}",
                    "--synthetic-point-count",
                    "3",
                    *control_args_flat,
                ]
            )
            == 0
        )

        manifest_candidates = list(
            (repo_root / "_runs" / "gv" / experiment / "gv_rad2_height14_28").rglob("reference_manifest.json")
        )
        assert manifest_candidates, f"Missing materialized GV reference manifest for {experiment}."
        reference_manifest_path = manifest_candidates[0]
        reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
        assert reference_manifest["experiment"] == experiment

        experiment_smoke_root = output_root / experiment
        assert (
            smoke_module.main(
                [
                    "--reference-manifest",
                    str(reference_manifest_path),
                    "--output-root",
                    str(experiment_smoke_root),
                    "--num-curves",
                    "2",
                    "--points-per-curve",
                    "2",
                    "--max-epoch",
                    "1",
                    "--width",
                    "4",
                    "--depth",
                    "1",
                    "--batch-size",
                    "2",
                    "--val-fraction",
                    "0.5",
                ]
            )
            == 0
        )

        smoke_manifest = json.loads(
            (experiment_smoke_root / "gv_dnn_surrogate_smoke_manifest.json").read_text(encoding="utf-8")
        )
        reference_dataset = reference_manifest_path.with_name("reference_dataset.npz")
        assert smoke_manifest["experiment"] == experiment
        assert smoke_manifest["input_manifests"]["reference_manifest"] == str(reference_manifest_path.resolve())
        assert smoke_manifest["reference_stage"]["status"] == "available"
        assert smoke_manifest["artifacts"]["reference_dataset"] == str(reference_dataset.resolve())
        assert Path(smoke_manifest["artifacts"]["reference_dataset"]).is_file()
