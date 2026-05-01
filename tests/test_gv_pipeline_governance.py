from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.structures.gv.runtime import RUNTIME_EXPERIMENTS, plan_runtime, runtime_module_name


GV_RUNTIME_ROOT = REPO_ROOT / "src" / "meso_uq" / "structures" / "gv"
GV_SIMULATION_ROOT = REPO_ROOT / "gv_simulation_files"
FORBIDDEN_GV_BNN_SURFACE_TOKENS = (
    "variationalbnnpredictor",
    "pyro",
    "promote_certified_bnn",
    "run_bnn_",
)
FORBIDDEN_GV_HBI_SURFACE_TOKENS = (
    "hierarchical inference",
    "hbi",
)
GV_HBI_SURFACE_ALLOWLIST = {
    REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_operational_canary.py",
}


def test_gv_runtime_surface_uses_python_modules_not_raw_staging_imports() -> None:
    runtime_sources = sorted((GV_RUNTIME_ROOT / "runtime").glob("*.py"))
    assert runtime_sources

    for experiment_name in RUNTIME_EXPERIMENTS:
        assert runtime_module_name(experiment_name) == f"meso_uq.structures.gv.runtime.{experiment_name}"

    for source_path in runtime_sources:
        text = source_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import gv_simulation_files")
            assert not stripped.startswith("from gv_simulation_files")


def test_gv_runtime_plans_route_generated_outputs_outside_canonical_source(tmp_path: Path) -> None:
    for experiment_name in RUNTIME_EXPERIMENTS:
        dry_run = plan_runtime(
            experiment_name,
            output_root=tmp_path / experiment_name,
            include_experimental=experiment_name == "shear_flow",
        )
        work_dir = Path(dry_run.work_dir)
        output_root = Path(dry_run.output_root)
        provenance_root = Path(dry_run.provenance_root)

        assert GV_SIMULATION_ROOT not in output_root.parents
        assert GV_SIMULATION_ROOT not in work_dir.parents
        assert work_dir == output_root / experiment_name / dry_run.geometry / dry_run.control_id / "work"
        assert provenance_root == REPO_ROOT / "gv" / experiment_name / "src"
        assert dry_run.legacy_import_root.startswith(str(GV_SIMULATION_ROOT))
        for generated_subdir in dry_run.generated_subdirs:
            generated_path = work_dir / generated_subdir
            assert GV_SIMULATION_ROOT not in generated_path.parents
            assert provenance_root not in generated_path.parents


def test_gv_surface_does_not_claim_bnn_promotion_or_ungated_hbi_support() -> None:
    surface_files = list(GV_RUNTIME_ROOT.rglob("*.py")) + list((REPO_ROOT / "scripts" / "workflows" / "gv").glob("*.py"))
    assert surface_files

    for path in surface_files:
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_GV_BNN_SURFACE_TOKENS:
            assert token not in text, f"Unexpected GV surface token {token!r} in {path}"
        if path in GV_HBI_SURFACE_ALLOWLIST:
            continue
        for token in FORBIDDEN_GV_HBI_SURFACE_TOKENS:
            assert token not in text, f"Unexpected ungated GV HBI token {token!r} in {path}"
