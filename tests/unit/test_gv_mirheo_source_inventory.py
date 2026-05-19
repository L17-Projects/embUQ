from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GV_ROOT = REPO_ROOT / "gv"
INVENTORY_PATH = GV_ROOT / "source_inventory.json"

REQUIRED_FILES = {
    "stretching": (
        "generate.py",
        "parameters.py",
        "equil.py",
        "run.sh",
        "run_all.sh",
        "clean_all.sh",
        "parameters-default.gv.yaml",
        "gas_vesicle/create_gv.py",
        "gas_vesicle/parameters.yaml",
        "gas_vesicle/parameters.py",
        "gas_vesicle/run.sh",
        "gas_vesicle/statistics.py",
        "gas_vesicle/add_to_off.py",
    ),
    "buckling": (
        "generate.py",
        "parameters.py",
        "equil.py",
        "run.sh",
        "run_all.sh",
        "clean_all.sh",
        "parameters-default.gv.yaml",
        "parameters-default.emb.yaml",
        "gas_vesicle/create_gv.py",
        "gas_vesicle/parameters.yaml",
        "gas_vesicle/parameters.py",
        "gas_vesicle/run.sh",
        "gas_vesicle/statistics.py",
        "gas_vesicle/add_to_off.py",
    ),
    "torsion": (
        "generate.py",
        "parameters.py",
        "equil.py",
        "run.sh",
        "run_all.sh",
        "clean_all.sh",
        "parameters-default.gv.yaml",
        "gas_vesicle/create_gv.py",
        "gas_vesicle/parameters.yaml",
        "gas_vesicle/parameters.py",
        "gas_vesicle/run.sh",
        "gas_vesicle/statistics.py",
        "gas_vesicle/add_to_off.py",
    ),
    "eigenmodes": (
        "generate.py",
        "parameters.py",
        "equil.py",
        "run.sh",
        "run_all.sh",
        "clean_all.sh",
        "parameters-default.gv.yaml",
        "gas_vesicle/create_gv.py",
        "gas_vesicle/parameters.yaml",
        "gas_vesicle/run.sh",
        "gas_vesicle/statistics.py",
        "analysis/all.sh",
        "analysis/all_analysis.py",
        "analysis/average.py",
        "analysis/combine.py",
        "analysis/initial.py",
        "analysis/plot_freq.py",
        "analysis/plot_modes.py",
        "analysis/run.sh",
        "analysis/run_an.sh",
        "analysis/run_eig.sh",
        "analysis/trim_eigenmodes.py",
        "analysis/trim.sh",
        "analysis/trim_svd.sh",
    ),
    "shear_flow": (
        "README.md",
        "copy_and_modify.py",
        "run_all.py",
        "run_all_HPC.sh",
        "generate.py",
        "parameters.py",
        "equil.py",
        "run.sh",
        "parameters-default.gv.yaml",
        "gas_vesicle/create_gv.py",
        "gas_vesicle/parameters.yaml",
        "gas_vesicle/run.sh",
        "gas_vesicle/statistics.py",
        "fixtures/bouncer_collision_candidates_coarse_excerpt.txt",
    ),
}


def test_inventory_tracks_all_gv_experiments() -> None:
    assert INVENTORY_PATH.is_file(), INVENTORY_PATH
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert "experiments" in payload
    assert set(payload["experiments"]) == set(REQUIRED_FILES)


def test_gv_source_inventory_matches_expected_files() -> None:
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    for experiment_name, required_files in REQUIRED_FILES.items():
        exp_entry = payload["experiments"][experiment_name]
        listed = set(
            rel_file
            for files in exp_entry["file_roles"].values()
            for rel_file in files
        )
        assert listed.issuperset(required_files), (experiment_name, required_files)


def test_gv_inventory_catalogued_files_exist_on_disk() -> None:
    for experiment_name, required_files in REQUIRED_FILES.items():
        src_root = GV_ROOT / experiment_name / "src"
        assert src_root.is_dir(), src_root
        for rel_file in required_files:
            assert (src_root / rel_file).is_file(), (experiment_name, rel_file)


def test_gv_source_tree_excludes_obvious_generated_artifacts() -> None:
    disallowed_suffixes = {".out", ".off", ".png", ".xyz", ".bin", ".o", ".d"}
    disallowed_filenames = {"commands.txt", "posq.txt", "output.out", "run_HPC.sbatch"}
    disallowed_dirs = {"mesh", "parameter", "restart", "logs", "anchor", "stats", "trj_eq", "force"}

    for path in GV_ROOT.rglob("*"):
        if path.name == "source_inventory.json":
            continue

        if path.is_file():
            assert path.name not in disallowed_filenames, path
            assert path.suffix.lower() not in disallowed_suffixes, path

        if path.is_dir():
            assert path.name not in disallowed_dirs, path


def test_gv_mirheo_experiments_register_pin_object_plugin() -> None:
    for experiment_name in REQUIRED_FILES:
        equil_path = GV_ROOT / experiment_name / "src" / "equil.py"
        active_pin_lines = [
            line
            for line in equil_path.read_text(encoding="utf-8").splitlines()
            if "createPinObject" in line and not line.lstrip().startswith("#")
        ]
        assert active_pin_lines, experiment_name


def test_gv_parameter_scripts_create_mesh_output_directory() -> None:
    for experiment_name in REQUIRED_FILES:
        parameters_path = GV_ROOT / experiment_name / "src" / "parameters.py"
        text = parameters_path.read_text(encoding="utf-8")
        assert 'os.makedirs("mesh", exist_ok=True)' in text, experiment_name


def test_gv_run_scripts_create_runtime_output_directories() -> None:
    for experiment_name in REQUIRED_FILES:
        run_script_path = GV_ROOT / experiment_name / "src" / "run.sh"
        text = run_script_path.read_text(encoding="utf-8")
        assert "mkdir -p" in text, experiment_name
        assert "logs" in text, experiment_name
        assert "restart" in text, experiment_name


def test_gv_lj_cutoff_prefers_configured_default_when_self_repulsion_is_used() -> None:
    for experiment_name in ("stretching", "eigenmodes", "shear_flow"):
        equil_path = GV_ROOT / experiment_name / "src" / "equil.py"
        text = equil_path.read_text(encoding="utf-8")
        assert 'parameters_default.get("lj_fac", mesh_lj_fac)' in text, experiment_name
