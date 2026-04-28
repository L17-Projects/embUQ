from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCRIPT_SETS = {
    PROJECT_ROOT / "inference" / "scripts": {
        "run_phase_1.py",
        "run_phase_2.py",
        "run_phase_3b.py",
    },
    PROJECT_ROOT / "propagation" / "scripts": {
        "plot_d0_correlations.py",
        "plot_posterior_marginals.py",
        "plot_validation_overlay.py",
        "run_phase1_propagation.py",
        "run_phase3b_propagation.py",
    },
    PROJECT_ROOT / "scripts" / "shared" / "config": {
        "list_experiment_datasets.py",
    },
}


def test_public_script_manifests_exist() -> None:
    for directory, expected in EXPECTED_SCRIPT_SETS.items():
        actual = {path.name for path in directory.iterdir() if path.suffix == ".py"}
        missing = expected - actual
        assert not missing, f"Missing scripts in {directory}: {sorted(missing)}"


def test_public_scripts_have_python_shebang() -> None:
    for directory, expected in EXPECTED_SCRIPT_SETS.items():
        for script_name in expected:
            first_line = (directory / script_name).read_text(encoding="utf-8").splitlines()[0]
            assert first_line == "#!/usr/bin/env python3"
