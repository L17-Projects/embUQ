from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PCA_ROOT = REPO_ROOT / "scripts/workflows/emb/uq_emb/pca_table_s7"
SOURCE_MANIFEST = PCA_ROOT / "SOURCE_MANIFEST.json"
RUNNER = PCA_ROOT / "run_replay.py"
SELECTOR = PCA_ROOT / "select_individual_breathing_mode.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovered_pca_sources_and_inputs_match_bundle_hashes() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    for entry in manifest["files"] + manifest["tracked_inputs"]:
        path = REPO_ROOT / entry["path"]
        assert path.is_file()
        assert _sha256(path) == entry["sha256"]


def test_pca_runner_builds_site_neutral_plan(tmp_path: Path) -> None:
    case = tmp_path / "cases/d1/case"
    for relative in (
        "output/positions.xyz",
        "xyz0.xyz",
        "mesh/emb00001.off",
        "parameter/parameters-default00001.yaml",
        "parameter/parameters00001.yaml",
    ):
        path = case / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--case", str(case)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["status"] == "ready"
    assert plan["mode_limit"] == 300
    assert plan["selected_map"]["symbol"] == "d1"
    assert plan["selected_map_will_be_created"] is True
    assert len(plan["commands"]) == 4
    serialized = json.dumps(plan)
    assert "/ceph" not in serialized
    assert "--site" not in serialized


def test_recovered_selector_reconstructs_frequency_and_receipt(tmp_path: Path) -> None:
    case = tmp_path / "cases/d1/case"
    (case / "parameter").mkdir(parents=True)
    (case.parent / "selected_map_row.json").write_text(
        json.dumps({"symbol": "d1", "agent": "definity"}) + "\n"
    )
    parameters = {"kbt": 2.0, "mass_factor": 160.0, "ut": 8.0e-6}
    (case / "parameter/parameters00001.yaml").write_text(
        yaml.safe_dump(parameters)
    )

    eigenvalues = np.linspace(1.0, 2.0, 300)
    scores = np.zeros((300, 10))
    scores[:, 0] = np.arange(300)
    scores[:, 1] = np.linspace(0.0, 1.0, 300)
    scores[:, 3] = 0.9
    scores[:, 4] = 0.96
    np.savetxt(case / "eigvalues_new.txt", eigenvalues)
    np.savetxt(case / "individual_mode_audit_300_mode_scores.txt", scores)

    subprocess.run(
        [sys.executable, str(SELECTOR), "--case", str(case)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    receipt = json.loads(
        (case / "provenance/individual_breathing_mode.json").read_text()
    )
    selected = receipt["selected"]
    expected_hz = (
        math.sqrt(parameters["kbt"] / eigenvalues[-1])
        * math.sqrt(parameters["mass_factor"])
        / (2.0 * math.pi * parameters["ut"])
    )
    assert selected["mode_zero_based"] == 299
    assert selected["frequency_mhz_mass_corrected"] == expected_hz / 1.0e6
    assert selected["clean_spatial_candidate"] is True
