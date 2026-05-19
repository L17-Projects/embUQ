from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "ACTIVE_LEARNING_FINAL_GATE.md"
CONFIG = REPO_ROOT / "configs" / "active_learning" / "final_gate_plan.example.yaml"


def _normalized_text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_final_gate_document_contains_core_run_and_gate_requirements() -> None:
    doc = _normalized_text(DOC)

    assert "EMB indentation" in doc
    assert "3.4um" in doc
    assert "3 rounds × 30 curves / round" in doc
    assert "full force sweep" in doc
    assert "LHS 90-curve comparator" in doc
    assert "Prior data state:" in doc
    assert "no prior training data" in doc
    assert "Yt" in doc and "kb" in doc
    assert "canary" in doc.lower()
    assert "1 curve × 3 force points" in doc
    assert "Karolina" in doc
    assert "30 concurrent jobs" in doc
    assert "retry limit" in doc
    assert "grouped-holdout median curve relative" in doc
    assert "relative `L2` primary" in doc
    assert "mean curve relative" in doc
    assert "max curve relative" in doc
    assert "predicted/reference curves" in doc
    assert "coverage/acquisition" in doc
    assert "failure/quarantine" in doc
    assert "model-selection" in doc
    assert "AL-vs-LHS" in doc
    assert "Linear/vault ready" in doc


def test_final_gate_document_includes_exact_34um_force_grid_values() -> None:
    doc = _normalized_text(DOC)
    expected_values = [
        "0.0",
        "357.14285714285717",
        "714.2857142857143",
        "1071.4285714285716",
        "1428.5714285714287",
        "1785.7142857142858",
        "2142.857142857143",
        "2500.0",
        "2857.1428571428573",
        "3214.2857142857147",
        "3571.4285714285716",
        "3928.571428571429",
        "4285.714285714286",
        "4642.857142857143",
        "5000.0",
    ]
    for value in expected_values:
        assert value in doc


def test_final_gate_config_matches_required_gates() -> None:
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert spec["kind"] == "active_learning"
    assert spec["spec"]["target"]["diameter_um"] == 3.4
    assert spec["spec"]["target"]["use_prior_training_data"] is False
    assert spec["spec"]["active_learning"]["rounds"] == 3
    assert spec["spec"]["active_learning"]["curves_per_round"] == 30
    assert spec["spec"]["active_learning"]["full_force_sweep"] is True
    assert spec["spec"]["active_learning"]["comparator"]["type"] == "lhs"
    assert spec["spec"]["active_learning"]["comparator"]["curve_count"] == 90
    assert spec["spec"]["candidate_dimensions"]["Yt"]["min"] == 100000.0
    assert spec["spec"]["candidate_dimensions"]["Yt"]["max"] == 1000000000.0
    assert spec["spec"]["candidate_dimensions"]["kb"]["min"] == 400.0
    assert spec["spec"]["candidate_dimensions"]["kb"]["max"] == 70000.0
    assert spec["spec"]["platform"]["name"] == "karolina"
    assert spec["spec"]["platform"]["concurrent_jobs"] == 30
    assert spec["spec"]["platform"]["retry_limit"] == 3
    assert spec["spec"]["canary"]["curves"] == 1
    assert spec["spec"]["canary"]["force_points"] == 3

    runtime_fingerprint = spec["spec"]["runtime_fingerprint"]
    assert runtime_fingerprint["radp"] == 6.8
    assert runtime_fingerprint["L"] == 25
    assert runtime_fingerprint["fscale"] == 0.0074
    assert float(runtime_fingerprint["shell_th"]) == 5e-9
    assert runtime_fingerprint["numsteps"] == 5000
    assert runtime_fingerprint["numsteps_eq"] == 10000

    expected_grid = [
        0.0,
        357.14285714285717,
        714.2857142857143,
        1071.4285714285716,
        1428.5714285714287,
        1785.7142857142858,
        2142.857142857143,
        2500.0,
        2857.1428571428573,
        3214.2857142857147,
        3571.4285714285716,
        3928.571428571429,
        4285.714285714286,
        4642.857142857143,
        5000.0,
    ]
    assert runtime_fingerprint["force_grid_um_3_4"] == expected_grid

    required_plots = set(spec["spec"]["required_plots"])
    assert "grouped_holdout_median_curve_relative_l2_primary" in required_plots
    assert "mean_curve_relative_l2" in required_plots
    assert "max_curve_relative_l2" in required_plots
    assert "residuals" in required_plots
    assert "predicted_vs_reference_curves" in required_plots
    assert "coverage_acquisition" in required_plots
    assert "failure_quarantine" in required_plots
    assert "model_selection" in required_plots
    assert "al_vs_lhs_summary" in required_plots
    assert "final_gate_acceptance" in required_plots
