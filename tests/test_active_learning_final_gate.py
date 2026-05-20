from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "ACTIVE_LEARNING_FINAL_GATE.md"
CONFIG = REPO_ROOT / "configs" / "active_learning" / "final_gate_plan.example.yaml"


def _normalized_text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _expected_force_grid() -> list[float]:
    return [
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


def test_final_gate_document_contains_core_run_and_gate_requirements() -> None:
    doc = _normalized_text(DOC)

    assert "EMB indentation" in doc
    assert "3.4um" in doc
    assert "log-space" in doc
    assert "ka" in doc and "kb" in doc
    assert "1e2" in doc
    assert "6e5" in doc
    assert "400" in doc
    assert "7e4" in doc
    assert "initial_sobol_maximin" in doc
    assert "ensemble_disagreement_diversity" in doc
    assert "greedy diversity" in doc
    assert "3 rounds × 30 curves / round" in doc
    assert "6" in doc
    assert "24" in doc
    assert "full force sweep" in doc
    assert "LHS 90-curve comparator" in doc
    assert "Prior data state:" in doc
    assert "no prior training data" in doc
    assert "it is never an active selected candidate dimension" in doc
    assert "canary" in doc.lower()
    assert "1 curve × 3 force points" in doc
    assert "Karolina" in doc
    assert "30 concurrent jobs" in doc
    assert "retry limit" in doc
    assert "AL prefixes" in doc
    assert "LHS prefixes" in doc
    assert "fresh DPD labels only" in doc
    assert "samples_all.dat" in doc
    assert "runtime-per-curve" in doc
    assert "ka/kb" in doc
    assert "per-round additions" in doc
    assert "exploration vs acquisition" in doc
    assert "disagreement-acquisition map" in doc
    assert "force overlays" in doc
    assert "AL-vs-LHS relative `L2`" in doc
    assert "failure/quarantine/replacement" in doc
    assert "Linear/vault ready" in doc


def test_final_gate_document_includes_exact_34um_force_grid_values() -> None:
    doc = _normalized_text(DOC)
    for value in map(str, _expected_force_grid()):
        assert value in doc


def test_final_gate_config_matches_required_gates() -> None:
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert spec["kind"] == "active_learning"
    assert spec["spec"]["target"]["diameter_um"] == 3.4
    assert spec["spec"]["target"]["use_prior_training_data"] is False
    assert list(spec["spec"]["candidate_dimensions"]) == ["ka", "kb"]
    assert spec["spec"]["candidate_dimensions"]["ka"]["min"] == 100.0
    assert spec["spec"]["candidate_dimensions"]["ka"]["max"] == 600000.0
    assert spec["spec"]["candidate_dimensions"]["kb"]["min"] == 400.0
    assert spec["spec"]["candidate_dimensions"]["kb"]["max"] == 70000.0
    assert spec["spec"]["runtime_parameterization"]["active_controls"] == ["ka", "kb"]
    assert spec["spec"]["runtime_parameterization"]["legacy_derived_controls"] == ["Yt"]

    assert spec["spec"]["candidate_selection"]["sampling"] == "log10"
    assert spec["spec"]["candidate_selection"]["source_constraints"]["prior_training_data"] is False
    assert spec["spec"]["candidate_selection"]["source_constraints"]["fresh_dpd_labels_only"] is True

    assert spec["spec"]["active_learning"]["rounds"] == 3
    assert spec["spec"]["active_learning"]["curves_per_round"] == 30
    assert spec["spec"]["active_learning"]["full_force_sweep"] is True
    assert spec["spec"]["active_learning"]["comparator"]["type"] == "lhs"
    assert spec["spec"]["active_learning"]["comparator"]["curve_count"] == 90
    assert spec["spec"]["active_learning"]["comparator"]["al_prefixes"] == [30, 60, 90]
    assert spec["spec"]["active_learning"]["comparator"]["lhs_prefixes"] == [30, 60, 90]

    round_plan = spec["spec"]["active_learning"]["round_plan"]
    assert len(round_plan) == 3
    assert round_plan[0]["index"] == 1
    assert round_plan[0]["source"] == "initial_sobol_maximin"
    assert round_plan[0]["sample_count"] == 30
    assert round_plan[1]["index"] == 2
    assert round_plan[1]["source"] == "ensemble_disagreement_diversity"
    assert round_plan[1]["exploration_count"] == 6
    assert round_plan[1]["acquisition_count"] == 24
    assert round_plan[1]["source_distribution"] == {
        "exploration": 6,
        "ensemble_disagreement_diversity": 24,
    }
    assert round_plan[2]["index"] == 3
    assert round_plan[2]["source"] == "ensemble_disagreement_diversity"
    assert round_plan[2]["exploration_count"] == 6
    assert round_plan[2]["acquisition_count"] == 24
    assert round_plan[2]["source_distribution"] == {
        "exploration": 6,
        "ensemble_disagreement_diversity": 24,
    }

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
    assert runtime_fingerprint["force_grid_input"] == "samples_all.dat"
    assert runtime_fingerprint["force_grid_mode"] == "only"
    assert runtime_fingerprint["force_grid_um_3_4"] == _expected_force_grid()

    required_plots = set(spec["spec"]["required_plots"])
    assert required_plots == {
        "runtime_per_curve",
        "samples_ka_kb",
        "per_round_additions",
        "exploration_vs_acquisition",
        "disagreement_acquisition_map",
        "force_curve_overlays",
        "al_vs_lhs_relative_l2",
        "failure_quarantine_replacement",
    }
