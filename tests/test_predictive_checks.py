import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from meso_uq.noise import (
    PredictiveCheckInputs,
    PredictiveCheckThresholds,
    SbcRankRecord,
    evaluate_predictive_checks,
)
from scripts.qa.noise_m6_predictive_checks_diagnostics import _nonnegative_interval_errors


def _tuple_vector(values):
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _tuple_matrix(values):
    matrix = np.asarray(values, dtype=float)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _rank_records(*, edge=False):
    samples = np.linspace(-1.0, 1.0, 31)
    records = []
    quantiles = np.linspace(0.10, 0.90, 15)
    for index, quantile in enumerate(quantiles):
        true_value = -1.4 if edge else float(np.quantile(samples, quantile))
        records.append(SbcRankRecord(f"theta_{index:02d}", true_value, _tuple_vector(samples)))
    return tuple(records)


def _inputs(*, observed_shift=0.0, edge_ranks=False, nonfinite=False):
    x = np.linspace(-1.0, 1.0, 24)
    observed = 1.0 + 0.25 * x + observed_shift
    rng = np.random.default_rng(3517)
    samples = np.asarray([1.0 + 0.25 * x + rng.normal(0.0, 0.04, size=x.size) for _ in range(96)], dtype=float)
    if nonfinite:
        samples[0, 0] = np.nan
    return PredictiveCheckInputs(
        scenario_id="baseline",
        seed=3517,
        observed=_tuple_vector(observed),
        predictive_samples=_tuple_matrix(samples),
        sbc_rank_records=_rank_records(edge=edge_ranks),
    )


def test_predictive_checks_pass_calibrated_fixture():
    result = evaluate_predictive_checks(_inputs())

    assert result.gate_status == "pass"
    assert result.runtime_failures == ()
    assert result.calibration_failures == ()
    assert result.ppc_metrics["finite_fraction"] == pytest.approx(1.0)
    assert result.ppc_metrics["pointwise_metrics"]["interval_coverage"] >= 0.85
    assert result.sbc_metrics["record_count"] == 15
    assert result.sbc_metrics["rank_histogram_l1"] <= 0.45


def test_predictive_checks_separate_calibration_failures():
    result = evaluate_predictive_checks(_inputs(observed_shift=0.8))

    assert result.gate_status == "fail"
    assert result.runtime_failures == ()
    assert result.calibration_failures
    assert any("posterior predictive summary" in failure for failure in result.calibration_failures)


def test_predictive_checks_separate_runtime_failures():
    result = evaluate_predictive_checks(_inputs(nonfinite=True))

    assert result.gate_status == "fail"
    assert result.runtime_failures
    assert any("nonfinite" in failure for failure in result.runtime_failures)


def test_predictive_checks_fail_edge_rank_fixture():
    result = evaluate_predictive_checks(_inputs(edge_ranks=True))

    assert result.gate_status == "fail"
    assert any("SBC" in failure for failure in result.calibration_failures)


def test_sbc_rank_record_uses_midrank_for_ties():
    record = SbcRankRecord("theta", 0.0, (-1.0, 0.0, 0.0, 1.0))

    assert record.rank() == pytest.approx(2.0)
    assert record.rank_quantile() == pytest.approx(0.5)
    assert record.as_dict(alpha=0.1)["rank"] == pytest.approx(2.0)


def test_pointwise_rank_quantiles_use_midrank_for_ties():
    result = evaluate_predictive_checks(
        PredictiveCheckInputs(
            scenario_id="pointwise_ties",
            seed=1,
            observed=(0.0,),
            predictive_samples=((-1.0,), (0.0,), (0.0,), (1.0,)),
            sbc_rank_records=(SbcRankRecord("theta", 0.0, (-1.0, 0.0, 0.0, 1.0)),),
        ),
        PredictiveCheckThresholds(
            min_pointwise_interval_coverage=0.0,
            max_pointwise_rank_edge_fraction=1.0,
            max_sbc_rank_histogram_l1=2.0,
            max_sbc_mean_rank_quantile_error=1.0,
            max_sbc_edge_fraction=1.0,
            min_sbc_interval_coverage=0.0,
        ),
    )

    assert result.ppc_metrics["pointwise_metrics"]["rank_quantiles"] == pytest.approx((0.5,))


def test_predictive_check_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="column count"):
        PredictiveCheckInputs(
            scenario_id="bad",
            seed=1,
            observed=(1.0, 2.0),
            predictive_samples=((1.0,), (2.0,)),
        )
    with pytest.raises(ValueError, match="at least two draws"):
        SbcRankRecord("bad", 0.0, (1.0,))
    with pytest.raises(ValueError, match="rank_bin_count"):
        PredictiveCheckThresholds(rank_bin_count=1)


def test_predictive_checks_diagnostics_script_writes_required_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "m6_predictive"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_m6_predictive_checks_diagnostics.py"),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repo_root,
    )

    manifest = json.loads((output_root / "predictive_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_root / "predictive_check_metrics.json").read_text(encoding="utf-8"))
    assert manifest["required_scenarios"] == ["baseline_legacy", "full_hierarchy"]
    assert metrics["all_scenarios_passed"] is True
    assert set(metrics["scenario_gate_statuses"].values()) == {"pass"}
    for scenario_id in manifest["required_scenarios"]:
        artifacts = manifest["scenario_artifacts"][scenario_id]
        for path in artifacts.values():
            assert Path(path).exists()
        report = json.loads(Path(artifacts["predictive_report"]).read_text(encoding="utf-8"))
        assert report["summary"]["failure_classes"] == {"calibration": [], "runtime_or_numerical": []}
    for key in (
        "ppc_summary_intervals",
        "ppc_observable_overlay",
        "sbc_rank_histogram",
        "calibration_summary",
    ):
        path = Path(manifest["artifacts"][key])
        assert path.exists()
        assert path.stat().st_size > 0


def test_predictive_checks_diagnostics_metrics_are_reproducible(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    first = tmp_path / "first"
    second = tmp_path / "second"
    command = [sys.executable, str(repo_root / "scripts/qa/noise_m6_predictive_checks_diagnostics.py")]
    subprocess.run([*command, "--output-root", str(first)], check=True, cwd=repo_root)
    subprocess.run([*command, "--output-root", str(second)], check=True, cwd=repo_root)

    first_metrics = json.loads((first / "predictive_check_metrics.json").read_text(encoding="utf-8"))
    second_metrics = json.loads((second / "predictive_check_metrics.json").read_text(encoding="utf-8"))
    assert first_metrics == second_metrics


def test_predictive_summary_interval_errors_are_nonnegative_for_skewed_draws():
    lower, upper = _nonnegative_interval_errors(
        centers=(10.0, -10.0, 0.5),
        lows=(0.0, -1.0, 0.0),
        highs=(1.0, 0.0, 1.0),
    )

    assert tuple(lower) == pytest.approx((10.0, 0.0, 0.5))
    assert tuple(upper) == pytest.approx((0.0, 10.0, 0.5))
