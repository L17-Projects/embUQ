from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS = REPO_ROOT / "scripts" / "postprocess"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# generate_surrogate_comparison.py
# ---------------------------------------------------------------------------

def _make_holdout_l2_summary(tmp_path: Path) -> Path:
    """Make a summary CSV with DNN values within 5% of UQ_DPD reference and BNN slightly better."""
    # DNN values chosen to be within ±3% of UQ_DPD reference for parity checks.
    _dnn = {
        ("compression", "2.1"): 0.93,
        ("compression", "2.9"): 2.10,
        ("compression", "3.0"): 2.07,
        ("indentation", "3.2"): 1.93,
        ("indentation", "3.4"): 2.22,
        ("indentation", "5.8"): 1.27,
    }
    rows = []
    for (modality, diam), dnn_val in _dnn.items():
        for family, val in [("dnn", dnn_val), ("bnn", round(dnn_val * 0.97, 4))]:
            rows.append({
                "modality": modality,
                "surrogate_family": family,
                "diameter_um": diam,
                "metric_rel_l2_pct": val,
                "summary_path": f"/{modality}/{diam}um/{family}/summary.json",
            })
    csv = tmp_path / "surrogate_holdout_l2_summary.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_comparison_go_verdict(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_surrogate_comparison.py", "gen_comparison_go")
    csv = _make_holdout_l2_summary(tmp_path)
    df = mod._load_summary(csv)
    comp = mod._build_comparison(df)
    verdict, reasons = mod._decision(comp)
    assert verdict == "GO", f"Expected GO, got {verdict}: {reasons}"
    assert not reasons


def test_comparison_nogo_worse_than_10pct(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_surrogate_comparison.py", "gen_comparison_nogo")
    rows = []
    for modality, diameters in [("compression", ["2.1", "2.9", "3.0"]), ("indentation", ["3.2", "3.4", "5.8"])]:
        for diam in diameters:
            rows.append({"modality": modality, "surrogate_family": "dnn", "diameter_um": diam, "metric_rel_l2_pct": 2.0, "summary_path": ""})
            # BNN is 15% worse on one case
            bnn_val = 2.3 if (modality == "compression" and diam == "2.1") else 1.9
            rows.append({"modality": modality, "surrogate_family": "bnn", "diameter_um": diam, "metric_rel_l2_pct": bnn_val, "summary_path": ""})
    csv = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    df = mod._load_summary(csv)
    comp = mod._build_comparison(df)
    verdict, reasons = mod._decision(comp)
    assert verdict == "NO-GO"
    assert any("15" in r or ">" in r or "10%" in r or ">10%" in r for r in reasons)


def test_comparison_main_writes_files(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module(POSTPROCESS / "generate_surrogate_comparison.py", "gen_comparison_main")
    csv = _make_holdout_l2_summary(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "generate_surrogate_comparison.py",
        "--input-csv", str(csv),
        "--output-dir", str(out_dir),
    ])
    rc = mod.main()
    assert rc == 0
    assert (out_dir / "surrogate_family_comparison.csv").exists()
    assert (out_dir / "bnn_insertion_decision.md").exists()
    comp_df = pd.read_csv(out_dir / "surrogate_family_comparison.csv")
    assert "dnn_median_rel_l2_pct" in comp_df.columns
    assert "bnn_median_rel_l2_pct" in comp_df.columns
    assert "winner" in comp_df.columns


def test_comparison_missing_columns_raises(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_surrogate_comparison.py", "gen_comparison_err")
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"foo": 1}]).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        mod._load_summary(bad_csv)


def test_comparison_nogo_for_incomplete_pairs_and_low_bnn_wins(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_surrogate_comparison.py", "gen_comparison_incomplete")
    rows = []
    for modality, diameters in [("compression", ["2.1", "2.9", "3.0"]), ("indentation", ["3.2", "3.4", "5.8"])]:
        for diam in diameters:
            # DNN-only rows force incomplete pairing and bnn_wins=0.
            rows.append({
                "modality": modality,
                "surrogate_family": "dnn",
                "diameter_um": diam,
                "metric_rel_l2_pct": 2.0,
                "summary_path": "",
            })
    csv = tmp_path / "summary_incomplete.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    comp = mod._build_comparison(mod._load_summary(csv))
    verdict, reasons = mod._decision(comp)
    assert verdict == "NO-GO"
    assert any("paired runs completed" in r for r in reasons)
    assert any("better or equal" in r for r in reasons)


# ---------------------------------------------------------------------------
# generate_uqdpd_parity_report.py
# ---------------------------------------------------------------------------

def _make_uqdpd_reference(tmp_path: Path) -> Path:
    rows = [
        {"modality": "compression", "diameter_um": "2.1", "best_median_curve_rel_l2_pct": 0.92},
        {"modality": "compression", "diameter_um": "2.9", "best_median_curve_rel_l2_pct": 2.09},
        {"modality": "compression", "diameter_um": "3.0", "best_median_curve_rel_l2_pct": 2.06},
        {"modality": "indentation", "diameter_um": "3.2", "best_median_curve_rel_l2_pct": 1.91},
        {"modality": "indentation", "diameter_um": "3.4", "best_median_curve_rel_l2_pct": 2.21},
        {"modality": "indentation", "diameter_um": "5.8", "best_median_curve_rel_l2_pct": 1.26},
    ]
    csv = tmp_path / "uqdpd_ref.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_parity_pass(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_uqdpd_parity_report.py", "gen_parity_pass")
    l2_csv = _make_holdout_l2_summary(tmp_path)
    ref_csv = _make_uqdpd_reference(tmp_path)
    mesouq = mod._load_mesouq(l2_csv)
    uqdpd = mod._load_uqdpd(ref_csv)
    metrics = mod._build_parity_metrics(mesouq, uqdpd)
    assert metrics["parity_ok"].all(), metrics[["modality", "diameter_um", "delta_rel_pct"]].to_string()


def test_parity_fail_large_deviation(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_uqdpd_parity_report.py", "gen_parity_fail")
    rows = []
    for modality, diameters in [("compression", ["2.1", "2.9", "3.0"]), ("indentation", ["3.2", "3.4", "5.8"])]:
        for diam in diameters:
            # Make one outlier 50% off reference
            val = 1.38 if (modality == "compression" and diam == "2.1") else 2.0
            rows.append({"modality": modality, "surrogate_family": "dnn", "diameter_um": diam, "metric_rel_l2_pct": val, "summary_path": ""})
    csv = tmp_path / "summary_fail.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    ref_csv = _make_uqdpd_reference(tmp_path)
    mesouq = mod._load_mesouq(csv)
    uqdpd = mod._load_uqdpd(ref_csv)
    metrics = mod._build_parity_metrics(mesouq, uqdpd)
    assert not metrics["parity_ok"].all()


def test_parity_main_writes_files(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module(POSTPROCESS / "generate_uqdpd_parity_report.py", "gen_parity_main")
    l2_csv = _make_holdout_l2_summary(tmp_path)
    ref_csv = _make_uqdpd_reference(tmp_path)
    out_dir = tmp_path / "parity_out"
    monkeypatch.setattr(sys, "argv", [
        "generate_uqdpd_parity_report.py",
        "--mesouq-summary-csv", str(l2_csv),
        "--uqdpd-reference-csv", str(ref_csv),
        "--output-dir", str(out_dir),
    ])
    rc = mod.main()
    assert rc == 0
    assert (out_dir / "uq_dpd_parity_metrics.csv").exists()
    assert (out_dir / "uq_dpd_parity_report.md").exists()
    md = (out_dir / "uq_dpd_parity_report.md").read_text()
    assert "PASS" in md


def test_parity_loaders_missing_columns_raise(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_uqdpd_parity_report.py", "gen_parity_loader_errors")
    bad_mesouq = tmp_path / "bad_mesouq.csv"
    bad_uqdpd = tmp_path / "bad_uqdpd.csv"
    pd.DataFrame([{"foo": 1}]).to_csv(bad_mesouq, index=False)
    pd.DataFrame([{"bar": 2}]).to_csv(bad_uqdpd, index=False)

    with pytest.raises(ValueError, match="MesoUQ summary CSV missing columns"):
        mod._load_mesouq(bad_mesouq)
    with pytest.raises(ValueError, match="UQ_DPD reference CSV missing columns"):
        mod._load_uqdpd(bad_uqdpd)


# ---------------------------------------------------------------------------
# generate_final_report.py
# ---------------------------------------------------------------------------

def test_final_report_writes_file(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module(POSTPROCESS / "generate_final_report.py", "gen_final_report")
    out_dir = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", [
        "generate_final_report.py",
        "--run-root", str(tmp_path),
        "--output-dir", str(out_dir),
    ])
    rc = mod.main()
    assert rc == 0
    report = out_dir / "final_report.md"
    assert report.exists()
    text = report.read_text()
    assert "BNN insertion recommendation" in text
    assert "Stage pass/fail" in text


def test_final_report_bnn_status_from_json(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_final_report.py", "gen_final_report_json")
    bnn_dir = tmp_path / "bnn_training"
    bnn_dir.mkdir()
    (bnn_dir / "bnn_training_matrix_report.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    out_dir = tmp_path / "comparison"
    assert mod.main(["--run-root", str(tmp_path), "--output-dir", str(out_dir)]) == 0
    text = (out_dir / "final_report.md").read_text()
    assert "PASSED" in text


def test_final_report_handles_unreadable_bnn_status_and_verdict_extraction(tmp_path: Path) -> None:
    mod = _load_module(POSTPROCESS / "generate_final_report.py", "gen_final_report_unreadable")
    bnn_dir = tmp_path / "bnn_training"
    out_dir = tmp_path / "comparison"
    bnn_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    # Invalid JSON exercises the unreadable-status fallback branch.
    (bnn_dir / "bnn_training_matrix_report.json").write_text("{not-json", encoding="utf-8")
    (out_dir / "bnn_insertion_decision.md").write_text(
        "# BNN Insertion Decision\n\n**Verdict: GO**\n", encoding="utf-8"
    )
    # Existing file without the keyword exercises the UNKNOWN path.
    (out_dir / "uq_dpd_parity_report.md").write_text(
        "# UQ_DPD Parity Report\n\nNo explicit verdict here.\n", encoding="utf-8"
    )

    assert mod.main(["--run-root", str(tmp_path), "--output-dir", str(out_dir)]) == 0
    text = (out_dir / "final_report.md").read_text(encoding="utf-8")
    assert "FAIL (unreadable)" in text
    assert "**Verdict: GO**" in text
    assert "**UNKNOWN**" in text
