from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _parse_arg(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def _write_selection(seed_dir: Path, *, family: str, artifact_path: Path, report_path: Path) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    if family == "dnn":
        payload = {
            "best_artifact_path": str(artifact_path),
            "best_report_path": str(report_path),
        }
    else:
        payload = {
            "best_candidate_artifact_path": str(artifact_path),
            "best_candidate_report_path": str(report_path),
        }
    (seed_dir / "selection.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_dnn_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "out": str(path.with_suffix(".pkl")),
                "train_loss": 0.1,
                "val_loss": 0.2,
                "val_rmse_phys": 1.0,
            }
        ),
        encoding="utf-8",
    )


def _write_bnn_report(path: Path, *, reload_passed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "out": str(path.with_suffix(".pt")),
                "training": {
                    "best_val_rmse": 1.0,
                    "final_val_rmse": 1.0,
                },
                "reload": {
                    "val_rmse": 1.04,
                    "degradation_abs": 0.01,
                    "degradation_rel": 0.01,
                    "pred_std_mean": 0.05,
                    "passed_rel_tol_0p05": reload_passed,
                },
            }
        ),
        encoding="utf-8",
    )


def _fake_holdout_outputs(output_dir: Path, *, family: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rel_l2 = [10.0, 12.0, 14.0]
    rmse = [1.0, 1.2, 1.4]
    metrics_df = pd.DataFrame(
        {
            "curve_id": [0, 1, 2],
            "n_points": [4, 4, 4],
            "rmse": rmse,
            "rel_l2_pct": rel_l2,
            "max_abs_err": [1.0, 1.1, 1.2],
            "pred_std_mean": [0.0, 0.0, 0.0] if family == "dnn" else [0.05, 0.05, 0.05],
            "pred_std_p95": [0.0, 0.0, 0.0] if family == "dnn" else [0.06, 0.06, 0.06],
        }
    )
    metrics_df.to_csv(output_dir / "per_curve_metrics.csv", index=False)
    pd.DataFrame(
        {
            "curve_id": [0, 0, 1, 1, 2, 2],
            "disp": [0, 1, 0, 1, 0, 1],
            "F_true": [0, 1, 0, 1, 0, 1],
            "F_pred": [0, 1, 0, 1, 0, 1],
            "pred_std": [0, 0, 0, 0, 0, 0],
        }
    ).to_csv(output_dir / "per_curve_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": f"{family}_candidate",
                "surrogate_family": family,
                "modality": "compression",
                "diameter_um": "2.1",
                "mean_curve_rmse": float(metrics_df["rmse"].mean()),
                "median_curve_rmse": float(metrics_df["rmse"].median()),
                "mean_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].mean()),
                "median_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].median()),
                "max_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].max()),
                "mean_curve_pred_std": float(metrics_df["pred_std_mean"].mean()),
                "model_path": str(output_dir / f"{family}.pkl"),
                "metrics_path": str(output_dir / "per_curve_metrics.csv"),
                "predictions_path": str(output_dir / "per_curve_predictions.csv"),
            }
        ]
    ).to_csv(output_dir / "training_results_group_holdout.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "modality": "compression",
                "surrogate_family": family,
                "diameter_um": "2.1",
                "seed": 101,
                "val_fraction": 0.10,
                "n_curves_total": 3,
                "n_curves_train": 2,
                "n_curves_validation": 1,
                "best_architecture": f"{family}_candidate",
                "best_mean_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].mean()),
                "best_median_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].median()),
                "best_max_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].max()),
                "representative_curve_id": 1,
                "representative_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].median()),
                "best_model_path": str(output_dir / f"{family}.pkl"),
            }
        ),
        encoding="utf-8",
    )


def test_certification_helpers_resolve_and_extract(tmp_path: Path) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_certification_matrix.py"),
        "run_bnn_certification_matrix_helpers_test",
    )

    dnn_root = tmp_path / "dnn"
    bnn_root = tmp_path / "bnn"
    for root in (dnn_root, bnn_root):
        (root / "spec_a" / "seed_101").mkdir(parents=True, exist_ok=True)
        (root / "spec_a" / "seed_102").mkdir(parents=True, exist_ok=True)
    assert module._resolve_seeds_for_spec(
        dnn_root=dnn_root,
        bnn_root=bnn_root,
        spec_name="spec_a",
        requested_seeds=[],
    ) == [101, 102]

    output_dir = tmp_path / "holdout"
    assert not module._holdout_completed(output_dir)
    _fake_holdout_outputs(output_dir, family="dnn")
    assert module._holdout_completed(output_dir)

    dnn_report = tmp_path / "dnn_report.json"
    bnn_report = tmp_path / "bnn_report.json"
    _write_dnn_report(dnn_report)
    _write_bnn_report(bnn_report)
    assert module._extract_validation_rmse(family="dnn", report_path=dnn_report) == pytest.approx(1.0)
    assert module._extract_validation_rmse(family="bnn", report_path=bnn_report) == pytest.approx(1.0)
    reload_payload = module._extract_reload_summary(bnn_report)
    assert reload_payload["reload_passed"] is True


def test_bnn_certification_matrix_runner_writes_gate_outputs(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_certification_matrix.py",
        "run_bnn_certification_matrix_main_test",
    )

    spec = {
        "name": "compression_2.1um",
        "modality": "compression",
        "diameter_um": "2.1",
        "data": str(tmp_path / "F_Delta.dat"),
        "dnn_artifact": str(tmp_path / "tracked_force_bnn.pt"),
        "bnn_artifact": str(tmp_path / "tracked_force_bnn.pt"),
        "dnn_train_script": str(tmp_path / "unused.py"),
        "dnn_multi_arch_script": str(tmp_path / "unused.py"),
        "bnn_train_script": str(tmp_path / "unused.py"),
        "group_holdout_script": str(tmp_path / "run_group_holdout.py"),
    }
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    dnn_root = tmp_path / "dnn_root"
    bnn_root = tmp_path / "bnn_root"
    dnn_seed_dir = dnn_root / spec["name"] / "seed_101"
    bnn_seed_dir = bnn_root / spec["name"] / "seed_101"
    dnn_artifact = tmp_path / "dnn_candidate.pkl"
    bnn_artifact = tmp_path / "bnn_candidate.pt"
    dnn_artifact.write_text("artifact", encoding="utf-8")
    bnn_artifact.write_text("artifact", encoding="utf-8")
    dnn_report = tmp_path / "reports" / "dnn.json"
    bnn_report = tmp_path / "reports" / "bnn.json"
    _write_dnn_report(dnn_report)
    _write_bnn_report(bnn_report, reload_passed=True)
    _write_selection(dnn_seed_dir, family="dnn", artifact_path=dnn_artifact, report_path=dnn_report)
    _write_selection(bnn_seed_dir, family="bnn", artifact_path=bnn_artifact, report_path=bnn_report)

    called: list[list[str]] = []

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        called.append(list(command))
        output_dir = Path(_parse_arg(command, "--output-dir"))
        family = _parse_arg(command, "--surrogate-family")
        _fake_holdout_outputs(output_dir, family=family)

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--dnn-root",
            str(dnn_root),
            "--bnn-root",
            str(bnn_root),
            "--output-root",
            str(tmp_path / "cert"),
            "--expected-seed-count",
            "1",
            "--bootstrap-resamples",
            "200",
        ]
    )
    assert rc == 0
    assert len(called) == 2
    assert "--surrogate-family dnn" in " ".join(called[0])
    assert "--surrogate-family bnn" in " ".join(called[1])

    per_dataset = pd.read_csv(tmp_path / "cert" / "certification_per_dataset.csv")
    assert per_dataset.loc[0, "certified"]
    assert int(per_dataset.loc[0, "candidate_seed"]) == 101

    promotion_candidates = json.loads((tmp_path / "cert" / "promotion_candidates.json").read_text(encoding="utf-8"))
    assert promotion_candidates[0]["tracked_bnn_artifact_path"].endswith("tracked_force_bnn.pt")


def test_bnn_certification_matrix_runner_fails_gate_when_reload_fails(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_certification_matrix.py",
        "run_bnn_certification_matrix_failure_test",
    )

    spec = {
        "name": "compression_2.1um",
        "modality": "compression",
        "diameter_um": "2.1",
        "data": str(tmp_path / "F_Delta.dat"),
        "dnn_artifact": str(tmp_path / "tracked_force_bnn.pt"),
        "bnn_artifact": str(tmp_path / "tracked_force_bnn.pt"),
        "dnn_train_script": str(tmp_path / "unused.py"),
        "dnn_multi_arch_script": str(tmp_path / "unused.py"),
        "bnn_train_script": str(tmp_path / "unused.py"),
        "group_holdout_script": str(tmp_path / "run_group_holdout.py"),
    }
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    dnn_root = tmp_path / "dnn_root"
    bnn_root = tmp_path / "bnn_root"
    dnn_seed_dir = dnn_root / spec["name"] / "seed_101"
    bnn_seed_dir = bnn_root / spec["name"] / "seed_101"
    dnn_artifact = tmp_path / "dnn_candidate.pkl"
    bnn_artifact = tmp_path / "bnn_candidate.pt"
    dnn_artifact.write_text("artifact", encoding="utf-8")
    bnn_artifact.write_text("artifact", encoding="utf-8")
    dnn_report = tmp_path / "reports" / "dnn.json"
    bnn_report = tmp_path / "reports" / "bnn.json"
    _write_dnn_report(dnn_report)
    _write_bnn_report(bnn_report, reload_passed=False)
    _write_selection(dnn_seed_dir, family="dnn", artifact_path=dnn_artifact, report_path=dnn_report)
    _write_selection(bnn_seed_dir, family="bnn", artifact_path=bnn_artifact, report_path=bnn_report)

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        output_dir = Path(_parse_arg(command, "--output-dir"))
        family = _parse_arg(command, "--surrogate-family")
        _fake_holdout_outputs(output_dir, family=family)

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--dnn-root",
            str(dnn_root),
            "--bnn-root",
            str(bnn_root),
            "--output-root",
            str(tmp_path / "cert"),
            "--expected-seed-count",
            "1",
            "--bootstrap-resamples",
            "200",
        ]
    )
    assert rc == 1
    per_dataset = pd.read_csv(tmp_path / "cert" / "certification_per_dataset.csv")
    assert not bool(per_dataset.loc[0, "certified"])


def test_hpc_bnn_certification_wrapper_dispatches_to_selected_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_bnn_certification_matrix.py"),
        "hpc_bnn_certification_dispatch_test",
    )
    monkeypatch.setenv("HPC_SITE", "karolina")
    captured: list[list[str]] = []

    def _fake_call(cmd):  # noqa: ANN001
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    rc = module.main(["--seed", "123"])
    assert rc == 0
    assert captured
    assert sys.executable in captured[0][0]
    assert "scripts/platforms/karolina/run_bnn_certification_matrix.py" in " ".join(captured[0])


def test_hpc_bnn_certification_wrapper_rejects_unknown_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_bnn_certification_matrix.py"),
        "hpc_bnn_certification_invalid_site_test",
    )
    monkeypatch.setenv("HPC_SITE", "unknown")
    with pytest.raises(SystemExit, match="Unsupported HPC_SITE"):
        module.main([])
