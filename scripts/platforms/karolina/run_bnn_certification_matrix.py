#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.hpc_paths import default_runs_root, detect_hpc_site  # noqa: E402
from meso_uq.surrogate.certification import (  # noqa: E402
    build_certification_summary,
    build_paired_metric_table,
    certify_paired_metrics,
)
from meso_uq.surrogate.emb_catalog import resolve_emb_dataset_specs  # noqa: E402

DEFAULT_METRIC_COLS = (
    "validation_rmse",
    "holdout_median_rel_l2_pct",
    "holdout_p95_rel_l2_pct",
    "holdout_max_rel_l2_pct",
)
HOLDOUT_SELECTION_CONTEXT = "selection_context.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _selection_path(root: Path, spec_name: str, seed: int) -> Path:
    return root / spec_name / f"seed_{seed}" / "selection.json"


def _discover_seed_values(root: Path, spec_name: str) -> set[int]:
    seed_values: set[int] = set()
    spec_root = root / spec_name
    if not spec_root.exists():
        return seed_values
    for candidate in spec_root.glob("seed_*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix("seed_")
        if suffix.isdigit():
            seed_values.add(int(suffix))
    return seed_values


def _resolve_seeds_for_spec(
    *,
    dnn_root: Path,
    bnn_root: Path,
    spec_name: str,
    requested_seeds: list[int],
) -> list[int]:
    if requested_seeds:
        return list(dict.fromkeys(int(seed) for seed in requested_seeds))
    common = sorted(_discover_seed_values(dnn_root, spec_name) & _discover_seed_values(bnn_root, spec_name))
    if not common:
        raise FileNotFoundError(
            f"No common seed selections found for {spec_name!r} under {dnn_root} and {bnn_root}."
        )
    return common


def _holdout_output_dir(output_root: Path, spec_name: str, seed: int, family: str) -> Path:
    return output_root / "holdout" / spec_name / f"seed_{seed}" / family


def _holdout_completed(output_dir: Path) -> bool:
    required = (
        output_dir / "summary.json",
        output_dir / "per_curve_metrics.csv",
        output_dir / "training_results_group_holdout.csv",
    )
    return all(path.exists() for path in required)


def _holdout_selection_context_path(output_dir: Path) -> Path:
    return output_dir / HOLDOUT_SELECTION_CONTEXT


def _selection_artifact_info(
    *,
    family: str,
    selection_path: Path,
) -> tuple[Path, Path]:
    payload = _read_json(selection_path)
    if family == "dnn":
        artifact_key = "best_artifact_path"
        report_key = "best_report_path"
    else:
        artifact_key = "best_candidate_artifact_path"
        report_key = "best_candidate_report_path"
    artifact_path = Path(str(payload[artifact_key])).resolve()
    report_path = Path(str(payload[report_key])).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Selected {family} artifact does not exist: {artifact_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Selected {family} report does not exist: {report_path}")
    return artifact_path, report_path


def _holdout_selection_context(
    *,
    family: str,
    artifact_path: Path,
    report_path: Path,
    selection_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "surrogate_family": family,
        "artifact_path": str(artifact_path.resolve()),
        "selection_report_path": str(report_path.resolve()),
        "selection_path": str(selection_path.resolve()),
    }


def _write_holdout_selection_context(
    *,
    output_dir: Path,
    family: str,
    artifact_path: Path,
    report_path: Path,
    selection_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = _holdout_selection_context_path(output_dir)
    payload = _holdout_selection_context(
        family=family,
        artifact_path=artifact_path,
        report_path=report_path,
        selection_path=selection_path,
    )
    context_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _holdout_outputs_match_selection(
    *,
    output_dir: Path,
    family: str,
    artifact_path: Path,
    report_path: Path,
    selection_path: Path,
) -> bool:
    context_path = _holdout_selection_context_path(output_dir)
    if not context_path.exists():
        return False
    try:
        payload = _read_json(context_path)
    except (OSError, json.JSONDecodeError):
        return False
    expected = _holdout_selection_context(
        family=family,
        artifact_path=artifact_path,
        report_path=report_path,
        selection_path=selection_path,
    )
    return all(payload.get(key) == value for key, value in expected.items())


def _build_holdout_command(
    *,
    python_bin: str,
    spec: dict[str, str],
    family: str,
    artifact_path: Path,
    output_dir: Path,
    seed: int,
    val_fraction: float,
    predictive_mc_samples: int,
    predictive_mc_chunk_size: int,
    device: str,
    disp_source: str,
    rupture_ratio: float,
) -> list[str]:
    command = [
        python_bin,
        spec["group_holdout_script"],
        "--diameter",
        spec["diameter_um"],
        "--data-file",
        spec["data"],
        "--model",
        str(artifact_path),
        "--surrogate-family",
        family,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--val-fraction",
        str(val_fraction),
        "--predictive-mc-samples",
        str(predictive_mc_samples),
        "--predictive-mc-chunk-size",
        str(predictive_mc_chunk_size),
        "--device",
        device,
    ]
    if spec["modality"] == "indentation":
        command.extend(["--disp-source", disp_source, "--rupture-ratio", str(rupture_ratio)])
    return command


def _extract_validation_rmse(*, family: str, report_path: Path) -> float:
    payload = _read_json(report_path)
    if family == "dnn":
        return float(payload["val_rmse_phys"])
    training = payload["training"]
    assert isinstance(training, dict)
    return float(training["final_val_rmse"])


def _extract_reload_summary(report_path: Path) -> dict[str, object]:
    payload = _read_json(report_path)
    reload_payload = payload.get("reload")
    if not isinstance(reload_payload, dict):
        return {
            "reload_passed": False,
            "reload_degradation_rel": float("nan"),
            "reload_degradation_abs": float("nan"),
            "reload_val_rmse": float("nan"),
            "reload_pred_std_mean": float("nan"),
        }
    return {
        "reload_passed": bool(reload_payload.get("passed_rel_tol_0p05", False)),
        "reload_degradation_rel": float(reload_payload.get("degradation_rel", float("nan"))),
        "reload_degradation_abs": float(reload_payload.get("degradation_abs", float("nan"))),
        "reload_val_rmse": float(reload_payload.get("val_rmse", float("nan"))),
        "reload_pred_std_mean": float(reload_payload.get("pred_std_mean", float("nan"))),
    }


def _build_family_metric_row(
    *,
    spec: dict[str, str],
    seed: int,
    family: str,
    artifact_path: Path,
    report_path: Path,
    selection_path: Path,
    holdout_output_dir: Path,
) -> dict[str, object]:
    metrics_df = pd.read_csv(holdout_output_dir / "per_curve_metrics.csv")
    summary = _read_json(holdout_output_dir / "summary.json")
    result_row = pd.read_csv(holdout_output_dir / "training_results_group_holdout.csv").iloc[0].to_dict()
    row: dict[str, object] = {
        "dataset_name": spec["name"],
        "modality": spec["modality"],
        "diameter_um": spec["diameter_um"],
        "seed": int(seed),
        "surrogate_family": family,
        "artifact_path": str(artifact_path),
        "selection_path": str(selection_path),
        "selection_report_path": str(report_path),
        "holdout_output_dir": str(holdout_output_dir),
        "validation_rmse": _extract_validation_rmse(family=family, report_path=report_path),
        "holdout_mean_rel_l2_pct": float(metrics_df["rel_l2_pct"].mean()),
        "holdout_median_rel_l2_pct": float(metrics_df["rel_l2_pct"].median()),
        "holdout_p95_rel_l2_pct": float(metrics_df["rel_l2_pct"].quantile(0.95)),
        "holdout_max_rel_l2_pct": float(metrics_df["rel_l2_pct"].max()),
        "holdout_mean_rmse": float(metrics_df["rmse"].mean()),
        "holdout_curve_count": int(len(metrics_df)),
        "holdout_summary_mean_rel_l2_pct": float(summary["best_mean_curve_rel_l2_pct"]),
        "holdout_summary_median_rel_l2_pct": float(summary["best_median_curve_rel_l2_pct"]),
        "holdout_summary_max_rel_l2_pct": float(summary["best_max_curve_rel_l2_pct"]),
        "holdout_group_result_name": str(result_row["name"]),
    }
    if family == "bnn":
        row.update(_extract_reload_summary(report_path))
    else:
        row.update(
            {
                "reload_passed": True,
                "reload_degradation_rel": 0.0,
                "reload_degradation_abs": 0.0,
                "reload_val_rmse": float(row["validation_rmse"]),
                "reload_pred_std_mean": 0.0,
            }
        )
    return row


def _final_candidate_rows(bnn_rows: pd.DataFrame, *, expected_seed_count: int) -> pd.DataFrame:
    sort_cols = [
        "holdout_median_rel_l2_pct",
        "holdout_p95_rel_l2_pct",
        "holdout_max_rel_l2_pct",
        "validation_rmse",
        "seed",
    ]
    winners: list[dict[str, object]] = []
    for dataset_name, group_df in bnn_rows.groupby("dataset_name", sort=True):
        ranked = group_df.sort_values(sort_cols).reset_index(drop=True)
        winner = ranked.iloc[0].to_dict()
        winner["expected_seed_count"] = int(expected_seed_count)
        winners.append(winner)
    if not winners:
        return pd.DataFrame()
    return pd.DataFrame(winners)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the seeded DNN-vs-BNN certification matrix over EMB surrogate datasets."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dnn-root", required=True, help="Output root from run_dnn_rebaseline_matrix.py")
    parser.add_argument("--bnn-root", required=True, help="Output root from run_bnn_sweep_matrix.py")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--expected-seed-count", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--predictive-mc-chunk-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--disp-source", choices=["auto", "diameter", "displacement"], default="auto")
    parser.add_argument("--rupture-ratio", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--acceptance-upper-bound", type=float, default=0.10)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed holdout output directories when present.",
    )
    args = parser.parse_args(argv)

    resolved_site = args.site if args.site is not None else detect_hpc_site()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else default_runs_root(REPO_ROOT, "bnn_certification", site=resolved_site, run_tag=args.run_tag)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dnn_root = Path(args.dnn_root).resolve()
    bnn_root = Path(args.bnn_root).resolve()
    specs = resolve_emb_dataset_specs(REPO_ROOT)
    if args.only:
        wanted = set(args.only)
        specs = [spec for spec in specs if spec["name"] in wanted]
        if not specs:
            raise ValueError(f"No EMB dataset specs matched --only values: {sorted(wanted)}")

    matrix_report_path = output_root / "bnn_certification_matrix_report.json"
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "output_root": str(output_root),
        "started_at": _now_iso(),
        "config": {
            "dnn_root": str(dnn_root),
            "bnn_root": str(bnn_root),
            "requested_seeds": [int(seed) for seed in args.seed],
            "expected_seed_count": int(args.expected_seed_count),
            "val_fraction": float(args.val_fraction),
            "predictive_mc_samples": int(args.predictive_mc_samples),
            "predictive_mc_chunk_size": int(args.predictive_mc_chunk_size),
            "device": str(args.device),
            "disp_source": str(args.disp_source),
            "rupture_ratio": float(args.rupture_ratio),
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "confidence_level": float(args.confidence_level),
            "acceptance_upper_bound": float(args.acceptance_upper_bound),
        },
        "runs": [],
    }
    _write_json(matrix_report_path, report)

    run_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []

    for spec in specs:
        seeds = _resolve_seeds_for_spec(
            dnn_root=dnn_root,
            bnn_root=bnn_root,
            spec_name=spec["name"],
            requested_seeds=[int(seed) for seed in args.seed],
        )
        for seed in seeds:
            for family, root in (("dnn", dnn_root), ("bnn", bnn_root)):
                selection_path = _selection_path(root, spec["name"], int(seed))
                if not selection_path.exists():
                    raise FileNotFoundError(f"Missing {family} selection file: {selection_path}")
                artifact_path, selection_report_path = _selection_artifact_info(
                    family=family,
                    selection_path=selection_path,
                )
                holdout_output_dir = _holdout_output_dir(output_root, spec["name"], int(seed), family)
                row = {
                    "dataset_name": spec["name"],
                    "modality": spec["modality"],
                    "diameter_um": spec["diameter_um"],
                    "seed": int(seed),
                    "surrogate_family": family,
                    "artifact_path": str(artifact_path),
                    "selection_path": str(selection_path),
                    "selection_report_path": str(selection_report_path),
                    "holdout_output_dir": str(holdout_output_dir),
                    "status": "pending",
                }
                can_resume = args.resume and _holdout_completed(holdout_output_dir)
                if can_resume and _holdout_outputs_match_selection(
                    output_dir=holdout_output_dir,
                    family=family,
                    artifact_path=artifact_path,
                    report_path=selection_report_path,
                    selection_path=selection_path,
                ):
                    row["status"] = "skipped_completed"
                    run_rows.append(row)
                else:
                    if can_resume:
                        print(
                            "[BNN certification] rerunning stale holdout outputs for "
                            f"{spec['name']} seed={seed} family={family} because the "
                            "selection context changed."
                        )
                    command = _build_holdout_command(
                        python_bin=args.python_bin,
                        spec=spec,
                        family=family,
                        artifact_path=artifact_path,
                        output_dir=holdout_output_dir,
                        seed=int(seed),
                        val_fraction=float(args.val_fraction),
                        predictive_mc_samples=int(args.predictive_mc_samples),
                        predictive_mc_chunk_size=int(args.predictive_mc_chunk_size),
                        device=str(args.device),
                        disp_source=str(args.disp_source),
                        rupture_ratio=float(args.rupture_ratio),
                    )
                    print(f"[BNN certification] running: {' '.join(command)}")
                    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
                    if not _holdout_completed(holdout_output_dir):
                        raise FileNotFoundError(
                            f"Holdout run did not produce required outputs in {holdout_output_dir}."
                        )
                    _write_holdout_selection_context(
                        output_dir=holdout_output_dir,
                        family=family,
                        artifact_path=artifact_path,
                        report_path=selection_report_path,
                        selection_path=selection_path,
                    )
                    row["status"] = "passed"
                    run_rows.append(row)

                if not _holdout_outputs_match_selection(
                    output_dir=holdout_output_dir,
                    family=family,
                    artifact_path=artifact_path,
                    report_path=selection_report_path,
                    selection_path=selection_path,
                ):
                    raise RuntimeError(
                        "Holdout outputs do not match the current selection context: "
                        f"{holdout_output_dir}"
                    )
                family_rows.append(
                    _build_family_metric_row(
                        spec=spec,
                        seed=int(seed),
                        family=family,
                        artifact_path=artifact_path,
                        report_path=selection_report_path,
                        selection_path=selection_path,
                        holdout_output_dir=holdout_output_dir,
                    )
                )

            report["runs"] = run_rows
            _write_json(matrix_report_path, report)

    family_metrics_df = pd.DataFrame(family_rows).sort_values(
        ["dataset_name", "seed", "surrogate_family"]
    ).reset_index(drop=True)
    family_metrics_path = output_root / "certification_seed_metrics.csv"
    family_metrics_df.to_csv(family_metrics_path, index=False)

    paired_metrics = build_paired_metric_table(
        family_metrics_df,
        metric_cols=DEFAULT_METRIC_COLS,
        group_cols=("dataset_name", "modality", "diameter_um", "seed"),
        family_col="surrogate_family",
    )
    paired_metrics_path = output_root / "certification_paired_metrics.csv"
    paired_metrics.to_csv(paired_metrics_path, index=False)

    certification_df = certify_paired_metrics(
        paired_metrics,
        diameter_col="dataset_name",
        n_resamples=int(args.bootstrap_resamples),
        confidence_level=float(args.confidence_level),
        random_seed=int(args.bootstrap_seed),
        acceptance_upper_bound=float(args.acceptance_upper_bound),
    )
    dataset_lookup = family_metrics_df[["dataset_name", "modality", "diameter_um"]].drop_duplicates("dataset_name")
    certification_df = certification_df.merge(dataset_lookup, on="dataset_name", how="left")
    certification_per_metric_path = output_root / "certification_per_metric.csv"
    certification_df.to_csv(certification_per_metric_path, index=False)

    certification_summary = build_certification_summary(certification_df, diameter_col="dataset_name")
    per_dataset_df = certification_summary.per_diameter.merge(dataset_lookup, on="dataset_name", how="left")

    bnn_rows = family_metrics_df[family_metrics_df["surrogate_family"] == "bnn"].copy()
    reload_summary = (
        bnn_rows.groupby(["dataset_name", "modality", "diameter_um"], sort=True)
        .agg(
            seed_count=("seed", "nunique"),
            reload_passed_count=("reload_passed", "sum"),
            all_reload_passed=("reload_passed", "all"),
        )
        .reset_index()
    )
    reload_summary["expected_seed_count"] = int(args.expected_seed_count)
    reload_summary["all_seeds_present"] = reload_summary["seed_count"].astype(int) == int(args.expected_seed_count)
    reload_summary["reload_failed_seed_count"] = (
        reload_summary["seed_count"].astype(int) - reload_summary["reload_passed_count"].astype(int)
    )

    winners_df = _final_candidate_rows(
        bnn_rows,
        expected_seed_count=int(args.expected_seed_count),
    )
    winners_df = winners_df.merge(
        dataset_lookup,
        on=["dataset_name", "modality", "diameter_um"],
        how="left",
    )

    if not winners_df.empty:
        spec_lookup = {
            spec["name"]: spec["bnn_artifact"] for spec in resolve_emb_dataset_specs(REPO_ROOT)
        }
        winners_df["tracked_bnn_artifact_path"] = winners_df["dataset_name"].map(spec_lookup)
        winners_df["candidate_seed"] = winners_df["seed"].astype(int)
        winners_df["candidate_artifact_path"] = winners_df["artifact_path"]
        winners_df["candidate_report_path"] = winners_df["selection_report_path"]
        winners_df = winners_df[
            [
                "dataset_name",
                "modality",
                "diameter_um",
                "candidate_seed",
                "candidate_artifact_path",
                "candidate_report_path",
                "tracked_bnn_artifact_path",
                "validation_rmse",
                "holdout_median_rel_l2_pct",
                "holdout_p95_rel_l2_pct",
                "holdout_max_rel_l2_pct",
                "reload_passed",
            ]
        ].sort_values("dataset_name")

    per_dataset_df = per_dataset_df.merge(
        reload_summary,
        on=["dataset_name", "modality", "diameter_um"],
        how="left",
    )
    if not winners_df.empty:
        per_dataset_df = per_dataset_df.merge(
            winners_df,
            on=["dataset_name", "modality", "diameter_um"],
            how="left",
        )
    per_dataset_df["seed_count"] = per_dataset_df["seed_count"].fillna(0).astype(int)
    per_dataset_df["reload_passed_count"] = per_dataset_df["reload_passed_count"].fillna(0).astype(int)
    per_dataset_df["reload_failed_seed_count"] = per_dataset_df["reload_failed_seed_count"].fillna(0).astype(int)
    per_dataset_df["all_reload_passed"] = per_dataset_df["all_reload_passed"].fillna(False).astype(bool)
    per_dataset_df["all_seeds_present"] = per_dataset_df["all_seeds_present"].fillna(False).astype(bool)
    per_dataset_df["certified"] = (
        per_dataset_df["all_metrics_passed"].astype(bool)
        & per_dataset_df["all_reload_passed"].astype(bool)
        & per_dataset_df["all_seeds_present"].astype(bool)
    )
    per_dataset_path = output_root / "certification_per_dataset.csv"
    per_dataset_df.to_csv(per_dataset_path, index=False)

    promotion_candidates = []
    if not winners_df.empty:
        merged_candidates = winners_df.merge(
            per_dataset_df[["dataset_name", "certified"]],
            on="dataset_name",
            how="left",
        )
        promotion_candidates = merged_candidates.sort_values("dataset_name").to_dict(orient="records")
    promotion_candidates_path = output_root / "promotion_candidates.json"
    promotion_candidates_path.write_text(json.dumps(promotion_candidates, indent=2), encoding="utf-8")

    overall = dict(certification_summary.overall)
    overall["all_reload_passed"] = bool(per_dataset_df["all_reload_passed"].all()) if len(per_dataset_df) else False
    overall["all_seeds_present"] = bool(per_dataset_df["all_seeds_present"].all()) if len(per_dataset_df) else False
    overall["all_datasets_certified"] = bool(per_dataset_df["certified"].all()) if len(per_dataset_df) else False
    overall["certified_dataset_count"] = int(per_dataset_df["certified"].sum()) if len(per_dataset_df) else 0

    summary_payload = {
        "output_root": str(output_root),
        "family_metrics_path": str(family_metrics_path),
        "paired_metrics_path": str(paired_metrics_path),
        "certification_per_metric_path": str(certification_per_metric_path),
        "certification_per_dataset_path": str(per_dataset_path),
        "promotion_candidates_path": str(promotion_candidates_path),
        "overall": overall,
    }
    summary_path = output_root / "certification_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report["status"] = "passed" if overall["all_datasets_certified"] else "failed"
    report["runs"] = run_rows
    report["summary_path"] = str(summary_path)
    _write_json(matrix_report_path, report)

    print(f"Certification summary: {summary_path}")
    print(f"Per-dataset verdicts: {per_dataset_path}")
    return 0 if overall["all_datasets_certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
