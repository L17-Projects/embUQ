#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from meso_uq.surrogate.group_holdout import (
    build_curve_split_manifest,
    build_holdout_outputs,
    find_representative_curve,
    predict_family_mean_std,
    read_indentation_table,
    resolve_modality_spec,
    resolve_surrogate_family,
    split_curves,
)


def _best_artifact_suffix(family: str) -> str:
    return ".pkl" if family == "dnn" else ".pt"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run grouped holdout evaluation for indentation surrogates."
    )
    ap.add_argument("--diameter", required=True, help="Diameter label (example: 3.2)")
    ap.add_argument("--data-file", required=True, help="Path to indentation training data table.")
    ap.add_argument("--model", required=True, help="Path to trained surrogate artifact.")
    ap.add_argument("--surrogate-family", default="dnn", choices=["dnn", "bnn"])
    ap.add_argument("--output-dir", required=True, help="Directory for grouped holdout outputs.")
    ap.add_argument("--seed", type=int, default=20260317)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument(
        "--disp-source",
        type=str,
        default="auto",
        choices=["auto", "diameter", "displacement"],
    )
    ap.add_argument(
        "--rupture-ratio",
        type=float,
        default=2.0,
        help="Drop curves with excessive consecutive displacement jumps; <=0 disables the filter.",
    )
    ap.add_argument("--predictive-mc-samples", type=int, default=64)
    ap.add_argument("--predictive-mc-chunk-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    family = resolve_surrogate_family(args.surrogate_family)
    spec = resolve_modality_spec("indentation")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rupture_ratio = None if args.rupture_ratio <= 0 else float(args.rupture_ratio)
    df = read_indentation_table(
        args.data_file,
        disp_source=args.disp_source,
        rupture_ratio_threshold=rupture_ratio,
    )
    df = split_curves(df, val_fraction=float(args.val_fraction), seed=int(args.seed))
    df_val = df[df["split"] == "validation"].copy().reset_index(drop=True)

    if len(df_val) == 0:
        raise RuntimeError("Validation split is empty; cannot compute holdout metrics.")

    X_val = df_val[list(spec.input_cols)].to_numpy(float)
    pred_mean, pred_std = predict_family_mean_std(
        family=family,
        model_path=args.model,
        X_raw=X_val,
        predictive_mc_samples=int(args.predictive_mc_samples),
        predictive_mc_chunk_size=int(args.predictive_mc_chunk_size),
        device=args.device,
    )
    metrics_df, preds_df = build_holdout_outputs(
        df_val,
        axis_col=spec.axis_col,
        target_col=spec.target_col,
        truth_col=spec.truth_col,
        pred_col=spec.pred_col,
        pred_mean=pred_mean,
        pred_std=pred_std,
    )

    split_manifest = build_curve_split_manifest(
        df, seed=int(args.seed), val_fraction=float(args.val_fraction)
    )
    split_manifest.to_csv(output_dir / "curve_split.csv", index=False)
    metrics_path = output_dir / "per_curve_metrics.csv"
    preds_path = output_dir / "per_curve_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    rep_curve_id = find_representative_curve(metrics_df)
    rep_curve = preds_df[preds_df["curve_id"] == rep_curve_id][
        [spec.axis_col, spec.truth_col, spec.pred_col, "pred_std"]
    ]
    rep_curve.to_csv(output_dir / "representative_curve.csv", index=False)
    rep_row = metrics_df[metrics_df["curve_id"] == rep_curve_id].iloc[0]

    candidate_name = Path(args.model).stem
    result_row = {
        "name": candidate_name,
        "surrogate_family": family,
        "modality": spec.modality,
        "diameter_um": args.diameter,
        "mean_curve_rmse": float(metrics_df["rmse"].mean()),
        "median_curve_rmse": float(metrics_df["rmse"].median()),
        "mean_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].mean()),
        "median_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].median()),
        "max_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].max()),
        "mean_curve_pred_std": float(metrics_df["pred_std_mean"].mean()),
        "model_path": str(Path(args.model).resolve()),
        "metrics_path": str(metrics_path),
        "predictions_path": str(preds_path),
    }
    pd.DataFrame([result_row]).to_csv(output_dir / "training_results_group_holdout.csv", index=False)

    best_dest = (
        output_dir
        / f"{spec.best_artifact_prefix}_GROUP_HOLDOUT_BEST{_best_artifact_suffix(family)}"
    )
    shutil.copy(Path(args.model), best_dest)

    summary = {
        "modality": spec.modality,
        "surrogate_family": family,
        "diameter_um": args.diameter,
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "n_curves_total": int(df["curve_id"].nunique()),
        "n_curves_train": int(df[df["split"] == "train"]["curve_id"].nunique()),
        "n_curves_validation": int(df["curve_id"].nunique() - df[df["split"] == "train"]["curve_id"].nunique()),
        "best_architecture": candidate_name,
        "best_mean_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].mean()),
        "best_median_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].median()),
        "best_max_curve_rel_l2_pct": float(metrics_df["rel_l2_pct"].max()),
        "representative_curve_id": int(rep_curve_id),
        "representative_curve_rel_l2_pct": float(rep_row["rel_l2_pct"]),
        "best_model_path": str(best_dest),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved grouped holdout outputs -> {output_dir}")
    print(
        "mean rel_l2_pct={:.3f}, median rel_l2_pct={:.3f}, max rel_l2_pct={:.3f}".format(
            summary["best_mean_curve_rel_l2_pct"],
            summary["best_median_curve_rel_l2_pct"],
            summary["best_max_curve_rel_l2_pct"],
        )
    )


if __name__ == "__main__":
    main()

