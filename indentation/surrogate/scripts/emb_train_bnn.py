#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from meso_uq.surrogate.bnn_training import train_tabular_bnn_surrogate
from meso_uq.surrogate.group_holdout import read_indentation_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train indentation variational BNN surrogate: disp = f(Yt, kb, b1, b2, a3, a4, F)"
    )
    parser.add_argument("data", help="Path to whitespace training table")
    parser.add_argument("--out", default="trained/microbubble_displacement_BNN.pt")
    parser.add_argument("--dnn-reference", default="trained/microbubble_displacement_BEST.pkl")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument(
        "--obs-noise-prior-scale",
        "--obs-noise",
        dest="obs_noise_prior_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--max-walltime-seconds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--parity-tol", type=float, default=1.20)
    parser.add_argument("--no-require-parity", action="store_true", default=False)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    out_path = Path(args.out).resolve()
    dnn_ref = Path(args.dnn_reference).resolve()
    report_path = Path(args.report_path).resolve() if args.report_path else None
    df = read_indentation_table(str(data_path), disp_source="auto", rupture_ratio_threshold=2.0)
    result = train_tabular_bnn_surrogate(
        df,
        input_cols=["Yt", "kb", "b1", "b2", "a3", "a4", "F"],
        target_col="disp",
        out_path=str(out_path),
        dnn_reference_path=str(dnn_ref),
        report_path=str(report_path) if report_path else None,
        width=args.width,
        depth=args.depth,
        prior_scale=args.prior_scale,
        obs_noise_prior_scale=args.obs_noise_prior_scale,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        predictive_mc_samples=args.predictive_mc_samples,
        max_walltime_seconds=args.max_walltime_seconds,
        seed=args.seed,
        parity_tol=args.parity_tol,
        require_parity=not args.no_require_parity,
        device=args.device,
    )
    t = result["training"]
    print(
        "Saved -> "
        f"{result['out']} | val_rmse={t['final_val_rmse']:.4e} | "
        f"dnn_rmse={t['parity_dnn_rmse']:.4e} | ratio={t['parity_ratio']:.4f} | "
        f"steps={t['step_count']} | reason={t['stop_reason']}"
    )


if __name__ == "__main__":
    main()
