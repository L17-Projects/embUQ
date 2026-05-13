from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meso_uq.core import AgentFamily, Modality, ModelBackend, coerce_modality


PARAMETER_COLUMNS: tuple[str, ...] = ("Yt", "kb", "b1", "b2", "a3", "a4")
SUPPORTED_EMB_SURROGATE_BACKENDS: tuple[ModelBackend, ...] = (
    ModelBackend.DNN,
    ModelBackend.BNN,
    ModelBackend.PYRO_BNN,
)


@dataclass(frozen=True)
class EmbSurrogateWorkflowSpec:
    modality: Modality
    dnn_description: str
    bnn_description: str
    group_holdout_description: str
    default_dnn_out: str
    default_bnn_out: str
    default_bnn_reference: str
    input_cols: tuple[str, ...]
    target_col: str
    loader_kind: str
    axis_col: str
    value_col: str
    best_artifact_prefix: str
    diameter_help: str
    has_indentation_loader_knobs: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", coerce_modality(self.modality))

    @property
    def family(self) -> AgentFamily:
        return AgentFamily.EMB

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "agent_family": self.family.value,
            "modality": self.modality.value,
            "supported_backends": [backend.value for backend in SUPPORTED_EMB_SURROGATE_BACKENDS],
            "default_dnn_out": self.default_dnn_out,
            "default_bnn_out": self.default_bnn_out,
            "default_bnn_reference": self.default_bnn_reference,
            "input_cols": list(self.input_cols),
            "target_col": self.target_col,
            "best_artifact_prefix": self.best_artifact_prefix,
        }


@dataclass(frozen=True)
class DatasetSplitMetadata:
    seed: int
    val_fraction: float

    def __post_init__(self) -> None:
        if self.val_fraction <= 0.0 or self.val_fraction >= 1.0:
            raise ValueError("val_fraction must be in (0, 1).")

    def as_dict(self) -> dict[str, int | float]:
        return {"seed": int(self.seed), "val_fraction": float(self.val_fraction)}


_WORKFLOW_SPECS: dict[Modality, EmbSurrogateWorkflowSpec] = {
    Modality.COMPRESSION: EmbSurrogateWorkflowSpec(
        modality=Modality.COMPRESSION,
        dnn_description="Train compression surrogate: force = f(Yt, kb, b1, b2, a3, a4, disp)",
        bnn_description="Train compression variational BNN surrogate: force = f(Yt, kb, b1, b2, a3, a4, disp)",
        group_holdout_description="Run grouped holdout evaluation for compression surrogates.",
        default_dnn_out="trained/microbubble_force_BEST.pkl",
        default_bnn_out="trained/microbubble_force_BNN.pt",
        default_bnn_reference="trained/microbubble_force_BEST.pkl",
        input_cols=(*PARAMETER_COLUMNS, "disp"),
        target_col="F",
        loader_kind="compression",
        axis_col="disp",
        value_col="F",
        best_artifact_prefix="microbubble_force",
        diameter_help="Diameter label (example: 2.9)",
    ),
    Modality.INDENTATION: EmbSurrogateWorkflowSpec(
        modality=Modality.INDENTATION,
        dnn_description="Train indentation surrogate: disp = f(Yt, kb, b1, b2, a3, a4, F)",
        bnn_description="Train indentation variational BNN surrogate: disp = f(Yt, kb, b1, b2, a3, a4, F)",
        group_holdout_description="Run grouped holdout evaluation for indentation surrogates.",
        default_dnn_out="trained/microbubble_disp_BEST.pkl",
        default_bnn_out="trained/microbubble_displacement_BNN.pt",
        default_bnn_reference="trained/microbubble_displacement_BEST.pkl",
        input_cols=(*PARAMETER_COLUMNS, "F"),
        target_col="disp",
        loader_kind="indentation",
        axis_col="F",
        value_col="disp",
        best_artifact_prefix="microbubble_displacement",
        diameter_help="Diameter label (example: 3.2)",
        has_indentation_loader_knobs=True,
    ),
}


def list_emb_surrogate_workflows() -> tuple[EmbSurrogateWorkflowSpec, ...]:
    return tuple(_WORKFLOW_SPECS[modality] for modality in (Modality.COMPRESSION, Modality.INDENTATION))


def get_emb_surrogate_workflow(modality: Modality | str) -> EmbSurrogateWorkflowSpec:
    selected = coerce_modality(modality)
    try:
        return _WORKFLOW_SPECS[selected]
    except KeyError as exc:
        supported = ", ".join(spec.modality.value for spec in list_emb_surrogate_workflows())
        raise ValueError(f"EMB surrogate workflow for modality '{selected.value}' is not registered. Expected one of: {supported}.") from exc


def resolve_emb_surrogate_backend(backend: ModelBackend | str) -> ModelBackend:
    selected = ModelBackend(backend)
    if selected not in SUPPORTED_EMB_SURROGATE_BACKENDS:
        supported = ", ".join(item.value for item in SUPPORTED_EMB_SURROGATE_BACKENDS)
        raise ValueError(f"Unsupported EMB surrogate backend '{selected.value}'. Expected one of: {supported}.")
    return selected


def _add_common_dnn_args(parser: argparse.ArgumentParser, spec: EmbSurrogateWorkflowSpec) -> None:
    parser.add_argument("data", help="Path to whitespace training table")
    parser.add_argument("--out", default=spec.default_dnn_out)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)


def _add_common_bnn_args(parser: argparse.ArgumentParser, spec: EmbSurrogateWorkflowSpec) -> None:
    parser.add_argument("data", help="Path to whitespace training table")
    parser.add_argument("--out", default=spec.default_bnn_out)
    parser.add_argument("--dnn-reference", default=spec.default_bnn_reference)
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
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--max-walltime-seconds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--parity-tol", type=float, default=1.20)
    parser.add_argument("--no-require-parity", action="store_true", default=False)
    parser.add_argument("--device", default="cpu")


def _add_indentation_loader_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--disp-source", choices=["auto", "diameter", "displacement"], default="auto")
    parser.add_argument(
        "--rupture-ratio",
        type=float,
        default=2.0,
        help="Drop curves with excessive consecutive displacement jumps; <=0 disables the filter.",
    )


def _rupture_ratio_threshold(value: float) -> float | None:
    return None if value <= 0 else float(value)


def _read_training_dataframe(
    spec: EmbSurrogateWorkflowSpec,
    path: str | Path,
    *,
    disp_source: str = "auto",
    rupture_ratio: float = 2.0,
):
    if spec.modality is Modality.COMPRESSION:
        from meso_uq.surrogate.cli import read_compression_training_table

        return read_compression_training_table(str(path), curve_axis_name=spec.axis_col, value_name=spec.target_col)
    from meso_uq.surrogate.cli import read_indentation_table

    return read_indentation_table(
        str(path),
        disp_source=disp_source,
        rupture_ratio_threshold=_rupture_ratio_threshold(rupture_ratio),
    )


def run_emb_dnn_training_cli(
    modality: Modality | str,
    *,
    argv: Sequence[str] | None = None,
    reader: Callable[..., Any] | None = None,
    trainer: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec = get_emb_surrogate_workflow(modality)
    parser = argparse.ArgumentParser(description=spec.dnn_description)
    _add_common_dnn_args(parser, spec)
    if spec.has_indentation_loader_knobs:
        _add_indentation_loader_args(parser)
    args = parser.parse_args(argv)

    if reader is None:
        df = _read_training_dataframe(
            spec,
            args.data,
            disp_source=getattr(args, "disp_source", "auto"),
            rupture_ratio=getattr(args, "rupture_ratio", 2.0),
        )
    elif spec.modality is Modality.COMPRESSION:
        df = reader(args.data, curve_axis_name=spec.axis_col, value_name=spec.target_col)
    else:
        df = reader(
            args.data,
            disp_source=args.disp_source,
            rupture_ratio_threshold=_rupture_ratio_threshold(args.rupture_ratio),
        )

    if trainer is None:
        from meso_uq.surrogate.cli import train_tabular_surrogate

        trainer = train_tabular_surrogate
    result = trainer(
        df,
        input_cols=list(spec.input_cols),
        target_col=spec.target_col,
        out_path=args.out,
        width=args.width,
        depth=args.depth,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epoch=args.max_epoch,
        seed=args.seed,
        report_path=args.report_path,
    )
    print(f"Saved -> {result['out']}. Final train={result['train_loss']:.3e}, valid={result['val_loss']:.3e}")
    return result


def run_emb_bnn_training_cli(
    modality: Modality | str,
    *,
    argv: Sequence[str] | None = None,
    reader: Callable[..., Any] | None = None,
    trainer: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec = get_emb_surrogate_workflow(modality)
    parser = argparse.ArgumentParser(description=spec.bnn_description)
    _add_common_bnn_args(parser, spec)
    if spec.has_indentation_loader_knobs:
        _add_indentation_loader_args(parser)
    args = parser.parse_args(argv)

    if reader is None:
        df = _read_training_dataframe(
            spec,
            Path(args.data).resolve(),
            disp_source=getattr(args, "disp_source", "auto"),
            rupture_ratio=getattr(args, "rupture_ratio", 2.0),
        )
    elif spec.modality is Modality.COMPRESSION:
        df = reader(str(Path(args.data).resolve()), curve_axis_name=spec.axis_col, value_name=spec.target_col)
    else:
        df = reader(
            str(Path(args.data).resolve()),
            disp_source=args.disp_source,
            rupture_ratio_threshold=_rupture_ratio_threshold(args.rupture_ratio),
        )

    if trainer is None:
        from meso_uq.surrogate.bnn_training import train_tabular_bnn_surrogate

        trainer = train_tabular_bnn_surrogate
    out_path = Path(args.out).resolve()
    dnn_ref = Path(args.dnn_reference).resolve()
    report_path = Path(args.report_path).resolve() if args.report_path else None
    result = trainer(
        df,
        input_cols=list(spec.input_cols),
        target_col=spec.target_col,
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
        max_epochs=args.max_epochs,
        eval_every=args.eval_every,
        predictive_mc_samples=args.predictive_mc_samples,
        max_walltime_seconds=args.max_walltime_seconds,
        seed=args.seed,
        parity_tol=args.parity_tol,
        require_parity=not args.no_require_parity,
        device=args.device,
    )
    training = result["training"]
    print(
        "Saved -> "
        f"{result['out']} | val_rmse={training['final_val_rmse']:.4e} | "
        f"dnn_rmse={training['parity_dnn_rmse']:.4e} | ratio={training['parity_ratio']:.4f} | "
        f"steps={training['step_count']} | reason={training['stop_reason']}"
    )
    return result


def best_artifact_suffix(family: str) -> str:
    return ".pkl" if family == "dnn" else ".pt"


def run_emb_group_holdout_cli(
    modality: Modality | str,
    *,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    import pandas as pd

    from meso_uq.surrogate.group_holdout import (
        build_curve_split_manifest,
        build_holdout_outputs,
        find_representative_curve,
        predict_family_mean_std,
        read_compression_table,
        resolve_modality_spec,
        resolve_surrogate_family,
        split_curves,
    )

    spec = get_emb_surrogate_workflow(modality)
    parser = argparse.ArgumentParser(description=spec.group_holdout_description)
    parser.add_argument("--diameter", required=True, help=spec.diameter_help)
    parser.add_argument("--data-file", required=True, help=f"Path to {spec.modality.value} training data table.")
    parser.add_argument("--model", required=True, help="Path to trained surrogate artifact.")
    parser.add_argument("--surrogate-family", default="dnn", choices=["dnn", "bnn"])
    parser.add_argument("--output-dir", required=True, help="Directory for grouped holdout outputs.")
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    if spec.has_indentation_loader_knobs:
        _add_indentation_loader_args(parser)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--predictive-mc-chunk-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    split_metadata = DatasetSplitMetadata(seed=int(args.seed), val_fraction=float(args.val_fraction))
    family = resolve_surrogate_family(args.surrogate_family)
    group_spec = resolve_modality_spec(spec.modality.value)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if spec.modality is Modality.COMPRESSION:
        df = read_compression_table(args.data_file)
    else:
        from meso_uq.surrogate.cli import read_indentation_table

        df = read_indentation_table(
            args.data_file,
            disp_source=args.disp_source,
            rupture_ratio_threshold=_rupture_ratio_threshold(args.rupture_ratio),
        )
    df = split_curves(df, val_fraction=split_metadata.val_fraction, seed=split_metadata.seed)
    df_val = df[df["split"] == "validation"].copy().reset_index(drop=True)

    if len(df_val) == 0:
        raise RuntimeError("Validation split is empty; cannot compute holdout metrics.")

    x_val = df_val[list(group_spec.input_cols)].to_numpy(float)
    pred_mean, pred_std = predict_family_mean_std(
        family=family,
        model_path=args.model,
        X_raw=x_val,
        predictive_mc_samples=int(args.predictive_mc_samples),
        predictive_mc_chunk_size=int(args.predictive_mc_chunk_size),
        device=args.device,
    )
    metrics_df, preds_df = build_holdout_outputs(
        df_val,
        axis_col=group_spec.axis_col,
        target_col=group_spec.target_col,
        truth_col=group_spec.truth_col,
        pred_col=group_spec.pred_col,
        pred_mean=pred_mean,
        pred_std=pred_std,
    )

    split_manifest = build_curve_split_manifest(
        df, seed=split_metadata.seed, val_fraction=split_metadata.val_fraction
    )
    split_manifest.to_csv(output_dir / "curve_split.csv", index=False)
    metrics_path = output_dir / "per_curve_metrics.csv"
    preds_path = output_dir / "per_curve_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    rep_curve_id = find_representative_curve(metrics_df)
    rep_curve = preds_df[preds_df["curve_id"] == rep_curve_id][
        [group_spec.axis_col, group_spec.truth_col, group_spec.pred_col, "pred_std"]
    ]
    rep_curve.to_csv(output_dir / "representative_curve.csv", index=False)
    rep_row = metrics_df[metrics_df["curve_id"] == rep_curve_id].iloc[0]

    candidate_name = Path(args.model).stem
    result_row = {
        "name": candidate_name,
        "surrogate_family": family,
        "modality": group_spec.modality,
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

    best_dest = output_dir / f"{group_spec.best_artifact_prefix}_GROUP_HOLDOUT_BEST{best_artifact_suffix(family)}"
    shutil.copy(Path(args.model), best_dest)

    summary = {
        "modality": group_spec.modality,
        "surrogate_family": family,
        "diameter_um": args.diameter,
        **split_metadata.as_dict(),
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
        "checkpoint_metadata": spec.checkpoint_metadata,
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
    return summary


__all__ = [
    "DatasetSplitMetadata",
    "EmbSurrogateWorkflowSpec",
    "PARAMETER_COLUMNS",
    "SUPPORTED_EMB_SURROGATE_BACKENDS",
    "best_artifact_suffix",
    "get_emb_surrogate_workflow",
    "list_emb_surrogate_workflows",
    "resolve_emb_surrogate_backend",
    "run_emb_bnn_training_cli",
    "run_emb_dnn_training_cli",
    "run_emb_group_holdout_cli",
]
