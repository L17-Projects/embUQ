#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.hpc_paths import default_runs_root, detect_hpc_site  # noqa: E402
from meso_uq.surrogate.bnn import VariationalBNNPredictor  # noqa: E402
from meso_uq.surrogate.bnn_training import _split_like_dnn, train_tabular_bnn_surrogate  # noqa: E402
from meso_uq.surrogate.cli import make_tensors, read_wide_curve_table  # noqa: E402
from meso_uq.surrogate.group_holdout import read_indentation_table  # noqa: E402

_SELECTIONS: dict[str, dict[str, str]] = {
    "compression_2.1um": {
        "modality": "compression",
        "diameter_um": "2.1",
        "data": "compression/surrogate/diameters/2.1um/data/F_Delta.dat",
        "dnn": "compression/surrogate/diameters/2.1um/trained/microbubble_force_BEST.pkl",
        "out_stem": "microbubble_force",
    },
    "compression_2.9um": {
        "modality": "compression",
        "diameter_um": "2.9",
        "data": "compression/surrogate/diameters/2.9um/data/F_Delta.dat",
        "dnn": "compression/surrogate/diameters/2.9um/trained/microbubble_force_BEST.pkl",
        "out_stem": "microbubble_force",
    },
    "compression_3.0um": {
        "modality": "compression",
        "diameter_um": "3.0",
        "data": "compression/surrogate/diameters/3.0um/data/F_Delta.dat",
        "dnn": "compression/surrogate/diameters/3.0um/trained/microbubble_force_BEST.pkl",
        "out_stem": "microbubble_force",
    },
    "indentation_3.2um": {
        "modality": "indentation",
        "diameter_um": "3.2",
        "data": "indentation/surrogate/diameters/3.2um/data/samples_all.dat",
        "dnn": "indentation/surrogate/diameters/3.2um/trained/microbubble_displacement_BEST.pkl",
        "out_stem": "microbubble_displacement",
    },
    "indentation_3.4um": {
        "modality": "indentation",
        "diameter_um": "3.4",
        "data": "indentation/surrogate/diameters/3.4um/data/samples_all.dat",
        "dnn": "indentation/surrogate/diameters/3.4um/trained/microbubble_displacement_BEST.pkl",
        "out_stem": "microbubble_displacement",
    },
    "indentation_5.8um": {
        "modality": "indentation",
        "diameter_um": "5.8",
        "data": "indentation/surrogate/diameters/5.8um/data/samples_all.dat",
        "dnn": "indentation/surrogate/diameters/5.8um/trained/microbubble_displacement_BEST.pkl",
        "out_stem": "microbubble_displacement",
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_selection_spec(selection: str) -> dict[str, str]:
    if selection not in _SELECTIONS:
        allowed = ", ".join(sorted(_SELECTIONS.keys()))
        raise ValueError(f"Unknown --selection={selection!r}. Available: {allowed}")
    spec = dict(_SELECTIONS[selection])
    spec["name"] = selection
    spec["data"] = str((REPO_ROOT / spec["data"]).resolve())
    spec["dnn"] = str((REPO_ROOT / spec["dnn"]).resolve())
    return spec


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)))))


def _compute_degradation(inprocess_rmse: float, reload_rmse: float) -> tuple[float, float]:
    degradation_abs = float(reload_rmse - inprocess_rmse)
    if inprocess_rmse <= 0:
        return degradation_abs, float("inf")
    return degradation_abs, float(degradation_abs / inprocess_rmse)


def _load_training_dataframe(spec: dict[str, str], *, disp_source: str, rupture_ratio: float) -> Any:
    if spec["modality"] == "compression":
        return read_wide_curve_table(spec["data"], curve_axis_name="disp", value_name="F")
    return read_indentation_table(
        spec["data"],
        disp_source=disp_source,
        rupture_ratio_threshold=rupture_ratio,
    )


def _split_validation_phys(
    df: Any,
    *,
    input_cols: list[str],
    target_col: str,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    Xz, yz, _, _, _, _ = make_tensors(df, list(input_cols), target_col)
    X_phys = df[list(input_cols)].to_numpy(float)
    y_phys = df[[target_col]].to_numpy(float).reshape(-1)
    split = _split_like_dnn(Xz, yz, X_phys, y_phys, seed=seed)
    return np.asarray(split["X_val_phys"], dtype=np.float32), np.asarray(split["y_val_phys"], dtype=np.float64)


def _collect_env() -> dict[str, Any]:
    import platform
    import torch

    payload: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        payload["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    return payload


def run_roundtrip_check(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    spec = _resolve_selection_spec(args.selection)
    df = _load_training_dataframe(
        spec,
        disp_source=args.disp_source,
        rupture_ratio=args.rupture_ratio,
    )

    if spec["modality"] == "compression":
        input_cols = ["Yt", "kb", "b1", "b2", "a3", "a4", "disp"]
        target_col = "F"
    else:
        input_cols = ["Yt", "kb", "b1", "b2", "a3", "a4", "F"]
        target_col = "disp"

    artifact_path = run_dir / f"{spec['out_stem']}_BNN_roundtrip_strict.pt"
    train_report_path = run_dir / "roundtrip_train_report.json"
    train_result = train_tabular_bnn_surrogate(
        df,
        input_cols=input_cols,
        target_col=target_col,
        out_path=str(artifact_path),
        dnn_reference_path=spec["dnn"],
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
        require_parity=True,
        device=args.device,
        report_path=str(train_report_path),
    )

    inprocess_rmse = float(train_result["training"]["final_val_rmse"])
    X_val_phys, y_val_phys = _split_validation_phys(
        df,
        input_cols=input_cols,
        target_col=target_col,
        seed=args.seed,
    )
    predictor = VariationalBNNPredictor(str(artifact_path), device=args.device)
    mean, std = predictor.predict_mean_std(
        X_val_phys,
        predictive_mc_samples=args.predictive_mc_samples,
        predictive_mc_chunk_size=args.predictive_mc_chunk_size,
    )
    mean = np.maximum(0.0, np.asarray(mean, dtype=np.float64))
    reload_rmse = _compute_rmse(y_val_phys, mean)
    degradation_abs, degradation_rel = _compute_degradation(inprocess_rmse, reload_rmse)

    status = "passed" if (math.isfinite(degradation_rel) and degradation_rel <= args.reload_tol_rel) else "failed"
    if not math.isfinite(degradation_rel):
        status = "failed"

    return {
        "status": status,
        "selection": args.selection,
        "inprocess_val_rmse": inprocess_rmse,
        "reload_val_rmse": reload_rmse,
        "degradation_abs": degradation_abs,
        "degradation_rel": degradation_rel,
        "reload_tol_rel": float(args.reload_tol_rel),
        "validation_pred_std_mean_inprocess": float(train_result["validation"]["val_predictive_std_mean"]),
        "validation_pred_std_mean_reload": float(np.mean(np.asarray(std, dtype=np.float64))),
        "artifact_path": str(artifact_path),
        "training_report_path": str(train_report_path),
        "run_dir": str(run_dir),
        "config": {
            "seed": args.seed,
            "device": args.device,
            "predictive_mc_samples": args.predictive_mc_samples,
            "predictive_mc_chunk_size": args.predictive_mc_chunk_size,
            "parity_tol": args.parity_tol,
        },
        "env": _collect_env(),
        "updated_at": _now_iso(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic BNN train-vs-reload round-trip checker (site-agnostic runner)."
    )
    parser.add_argument("--selection", default="indentation_3.2um")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reload-tol-rel", type=float, default=0.05)

    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--obs-noise-prior-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--predictive-mc-chunk-size", type=int, default=8)
    parser.add_argument("--max-walltime-seconds", type=int, default=1200)
    parser.add_argument("--parity-tol", type=float, default=1.2)

    parser.add_argument("--disp-source", choices=["auto", "diameter", "displacement"], default="auto")
    parser.add_argument("--rupture-ratio", type=float, default=2.0)
    args = parser.parse_args(argv)

    resolved_site = args.site if args.site is not None else detect_hpc_site()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else default_runs_root(REPO_ROOT, "bnn_roundtrip", site=resolved_site, run_tag=args.run_tag)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / args.selection
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "roundtrip_strict_result.json"

    try:
        result = run_roundtrip_check(args, run_dir)
    except Exception as exc:  # pragma: no cover - error path exercised in integration usage
        result = {
            "status": "failed",
            "selection": args.selection,
            "error": f"{type(exc).__name__}: {exc}",
            "run_dir": str(run_dir),
            "updated_at": _now_iso(),
        }

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote round-trip result: {result_path}")
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
