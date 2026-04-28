#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.hpc_paths import default_runs_root, detect_hpc_site  # noqa: E402
from meso_uq.surrogate.emb_catalog import (  # noqa: E402
    DEFAULT_BNN_ARCHITECTURES,
    resolve_emb_dataset_specs,
)

DEFAULT_SEEDS = (20260317, 20260318, 20260319, 20260320, 20260321)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_architecture_text() -> str:
    return ",".join(arch.name for arch in DEFAULT_BNN_ARCHITECTURES)


def _architecture_lookup() -> dict[str, tuple[int, int, str]]:
    return {arch.name: (arch.width, arch.depth, arch.name) for arch in DEFAULT_BNN_ARCHITECTURES}


def _parse_architecture_names(text: str) -> list[tuple[int, int, str]]:
    by_name = _architecture_lookup()
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("Architecture list must be non-empty.")
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown BNN architecture name(s): {unknown}")
    return [by_name[name] for name in names]


def _parse_float_list(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Float grid must be non-empty.")
    return [float(item) for item in values]


def _resolve_stage1_grid(
    *,
    grid_text: str | None,
    scalar_value: float | None,
    default_text: str,
    option_name: str,
    legacy_option_name: str,
) -> list[float]:
    if grid_text is not None and scalar_value is not None:
        raise ValueError(f"Use either {option_name} or {legacy_option_name}, not both.")
    if grid_text is not None:
        return _parse_float_list(grid_text)
    if scalar_value is not None:
        return [float(scalar_value)]
    return _parse_float_list(default_text)


def _resolve_seeds(values: list[int]) -> list[int]:
    if values:
        return [int(value) for value in values]
    return [int(seed) for seed in DEFAULT_SEEDS]


def _candidate_key(
    *,
    stage: str,
    arch_name: str,
    prior_scale: float,
    obs_noise_prior_scale: float,
    lr: float,
) -> str:
    return (
        f"{stage}__{arch_name}"
        f"__prior{prior_scale:g}"
        f"__obs{obs_noise_prior_scale:g}"
        f"__lr{lr:g}"
    )


def _candidate_paths(seed_root: Path, key: str) -> tuple[Path, Path]:
    artifact_path = seed_root / "candidates" / f"{key}.pt"
    report_path = seed_root / "reports" / f"{key}.json"
    return artifact_path, report_path


def _selection_path(root: Path, spec_name: str, seed: int) -> Path:
    return root / spec_name / f"seed_{seed}" / "selection.json"


def _selection_string_value(payload: dict[str, object], key: str, *, selection_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {key!r} in DNN selection file: {selection_path}")
    return value.strip()


def _resolve_selection_path_value(raw_path: str, *, selection_path: Path, label: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (selection_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Selected {label} does not exist: {candidate}")
    return candidate


def _load_dnn_selection(*, dnn_root: Path, spec_name: str, seed: int) -> tuple[Path, dict[str, object]]:
    selection_path = _selection_path(dnn_root, spec_name, seed)
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing DNN selection file: {selection_path}")
    return selection_path, _read_json(selection_path)


def _resolve_dnn_reference_from_selection(
    payload: dict[str, object], *, selection_path: Path
) -> Path:
    artifact_value = _selection_string_value(payload, "best_artifact_path", selection_path=selection_path)
    return _resolve_selection_path_value(artifact_value, selection_path=selection_path, label="DNN artifact")


def _resolve_architecture_from_selection(
    payload: dict[str, object], *, selection_path: Path
) -> tuple[int, int, str]:
    arch_name = _selection_string_value(payload, "best_architecture", selection_path=selection_path)
    architecture = _architecture_lookup().get(arch_name)
    if architecture is None:
        raise ValueError(f"Unknown BNN architecture {arch_name!r} in DNN selection file: {selection_path}")
    return architecture


def _candidate_completed(report_path: Path, artifact_path: Path) -> bool:
    if not report_path.exists() or not artifact_path.exists():
        return False
    try:
        payload = _read_json(report_path)
    except Exception:
        return False
    training = payload.get("training")
    return isinstance(training, dict) and isinstance(training.get("best_val_rmse"), (float, int))


def _build_command(
    *,
    python_bin: str,
    spec: dict[str, str],
    seed_root: Path,
    dnn_reference_path: str | None,
    stage: str,
    arch_name: str,
    width: int,
    depth: int,
    prior_scale: float,
    obs_noise_prior_scale: float,
    batch_size: int,
    lr: float,
    max_steps: int,
    max_epochs: int | None = None,
    eval_every: int,
    predictive_mc_samples: int,
    max_walltime_seconds: int,
    seed: int,
    parity_tol: float,
    require_parity: bool,
    device: str,
) -> tuple[list[str], Path, Path]:
    key = _candidate_key(
        stage=stage,
        arch_name=arch_name,
        prior_scale=prior_scale,
        obs_noise_prior_scale=obs_noise_prior_scale,
        lr=lr,
    )
    artifact_path, report_path = _candidate_paths(seed_root, key)
    command = [
        python_bin,
        spec["bnn_train_script"],
        spec["data"],
        "--out",
        str(artifact_path),
        "--dnn-reference",
        dnn_reference_path if dnn_reference_path is not None else spec["dnn_artifact"],
        "--report-path",
        str(report_path),
        "--width",
        str(width),
        "--depth",
        str(depth),
        "--prior-scale",
        str(prior_scale),
        "--obs-noise-prior-scale",
        str(obs_noise_prior_scale),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--eval-every",
        str(eval_every),
        "--predictive-mc-samples",
        str(predictive_mc_samples),
        "--max-walltime-seconds",
        str(max_walltime_seconds),
        "--seed",
        str(seed),
        "--parity-tol",
        str(parity_tol),
        "--device",
        str(device),
    ]
    if max_epochs is None:
        command.extend(["--max-steps", str(max_steps)])
    else:
        command.extend(["--max-epochs", str(int(max_epochs))])
    if not require_parity:
        command.append("--no-require-parity")
    return command, artifact_path, report_path


def _candidate_metric(report_path: Path) -> float:
    payload = _read_json(report_path)
    training = payload["training"]
    assert isinstance(training, dict)
    best_val_rmse = training.get("best_val_rmse", training.get("final_val_rmse"))
    return float(best_val_rmse)


def _iter_hyperparameter_grid(
    *,
    prior_scales: list[float],
    obs_noise_prior_scales: list[float],
    lrs: list[float],
) -> list[tuple[float, float, float]]:
    return [
        (float(prior_scale), float(obs_noise_prior_scale), float(lr))
        for prior_scale in prior_scales
        for obs_noise_prior_scale in obs_noise_prior_scales
        for lr in lrs
    ]


def _stage1_top_architectures(
    *,
    seed_root: Path,
    architectures: list[tuple[int, int, str]],
    top_k: int,
    prior_scales: list[float],
    obs_noise_prior_scales: list[float],
    lrs: list[float],
) -> list[tuple[int, int, str]]:
    rows: list[tuple[float, tuple[int, int, str]]] = []
    for width, depth, arch_name in architectures:
        best_metric = min(
            _candidate_metric(
                _candidate_paths(
                    seed_root,
                    _candidate_key(
                        stage="stage1",
                        arch_name=arch_name,
                        prior_scale=prior_scale,
                        obs_noise_prior_scale=obs_noise_prior_scale,
                        lr=lr,
                    ),
                )[1]
            )
            for prior_scale, obs_noise_prior_scale, lr in _iter_hyperparameter_grid(
                prior_scales=prior_scales,
                obs_noise_prior_scales=obs_noise_prior_scales,
                lrs=lrs,
            )
        )
        rows.append((best_metric, (width, depth, arch_name)))
    rows.sort(key=lambda item: (item[0], item[1][0], item[1][1], item[1][2]))
    return [arch for _, arch in rows[: max(1, min(top_k, len(rows)))]]


def _resolve_seed_inputs(
    *,
    spec: dict[str, str],
    seed: int,
    explicit_architectures: list[tuple[int, int, str]],
    architecture_source: str,
    dnn_root: Path | None,
) -> dict[str, object]:
    dnn_reference_path = Path(spec["dnn_artifact"]).resolve()
    dnn_selection_path: Path | None = None
    dnn_selected_architecture: str | None = None
    if dnn_root is not None:
        dnn_selection_path, selection_payload = _load_dnn_selection(
            dnn_root=dnn_root,
            spec_name=spec["name"],
            seed=seed,
        )
        dnn_reference_path = _resolve_dnn_reference_from_selection(
            selection_payload,
            selection_path=dnn_selection_path,
        )
        best_architecture = selection_payload.get("best_architecture")
        if isinstance(best_architecture, str) and best_architecture.strip():
            dnn_selected_architecture = best_architecture.strip()
        if architecture_source == "dnn-selection":
            architecture = _resolve_architecture_from_selection(
                selection_payload,
                selection_path=dnn_selection_path,
            )
            return {
                "architectures": [architecture],
                "dnn_reference_path": str(dnn_reference_path),
                "dnn_selection_path": str(dnn_selection_path),
                "dnn_selected_architecture": architecture[2],
            }
    elif architecture_source == "dnn-selection":
        raise ValueError("--architecture-source=dnn-selection requires --dnn-root.")

    return {
        "architectures": explicit_architectures,
        "dnn_reference_path": str(dnn_reference_path),
        "dnn_selection_path": str(dnn_selection_path) if dnn_selection_path is not None else None,
        "dnn_selected_architecture": dnn_selected_architecture,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a two-stage seeded BNN candidate sweep across all EMB surrogate datasets."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument(
        "--dnn-root",
        default=None,
        help="Optional output root from run_dnn_rebaseline_matrix.py to use seeded DNN selections as references.",
    )
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--architectures", default=None)
    parser.add_argument(
        "--architecture-source",
        choices=["explicit", "dnn-selection"],
        default="explicit",
        help="Choose BNN candidate architectures from an explicit list or from each seeded DNN rebaseline winner.",
    )
    parser.add_argument("--stage1-prior-scales", default=None)
    parser.add_argument("--stage1-obs-noise-prior-scales", default=None)
    parser.add_argument("--stage1-lrs", default=None)
    parser.add_argument("--stage1-prior-scale", type=float, default=None)
    parser.add_argument("--stage1-obs-noise-prior-scale", type=float, default=None)
    parser.add_argument("--stage1-lr", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--prior-scales", default="0.5,1.0,2.0")
    parser.add_argument("--obs-noise-prior-scales", default="0.1,1.0")
    parser.add_argument("--lrs", default="5e-4,1e-3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--max-walltime-seconds", type=int, default=1200)
    parser.add_argument("--parity-tol", type=float, default=1.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--require-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When false, candidate search records parity metrics but does not hard-stop on them.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip candidate runs that already have artifact+report outputs.",
    )
    args = parser.parse_args(argv)

    if args.architecture_source == "dnn-selection" and args.architectures is not None:
        raise ValueError(
            "Do not combine --architectures with --architecture-source=dnn-selection; "
            "the DNN winner architecture is selected automatically."
        )

    resolved_site = args.site if args.site is not None else detect_hpc_site()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else default_runs_root(REPO_ROOT, "bnn_sweep", site=resolved_site, run_tag=args.run_tag)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    dnn_root = Path(args.dnn_root).resolve() if args.dnn_root is not None else None

    seeds = _resolve_seeds(list(args.seed))
    architecture_text = _default_architecture_text() if args.architectures is None else args.architectures
    explicit_architectures = _parse_architecture_names(architecture_text)
    stage1_prior_scales = _resolve_stage1_grid(
        grid_text=args.stage1_prior_scales,
        scalar_value=args.stage1_prior_scale,
        default_text="1.0",
        option_name="--stage1-prior-scales",
        legacy_option_name="--stage1-prior-scale",
    )
    stage1_obs_noise_prior_scales = _resolve_stage1_grid(
        grid_text=args.stage1_obs_noise_prior_scales,
        scalar_value=args.stage1_obs_noise_prior_scale,
        default_text="1.0",
        option_name="--stage1-obs-noise-prior-scales",
        legacy_option_name="--stage1-obs-noise-prior-scale",
    )
    stage1_lrs = _resolve_stage1_grid(
        grid_text=args.stage1_lrs,
        scalar_value=args.stage1_lr,
        default_text="1e-3",
        option_name="--stage1-lrs",
        legacy_option_name="--stage1-lr",
    )
    prior_scales = _parse_float_list(args.prior_scales)
    obs_noise_prior_scales = _parse_float_list(args.obs_noise_prior_scales)
    lrs = _parse_float_list(args.lrs)
    specs = resolve_emb_dataset_specs(REPO_ROOT)
    if args.only:
        wanted = set(args.only)
        specs = [spec for spec in specs if spec["name"] in wanted]
        if not specs:
            raise ValueError(f"No EMB dataset specs matched --only values: {sorted(wanted)}")

    matrix_report_path = output_root / "bnn_sweep_matrix_report.json"
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "output_root": str(output_root),
        "started_at": _now_iso(),
        "config": {
            "seeds": seeds,
            "architectures": (
                [name for _, _, name in explicit_architectures] if args.architecture_source == "explicit" else None
            ),
            "architecture_source": args.architecture_source,
            "dnn_root": str(dnn_root) if dnn_root is not None else None,
            "stage1_prior_scale": float(stage1_prior_scales[0]) if len(stage1_prior_scales) == 1 else None,
            "stage1_obs_noise_prior_scale": (
                float(stage1_obs_noise_prior_scales[0]) if len(stage1_obs_noise_prior_scales) == 1 else None
            ),
            "stage1_lr": float(stage1_lrs[0]) if len(stage1_lrs) == 1 else None,
            "stage1_prior_scales": stage1_prior_scales,
            "stage1_obs_noise_prior_scales": stage1_obs_noise_prior_scales,
            "stage1_lrs": stage1_lrs,
            "top_k": int(args.top_k),
            "prior_scales": prior_scales,
            "obs_noise_prior_scales": obs_noise_prior_scales,
            "lrs": lrs,
            "batch_size": int(args.batch_size),
            "max_steps": int(args.max_steps),
            "max_epochs": None if args.max_epochs is None else int(args.max_epochs),
            "eval_every": int(args.eval_every),
            "predictive_mc_samples": int(args.predictive_mc_samples),
            "max_walltime_seconds": int(args.max_walltime_seconds),
            "parity_tol": float(args.parity_tol),
            "require_parity": bool(args.require_parity),
            "device": args.device,
        },
        "runs": [],
    }
    _write_json(matrix_report_path, report)

    run_rows: list[dict[str, object]] = []
    for spec in specs:
        for seed in seeds:
            seed_root = output_root / spec["name"] / f"seed_{seed}"
            seed_inputs = _resolve_seed_inputs(
                spec=spec,
                seed=int(seed),
                explicit_architectures=explicit_architectures,
                architecture_source=str(args.architecture_source),
                dnn_root=dnn_root,
            )
            seed_architectures = seed_inputs["architectures"]
            assert isinstance(seed_architectures, list)
            dnn_reference_path = str(seed_inputs["dnn_reference_path"])
            dnn_selection_path = seed_inputs["dnn_selection_path"]
            dnn_selected_architecture = seed_inputs["dnn_selected_architecture"]

            for width, depth, arch_name in seed_architectures:
                for prior_scale, obs_noise_prior_scale, lr in _iter_hyperparameter_grid(
                    prior_scales=stage1_prior_scales,
                    obs_noise_prior_scales=stage1_obs_noise_prior_scales,
                    lrs=stage1_lrs,
                ):
                    command, artifact_path, report_path = _build_command(
                        python_bin=args.python_bin,
                        spec=spec,
                        seed_root=seed_root,
                        dnn_reference_path=dnn_reference_path,
                        stage="stage1",
                        arch_name=arch_name,
                        width=width,
                        depth=depth,
                        prior_scale=prior_scale,
                        obs_noise_prior_scale=obs_noise_prior_scale,
                        batch_size=int(args.batch_size),
                        lr=lr,
                        max_steps=int(args.max_steps),
                        max_epochs=None if args.max_epochs is None else int(args.max_epochs),
                        eval_every=int(args.eval_every),
                        predictive_mc_samples=int(args.predictive_mc_samples),
                        max_walltime_seconds=int(args.max_walltime_seconds),
                        seed=int(seed),
                        parity_tol=float(args.parity_tol),
                        require_parity=bool(args.require_parity),
                        device=str(args.device),
                    )
                    row = {
                        "name": spec["name"],
                        "seed": int(seed),
                        "stage": "stage1",
                        "architecture": arch_name,
                        "prior_scale": float(prior_scale),
                        "obs_noise_prior_scale": float(obs_noise_prior_scale),
                        "lr": float(lr),
                        "artifact_path": str(artifact_path),
                        "report_path": str(report_path),
                        "dnn_reference_path": dnn_reference_path,
                        "dnn_selection_path": dnn_selection_path,
                        "status": "pending",
                    }
                    if args.resume and _candidate_completed(report_path, artifact_path):
                        row["status"] = "skipped_completed"
                        run_rows.append(row)
                        continue
                    print(f"[BNN sweep stage1] running: {' '.join(command)}")
                    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
                    row["status"] = "passed"
                    run_rows.append(row)

            top_architectures = _stage1_top_architectures(
                seed_root=seed_root,
                architectures=seed_architectures,
                top_k=int(args.top_k),
                prior_scales=stage1_prior_scales,
                obs_noise_prior_scales=stage1_obs_noise_prior_scales,
                lrs=stage1_lrs,
            )

            for width, depth, arch_name in top_architectures:
                for prior_scale, obs_noise_prior_scale, lr in _iter_hyperparameter_grid(
                    prior_scales=prior_scales,
                    obs_noise_prior_scales=obs_noise_prior_scales,
                    lrs=lrs,
                ):
                    command, artifact_path, report_path = _build_command(
                        python_bin=args.python_bin,
                        spec=spec,
                        seed_root=seed_root,
                        dnn_reference_path=dnn_reference_path,
                        stage="stage2",
                        arch_name=arch_name,
                        width=width,
                        depth=depth,
                        prior_scale=prior_scale,
                        obs_noise_prior_scale=obs_noise_prior_scale,
                        batch_size=int(args.batch_size),
                        lr=lr,
                        max_steps=int(args.max_steps),
                        max_epochs=None if args.max_epochs is None else int(args.max_epochs),
                        eval_every=int(args.eval_every),
                        predictive_mc_samples=int(args.predictive_mc_samples),
                        max_walltime_seconds=int(args.max_walltime_seconds),
                        seed=int(seed),
                        parity_tol=float(args.parity_tol),
                        require_parity=bool(args.require_parity),
                        device=str(args.device),
                    )
                    row = {
                        "name": spec["name"],
                        "seed": int(seed),
                        "stage": "stage2",
                        "architecture": arch_name,
                        "prior_scale": float(prior_scale),
                        "obs_noise_prior_scale": float(obs_noise_prior_scale),
                        "lr": float(lr),
                        "artifact_path": str(artifact_path),
                        "report_path": str(report_path),
                        "dnn_reference_path": dnn_reference_path,
                        "dnn_selection_path": dnn_selection_path,
                        "status": "pending",
                    }
                    if args.resume and _candidate_completed(report_path, artifact_path):
                        row["status"] = "skipped_completed"
                        run_rows.append(row)
                        continue
                    print(f"[BNN sweep stage2] running: {' '.join(command)}")
                    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
                    row["status"] = "passed"
                    run_rows.append(row)

            stage2_rows = [row for row in run_rows if row["name"] == spec["name"] and row["seed"] == int(seed) and row["stage"] == "stage2"]
            best = min(
                stage2_rows,
                key=lambda row: _candidate_metric(Path(str(row["report_path"]))),
            )
            selection_payload = {
                "name": spec["name"],
                "modality": spec["modality"],
                "diameter_um": spec["diameter_um"],
                "seed": int(seed),
                "architecture_source": args.architecture_source,
                "candidate_architectures": [arch_name for _, _, arch_name in seed_architectures],
                "dnn_reference_path": dnn_reference_path,
                "dnn_selection_path": dnn_selection_path,
                "dnn_selected_architecture": dnn_selected_architecture,
                "best_candidate_report_path": str(best["report_path"]),
                "best_candidate_artifact_path": str(best["artifact_path"]),
                "best_candidate_metric": float(_candidate_metric(Path(str(best["report_path"])))),
                "top_architectures": [arch_name for _, _, arch_name in top_architectures],
            }
            _write_json(seed_root / "selection.json", selection_payload)

        report["runs"] = run_rows
        _write_json(matrix_report_path, report)

    report["status"] = "passed"
    report["runs"] = run_rows
    _write_json(matrix_report_path, report)
    print(f"BNN sweep matrix report: {matrix_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
