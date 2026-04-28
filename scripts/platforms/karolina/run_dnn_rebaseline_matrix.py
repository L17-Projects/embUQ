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
    DEFAULT_DNN_ARCHITECTURES,
    resolve_emb_dataset_specs,
)

DEFAULT_SEEDS = (20260317, 20260318, 20260319, 20260320, 20260321)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_dnn_architecture_text() -> str:
    return ",".join(arch.name for arch in DEFAULT_DNN_ARCHITECTURES)


def _parse_architecture_names(text: str) -> list[tuple[int, int, str]]:
    by_name = {arch.name: (arch.width, arch.depth, arch.name) for arch in DEFAULT_DNN_ARCHITECTURES}
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("Architecture list must be non-empty.")
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown DNN architecture name(s): {unknown}")
    return [by_name[name] for name in names]


def _resolve_seeds(values: list[int]) -> list[int]:
    if values:
        return [int(value) for value in values]
    return [int(seed) for seed in DEFAULT_SEEDS]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_paths(seed_root: Path, arch_name: str) -> tuple[Path, Path]:
    artifact_path = seed_root / "candidates" / f"{arch_name}.pkl"
    report_path = seed_root / "reports" / f"{arch_name}.json"
    return artifact_path, report_path


def _candidate_completed(report_path: Path, artifact_path: Path) -> bool:
    if not report_path.exists() or not artifact_path.exists():
        return False
    try:
        payload = _read_json(report_path)
    except Exception:
        return False
    return isinstance(payload.get("val_loss"), (float, int))


def _build_command(
    *,
    python_bin: str,
    spec: dict[str, str],
    seed_root: Path,
    width: int,
    depth: int,
    arch_name: str,
    batch_size: int,
    lr: float,
    max_epoch: int,
    seed: int,
) -> tuple[list[str], Path, Path]:
    artifact_path, report_path = _candidate_paths(seed_root, arch_name)
    command = [
        python_bin,
        spec["dnn_train_script"],
        spec["data"],
        "--out",
        str(artifact_path),
        "--width",
        str(width),
        "--depth",
        str(depth),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--max-epoch",
        str(max_epoch),
        "--seed",
        str(seed),
        "--report-path",
        str(report_path),
    ]
    return command, artifact_path, report_path


def _best_candidate(seed_root: Path, architectures: list[tuple[int, int, str]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for width, depth, arch_name in architectures:
        artifact_path, report_path = _candidate_paths(seed_root, arch_name)
        payload = _read_json(report_path)
        rows.append(
            {
                "name": arch_name,
                "width": int(width),
                "depth": int(depth),
                "artifact_path": str(artifact_path),
                "report_path": str(report_path),
                "train_loss": float(payload["train_loss"]),
                "val_loss": float(payload["val_loss"]),
            }
        )
    rows.sort(key=lambda row: (float(row["val_loss"]), float(row["train_loss"]), int(row["width"]), int(row["depth"])))
    return rows[0]


def _seed_output_root(output_root: Path, spec_name: str, seed: int) -> Path:
    return output_root / spec_name / f"seed_{seed}"


def _selection_path(seed_root: Path) -> Path:
    return seed_root / "selection.json"


def _sanitize_report_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "+"} else "-" for ch in text)


def _matrix_report_path(
    output_root: Path,
    *,
    requested_specs: list[str],
    requested_seeds: list[int],
) -> Path:
    if not requested_specs and not requested_seeds:
        return output_root / "dnn_rebaseline_matrix_report.json"

    parts: list[str] = []
    if requested_specs:
        spec_token = "+".join(_sanitize_report_token(name) for name in sorted(requested_specs))
        parts.append(f"specs-{spec_token}")
    if requested_seeds:
        seed_token = "+".join(str(int(seed)) for seed in requested_seeds)
        parts.append(f"seeds-{seed_token}")
    suffix = "__".join(parts) if parts else "invocation"
    return output_root / f"dnn_rebaseline_matrix_report__{suffix}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run seeded DNN rebaseline sweeps for all EMB surrogate datasets without promoting tracked BEST artifacts."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--architectures", default=_default_dnn_architecture_text())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip candidates that already have artifact+report outputs.",
    )
    args = parser.parse_args(argv)

    resolved_site = args.site if args.site is not None else detect_hpc_site()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else default_runs_root(REPO_ROOT, "dnn_rebaseline", site=resolved_site, run_tag=args.run_tag)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = _resolve_seeds(list(args.seed))
    architectures = _parse_architecture_names(args.architectures)
    specs = resolve_emb_dataset_specs(REPO_ROOT)
    if args.only:
        wanted = set(args.only)
        specs = [spec for spec in specs if spec["name"] in wanted]
        if not specs:
            raise ValueError(f"No EMB dataset specs matched --only values: {sorted(wanted)}")

    requested_specs = list(args.only)
    requested_seed_values = [int(seed) for seed in args.seed]
    matrix_report_path = _matrix_report_path(
        output_root,
        requested_specs=requested_specs,
        requested_seeds=requested_seed_values,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "completed_at": None,
        "output_root": str(output_root),
        "started_at": _now_iso(),
        "config": {
            "requested_specs": requested_specs,
            "requested_seeds": requested_seed_values,
            "resolved_specs": [str(spec["name"]) for spec in specs],
            "seeds": seeds,
            "architectures": [name for _, _, name in architectures],
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "max_epoch": int(args.max_epoch),
        },
        "runs": [],
    }
    _write_json(matrix_report_path, report)

    run_rows: list[dict[str, object]] = []
    for spec in specs:
        for seed in seeds:
            seed_root = _seed_output_root(output_root, spec["name"], seed)
            for width, depth, arch_name in architectures:
                command, artifact_path, report_path = _build_command(
                    python_bin=args.python_bin,
                    spec=spec,
                    seed_root=seed_root,
                    width=width,
                    depth=depth,
                    arch_name=arch_name,
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    max_epoch=int(args.max_epoch),
                    seed=int(seed),
                )
                row = {
                    "name": spec["name"],
                    "seed": int(seed),
                    "architecture": arch_name,
                    "artifact_path": str(artifact_path),
                    "report_path": str(report_path),
                    "status": "pending",
                }
                if args.resume and _candidate_completed(report_path, artifact_path):
                    row["status"] = "skipped_completed"
                    run_rows.append(row)
                    continue

                print(f"[DNN rebaseline] running: {' '.join(command)}")
                subprocess.run(command, cwd=str(REPO_ROOT), check=True)
                row["status"] = "passed"
                run_rows.append(row)

            best = _best_candidate(seed_root, architectures)
            selection_payload = {
                "name": spec["name"],
                "modality": spec["modality"],
                "diameter_um": spec["diameter_um"],
                "seed": int(seed),
                "best_architecture": str(best["name"]),
                "best_artifact_path": str(best["artifact_path"]),
                "best_report_path": str(best["report_path"]),
                "best_train_loss": float(best["train_loss"]),
                "best_val_loss": float(best["val_loss"]),
                "candidate_count": len(architectures),
            }
            _write_json(_selection_path(seed_root), selection_payload)

        report["runs"] = run_rows
        _write_json(matrix_report_path, report)

    report["status"] = "passed"
    report["completed_at"] = _now_iso()
    report["runs"] = run_rows
    _write_json(matrix_report_path, report)
    print(f"DNN rebaseline matrix report: {matrix_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
