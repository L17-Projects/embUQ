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

SPECS = [
    {
        "name": "compression_2.1um",
        "script": REPO_ROOT / "emb" / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.1um" / "data" / "F_Delta.dat",
        "out": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BNN.pt",
        "dnn": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "compression_2.9um",
        "script": REPO_ROOT / "emb" / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.9um" / "data" / "F_Delta.dat",
        "out": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BNN.pt",
        "dnn": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "compression_3.0um",
        "script": REPO_ROOT / "emb" / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "3.0um" / "data" / "F_Delta.dat",
        "out": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BNN.pt",
        "dnn": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "indentation_3.2um",
        "script": REPO_ROOT / "emb" / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.2um" / "data" / "samples_all.dat",
        "out": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BNN.pt",
        "dnn": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
    {
        "name": "indentation_3.4um",
        "script": REPO_ROOT / "emb" / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "data" / "samples_all.dat",
        "out": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BNN.pt",
        "dnn": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
    {
        "name": "indentation_5.8um",
        "script": REPO_ROOT / "emb" / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "5.8um" / "data" / "samples_all.dat",
        "out": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BNN.pt",
        "dnn": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, object]) -> None:
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_map(report: dict[str, object]) -> dict[str, dict[str, object]]:
    runs = report.get("runs", [])
    if not isinstance(runs, list):
        return {}
    mapping: dict[str, dict[str, object]] = {}
    for item in runs:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str):
            mapping[name] = item
    return mapping


def _load_report(path: Path, output_root: Path, args: argparse.Namespace) -> dict[str, object]:
    if path.exists():
        report = _read_json(path)
        if "runs" not in report or not isinstance(report["runs"], list):
            report["runs"] = []
        report.setdefault("schema_version", 2)
        report.setdefault("output_root", str(output_root))
        report.setdefault("started_at", _now_iso())
        report.setdefault("config", {})
        report["config"] = {
            "width": args.width,
            "depth": args.depth,
            "prior_scale": args.prior_scale,
            "obs_noise": args.obs_noise,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "predictive_mc_samples": args.predictive_mc_samples,
            "max_walltime_seconds": args.max_walltime_seconds,
            "seed": args.seed,
            "parity_tol": args.parity_tol,
            "device": args.device,
        }
        return report
    return {
        "schema_version": 2,
        "status": "created",
        "output_root": str(output_root),
        "started_at": _now_iso(),
        "runs": [],
        "config": {
            "width": args.width,
            "depth": args.depth,
            "prior_scale": args.prior_scale,
            "obs_noise": args.obs_noise,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "predictive_mc_samples": args.predictive_mc_samples,
            "max_walltime_seconds": args.max_walltime_seconds,
            "seed": args.seed,
            "parity_tol": args.parity_tol,
            "device": args.device,
        },
    }


def _training_report_passed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = _read_json(path)
    except Exception:
        return False
    training = payload.get("training")
    if not isinstance(training, dict):
        return False
    return bool(training.get("parity_passed", False))


def _is_completed(spec: dict[str, Path | str], item_report: Path, run_entry: dict[str, object] | None) -> bool:
    artifact_exists = Path(spec["out"]).exists()
    report_passed = _training_report_passed(item_report)
    run_passed = bool(run_entry and run_entry.get("status") == "passed")
    return bool(artifact_exists and report_passed and (run_passed or run_entry is None))


def _select_specs(
    specs: list[dict[str, Path | str]],
    *,
    start_from: str | None,
    only: list[str],
) -> list[dict[str, Path | str]]:
    selected = list(specs)
    if only:
        wanted = set(only)
        selected = [spec for spec in selected if str(spec["name"]) in wanted]
    if start_from is not None:
        names = [str(spec["name"]) for spec in selected]
        if start_from not in names:
            raise ValueError(
                f"--start-from={start_from!r} not found in selected specs. Available: {', '.join(names)}"
            )
        start_idx = names.index(start_from)
        selected = selected[start_idx:]
    return selected


def _build_command(
    *,
    python_bin: str,
    spec: dict[str, Path | str],
    output_root: Path,
    width: int,
    depth: int,
    prior_scale: float,
    obs_noise: float,
    batch_size: int,
    lr: float,
    max_steps: int,
    eval_every: int,
    predictive_mc_samples: int,
    max_walltime_seconds: int,
    seed: int | None,
    parity_tol: float,
    device: str,
) -> tuple[list[str], Path]:
    report_path = output_root / f"{spec['name']}.json"
    command = [
        python_bin,
        str(spec["script"]),
        str(spec["data"]),
        "--out",
        str(spec["out"]),
        "--dnn-reference",
        str(spec["dnn"]),
        "--report-path",
        str(report_path),
        "--width",
        str(width),
        "--depth",
        str(depth),
        "--prior-scale",
        str(prior_scale),
        "--obs-noise",
        str(obs_noise),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--max-steps",
        str(max_steps),
        "--eval-every",
        str(eval_every),
        "--predictive-mc-samples",
        str(predictive_mc_samples),
        "--max-walltime-seconds",
        str(max_walltime_seconds),
        "--parity-tol",
        str(parity_tol),
        "--device",
        str(device),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command, report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train BNN surrogates for all EMB diameters with hard-stop on first failure."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--obs-noise", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--max-walltime-seconds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--parity-tol", type=float, default=1.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing per-diameter artifacts/reports and skip completed entries.",
    )
    parser.add_argument(
        "--start-from",
        default=None,
        help="Start from this spec name (e.g. indentation_3.2um) within selected specs.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only these spec names (repeatable).",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional explicit path for matrix state JSON (default: <output-root>/bnn_training_matrix_report.json).",
    )
    args = parser.parse_args(argv)

    resolved_site = args.site if args.site is not None else detect_hpc_site()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else default_runs_root(REPO_ROOT, "bnn_training", site=resolved_site, run_tag=args.run_tag)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(args.state_path).resolve()
        if args.state_path is not None
        else output_root / "bnn_training_matrix_report.json"
    )
    report = _load_report(report_path, output_root, args)
    runs_by_name = _run_map(report)

    selected_specs = _select_specs(SPECS, start_from=args.start_from, only=list(args.only))
    if not selected_specs:
        raise ValueError("No specs selected to run.")

    report["status"] = "running"
    report["selected"] = [str(spec["name"]) for spec in selected_specs]
    _write_report(report_path, report)

    for spec in selected_specs:
        command, item_report = _build_command(
            python_bin=args.python_bin,
            spec=spec,
            output_root=output_root,
            width=args.width,
            depth=args.depth,
            prior_scale=args.prior_scale,
            obs_noise=args.obs_noise,
            batch_size=args.batch_size,
            lr=args.lr,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            predictive_mc_samples=args.predictive_mc_samples,
            max_walltime_seconds=args.max_walltime_seconds,
            seed=args.seed,
            parity_tol=args.parity_tol,
            device=args.device,
        )

        name = str(spec["name"])
        run_entry = runs_by_name.get(name)
        if args.resume and _is_completed(spec, item_report, run_entry):
            if run_entry is None:
                run_entry = {
                    "name": name,
                    "artifact": str(spec["out"]),
                    "report": str(item_report),
                    "status": "passed",
                    "skipped": True,
                    "started_at": None,
                    "finished_at": _now_iso(),
                    "skip_reason": "already_completed",
                }
                report["runs"].append(run_entry)
                runs_by_name[name] = run_entry
            else:
                run_entry["skipped"] = True
                run_entry["skip_reason"] = "already_completed"
                run_entry["status"] = "passed"
            _write_report(report_path, report)
            print(f"[BNN] Skipping {name}: already completed")
            continue

        if run_entry is None:
            run_entry = {
                "name": name,
                "artifact": str(spec["out"]),
                "report": str(item_report),
            }
            report["runs"].append(run_entry)
            runs_by_name[name] = run_entry

        run_entry["status"] = "running"
        run_entry["started_at"] = _now_iso()
        run_entry["finished_at"] = None
        run_entry["skipped"] = False
        run_entry.pop("skip_reason", None)
        report["current"] = name
        report.pop("failure", None)
        _write_report(report_path, report)

        print(f"[BNN] Training {spec['name']}")
        print(f"[BNN] Command: {' '.join(command)}")
        try:
            subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                check=True,
                timeout=int(args.max_walltime_seconds) + 300,
            )
        except subprocess.TimeoutExpired as exc:
            report["status"] = "failed"
            report["failure"] = {
                "name": name,
                "reason": "timeout",
                "seconds": exc.timeout,
            }
            run_entry["status"] = "failed"
            run_entry["finished_at"] = _now_iso()
            run_entry["error"] = "timeout"
            _write_report(report_path, report)
            raise
        except subprocess.CalledProcessError as exc:
            report["status"] = "failed"
            report["failure"] = {
                "name": name,
                "reason": "command_failed",
                "returncode": exc.returncode,
                "command": exc.cmd,
            }
            run_entry["status"] = "failed"
            run_entry["finished_at"] = _now_iso()
            run_entry["error"] = f"returncode={exc.returncode}"
            _write_report(report_path, report)
            raise
        run_entry["status"] = "passed"
        run_entry["finished_at"] = _now_iso()
        _write_report(report_path, report)

    report["status"] = "passed"
    report.pop("current", None)
    _write_report(report_path, report)
    print(f"BNN training matrix report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
