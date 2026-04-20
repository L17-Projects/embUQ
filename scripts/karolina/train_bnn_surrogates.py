#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT_DEFAULT = REPO_ROOT / "_vega" / "bnn_training"

SPECS = [
    {
        "name": "compression_2.1um",
        "script": REPO_ROOT / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.1um" / "data" / "F_Delta.dat",
        "out": REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BNN.pt",
        "dnn": REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "compression_2.9um",
        "script": REPO_ROOT / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.9um" / "data" / "F_Delta.dat",
        "out": REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BNN.pt",
        "dnn": REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "compression_3.0um",
        "script": REPO_ROOT / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "compression" / "surrogate" / "diameters" / "3.0um" / "data" / "F_Delta.dat",
        "out": REPO_ROOT / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BNN.pt",
        "dnn": REPO_ROOT / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "indentation_3.2um",
        "script": REPO_ROOT / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.2um" / "data" / "samples_all.dat",
        "out": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BNN.pt",
        "dnn": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
    {
        "name": "indentation_3.4um",
        "script": REPO_ROOT / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.4um" / "data" / "samples_all.dat",
        "out": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BNN.pt",
        "dnn": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
    {
        "name": "indentation_5.8um",
        "script": REPO_ROOT / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "data": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "5.8um" / "data" / "samples_all.dat",
        "out": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BNN.pt",
        "dnn": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
]


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
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT_DEFAULT))
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "status": "running",
        "output_root": str(output_root),
        "runs": [],
    }
    report_path = output_root / "bnn_training_matrix_report.json"

    for spec in SPECS:
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
                "name": spec["name"],
                "reason": "timeout",
                "seconds": exc.timeout,
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise
        except subprocess.CalledProcessError as exc:
            report["status"] = "failed"
            report["failure"] = {
                "name": spec["name"],
                "reason": "command_failed",
                "returncode": exc.returncode,
                "command": exc.cmd,
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise
        run_item = {
            "name": spec["name"],
            "artifact": str(spec["out"]),
            "report": str(item_report),
        }
        report["runs"].append(run_item)

    report["status"] = "passed"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"BNN training matrix report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
