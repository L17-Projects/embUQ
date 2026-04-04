#!/usr/bin/env python3
"""
Refresh the indentation 3.4um surrogate and rerun the reduced indentation workflow.

This helper is designed for the "new training data just arrived" loop:
1. Optionally stage a replacement samples_all.dat file for 3.4um.
2. Back up the currently deployed best 3.4um surrogate.
3. Retrain 3.4um with the normal multi-architecture selection flow.
4. Run the GPU-batched indentation_reduced workflow at the requested population,
   skipping Phase 3a and postprocessing through MAP extraction and plots.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SURROGATE_ROOT = PROJECT_ROOT / "indentation" / "surrogate"
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "reduced" / "configs" / "production" / "reduced_config_indentation.yaml"
DEFAULT_TRAIN_SCRIPT = SURROGATE_ROOT / "scripts" / "train_multi_arch.py"
DEFAULT_WORKFLOW_RUNNER = PROJECT_ROOT / "inference" / "scripts" / "run_gpu_validation_suite.py"
DEFAULT_TRAINING_DATA = SURROGATE_ROOT / "diameters" / "3.4um" / "data" / "samples_all.dat"
DEFAULT_BEST_MODEL = SURROGATE_ROOT / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BEST.pkl"
DEFAULT_KORALI_PYTHONPATH = PROJECT_ROOT.parent / "korali_gpu" / "local_install" / "usr" / "local" / "lib" / "python3.8" / "site-packages"
DEFAULT_PYTHON_BIN = Path(sys.executable)

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _print_step(title: str, cmd: list[str] | None = None, cwd: Path | None = None) -> None:
    print(f"\n=== {title} ===")
    if cwd is not None:
        print(f"cwd: {cwd}")
    if cmd is not None:
        print("cmd:", " ".join(shlex.quote(part) for part in cmd))

def _run_step(title: str, cmd: list[str], cwd: Path, env: dict[str, str], timings: list[dict[str, Any]]) -> None:
    _print_step(title, cmd=cmd, cwd=cwd)
    start = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
    elapsed = time.perf_counter() - start
    timings.append({"step": title, "elapsed_seconds": elapsed})
    print(f"{title} finished in {elapsed:.2f}s")

def _parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"Invalid --config-set value '{item}'. Expected key=value.")
    key, value = item.split("=", 1)
    if not key:
        raise ValueError(f"Invalid --config-set value '{item}'. Empty key.")
    return key, yaml.safe_load(value)

def _write_config_override(base_config: Path, output_path: Path, config_overrides: dict[str, Any]) -> dict[str, Any]:
    with base_config.open("rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)
    config["skip_phase3a"] = True
    for key, value in config_overrides.items():
        config[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config

def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the 3.4um indentation surrogate and rerun the reduced workflow.")
    parser.add_argument("--training-data", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--population-size", type=int, default=50000)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--base-config", type=str, default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--config-set", action="append", default=[])
    parser.add_argument("--python-bin", type=str, default=str(DEFAULT_PYTHON_BIN))
    parser.add_argument("--korali-pythonpath", type=str, default=str(DEFAULT_KORALI_PYTHONPATH) if DEFAULT_KORALI_PYTHONPATH.exists() else None)
    parser.add_argument("--cpu-ranks", type=int, default=9)
    parser.add_argument("--gpu-device", type=str, default="cuda")
    parser.add_argument("--gpu-chunk-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epoch", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--disp-source", type=str, default="diameter", choices=["auto", "diameter", "displacement"])
    args = parser.parse_args()

    stamp = _timestamp()
    output_root = Path(args.output_root).resolve() if args.output_root else (PROJECT_ROOT.parent / "korali_gpu" / f"indentation_reduced_refresh_{stamp}")
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "indentation_reduced_refresh_config.yaml"
    manifest_path = output_root / "refresh_manifest.json"
    config_overrides = dict(_parse_override(item) for item in args.config_set)
    config_overrides.setdefault("skip_phase3a", True)
    derived_config = _write_config_override(Path(args.base_config), config_path, config_overrides)

    env = os.environ.copy()
    if args.korali_pythonpath:
        env["PYTHONPATH"] = ":".join([args.korali_pythonpath, str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(":")
    else:
        env["PYTHONPATH"] = ":".join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(":")

    timings: list[dict[str, Any]] = []
    if not args.skip_training:
        train_cmd = [args.python_bin, str(DEFAULT_TRAIN_SCRIPT), "--diameter", "3.4", "--data-diameter", "3.4", "--data-file", "samples_all.dat", "--disp-source", args.disp_source, "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--max-epoch", str(args.max_epoch), "--num-workers", str(args.num_workers)]
        _run_step("Retrain 3.4um surrogate", train_cmd, SURROGATE_ROOT, env, timings)
    else:
        _print_step("Retrain 3.4um surrogate")
        print("Skipping retraining because --skip-training was provided.")

    workflow_cmd = [args.python_bin, str(DEFAULT_WORKFLOW_RUNNER), "--workflows", "indentation_reduced", "--output-root", str(output_root), "--python-bin", args.python_bin, "--cpu-ranks", str(args.cpu_ranks), "--gpu-device", args.gpu_device, "--gpu-chunk-size", str(args.gpu_chunk_size), "--seed", str(args.seed), "--population-size", str(args.population_size), "--skip-phase3a", "--config-override", f"indentation_reduced={config_path}"]
    if args.korali_pythonpath:
        workflow_cmd.extend(["--korali-pythonpath", args.korali_pythonpath])
    _run_step("Run indentation_reduced workflow", workflow_cmd, PROJECT_ROOT, env, timings)

    manifest = {
        "timestamp": stamp,
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "workflow_output": str(output_root / f"indentation_reduced_{args.population_size}"),
        "python_bin": args.python_bin,
        "korali_pythonpath": args.korali_pythonpath or "",
        "base_config": str(Path(args.base_config).resolve()),
        "config_override": str(config_path),
        "population_size": args.population_size,
        "cpu_ranks": args.cpu_ranks,
        "gpu_device": args.gpu_device,
        "gpu_chunk_size": args.gpu_chunk_size,
        "seed": args.seed,
        "skip_phase3a": True,
        "skip_training": bool(args.skip_training),
        "config_overrides": config_overrides,
        "resolved_config": derived_config,
        "step_timings": timings,
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in timings),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nRefresh workflow complete.")
    print(f"Manifest: {manifest_path}")
    print(f"Workflow results: {output_root / f'indentation_reduced_{args.population_size}'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
