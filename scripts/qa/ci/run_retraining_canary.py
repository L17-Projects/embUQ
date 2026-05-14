#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "emb" / "compression" / "surrogate" / "ci" / "retraining_smoke.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_ci" / "surrogate_retraining"


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _loss_history_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_loss_hist.pkl")


def _run_logged_command(command: list[str], cwd: Path, output_root: Path) -> dict[str, str]:
    stdout_log = output_root / "retraining_canary.stdout.log"
    stderr_log = output_root / "retraining_canary.stderr.log"
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return {
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the public surrogate retraining canary and assert output artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args(argv)

    config_path = _resolve_repo_path(args.config)
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with config_path.open("rb") as handle:
        config = yaml.safe_load(handle)

    entrypoint = _resolve_repo_path(config["entrypoint"])
    data_path = _resolve_repo_path(config["data"])
    model_path = output_root / config["output_name"]

    command = [
        args.python_bin,
        str(entrypoint),
        str(data_path),
        "--out",
        str(model_path),
        "--width",
        str(config["width"]),
        "--depth",
        str(config["depth"]),
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--max-epoch",
        str(config["max_epoch"]),
    ]
    log_artifacts = _run_logged_command(command, REPO_ROOT, output_root)

    artifacts = {
        "model": str(model_path),
        "loss_history": str(_loss_history_path(model_path)),
    }
    artifacts.update(log_artifacts)
    missing = [path for path in artifacts.values() if not Path(path).exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Retraining canary completed but required artifacts are missing:\n{formatted}")

    report = {
        "status": "passed",
        "config": str(config_path),
        "data": str(data_path),
        "output_root": str(output_root),
        "command": command,
        "artifacts": artifacts,
    }
    report_path = output_root / "retraining_canary_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Retraining canary report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
