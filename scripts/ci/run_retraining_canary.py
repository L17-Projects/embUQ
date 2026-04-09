#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "compression" / "surrogate" / "ci" / "retraining_smoke.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_ci" / "surrogate_retraining"


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _loss_history_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_loss_hist.pkl")


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
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)

    artifacts = {
        "model": str(model_path),
        "loss_history": str(_loss_history_path(model_path)),
    }
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
