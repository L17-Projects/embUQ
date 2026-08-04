#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.config.models import InferenceConfig  # noqa: E402
from meso_uq.platforms.site_selector import resolve_hpc_site  # noqa: E402
from replay_provenance import (  # noqa: E402
    load_materialization_binding,
    runtime_provenance,
)
from meso_uq.vega_workflows import (  # noqa: E402
    VegaWorkflowSelection,
    build_inference_command,
    format_command,
)

SCHEMA_VERSION = "mesouq.uq_emb.hbi_replay.v1"
VALID_STAGES = ("phase1", "phase2", "phase3b")
AGENT_EXPERIMENT = {"sonovue": "indentation", "definity": "compression"}

def _load_and_validate_config(config_path: Path) -> tuple[dict[str, Any], str, int]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    InferenceConfig.model_validate(config)
    agent = str(config.get("resonance", {}).get("agent", "")).strip().lower()
    if agent not in AGENT_EXPERIMENT:
        raise ValueError(f"Unsupported or missing resonance.agent in {config_path}: {agent!r}.")
    populations = {
        int(config[name])
        for name in ("pop_size", "hbi_pop_size", "phase3b_pop_size")
    }
    if len(populations) != 1:
        raise ValueError(
            "UQ_EMB replay requires identical pop_size, hbi_pop_size, and phase3b_pop_size."
        )
    population = populations.pop()
    if population not in {10000, 50000}:
        raise ValueError(f"UQ_EMB replay population must be 10000 or 50000, got {population}.")
    return config, agent, population


def build_replay_commands(
    *,
    config_path: Path,
    output_root: Path,
    python_bin: str,
    stages: list[str],
) -> tuple[str, int, list[list[str]]]:
    _config, agent, population = _load_and_validate_config(config_path)
    selection = VegaWorkflowSelection(
        AGENT_EXPERIMENT[agent],
        "reduced-model",
        "production",
    )
    commands = [
        build_inference_command(
            REPO_ROOT,
            selection,
            stage=stage,
            python_bin=python_bin,
            config_path=config_path,
            output_root=output_root,
            cpu_ranks=1,
            device="gpu",
            phase2_backend="native-cuda" if stage == "phase2" else None,
        )
        for stage in stages
    ]
    return agent, population, commands


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_replay(
    *,
    config_path: Path,
    output_root: Path,
    python_bin: str,
    site: str,
    stages: list[str],
    execute: bool,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    agent, population, commands = build_replay_commands(
        config_path=config_path,
        output_root=output_root,
        python_bin=python_bin,
        stages=stages,
    )
    config_binding = load_materialization_binding(config_path, repo_root=REPO_ROOT)
    if execute and "phase1" in stages and (output_root / "results_phase_1").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Phase 1 output under {output_root}. "
            "Choose a fresh replay root."
        )

    receipt_path = output_root / "uq_emb_hbi_replay_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running" if execute else "planned",
        "agent": agent,
        "population": population,
        "site": site,
        "config_path": str(config_path),
        **config_binding,
        "provenance": runtime_provenance(
            repo_root=REPO_ROOT,
            site=site,
            requested_python_bin=python_bin,
        ),
        "output_root": str(output_root),
        "stages": stages,
        "commands": [format_command(command) for command in commands],
        "stage_results": [],
    }
    _write_receipt(receipt_path, receipt)
    if not execute:
        return receipt

    os.environ["HUQ_INFERENCE_CONFIG"] = str(config_path)
    started = time.monotonic()
    try:
        for stage, command in zip(stages, commands, strict=True):
            stage_started = time.monotonic()
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            receipt["stage_results"].append(
                {
                    "stage": stage,
                    "status": "passed",
                    "wall_seconds": time.monotonic() - stage_started,
                }
            )
            _write_receipt(receipt_path, receipt)
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["wall_seconds"] = time.monotonic() - started
        _write_receipt(receipt_path, receipt)
        raise
    receipt["status"] = "passed"
    receipt["wall_seconds"] = time.monotonic() - started
    _write_receipt(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay the accepted UQ_EMB reduced HBI Phase 1, 2, and 3b sequence."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--site", choices=("karolina", "vega"), default=None)
    parser.add_argument("--stages", nargs="+", choices=VALID_STAGES, default=list(VALID_STAGES))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    site = resolve_hpc_site(
        cli_site=args.site,
        env=os.environ,
        allow_hostname=False,
        default=None,
    )
    receipt = run_replay(
        config_path=args.config,
        output_root=args.output_root,
        python_bin=args.python_bin,
        site=site,
        stages=list(args.stages),
        execute=args.execute,
    )
    print(
        f"UQ_EMB {receipt['agent']} {receipt['population']} replay {receipt['status']}: "
        f"{receipt['output_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
