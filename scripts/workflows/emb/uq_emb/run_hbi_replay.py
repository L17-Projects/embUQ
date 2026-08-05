#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    sha256,
)
from meso_uq.vega_workflows import (  # noqa: E402
    VegaWorkflowSelection,
    build_inference_command,
    format_command,
)

SCHEMA_VERSION = "mesouq.uq_emb.hbi_replay.v1"
VALID_STAGES = ("phase1", "phase2", "phase3b")
AGENT_EXPERIMENT = {"sonovue": "indentation", "definity": "compression"}
STAGE_OUTPUTS = {
    "phase1": "results_phase_1",
    "phase2": "results_phase_2",
    "phase3b": "results_phase_3b",
}

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
    stage_seeds: dict[str, int],
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
            korali_random_seed=stage_seeds[stage],
        )
        for stage in stages
    ]
    return agent, population, commands


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_stage_selection_and_freshness(output_root: Path, stages: list[str]) -> None:
    if not stages or len(set(stages)) != len(stages):
        raise ValueError("Replay stages must be a non-empty sequence without duplicates.")
    stage_indices = [VALID_STAGES.index(stage) for stage in stages]
    if stage_indices != sorted(stage_indices):
        raise ValueError("Replay stages must follow Phase 1, Phase 2, Phase 3b order.")

    selected = set(stages)
    for index, stage in zip(stage_indices, stages, strict=True):
        stage_output = output_root / STAGE_OUTPUTS[stage]
        if stage_output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing {stage} output under {output_root}: {stage_output}. "
                "Choose a fresh replay root."
            )
        for prerequisite in VALID_STAGES[:index]:
            prerequisite_output = output_root / STAGE_OUTPUTS[prerequisite]
            if prerequisite not in selected and not prerequisite_output.is_dir():
                raise FileNotFoundError(
                    f"{stage} requires existing {prerequisite} output when {prerequisite} "
                    f"is not selected for replay: {prerequisite_output}"
                )


def _snapshot_run_input(
    *,
    config_path: Path,
    output_root: Path,
    config_binding: dict[str, Any],
) -> dict[str, str]:
    """Copy the verified materialized config once and execute only that private copy."""
    snapshot_dir = output_root / "runtime_inputs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "hbi_config.yaml"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=snapshot_dir,
        prefix=f".{snapshot_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with config_path.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, snapshot_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to replace an existing replay run-input snapshot: {snapshot_path}"
            ) from exc
        snapshot_path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)

    snapshot_sha256 = sha256(snapshot_path)
    if snapshot_sha256 != config_binding["config_sha256"]:
        raise ValueError(
            "Private replay run-input snapshot differs from the verified materialized config: "
            f"{snapshot_path}"
        )
    return {
        "path": str(snapshot_path),
        "sha256": snapshot_sha256,
        "source_config_sha256": config_binding["config_sha256"],
    }


def _verify_run_input_snapshot(snapshot: dict[str, str]) -> dict[str, str]:
    path = Path(snapshot["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Private replay run-input snapshot is missing: {path}")
    current_sha256 = sha256(path)
    if current_sha256 != snapshot["sha256"]:
        raise ValueError(
            "Private replay run-input snapshot changed after verification: "
            f"{path}"
        )
    return {"path": str(path), "sha256": current_sha256}


def run_replay(
    *,
    config_path: Path,
    output_root: Path,
    python_bin: str,
    site: str,
    stages: list[str],
    execute: bool,
) -> dict[str, Any]:
    # Preserve the caller's lexical spelling until provenance validation has
    # rejected direct or ancestor symlink aliases.
    config_path = config_path.expanduser().absolute()
    output_root = output_root.expanduser().resolve()
    _validate_stage_selection_and_freshness(output_root, stages)
    config_binding = load_materialization_binding(config_path, repo_root=REPO_ROOT)
    stage_seeds = config_binding["accepted_stage_seeds"]
    if set(stage_seeds) != set(VALID_STAGES):
        raise ValueError(f"Materialization binding has invalid accepted stage seeds: {stage_seeds}")
    run_input_snapshot = _snapshot_run_input(
        config_path=config_path,
        output_root=output_root,
        config_binding=config_binding,
    )
    snapshot_config_path = Path(run_input_snapshot["path"])
    agent, population, commands = build_replay_commands(
        config_path=snapshot_config_path,
        output_root=output_root,
        python_bin=python_bin,
        stages=stages,
        stage_seeds=stage_seeds,
    )

    receipt_path = output_root / "uq_emb_hbi_replay_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running" if execute else "planned",
        "agent": agent,
        "population": population,
        "site": site,
        "config_path": str(config_path),
        "run_input_snapshot": run_input_snapshot,
        "accepted_stage_seeds": stage_seeds,
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

    stage_env = os.environ.copy()
    stage_env["HUQ_INFERENCE_CONFIG"] = str(snapshot_config_path)
    started = time.monotonic()
    try:
        for stage, command in zip(stages, commands, strict=True):
            stage_started = time.monotonic()
            before = _verify_run_input_snapshot(run_input_snapshot)
            subprocess.run(command, cwd=REPO_ROOT, check=True, env=stage_env)
            after = _verify_run_input_snapshot(run_input_snapshot)
            receipt["stage_results"].append(
                {
                    "stage": stage,
                    "korali_random_seed": stage_seeds[stage],
                    "status": "passed",
                    "wall_seconds": time.monotonic() - stage_started,
                    "run_input_before": before,
                    "run_input_after": after,
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
