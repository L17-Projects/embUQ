#!/usr/bin/env python3
"""Verify a materialized UQ_EMB direct-DPD replay plan without running DPD."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


EXPECTED_SCHEMA = "mesouq.uq_emb.direct_dpd_replay.v1"
EXPECTED_BUBBLES = ("d1", "d2", "d3", "d4", "d5", "d6")
DEFAULT_ACOUSTIC_RUNNER = (
    Path(__file__).resolve().parents[1] / "run_emb_free_shell_breathing_protocol.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runner(path: Path) -> Callable[[Path], list[Any]]:
    spec = importlib.util.spec_from_file_location("uq_emb_frozen_breathing_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen breathing runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_bubbles


def _verify_hash(path: Path, expected: str, context: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{context} hash mismatch: expected {expected}, got {actual}")


def _command_environment(command: list[str]) -> tuple[str, str]:
    path_prefix = "MESOUQ_BOUND_MAP_SOURCE_PATH="
    hash_prefix = "MESOUQ_BOUND_MAP_SOURCE_SHA256="
    paths = [item[len(path_prefix) :] for item in command if item.startswith(path_prefix)]
    hashes = [item[len(hash_prefix) :] for item in command if item.startswith(hash_prefix)]
    if len(paths) != 1 or len(hashes) != 1:
        raise ValueError("Acoustic command must bind exactly one MAP source path and SHA-256")
    return paths[0], hashes[0]


def verify(plan_path: Path, acoustic_runner: Path) -> dict[str, Any]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected replay schema in {plan_path}")
    if payload.get("mode") != "static_dry_run_only" or payload.get("submission") != "not performed":
        raise ValueError("Replay plan must remain static and unsubmitted")
    activation = payload.get("runtime_activation") or {}
    if activation.get("required_before_execution") is not True:
        raise ValueError("Replay plan does not declare the required site-runtime activation")

    for relative, expected in (payload.get("runtime_source_hashes") or {}).items():
        repository = Path(__file__).resolve().parents[4]
        _verify_hash(repository / relative, str(expected), f"Runtime source {relative}")

    load_bubbles = _load_runner(acoustic_runner)
    entries = payload.get("bubbles")
    if not isinstance(entries, list):
        raise ValueError("Replay plan bubbles must be a list")
    ids = tuple(str(entry.get("bubble_id")) for entry in entries)
    if ids != EXPECTED_BUBBLES:
        raise ValueError(f"Expected bubbles {EXPECTED_BUBBLES}, got {ids}")

    loaded: list[dict[str, Any]] = []
    for entry in entries:
        bubble_id = str(entry["bubble_id"])
        for modality in ("mechanical", "acoustic"):
            item = entry.get(modality) or {}
            command = item.get("command")
            if not isinstance(command, list) or not command:
                raise ValueError(f"{bubble_id} {modality} command is missing")
            for source in item.get("source_hashes") or []:
                _verify_hash(
                    Path(source["accepted_path"]),
                    str(source["accepted_sha256"]),
                    f"{bubble_id} {modality} accepted source",
                )

        mechanical = entry["mechanical"]
        _verify_hash(
            Path(mechanical["materialized_phase3b_map_manifest"]),
            str(mechanical["materialized_phase3b_map_manifest_sha256"]),
            f"{bubble_id} materialized mechanical input",
        )

        acoustic = entry["acoustic"]
        map_values = Path(acoustic["materialized_map_values"])
        _verify_hash(
            map_values,
            str(acoustic["materialized_map_values_sha256"]),
            f"{bubble_id} materialized acoustic input",
        )
        bound_path, bound_sha = _command_environment(acoustic["command"])
        old_path = os.environ.get("MESOUQ_BOUND_MAP_SOURCE_PATH")
        old_sha = os.environ.get("MESOUQ_BOUND_MAP_SOURCE_SHA256")
        os.environ["MESOUQ_BOUND_MAP_SOURCE_PATH"] = bound_path
        os.environ["MESOUQ_BOUND_MAP_SOURCE_SHA256"] = bound_sha
        try:
            bubbles = load_bubbles(map_values)
        finally:
            if old_path is None:
                os.environ.pop("MESOUQ_BOUND_MAP_SOURCE_PATH", None)
            else:
                os.environ["MESOUQ_BOUND_MAP_SOURCE_PATH"] = old_path
            if old_sha is None:
                os.environ.pop("MESOUQ_BOUND_MAP_SOURCE_SHA256", None)
            else:
                os.environ["MESOUQ_BOUND_MAP_SOURCE_SHA256"] = old_sha
        if len(bubbles) != 1:
            raise ValueError(f"{bubble_id} acoustic input loaded {len(bubbles)} cases instead of one")
        bubble = bubbles[0]
        loaded.append(
            {
                "bubble_id": bubble_id,
                "symbol": bubble.symbol,
                "ka": bubble.ka,
                "kb": bubble.kb,
            }
        )

    return {
        "schema_version": "mesouq.uq_emb.direct_dpd_replay_verification.v1",
        "status": "PASS",
        "plan": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "site": payload.get("site"),
        "bubble_count": len(loaded),
        "planned_command_count": len(loaded) * 2,
        "dpd_executed": False,
        "loaded_acoustic_inputs": loaded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acoustic-runner", type=Path, default=DEFAULT_ACOUSTIC_RUNNER)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.plan.resolve(), args.acoustic_runner.resolve())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PASS: verified {report['planned_command_count']} direct-DPD commands and "
        f"loaded {report['bubble_count']} acoustic inputs; no DPD was executed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
