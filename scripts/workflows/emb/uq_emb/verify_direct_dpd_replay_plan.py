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
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from replay_provenance import (  # noqa: E402
    checked_replay_path,
    require_output_distinct_from_inputs,
    require_output_outside_known_locked_roots,
    require_output_outside_locked_root,
)

DEFAULT_ACOUSTIC_RUNNER = (
    Path(__file__).resolve().parents[1] / "run_emb_free_shell_breathing_protocol.py"
)
DEFAULT_MATERIALIZER = Path(__file__).resolve().parent / "materialize_direct_dpd_replay.py"


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


def _load_materializer(path: Path):
    spec = importlib.util.spec_from_file_location("uq_emb_direct_dpd_materializer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import direct-DPD materializer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _require_receipt_outside_accepted_root(*, receipt: Path, plan_path: Path) -> None:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    accepted_root = checked_replay_path(
        Path(str(payload.get("accepted_artifact_root", ""))),
        label="accepted artifact root",
    )
    require_output_outside_locked_root(
        output_path=receipt,
        locked_root=accepted_root,
        label="Direct-DPD verification receipt",
        locked_root_label="accepted artifact root",
    )


def verify(
    plan_path: Path,
    acoustic_runner: Path,
    materializer_path: Path = DEFAULT_MATERIALIZER,
) -> dict[str, Any]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected replay schema in {plan_path}")
    if payload.get("mode") != "static_dry_run_only" or payload.get("submission") != "not performed":
        raise ValueError("Replay plan must remain static and unsubmitted")
    activation = payload.get("runtime_activation") or {}
    if activation.get("required_before_execution") is not True:
        raise ValueError("Replay plan does not declare the required site-runtime activation")

    if materializer_path.resolve() != DEFAULT_MATERIALIZER.resolve():
        raise ValueError("Direct-DPD verification must use the repository materializer")
    materializer = _load_materializer(materializer_path)
    if acoustic_runner.resolve() != materializer.SONOVUE_ACOUSTIC_RUNNER.resolve():
        raise ValueError("Direct-DPD verification must use the hashed acoustic runner")
    accepted_root = Path(str(payload.get("accepted_artifact_root", ""))).resolve()
    accepted_manifest = Path(str(payload.get("accepted_artifact_manifest", ""))).resolve()
    accepted_index = materializer._load_accepted_manifest(accepted_manifest, accepted_root)
    if payload.get("accepted_artifact_manifest_sha256") != accepted_index["sha256"]:
        raise ValueError("Replay plan accepted-manifest hash does not match the locked manifest")
    if payload.get("accepted_artifact_verification") != accepted_index["verification"]:
        raise ValueError(
            "Replay plan accepted-artifact verification does not match the complete locked root"
        )
    provenance = payload.get("provenance") or {}
    current_provenance = materializer.runtime_provenance(
        repo_root=materializer.REPO_ROOT,
        site=str(payload.get("site", "")),
        requested_python_bin=str(payload.get("python_bin", "")),
    )
    if provenance.get("git_commit") != current_provenance["git_commit"]:
        raise ValueError("Replay plan Git commit does not match the current materializer checkout")

    runtime_source_hashes = payload.get("runtime_source_hashes")
    expected_runtime_sources = {
        str(path.relative_to(materializer.REPO_ROOT))
        for path in (
            materializer.MECHANICAL_RUNNER,
            *materializer.MECHANICAL_RUNNER_HELPERS,
            *materializer.MECHANICAL_EVALUATORS.values(),
            *materializer.ACOUSTIC_RUNTIME_SOURCES,
        )
    }
    if (
        not isinstance(runtime_source_hashes, dict)
        or set(runtime_source_hashes) != expected_runtime_sources
    ):
        raise ValueError(
            "Replay plan runtime_source_hashes must contain the complete runtime source set"
        )
    for relative, expected in runtime_source_hashes.items():
        repository = materializer.REPO_ROOT
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
                verified = materializer._accepted_source(
                    accepted_index,
                    Path(source["accepted_path"]),
                )
                if verified != source:
                    raise ValueError(
                        f"{bubble_id} {modality} source binding differs from the locked manifest"
                    )

        expected_bubble = materializer.BUBBLES[len(loaded)]
        for field in ("id", "agent", "diameter_um", "experiment"):
            if entry.get(field) != expected_bubble[field]:
                raise ValueError(f"{bubble_id} metadata differs from the canonical bubble table")

        output_root = plan_path.parent.resolve()
        direct_map = None
        if entry["agent"] == "sonovue":
            direct_map = (
                output_root
                / bubble_id
                / "mechanical/map_json"
                / f"indentation_{float(entry['diameter_um']):.1f}um_map.json"
            )
        expected_mechanical_command = materializer._mechanical_command(
            site=str(payload["site"]),
            python_bin=str(payload["python_bin"]),
            accepted_root=accepted_root,
            accepted_index=accepted_index,
            bubble=expected_bubble,
            output_root=output_root,
            direct_map=direct_map,
        )
        if entry["mechanical"].get("command") != expected_mechanical_command:
            raise ValueError(f"{bubble_id} mechanical command differs from the canonical command")

        _setup_source, map_payload, protocol, _sources = materializer._acoustic_sources(
            accepted_root=accepted_root,
            accepted_index=accepted_index,
            bubble=expected_bubble,
        )
        state_source = materializer._accepted_hbi_state(
            accepted_root,
            accepted_index,
            expected_bubble,
        )
        symbol = str(map_payload.get("symbol", map_payload.get("diameter_symbol")))
        expected_acoustic_command = materializer._acoustic_command(
            site=str(payload["site"]),
            python_bin=str(payload["python_bin"]),
            state_source=state_source,
            map_copy=Path(entry["acoustic"]["materialized_map_values"]),
            output_root=output_root,
            bubble_id=bubble_id,
            symbol=symbol,
            protocol=protocol,
        )
        if entry["acoustic"].get("command") != expected_acoustic_command:
            raise ValueError(f"{bubble_id} acoustic command differs from the canonical command")

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
        "accepted_artifact_manifest": str(accepted_manifest),
        "accepted_artifact_manifest_sha256": accepted_index["sha256"],
        "accepted_artifact_verification": accepted_index["verification"],
        "git_commit": provenance.get("git_commit"),
        "bubble_count": len(loaded),
        "planned_command_count": len(loaded) * 2,
        "dpd_executed": False,
        "loaded_acoustic_inputs": loaded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acoustic-runner", type=Path, default=DEFAULT_ACOUSTIC_RUNNER)
    parser.add_argument("--materializer", type=Path, default=DEFAULT_MATERIALIZER)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    plan_path = checked_replay_path(args.plan, label="direct-DPD replay plan")
    acoustic_runner = checked_replay_path(
        args.acoustic_runner, label="direct-DPD acoustic runner"
    )
    materializer = checked_replay_path(args.materializer, label="direct-DPD materializer")
    repo_root = checked_replay_path(
        Path(__file__).resolve().parents[4], label="repository root"
    )
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    accepted_manifest = checked_replay_path(
        Path(str(plan_payload.get("accepted_artifact_manifest", ""))),
        label="accepted artifact manifest",
    )
    receipt = checked_replay_path(args.receipt, label="direct-DPD verification receipt")
    try:
        _require_receipt_outside_accepted_root(receipt=receipt, plan_path=plan_path)
        receipt = require_output_outside_known_locked_roots(
            output_path=receipt,
            repo_root=repo_root,
            label="Direct-DPD verification receipt",
        )
        receipt = require_output_distinct_from_inputs(
            output_path=receipt,
            input_paths=[plan_path, accepted_manifest, acoustic_runner, materializer],
            label="Direct-DPD verification receipt",
        )
        report = verify(
            plan_path,
            acoustic_runner,
            materializer,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PASS: verified {report['planned_command_count']} direct-DPD commands and "
        f"loaded {report['bubble_count']} acoustic inputs; no DPD was executed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
