#!/usr/bin/env python3
"""Compare UQ_EMB forward-canary science across relocated HPC sites."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from replay_provenance import (  # noqa: E402
    checked_replay_path,
    require_output_distinct_from_inputs,
    require_output_outside_known_locked_roots,
)


SCHEMA_VERSION = "mesouq.uq_emb.forward_canary_comparison.v1"
FORWARD_RTOL = 1.0e-6
FORWARD_ATOL = 1.0e-8
IGNORED_KEYS = frozenset(
    {
        "accepted_artifact_root",
        "config_path",
        "config_sha256",
        "data_file",
        "dependency_artifact_root",
        "git_branch",
        "hostname",
        "materialization_receipt",
        "materialization_receipt_sha256",
        "manifest",
        "path",
        "python_executable",
        "python_version",
        "repo_root",
        "requested_python_bin",
        "root",
        "site",
        "site_runtime_root",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "slurm_job_id",
        "training_data_file",
        "virtual_env",
        "conda_prefix",
        "wall_seconds",
        "prediction_min",
        "prediction_max",
        "predictions",
        "standard_deviation_min",
        "standard_deviation_max",
        "standard_deviations",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(payload: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing integrity mapping {context}.{key}")
    return value


def _sha256_field(payload: dict[str, Any], key: str, context: str) -> str:
    value = str(payload.get(key, ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Missing or invalid integrity field {context}.{key}")
    return value


def _integer_field(payload: dict[str, Any], key: str, context: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Missing or invalid integrity field {context}.{key}")
    return value


def _validate_integrity_fields(payload: dict[str, Any], context: str) -> None:
    for key in (
        "config_sha256",
        "config_semantic_sha256",
        "materialization_receipt_sha256",
        "source_config_sha256",
        "accepted_manifest_sha256",
        "dependency_manifest_sha256",
    ):
        _sha256_field(payload, key, context)

    accepted = _mapping(payload, "accepted_source_verification", context)
    accepted_context = f"{context}.accepted_source_verification"
    if accepted.get("status") != "PASS":
        raise ValueError(f"Integrity verification did not pass: {accepted_context}")
    _sha256_field(accepted, "manifest_sha256", accepted_context)
    accepted_count = _integer_field(accepted, "file_count", accepted_context)
    accepted_size = _integer_field(accepted, "logical_size_bytes", accepted_context)
    members = accepted.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"Missing integrity members in {accepted_context}")
    member_size = 0
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError(f"Invalid integrity member {accepted_context}.members[{index}]")
        member_context = f"{accepted_context}.members[{index}]"
        _sha256_field(member, "sha256", member_context)
        member_size += _integer_field(member, "size_bytes", member_context)
    if accepted_count != len(members) or accepted_size != member_size:
        raise ValueError(f"Integrity totals differ from members in {accepted_context}")

    dependencies = _mapping(payload, "dependency_verification", context)
    dependency_context = f"{context}.dependency_verification"
    if dependencies.get("status") != "PASS":
        raise ValueError(f"Integrity verification did not pass: {dependency_context}")
    _sha256_field(dependencies, "manifest_sha256", dependency_context)
    if _integer_field(dependencies, "file_count", dependency_context) == 0:
        raise ValueError(f"Integrity file count is empty in {dependency_context}")
    if _integer_field(dependencies, "logical_size_bytes", dependency_context) == 0:
        raise ValueError(f"Integrity byte count is empty in {dependency_context}")

    acoustic_artifacts = _mapping(payload, "acoustic_artifacts", context)
    if not acoustic_artifacts:
        raise ValueError(f"Missing acoustic artifact integrity records in {context}")
    for name, artifact in acoustic_artifacts.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"Invalid acoustic artifact record {context}.{name}")
        artifact_context = f"{context}.acoustic_artifacts.{name}"
        _sha256_field(artifact, "sha256", artifact_context)
        _integer_field(artifact, "size_bytes", artifact_context)

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Missing datasets in {context}")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"Invalid dataset record {context}.datasets[{index}]")
        dataset_context = f"{context}.datasets[{index}]"
        _sha256_field(dataset, "parameter_batch_sha256", dataset_context)
        _sha256_field(dataset, "reference_input_sha256", dataset_context)
        artifacts = dataset.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(f"Missing artifact list in {dataset_context}")
        if dataset.get("experiment") in {"compression", "indentation"} and not artifacts:
            raise ValueError(f"Missing mechanical artifact integrity records in {dataset_context}")
        for artifact_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                raise ValueError(
                    f"Invalid artifact record {dataset_context}.artifacts[{artifact_index}]"
                )
            artifact_context = f"{dataset_context}.artifacts[{artifact_index}]"
            _sha256_field(artifact, "sha256", artifact_context)
            _integer_field(artifact, "size_bytes", artifact_context)


def _load_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mesouq.uq_emb.forward_canary.v1":
        raise ValueError(f"Unsupported forward-canary schema in {path}")
    if payload.get("status") != "passed":
        raise ValueError(f"Forward canary did not pass: {path}")
    _validate_integrity_fields(payload, str(path))
    return payload


def _scientific_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scientific_payload(item)
            for key, item in value.items()
            if key not in IGNORED_KEYS
        }
    if isinstance(value, list):
        return [_scientific_payload(item) for item in value]
    return value


def _compare_forward_arrays(
    karolina: dict[str, Any],
    vega: dict[str, Any],
) -> list[dict[str, Any]]:
    karolina_datasets = karolina.get("datasets")
    vega_datasets = vega.get("datasets")
    if not isinstance(karolina_datasets, list) or not isinstance(vega_datasets, list):
        raise ValueError("Forward-canary receipts must contain dataset lists")
    if len(karolina_datasets) != len(vega_datasets):
        raise ValueError("Forward-canary dataset counts differ across sites")

    comparisons: list[dict[str, Any]] = []
    for karolina_row, vega_row in zip(karolina_datasets, vega_datasets, strict=True):
        dataset_name = str(karolina_row.get("dataset_name", ""))
        if not dataset_name or dataset_name != str(vega_row.get("dataset_name", "")):
            raise ValueError("Forward-canary dataset ordering differs across sites")
        comparison: dict[str, Any] = {"dataset_name": dataset_name}
        for key in ("predictions", "standard_deviations"):
            left = np.asarray(karolina_row.get(key), dtype=np.float64)
            right = np.asarray(vega_row.get(key), dtype=np.float64)
            if left.shape != right.shape or left.size == 0:
                raise ValueError(
                    f"Forward-canary {key} shapes differ for {dataset_name}: "
                    f"{left.shape} != {right.shape}"
                )
            if not np.allclose(left, right, rtol=FORWARD_RTOL, atol=FORWARD_ATOL):
                difference = np.abs(left - right)
                raise ValueError(
                    f"Full forward-canary {key} differ for {dataset_name}: "
                    f"max_abs={float(np.max(difference))}"
                )
            difference = np.abs(left - right)
            scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), FORWARD_ATOL)
            comparison[key] = {
                "shape": list(left.shape),
                "max_abs_difference": float(np.max(difference)),
                "max_relative_difference": float(np.max(difference / scale)),
            }
        comparisons.append(comparison)
    return comparisons


def compare_receipts(
    karolina_path: Path,
    vega_path: Path,
    *,
    expected_agent: str | None = None,
) -> dict[str, Any]:
    karolina = _load_receipt(karolina_path)
    vega = _load_receipt(vega_path)
    agent = str(karolina.get("agent", ""))
    if agent != str(vega.get("agent", "")):
        raise ValueError("Forward-canary agents differ across sites")
    if expected_agent is not None and agent != expected_agent:
        raise ValueError(f"Expected agent {expected_agent!r}, got {agent!r}")
    if karolina.get("site") != "karolina" or vega.get("site") != "vega":
        raise ValueError("Forward-canary receipts are not ordered Karolina then Vega")
    semantic_sha = str(karolina.get("config_semantic_sha256", ""))
    if len(semantic_sha) != 64 or semantic_sha != str(vega.get("config_semantic_sha256", "")):
        raise ValueError("Forward canaries do not share the same semantic config digest")
    karolina_commit = str((karolina.get("provenance") or {}).get("git_commit", ""))
    vega_commit = str((vega.get("provenance") or {}).get("git_commit", ""))
    if len(karolina_commit) != 40 or karolina_commit != vega_commit:
        raise ValueError("Forward canaries were not executed from the same Git commit")

    numerical_comparisons = _compare_forward_arrays(karolina, vega)

    karolina_science = _scientific_payload(karolina)
    vega_science = _scientific_payload(vega)
    karolina_canonical = _canonical_json(karolina_science)
    vega_canonical = _canonical_json(vega_science)
    if karolina_canonical != vega_canonical:
        raise ValueError(f"Scientific forward-canary payloads differ for {agent}")

    digest = _sha256_text(karolina_canonical)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "agent": agent,
        "config_semantic_sha256": semantic_sha,
        "git_commit": karolina_commit,
        "scientific_payload_sha256": digest,
        "forward_array_tolerances": {"rtol": FORWARD_RTOL, "atol": FORWARD_ATOL},
        "forward_array_comparisons": numerical_comparisons,
        "ignored_location_or_timing_keys": sorted(IGNORED_KEYS),
        "karolina": {
            "receipt": str(karolina_path.resolve()),
            "site": karolina["site"],
            "device": karolina.get("device"),
            "wall_seconds": karolina.get("wall_seconds"),
        },
        "vega": {
            "receipt": str(vega_path.resolve()),
            "site": vega["site"],
            "device": vega.get("device"),
            "wall_seconds": vega.get("wall_seconds"),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("definity", "sonovue"), required=True)
    parser.add_argument("--karolina", type=Path, required=True)
    parser.add_argument("--vega", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    karolina = checked_replay_path(args.karolina, label="Karolina canary receipt")
    vega = checked_replay_path(args.vega, label="Vega canary receipt")
    output = require_output_distinct_from_inputs(
        output_path=args.output,
        input_paths=[karolina, vega],
        label="Forward-canary comparison output",
    )
    output = require_output_outside_known_locked_roots(
        output_path=output,
        repo_root=SCRIPT_DIR.parents[3],
        label="Forward-canary comparison output",
    )
    report = compare_receipts(
        karolina,
        vega,
        expected_agent=args.agent,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"UQ_EMB {args.agent} forward canaries match: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
