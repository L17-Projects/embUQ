#!/usr/bin/env python3
"""Compare UQ_EMB forward-canary science across relocated HPC sites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "mesouq.uq_emb.forward_canary_comparison.v1"
FORWARD_RTOL = 1.0e-6
FORWARD_ATOL = 1.0e-8
IGNORED_KEYS = frozenset(
    {
        "config_path",
        "config_sha256",
        "data_file",
        "git_branch",
        "hostname",
        "materialization_receipt",
        "materialization_receipt_sha256",
        "path",
        "python_executable",
        "python_version",
        "repo_root",
        "requested_python_bin",
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


def _load_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mesouq.uq_emb.forward_canary.v1":
        raise ValueError(f"Unsupported forward-canary schema in {path}")
    if payload.get("status") != "passed":
        raise ValueError(f"Forward canary did not pass: {path}")
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
    report = compare_receipts(
        args.karolina.expanduser().resolve(),
        args.vega.expanduser().resolve(),
        expected_agent=args.agent,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"UQ_EMB {args.agent} forward canaries match: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
