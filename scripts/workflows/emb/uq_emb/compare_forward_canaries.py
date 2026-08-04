#!/usr/bin/env python3
"""Compare UQ_EMB forward-canary science across relocated HPC sites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mesouq.uq_emb.forward_canary_comparison.v1"
IGNORED_KEYS = frozenset(
    {
        "config_path",
        "config_sha256",
        "data_file",
        "path",
        "site",
        "training_data_file",
        "wall_seconds",
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
        "scientific_payload_sha256": digest,
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
