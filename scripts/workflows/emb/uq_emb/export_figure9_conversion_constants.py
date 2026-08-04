#!/usr/bin/env python3
"""Export the indentation diameter constants used by the frozen Figure 9 renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "mesouq.uq_emb.figure9_conversion_constants.v1"
DIAMETERS = ("3.2", "3.4", "5.8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_constants(*, legacy_root: Path, output: Path) -> dict:
    legacy_root = legacy_root.expanduser().resolve()
    output = output.expanduser().resolve()
    constants = {}
    for diameter in DIAMETERS:
        source = (
            legacy_root
            / f"indentation/surrogate/diameters/{diameter}um/data/samples_all.dat"
        )
        samples = np.loadtxt(source, ndmin=2)
        if samples.shape[1] <= 7:
            raise ValueError(f"Expected at least eight columns in {source}")
        constants[diameter] = {
            "initial_diameter_dpd": 2.0 * float(np.median(samples[:, 7])),
            "source": str(source),
            "source_sha256": _sha256(source),
            "row_count": int(samples.shape[0]),
            "column_count": int(samples.shape[1]),
            "source_column_zero_based": 7,
            "reduction": "2 * median(column)",
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "diameters": constants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = export_constants(legacy_root=args.legacy_root, output=args.output)
    print(json.dumps({"diameters": payload["diameters"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
