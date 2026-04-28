#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_candidates(certification_root: Path) -> list[dict[str, object]]:
    candidates_path = certification_root / "promotion_candidates.json"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing promotion candidates file: {candidates_path}")
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("promotion_candidates.json must contain a JSON list.")
    return payload


def _read_per_dataset(certification_root: Path) -> list[dict[str, object]]:
    per_dataset_path = certification_root / "certification_per_dataset.csv"
    if not per_dataset_path.exists():
        raise FileNotFoundError(f"Missing certification per-dataset file: {per_dataset_path}")
    import pandas as pd

    return pd.read_csv(per_dataset_path).to_dict(orient="records")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote certified BNN artifacts from a certification run into tracked trained/ targets."
    )
    parser.add_argument("--certification-root", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)

    certification_root = Path(args.certification_root).resolve()
    candidates = _read_candidates(certification_root)
    per_dataset = _read_per_dataset(certification_root)

    uncertified = [row for row in per_dataset if not bool(row.get("certified", False))]
    if uncertified:
        names = ", ".join(sorted(str(row.get("dataset_name")) for row in uncertified))
        raise SystemExit(f"Refusing promotion because uncertified datasets remain: {names}")

    copied_rows: list[dict[str, object]] = []
    for candidate in candidates:
        source = Path(str(candidate["candidate_artifact_path"])).resolve()
        target = Path(str(candidate["tracked_bnn_artifact_path"])).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Certified BNN artifact does not exist: {source}")
        copied_rows.append(
            {
                "dataset_name": str(candidate["dataset_name"]),
                "candidate_seed": int(candidate["candidate_seed"]),
                "source_path": str(source),
                "target_path": str(target),
                "dry_run": bool(args.dry_run),
            }
        )
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    manifest = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "promoted",
        "certification_root": str(certification_root),
        "copied_rows": copied_rows,
        "updated_at": _now_iso(),
    }
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path is not None
        else certification_root / "promotion_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Promotion manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
