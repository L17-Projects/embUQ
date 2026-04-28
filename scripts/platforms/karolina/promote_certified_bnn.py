#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
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


def _coerce_certified(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan"}:
        return False
    raise ValueError(f"Unsupported certified value: {value!r}")


def _certification_index(per_dataset: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for row in per_dataset:
        dataset_name = str(row.get("dataset_name", "")).strip()
        if not dataset_name:
            raise ValueError("certification_per_dataset.csv rows must define dataset_name.")
        if dataset_name in index:
            raise ValueError(f"Duplicate certification rows for dataset: {dataset_name}")
        index[dataset_name] = row
    return index


def _validated_promotion_rows(
    candidates: list[dict[str, object]],
    per_dataset: list[dict[str, object]],
    *,
    dry_run: bool,
) -> list[dict[str, object]]:
    certification_by_dataset = _certification_index(per_dataset)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        dataset_name = str(candidate["dataset_name"])
        certification_row = certification_by_dataset.get(dataset_name)
        if certification_row is None:
            raise SystemExit(f"Refusing promotion because dataset lacks explicit certified row: {dataset_name}")
        if not _coerce_certified(certification_row.get("certified", False)):
            raise SystemExit(f"Refusing promotion because dataset is not certified: {dataset_name}")

        source = Path(str(candidate["candidate_artifact_path"])).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Certified BNN artifact does not exist: {source}")
        target = Path(str(candidate["tracked_bnn_artifact_path"])).resolve()
        rows.append(
            {
                "dataset_name": dataset_name,
                "candidate_seed": int(candidate["candidate_seed"]),
                "source_path": str(source),
                "target_path": str(target),
                "dry_run": bool(dry_run),
            }
        )
    return rows


def _stage_copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(source, tmp_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


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

    uncertified = [row for row in per_dataset if not _coerce_certified(row.get("certified", False))]
    if uncertified:
        names = ", ".join(sorted(str(row.get("dataset_name")) for row in uncertified))
        raise SystemExit(f"Refusing promotion because uncertified datasets remain: {names}")

    copied_rows = _validated_promotion_rows(candidates, per_dataset, dry_run=bool(args.dry_run))
    if not args.dry_run:
        staged_copies: list[tuple[Path, Path]] = []
        try:
            for row in copied_rows:
                source = Path(str(row["source_path"]))
                target = Path(str(row["target_path"]))
                staged_copies.append((target, _stage_copy(source, target)))
            for target, tmp_path in staged_copies:
                tmp_path.replace(target)
        finally:
            for _target, tmp_path in staged_copies:
                tmp_path.unlink(missing_ok=True)

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
