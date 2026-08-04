#!/usr/bin/env python3
"""Flatten the frozen Definity Phase-1 ridge sources used by UQ_EMB Figure 7."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "mesouq.uq_emb.figure7_phase1_overlay.v1"
REQUIRED_COLUMNS = {"state_path", "source_label", "dataset", "plotted_samples"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_state(
    *, state_path: Path, requested_samples: int, row_index: int
) -> tuple[np.ndarray, np.ndarray, int]:
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    variables = [str(variable.get("Name", "")) for variable in state.get("Variables", [])]
    try:
        ka_index = variables.index("ka")
        kb_index = variables.index("kb")
    except ValueError as exc:
        raise ValueError(f"State file lacks ka/kb variables: {state_path}") from exc

    samples = np.asarray(
        state.get("Results", {}).get("Posterior Sample Database", []), dtype=float
    )
    if samples.ndim != 2 or samples.shape[1] <= max(ka_index, kb_index):
        raise ValueError(f"Unexpected posterior sample shape in {state_path}: {samples.shape}")
    finite = samples[np.isfinite(samples[:, ka_index]) & np.isfinite(samples[:, kb_index])]
    if finite.shape[0] > requested_samples:
        choice = np.random.default_rng(20260630 + row_index).choice(
            finite.shape[0], size=requested_samples, replace=False
        )
        finite = finite[np.sort(choice)]
    return finite[:, ka_index], finite[:, kb_index], int(samples.shape[0])


def export_overlay(
    *, sources_csv: Path, source_manifest: Path, output_csv: Path, output_manifest: Path
) -> dict[str, Any]:
    sources_csv = sources_csv.expanduser().resolve()
    source_manifest = source_manifest.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    output_manifest = output_manifest.expanduser().resolve()
    if output_csv.exists() or output_manifest.exists():
        raise FileExistsError("Refusing to overwrite an existing Figure 7 overlay export")

    source_contract = json.loads(source_manifest.read_text(encoding="utf-8"))
    sources = pd.read_csv(sources_csv)
    missing = REQUIRED_COLUMNS - set(sources.columns)
    if missing:
        raise ValueError(f"Missing source-table columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    source_receipts: list[dict[str, Any]] = []
    loaded_samples = 0
    for row_index, row in sources.reset_index(drop=True).iterrows():
        state_path = Path(str(row["state_path"])).expanduser().resolve()
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        ka, kb, loaded = _extract_state(
            state_path=state_path,
            requested_samples=int(row["plotted_samples"]),
            row_index=row_index,
        )
        loaded_samples += loaded
        dataset = str(row["dataset"])
        frames.append(
            pd.DataFrame(
                {
                    "ka": ka,
                    "kb": kb,
                    "diameter_label": dataset.removeprefix("compression_").replace(
                        "um", " um"
                    ),
                    "phase": "phase1",
                    "source_label": str(row["source_label"]),
                }
            )
        )
        source_receipts.append(
            {
                "row_index": int(row_index),
                "state_path_hint": str(row["state_path"]),
                "state_size_bytes": state_path.stat().st_size,
                "state_sha256": _sha256(state_path),
                "loaded_samples": loaded,
                "plotted_samples": len(ka),
            }
        )

    overlay = pd.concat(frames, ignore_index=True)
    expected_loaded = int(source_contract.get("total_loaded_samples", loaded_samples))
    expected_plotted = int(source_contract.get("total_plotted_samples", len(overlay)))
    expected_sources = int(source_contract.get("source_count", len(sources)))
    observed_counts = {
        "sources": len(sources),
        "loaded_samples": loaded_samples,
        "plotted_samples": len(overlay),
    }
    expected_counts = {
        "sources": expected_sources,
        "loaded_samples": expected_loaded,
        "plotted_samples": expected_plotted,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Figure 7 overlay count mismatch: observed={observed_counts}, "
            f"expected={expected_counts}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as zipped:
            overlay.to_csv(zipped, index=False)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "source_manifest_hint": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "sources_csv_hint": str(sources_csv),
        "sources_csv_sha256": _sha256(sources_csv),
        "counts": observed_counts,
        "counts_by_diameter": {
            str(label): int(count)
            for label, count in overlay.groupby("diameter_label").size().items()
        },
        "output_csv": str(output_csv),
        "output_csv_size_bytes": output_csv.stat().st_size,
        "output_csv_sha256": _sha256(output_csv),
        "source_states": source_receipts,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-csv", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    receipt = export_overlay(
        sources_csv=args.sources_csv,
        source_manifest=args.source_manifest,
        output_csv=args.output_csv,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(receipt["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
