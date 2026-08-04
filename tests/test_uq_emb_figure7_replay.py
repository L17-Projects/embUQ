from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/workflows/emb/uq_emb/render_figure7_replay.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("uq_emb_figure7_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_overlay_checks_manifest_count(tmp_path: Path) -> None:
    module = _module()
    overlay = tmp_path / "overlay.csv.gz"
    pd.DataFrame(
        {
            "ka": [1000.0, 2000.0],
            "kb": [100.0, 200.0],
            "diameter_label": ["2.1 um", "2.1 um"],
            "phase": ["phase1", "phase1"],
            "source_label": ["seed-00", "seed-00"],
        }
    ).to_csv(overlay, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "counts": {"sources": 1, "loaded_samples": 2, "plotted_samples": 2},
                "counts_by_diameter": {"2.1 um": 2},
            }
        ),
        encoding="utf-8",
    )

    table, report = module._load_overlay(overlay, manifest)

    assert len(table) == 2
    assert report["plotted_samples"] == 2
    assert report["counts_by_diameter"] == {"2.1 um": 2}
