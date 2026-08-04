from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/workflows/emb/uq_emb/export_figure7_phase1_overlay.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("uq_emb_figure7_export", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_overlay_preserves_source_order_and_counts(tmp_path: Path) -> None:
    module = _module()
    state = {
        "Variables": [{"Name": "kb"}, {"Name": "ka"}],
        "Results": {
            "Posterior Sample Database": [
                [100.0, 1000.0],
                [200.0, 2000.0],
                [300.0, 3000.0],
            ]
        },
    }
    state_path = tmp_path / "latest"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    sources = tmp_path / "sources.csv"
    pd.DataFrame(
        [
            {
                "state_path": state_path,
                "source_label": "seed-00",
                "dataset": "compression_2.1um",
                "plotted_samples": 3,
            }
        ]
    ).to_csv(sources, index=False)
    contract = tmp_path / "manifest.json"
    contract.write_text(
        json.dumps(
            {"source_count": 1, "total_loaded_samples": 3, "total_plotted_samples": 3}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "overlay.csv.gz"
    receipt_path = tmp_path / "receipt.json"

    receipt = module.export_overlay(
        sources_csv=sources,
        source_manifest=contract,
        output_csv=output,
        output_manifest=receipt_path,
    )

    table = pd.read_csv(output)
    assert table[["ka", "kb"]].values.tolist() == [
        [1000.0, 100.0],
        [2000.0, 200.0],
        [3000.0, 300.0],
    ]
    assert set(table["diameter_label"]) == {"2.1 um"}
    assert receipt["counts"] == {
        "sources": 1,
        "loaded_samples": 3,
        "plotted_samples": 3,
    }
    assert receipt_path.is_file()
