from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


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
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    state_path = input_root / "latest"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    sources = input_root / "sources.csv"
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
    contract = input_root / "manifest.json"
    contract.write_text(
        json.dumps(
            {"source_count": 1, "total_loaded_samples": 3, "total_plotted_samples": 3}
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    output = output_root / "overlay.csv.gz"
    receipt_path = output_root / "receipt.json"

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


@pytest.mark.parametrize("target_kind", ("csv", "manifest"))
def test_export_overlay_rejects_output_inside_consumed_source_root(
    tmp_path: Path, target_kind: str
) -> None:
    module = _module()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    state_path = input_root / "states" / "latest"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps(
            {
                "Variables": [{"Name": "ka"}, {"Name": "kb"}],
                "Results": {"Posterior Sample Database": [[1000.0, 100.0]]},
            }
        ),
        encoding="utf-8",
    )
    sources = input_root / "sources.csv"
    pd.DataFrame(
        [
            {
                "state_path": state_path,
                "source_label": "seed-00",
                "dataset": "compression_2.1um",
                "plotted_samples": 1,
            }
        ]
    ).to_csv(sources, index=False)
    contract = input_root / "manifest.json"
    contract.write_text(
        json.dumps(
            {"source_count": 1, "total_loaded_samples": 1, "total_plotted_samples": 1}
        ),
        encoding="utf-8",
    )
    outside = tmp_path / "outputs"
    output_csv = input_root / "forbidden.csv.gz" if target_kind == "csv" else outside / "overlay.csv.gz"
    output_manifest = input_root / "forbidden.json" if target_kind == "manifest" else outside / "receipt.json"

    with pytest.raises(ValueError, match="consumed Figure 7 source root"):
        module.export_overlay(
            sources_csv=sources,
            source_manifest=contract,
            output_csv=output_csv,
            output_manifest=output_manifest,
        )

    assert not output_csv.exists()
    assert not output_manifest.exists()


def test_export_overlay_rejects_identical_output_paths(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "overlay"

    with pytest.raises(ValueError, match="outputs must not overlap"):
        module.export_overlay(
            sources_csv=tmp_path / "sources.csv",
            source_manifest=tmp_path / "manifest.json",
            output_csv=output,
            output_manifest=output,
        )

    assert not output.exists()


def test_export_overlay_rejects_ancestor_output_paths(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "overlay"

    with pytest.raises(ValueError, match="outputs must not overlap"):
        module.export_overlay(
            sources_csv=tmp_path / "sources.csv",
            source_manifest=tmp_path / "manifest.json",
            output_csv=output,
            output_manifest=output / "receipt.json",
        )

    assert not output.exists()


def test_export_overlay_rejects_reverse_ancestor_output_paths(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "overlay"

    with pytest.raises(ValueError, match="outputs must not overlap"):
        module.export_overlay(
            sources_csv=tmp_path / "sources.csv",
            source_manifest=tmp_path / "manifest.json",
            output_csv=output / "overlay.csv.gz",
            output_manifest=output,
        )

    assert not output.exists()
