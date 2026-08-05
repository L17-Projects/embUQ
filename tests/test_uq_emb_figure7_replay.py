from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


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


def test_figure7_verifies_locked_inputs_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "require_output_outside_consumed_roots",
        lambda *, output_path, **_kwargs: output_path.resolve(),
    )

    def reject_before_write(**_kwargs):
        assert not output.exists()
        raise ValueError("locked root drift")

    monkeypatch.setattr(module, "replay_receipt_provenance", reject_before_write)
    with pytest.raises(ValueError, match="locked root drift"):
        module.render_figure7(
            code_root=tmp_path / "code",
            renderer_inputs=tmp_path / "inputs",
            phase1_overlay=tmp_path / "overlay.csv",
            phase1_manifest=tmp_path / "overlay.json",
            output_dir=output,
            tex_bin_dir=None,
            texdeps_dir=None,
            baseline_pdf=None,
        )
    assert not output.exists()
