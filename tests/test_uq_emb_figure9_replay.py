from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPLAY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/workflows/emb/uq_emb/render_figure9_replay.py"
)
EXPORT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/workflows/emb/uq_emb/export_figure9_conversion_constants.py"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copy_runtime_results_builds_historical_renderer_layout(
    tmp_path: Path,
) -> None:
    module = _module(REPLAY_SCRIPT, "uq_emb_figure9_replay")
    accepted = tmp_path / "accepted"
    collection = accepted / "definity/direct_dpd/postprocessed"
    collection.mkdir(parents=True)
    (collection / "collection_manifest.json").write_text("{}\n", encoding="utf-8")
    (collection / "definity_direct_map_acoustic_comparison.csv").write_text(
        "bubble_id\nd1\n", encoding="utf-8"
    )
    for diameter, bubble in module.DEFINITY_RESULTS.items():
        source = (
            accepted
            / f"definity/direct_dpd/mechanical/{bubble}/map_workflow/map_mirheo/results"
            / f"compression_{diameter}um_result.json"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps({"bubble_id": bubble}), encoding="utf-8")
    for _diameter, bubble in module.SONOVUE_RESULTS.items():
        source = accepted / f"sonovue/direct_dpd/mechanical/{bubble}/result.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps({"bubble_id": bubble}), encoding="utf-8")

    receipts, runtime_collection = module._copy_runtime_results(
        accepted, tmp_path / "runtime"
    )

    assert len(receipts) == 8
    assert (runtime_collection / "collection_manifest.json").is_file()
    assert (
        runtime_collection
        / "renderer_root/definity_compression/map_mirheo/results"
        / "compression_2.9um_result.json"
    ).is_file()
    assert (
        tmp_path / "runtime/sonovue/runs/force_spectroscopy/5.8um/result.json"
    ).is_file()


def test_export_conversion_constants_uses_twice_column_median(tmp_path: Path) -> None:
    module = _module(EXPORT_SCRIPT, "uq_emb_figure9_conversion_export")
    legacy = tmp_path / "legacy"
    for index, diameter in enumerate(module.DIAMETERS):
        source = (
            legacy
            / f"indentation/surrogate/diameters/{diameter}um/data/samples_all.dat"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        samples = np.zeros((3, 8), dtype=float)
        samples[:, 7] = [1.0 + index, 2.0 + index, 10.0 + index]
        np.savetxt(source, samples)

    output = tmp_path / "constants.json"
    payload = module.export_constants(legacy_root=legacy, output=output)

    assert payload["diameters"]["3.2"]["initial_diameter_dpd"] == 4.0
    assert payload["diameters"]["3.4"]["initial_diameter_dpd"] == 6.0
    assert payload["diameters"]["5.8"]["initial_diameter_dpd"] == 8.0
    assert output.is_file()


def test_figure9_verifies_locked_inputs_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module(REPLAY_SCRIPT, "uq_emb_figure9_prewrite")
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
        module.render_figure9(
            renderer=tmp_path / "renderer.py",
            accepted_root=tmp_path / "accepted",
            plotting_root=tmp_path / "plotting",
            old_generator=tmp_path / "legacy.py",
            conversion_constants=tmp_path / "constants.json",
            tex_bin_dir=tmp_path / "tex-bin",
            texdeps_dir=tmp_path / "texdeps",
            output_root=output,
            baseline_pdf=None,
        )
    assert not output.exists()
