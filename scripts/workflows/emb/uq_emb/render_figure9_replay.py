#!/usr/bin/env python3
"""Render the frozen UQ_EMB Figure 9 from accepted direct-DPD artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True


SCHEMA_VERSION = "mesouq.uq_emb.figure9_replay.v1"
DEFINITY_RESULTS = {
    "2.1": "d1",
    "2.9": "d2",
    "3.0": "d3",
}
SONOVUE_RESULTS = {
    "3.2": "d4",
    "3.4": "d5",
    "5.8": "d6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raster_sha256(path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="uq-emb-f9-raster-") as temporary:
        output = Path(temporary) / "page"
        subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-r", "150", "-png", str(path), str(output)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _sha256(output.with_suffix(".png"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("uq_emb_frozen_figure9", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_runtime_results(
    accepted_root: Path, runtime_root: Path
) -> tuple[list[dict[str, str]], Path]:
    receipts: list[dict[str, str]] = []
    accepted_collection = accepted_root / "definity/direct_dpd/postprocessed"
    runtime_collection = runtime_root / "definity_collected"
    runtime_collection.mkdir(parents=True)
    for name in (
        "collection_manifest.json",
        "definity_direct_map_acoustic_comparison.csv",
    ):
        source = accepted_collection / name
        shutil.copy2(source, runtime_collection / name)
        receipts.append({"source": str(source), "sha256": _sha256(source)})

    definity_target = (
        runtime_collection / "renderer_root/definity_compression/map_mirheo/results"
    )
    definity_target.mkdir(parents=True)
    for diameter, bubble in DEFINITY_RESULTS.items():
        source = (
            accepted_root
            / f"definity/direct_dpd/mechanical/{bubble}/map_workflow/map_mirheo/results"
            / f"compression_{diameter}um_result.json"
        )
        target = definity_target / source.name
        shutil.copy2(source, target)
        receipts.append({"source": str(source), "sha256": _sha256(source)})

    sonovue_target = runtime_root / "sonovue/runs/force_spectroscopy"
    for diameter, bubble in SONOVUE_RESULTS.items():
        source = accepted_root / f"sonovue/direct_dpd/mechanical/{bubble}/result.json"
        target = sonovue_target / f"{diameter}um/result.json"
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        receipts.append({"source": str(source), "sha256": _sha256(source)})
    return receipts, runtime_collection


def render_figure9(
    *,
    renderer: Path,
    accepted_root: Path,
    plotting_root: Path,
    old_generator: Path,
    conversion_constants: Path,
    tex_bin_dir: Path,
    texdeps_dir: Path,
    output_root: Path,
    baseline_pdf: Path | None,
) -> dict:
    renderer = renderer.expanduser().resolve()
    accepted_root = accepted_root.expanduser().resolve()
    plotting_root = plotting_root.expanduser().resolve()
    old_generator = old_generator.expanduser().resolve()
    conversion_constants = conversion_constants.expanduser().resolve()
    tex_bin_dir = tex_bin_dir.expanduser().resolve()
    texdeps_dir = texdeps_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty Figure 9 output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = output_root / "_runtime"
    runtime_receipts, runtime_collection = _copy_runtime_results(
        accepted_root, runtime_root
    )
    figure_root = output_root / "figures"

    module = _load(renderer)
    conversion_payload = json.loads(conversion_constants.read_text(encoding="utf-8"))
    if conversion_payload.get("schema_version") != (
        "mesouq.uq_emb.figure9_conversion_constants.v1"
    ):
        raise ValueError(f"Unexpected Figure 9 conversion schema: {conversion_constants}")
    indentation_diameters = {
        str(key): float(value["initial_diameter_dpd"])
        for key, value in conversion_payload["diameters"].items()
    }
    module.DEFINITY_INPUTS = plotting_root / "inputs/figure7/renderer_inputs/definity"
    module.COLLECTED = runtime_collection
    module.BASE = accepted_root / "sonovue/direct_dpd/paper_facing/render_direct_map_confirmation_july.py"
    module.RETAINED_DEFINITY_ACOUSTIC = (
        module.DEFINITY_INPUTS / "tables/definity_resonance_observations.csv"
    )
    module.OUT = figure_root

    original_loader = module.load_renderer

    def load_base_renderer():
        base = original_loader()
        base.OLD_GENERATOR = old_generator
        base.TINYTEX_BIN = tex_bin_dir
        base.TEXDEPS_DIR = texdeps_dir
        base.MAP_ROOT = runtime_root
        base.SONO_DIRECT_ROOT = runtime_root / "sonovue"
        base.SONO_DIRECT_ACOUSTIC = (
            accepted_root
            / "sonovue/direct_dpd/evidence/forward_model_comparison"
            / "sonovue_direct_map_acoustic_comparison.csv"
        )
        base.SONO_HOME = plotting_root / "inputs/figure7/renderer_inputs/sonovue"

        original_old_loader = base._load_old_generator

        def load_old_generator():
            old = original_old_loader()

            def indentation_initial_diameter_dpd(diameter: str) -> float:
                try:
                    return indentation_diameters[str(diameter)]
                except KeyError as exc:
                    raise KeyError(
                        f"No frozen indentation conversion constant for {diameter}"
                    ) from exc

            old.indentation_initial_diameter_dpd = indentation_initial_diameter_dpd
            return old

        base._load_old_generator = load_old_generator
        return base

    module.load_renderer = load_base_renderer
    module.main()

    output_pdf = figure_root / "Figure_9_main_candidate_direct_map_confirmation.pdf"
    raster_sha = _raster_sha256(output_pdf)
    baseline = None
    if baseline_pdf is not None:
        baseline_pdf = baseline_pdf.expanduser().resolve()
        baseline_raster = _raster_sha256(baseline_pdf)
        baseline = {
            "path": str(baseline_pdf),
            "sha256": _sha256(baseline_pdf),
            "raster_sha256": baseline_raster,
            "raster_matches": baseline_raster == raster_sha,
        }
        if not baseline["raster_matches"]:
            raise ValueError(f"Figure 9 raster differs from baseline: {baseline}")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "renderer": {"path": str(renderer), "sha256": _sha256(renderer)},
        "base_renderer": {"path": str(module.BASE), "sha256": _sha256(module.BASE)},
        "old_generator": {"path": str(old_generator), "sha256": _sha256(old_generator)},
        "conversion_constants": {
            "path": str(conversion_constants),
            "sha256": _sha256(conversion_constants),
        },
        "runtime_inputs": runtime_receipts,
        "output": {
            "path": str(output_pdf),
            "sha256": _sha256(output_pdf),
            "raster_sha256": raster_sha,
        },
        "baseline": baseline,
    }
    (output_root / "uq_emb_figure9_replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--plotting-root", type=Path, required=True)
    parser.add_argument("--old-generator", type=Path, required=True)
    parser.add_argument("--conversion-constants", type=Path, required=True)
    parser.add_argument("--tex-bin-dir", type=Path, required=True)
    parser.add_argument("--texdeps-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-pdf", type=Path)
    args = parser.parse_args()
    receipt = render_figure9(
        renderer=args.renderer,
        accepted_root=args.accepted_root,
        plotting_root=args.plotting_root,
        old_generator=args.old_generator,
        conversion_constants=args.conversion_constants,
        tex_bin_dir=args.tex_bin_dir,
        texdeps_dir=args.texdeps_dir,
        output_root=args.output_root,
        baseline_pdf=args.baseline_pdf,
    )
    print(json.dumps(receipt["output"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
