#!/usr/bin/env python3
"""Render the frozen UQ_EMB Figure 7 from portable, checksummed inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

from replay_provenance import replay_receipt_provenance  # noqa: E402


SCHEMA_VERSION = "mesouq.uq_emb.figure7_replay.v1"
FROZEN_BINS = {
    "definity_ka": 192,
    "definity_kb": 256,
    "sonovue_ka": 256,
    "sonovue_kb": 320,
}
RENDERER_FILES = {
    "top": "render_figure7_with_posteriors.py",
    "variant": "render_figure7_attempt081_variants.py",
    "base": "render_revision_draft_assets.py",
}
REQUIRED_OVERLAY_COLUMNS = {"ka", "kb", "diameter_label", "phase", "source_label"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raster_sha256(path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="uq-emb-f7-raster-") as temporary:
        output = Path(temporary) / "page"
        subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-r", "150", "-png", str(path), str(output)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _sha256(output.with_suffix(".png"))


def _load_renderer(code_root: Path):
    paths = {name: code_root / filename for name, filename in RENDERER_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Figure 7 renderer files: {missing}")

    original_spec = importlib.util.spec_from_file_location
    redirects = {
        paths["variant"].name: paths["variant"],
        paths["base"].name: paths["base"],
    }

    def redirected_spec(name: str, location: str | Path, *args, **kwargs):
        redirected = redirects.get(Path(location).name, Path(location))
        return original_spec(name, redirected, *args, **kwargs)

    importlib.util.spec_from_file_location = redirected_spec
    try:
        spec = original_spec("uq_emb_frozen_figure7", paths["top"])
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {paths['top']}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        importlib.util.spec_from_file_location = original_spec
    return module, paths


def _load_overlay(path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    overlay = pd.read_csv(path)
    missing = REQUIRED_OVERLAY_COLUMNS - set(overlay.columns)
    if missing:
        raise ValueError(f"Missing flattened Phase-1 columns: {sorted(missing)}")
    numeric = overlay[["ka", "kb"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Flattened Phase-1 overlay contains non-finite ka/kb values")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = int(manifest["counts"]["plotted_samples"])
    if len(overlay) != expected:
        raise ValueError(f"Phase-1 overlay count mismatch: {len(overlay)} != {expected}")
    return overlay, {
        "flat_overlay_csv": str(path),
        "flat_overlay_sha256": _sha256(path),
        "flat_overlay_manifest": str(manifest_path),
        "flat_overlay_manifest_sha256": _sha256(manifest_path),
        "source_count": int(manifest["counts"]["sources"]),
        "loaded_samples": int(manifest["counts"]["loaded_samples"]),
        "plotted_samples": expected,
        "counts_by_diameter": manifest["counts_by_diameter"],
    }


def render_figure7(
    *,
    code_root: Path,
    renderer_inputs: Path,
    phase1_overlay: Path,
    phase1_manifest: Path,
    output_dir: Path,
    tex_bin_dir: Path | None,
    texdeps_dir: Path | None,
    baseline_pdf: Path | None,
) -> dict[str, Any]:
    code_root = code_root.expanduser().resolve()
    renderer_inputs = renderer_inputs.expanduser().resolve()
    phase1_overlay = phase1_overlay.expanduser().resolve()
    phase1_manifest = phase1_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty Figure 7 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    definity = renderer_inputs / "definity"
    sonovue = renderer_inputs / "sonovue"
    sonovue_heavy = renderer_inputs / "sonovue_heavy"
    environment = {
        "JCP_RESONANCE_DEFINITY_HOME": str(definity),
        "JCP_RESONANCE_SONOVUE_HOME": str(sonovue),
        "JCP_RESONANCE_SONOVUE_HEAVY_TABLES": str(sonovue_heavy),
        "JCP_RESONANCE_DEFINITY_POSTERIOR_DENSITY_TABLE": str(
            definity / "tables/posterior_scatter_samples.csv"
        ),
        "JCP_RESONANCE_SONOVUE_POSTERIOR_DENSITY_TABLE": str(
            sonovue_heavy / "posterior_scatter_samples.csv"
        ),
        "JCP_RESONANCE_FIG7_POSTERIOR_DENSITY_SAMPLES": "50000",
        "JCP_RESONANCE_FIG7_DEFINITY_KA_MAX": "34500",
        "JCP_RESONANCE_FIGURE_INPUT_STATUS": "frozen combined 50k rendering replay",
    }
    old_environment = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    try:
        renderer, renderer_paths = _load_renderer(code_root)
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    if tex_bin_dir is not None:
        renderer.base.TINYTEX_BIN = tex_bin_dir.expanduser().resolve()
    if texdeps_dir is not None:
        renderer.base.TEXDEPS_DIR = texdeps_dir.expanduser().resolve()

    overlay, overlay_report = _load_overlay(phase1_overlay, phase1_manifest)

    def read_flat_overlay():
        return overlay.copy(), dict(overlay_report)

    renderer.fig7._read_definity_full_phase1_overlay = read_flat_overlay
    renderer.fig7.FIG7_DEFINITY_21UM_COLOR = "#AEB9C8"
    renderer.base.DIAMETER_PALETTES["Definity"] = {
        "2.1 um": "#AEB9C8",
        "2.9 um": "#566F91",
        "3.0 um": "#16233D",
    }
    renderer.base.DIAMETER_PALETTES["SonoVue"] = {
        "3.2 um": "#F0C44F",
        "3.4 um": "#D99500",
        "5.8 um": "#A96800",
    }

    original_inside_label = renderer._inside_label

    def inside_label(ax, label, *args, **kwargs):
        return original_inside_label(ax, label, *args, **kwargs)

    renderer._inside_label = inside_label

    def comma_tick(value: float, _position: int) -> str:
        if np.isclose(value, round(value), atol=1.0e-9):
            return f"{int(round(value)):,}"
        return f"{value:,.2f}".rstrip("0").rstrip(".")

    def save_with_padding(fig):
        renderer.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        renderer.VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
        png_path = renderer.FIGURES_DIR / f"{renderer.OUT_STEM}.png"
        pdf_path = renderer.FIGURES_DIR / f"{renderer.OUT_STEM}.pdf"
        variant_png = renderer.VARIANTS_DIR / f"{renderer.VARIANT_STEM}.png"
        variant_pdf = renderer.VARIANTS_DIR / f"{renderer.VARIANT_STEM}.pdf"
        formatter = FuncFormatter(comma_tick)
        for axis in fig.axes:
            x0, x1 = axis.get_xlim()
            y0, y1 = axis.get_ylim()
            if max(abs(x0), abs(x1)) >= 1000:
                axis.xaxis.set_major_formatter(formatter)
            if max(abs(y0), abs(y1)) >= 1000:
                axis.yaxis.set_major_formatter(formatter)
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
        renderer.plt.close(fig)
        shutil.copy2(png_path, variant_png)
        shutil.copy2(pdf_path, variant_pdf)
        return {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "variant_pdf": str(variant_pdf),
            "variant_png": str(variant_png),
            "pdf_sha256": renderer.base._sha256(pdf_path),
            "png_sha256": renderer.base._sha256(png_path),
            "variant_pdf_sha256": renderer.base._sha256(variant_pdf),
            "variant_png_sha256": renderer.base._sha256(variant_png),
            "pdf_page_size": renderer.base._pdf_page_size(str(pdf_path)),
            "pdf_bytes": str(pdf_path.stat().st_size),
            "png_bytes": str(png_path.stat().st_size),
        }

    renderer._save = save_with_padding
    renderer.FIGURES_DIR = output_dir
    renderer.VARIANTS_DIR = output_dir / "jun30_variants"
    renderer.COMPILE_DIR = output_dir / "_disabled_compile"
    renderer.BUNDLE_DIR = output_dir / "_disabled_bundle"
    renderer.DEFAULT_PROJECTION_BINS = FROZEN_BINS["definity_ka"]
    renderer.DEFI_KB_PROJECTION_BINS = FROZEN_BINS["definity_kb"]
    renderer.SONOVUE_KA_PROJECTION_BINS = FROZEN_BINS["sonovue_ka"]
    renderer.SONOVUE_KB_PROJECTION_BINS = FROZEN_BINS["sonovue_kb"]

    result = renderer.render()
    output_pdf = Path(result["pdf"])
    raster_sha = _raster_sha256(output_pdf)
    baseline = None
    if baseline_pdf is not None:
        baseline_pdf = baseline_pdf.expanduser().resolve()
        baseline_raster = _raster_sha256(baseline_pdf)
        baseline = {
            "path": str(baseline_pdf),
            "sha256": _sha256(baseline_pdf),
            "raster_sha256": baseline_raster,
            "raster_matches": raster_sha == baseline_raster,
        }
        if not baseline["raster_matches"]:
            raise ValueError(f"Figure 7 raster differs from baseline: {baseline}")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "execution_provenance": replay_receipt_provenance(
            repo_root=REPO_ROOT,
            runner=Path(__file__),
            consumed_paths=[
                code_root,
                renderer_inputs,
                phase1_overlay,
                phase1_manifest,
                *([tex_bin_dir] if tex_bin_dir is not None else []),
                *([texdeps_dir] if texdeps_dir is not None else []),
                *([baseline_pdf] if baseline_pdf is not None else []),
            ],
        ),
        "renderer_scripts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in renderer_paths.items()
        },
        "renderer_inputs": str(renderer_inputs),
        "phase1_overlay": overlay_report,
        "projection_bins": FROZEN_BINS,
        "output": {
            "path": str(output_pdf),
            "sha256": _sha256(output_pdf),
            "raster_sha256": raster_sha,
        },
        "baseline": baseline,
    }
    (output_dir / "uq_emb_figure7_replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--renderer-inputs", type=Path, required=True)
    parser.add_argument("--phase1-overlay", type=Path, required=True)
    parser.add_argument("--phase1-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tex-bin-dir", type=Path)
    parser.add_argument("--texdeps-dir", type=Path)
    parser.add_argument("--baseline-pdf", type=Path)
    args = parser.parse_args()
    receipt = render_figure7(
        code_root=args.code_root,
        renderer_inputs=args.renderer_inputs,
        phase1_overlay=args.phase1_overlay,
        phase1_manifest=args.phase1_manifest,
        output_dir=args.output_dir,
        tex_bin_dir=args.tex_bin_dir,
        texdeps_dir=args.texdeps_dir,
        baseline_pdf=args.baseline_pdf,
    )
    print(json.dumps(receipt["output"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
