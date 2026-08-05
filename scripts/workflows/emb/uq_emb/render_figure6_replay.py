#!/usr/bin/env python3
"""Render the frozen UQ_EMB Figure 6 acoustic-surrogate validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

from replay_provenance import (  # noqa: E402
    checked_replay_path,
    replay_receipt_provenance,
    require_output_outside_consumed_roots,
)


SCHEMA_VERSION = "mesouq.uq_emb.figure6_replay.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raster_sha256(path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="uq-emb-f6-raster-") as temporary:
        output = Path(temporary) / "page"
        subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-r", "150", "-png", str(path), str(output)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _sha256(output.with_suffix(".png"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("uq_emb_frozen_figure6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render_figure6(
    *,
    renderer: Path,
    paper_style: Path,
    paper_style_source: Path,
    rows: Path,
    tex_bin_dir: Path,
    texdeps_dir: Path,
    output_dir: Path,
    baseline_pdf: Path | None,
) -> dict:
    renderer = checked_replay_path(renderer, label="Figure 6 renderer")
    paper_style = checked_replay_path(paper_style, label="Figure 6 paper style")
    paper_style_source = checked_replay_path(
        paper_style_source, label="Figure 6 paper-style source"
    )
    rows = checked_replay_path(rows, label="Figure 6 rows")
    tex_bin_dir = checked_replay_path(tex_bin_dir, label="Figure 6 TeX binaries")
    texdeps_dir = checked_replay_path(texdeps_dir, label="Figure 6 TeX dependencies")
    baseline_pdf = (
        checked_replay_path(baseline_pdf, label="Figure 6 baseline")
        if baseline_pdf is not None
        else None
    )
    consumed_paths = [
        renderer,
        paper_style,
        paper_style_source,
        rows,
        tex_bin_dir,
        texdeps_dir,
        *([baseline_pdf] if baseline_pdf is not None else []),
    ]
    output_dir = require_output_outside_consumed_roots(
        output_path=output_dir,
        repo_root=REPO_ROOT,
        consumed_paths=consumed_paths,
    )
    execution_provenance = replay_receipt_provenance(
        repo_root=REPO_ROOT,
        runner=Path(__file__),
        consumed_paths=consumed_paths,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty Figure 6 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tex_bin_dir}:{old_path}"
    try:
        module = _load(renderer)
        module.MESOUQ_SNAPSHOT_SRC = paper_style_source
        module.PAPER_PLOT_SCRIPT = paper_style
        retained = module.retain_measurement_diameters(module.load_rows(rows))
        summary = module.summarize(retained)
        summary_path = output_dir / "acoustic_emulator_validation_summary.csv"
        output_stem = output_dir / "acoustic_emulator_validation"
        summary.to_csv(summary_path, index=False)

        original_loader = module.load_paper_plot_module

        def load_paper_style():
            paper = original_loader()
            paper.TEXDEPS_DIR = texdeps_dir
            return paper

        module.load_paper_plot_module = load_paper_style
        module.render(retained, summary, output_stem)
    finally:
        os.environ["PATH"] = old_path
    output_pdf = output_stem.with_suffix(".pdf")
    raster_sha = _raster_sha256(output_pdf)
    baseline = None
    if baseline_pdf is not None:
        baseline_raster = _raster_sha256(baseline_pdf)
        baseline = {
            "path": str(baseline_pdf),
            "sha256": _sha256(baseline_pdf),
            "raster_sha256": baseline_raster,
            "raster_matches": baseline_raster == raster_sha,
        }
        if not baseline["raster_matches"]:
            raise ValueError(f"Figure 6 raster differs from baseline: {baseline}")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "execution_provenance": execution_provenance,
        "renderer": {"path": str(renderer), "sha256": _sha256(renderer)},
        "paper_style": {"path": str(paper_style), "sha256": _sha256(paper_style)},
        "rows": {"path": str(rows), "sha256": _sha256(rows), "count": len(retained)},
        "output": {
            "path": str(output_pdf),
            "sha256": _sha256(output_pdf),
            "raster_sha256": raster_sha,
        },
        "baseline": baseline,
    }
    (output_dir / "uq_emb_figure6_replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--paper-style", type=Path, required=True)
    parser.add_argument("--paper-style-source", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--tex-bin-dir", type=Path, required=True)
    parser.add_argument("--texdeps-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-pdf", type=Path)
    args = parser.parse_args()
    receipt = render_figure6(
        renderer=args.renderer,
        paper_style=args.paper_style,
        paper_style_source=args.paper_style_source,
        rows=args.rows,
        tex_bin_dir=args.tex_bin_dir,
        texdeps_dir=args.texdeps_dir,
        output_dir=args.output_dir,
        baseline_pdf=args.baseline_pdf,
    )
    print(json.dumps(receipt["output"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
