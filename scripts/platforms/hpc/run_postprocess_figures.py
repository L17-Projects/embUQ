#!/usr/bin/env python3
"""Run all postprocessing steps for a completed surrogate validation run.

Steps:
  1. Holdout L2 figure
  2. Sensitivity figure
  3. Family comparison + BNN insertion decision
  4. UQ_DPD parity report
  5. Final report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SITE_HELPER_DIR = Path(__file__).resolve().parent
if str(_SITE_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_SITE_HELPER_DIR))
from _site_cli import add_site_argument, validate_site_argument

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.campaign_manifests import (  # noqa: E402
    MANDATORY_MAIN_FIGURES,
    MANDATORY_SUPPLEMENTARY_FIGURES,
    MANDATORY_TABLES,
    build_paper_release_manifest,
    build_required_asset_entries,
    derive_asset_source_map,
    load_asset_source_map,
    utc_now_iso,
    write_manifest,
)

POSTPROCESS = REPO_ROOT / "scripts" / "postprocess"

UQDPD_REFERENCE_CSV = (
    REPO_ROOT.parent
    / "UQ_DPD"
    / "Hierarchical_UQ_compression_dev"
    / "_paper"
    / "v4"
    / "generated"
    / "figures"
    / "surrogate_group_holdout_summary.csv"
)


def _run(cmd: list[str], label: str) -> int:
    print(f"\n[postprocess] {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  FAILED (returncode={result.returncode})", file=sys.stderr)
    return result.returncode


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_lane_manifest_paths(items: list[str], manifests_dir: str | None) -> list[Path]:
    paths = [_resolve_path(item) for item in items]
    if manifests_dir:
        root = _resolve_path(manifests_dir)
        if root.exists():
            paths.extend(sorted(root.glob("*.json")))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _emit_release_manifest(
    *,
    run_campaign_id: str,
    figures_main_root: Path,
    figures_supplementary_root: Path,
    tables_root: Path,
    lane_manifest_paths: list[Path],
    asset_source_map_path: Path | None,
    manifest_output: Path,
) -> dict[str, object]:
    source_map_overrides = load_asset_source_map(asset_source_map_path)
    lane_payloads = []
    for lane_manifest_path in lane_manifest_paths:
        if lane_manifest_path.exists():
            lane_payloads.append(
                json.loads(lane_manifest_path.read_text(encoding="utf-8"))
            )
    required_relative_paths = [
        *[f"figures/main/{name}" for name in MANDATORY_MAIN_FIGURES],
        *[f"figures/supplementary/{name}" for name in MANDATORY_SUPPLEMENTARY_FIGURES],
        *[f"tables/{name}" for name in MANDATORY_TABLES],
    ]
    source_map = derive_asset_source_map(
        required_relative_paths=required_relative_paths,
        lane_manifests=lane_payloads,
    )
    source_map.update(source_map_overrides)
    manifest_payload = build_paper_release_manifest(
        run_campaign_id=run_campaign_id,
        generated_at_utc=utc_now_iso(),
        lane_manifest_paths=lane_manifest_paths,
        figures_main_entries=build_required_asset_entries(
            category="figures/main",
            root=figures_main_root,
            required_files=MANDATORY_MAIN_FIGURES,
            source_map=source_map,
        ),
        figures_supplementary_entries=build_required_asset_entries(
            category="figures/supplementary",
            root=figures_supplementary_root,
            required_files=MANDATORY_SUPPLEMENTARY_FIGURES,
            source_map=source_map,
        ),
        table_entries=build_required_asset_entries(
            category="tables",
            root=tables_root,
            required_files=MANDATORY_TABLES,
            source_map=source_map,
        ),
    )
    write_manifest(manifest_output, manifest_payload)
    print(f"Release manifest: {manifest_output}")
    print(f"Release status: {manifest_payload['release_status']}")
    if manifest_payload["hard_failures"]:
        print("Hard failures:")
        for entry in manifest_payload["hard_failures"]:
            print(f"  - {entry}")
    return manifest_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Run root created by the pipeline.")
    parser.add_argument("--metric", default="median", choices=["mean", "median", "max"])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--uqdpd-reference-csv",
        default=str(UQDPD_REFERENCE_CSV),
        help="Path to UQ_DPD surrogate_group_holdout_summary.csv reference.",
    )
    parser.add_argument(
        "--emit-release-manifest",
        action="store_true",
        default=False,
        help="Emit paper_release_manifest.json after figure/table postprocessing.",
    )
    parser.add_argument(
        "--run-campaign-id",
        default=None,
        help="Optional release campaign id. Defaults to run-root directory name.",
    )
    parser.add_argument(
        "--figures-main-root",
        default=None,
        help="Root directory for main figures (default: <run-root>/figures/main).",
    )
    parser.add_argument(
        "--figures-supplementary-root",
        default=None,
        help="Root directory for supplementary figures (default: <run-root>/figures/supplementary).",
    )
    parser.add_argument(
        "--tables-root",
        default=None,
        help="Root directory for tables (default: <run-root>/tables).",
    )
    parser.add_argument(
        "--lane-manifest",
        action="append",
        default=[],
        help="Path to a lane manifest JSON. Can be passed multiple times.",
    )
    parser.add_argument(
        "--lane-manifests-dir",
        default=None,
        help="Optional directory containing lane manifest JSON files.",
    )
    parser.add_argument(
        "--asset-source-map",
        default=None,
        help="Optional JSON map of required assets to source lane/artifacts.",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Output path for paper release manifest (default: <run-root>/manifests/paper_release_manifest.json).",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        default=False,
        help="Skip legacy postprocess figure/table steps and emit only the release manifest.",
    )
    add_site_argument(parser)
    args = parser.parse_args(argv)
    validate_site_argument(parser, args.site)

    run_root = Path(args.run_root).resolve()
    py = args.python_bin

    if not args.manifest_only:
        holdout_input = run_root / "group_holdout"
        sobol_input = run_root / "sobol"
        figures_holdout = run_root / "figures" / "holdout_l2"
        figures_sobol = run_root / "figures" / "sensitivity"
        comparison_dir = run_root / "comparison"
        holdout_l2_csv = figures_holdout / "surrogate_holdout_l2_summary.csv"

        rc = _run(
            [py, str(POSTPROCESS / "generate_surrogate_holdout_l2_figure.py"),
             "--input-root", str(holdout_input),
             "--output-dir", str(figures_holdout),
             "--metric", args.metric],
            "Holdout L2 figure",
        )
        if rc != 0:
            return rc

        rc = _run(
            [py, str(POSTPROCESS / "generate_surrogate_sensitivity_figure.py"),
             "--input-root", str(sobol_input),
             "--output-dir", str(figures_sobol),
             "--index-type", "ST",
             "--parameters", "Yt", "kb", "b1", "b2", "a3", "a4"],
            "Sensitivity figure",
        )
        if rc != 0:
            return rc

        rc = _run(
            [py, str(POSTPROCESS / "generate_surrogate_comparison.py"),
             "--input-csv", str(holdout_l2_csv),
             "--output-dir", str(comparison_dir)],
            "Family comparison + BNN decision",
        )
        # Rollout gate: stop immediately when BNN insertion is NO-GO.
        if rc != 0:
            return rc

        rc = _run(
            [py, str(POSTPROCESS / "generate_uqdpd_parity_report.py"),
             "--mesouq-summary-csv", str(holdout_l2_csv),
             "--uqdpd-reference-csv", args.uqdpd_reference_csv,
             "--output-dir", str(comparison_dir)],
            "UQ_DPD parity report",
        )
        if rc != 0:
            return rc

        rc = _run(
            [py, str(POSTPROCESS / "generate_final_report.py"),
             "--run-root", str(run_root),
             "--output-dir", str(comparison_dir)],
            "Final report",
        )
        if rc != 0:
            return rc
    elif not args.emit_release_manifest:
        raise ValueError("--manifest-only requires --emit-release-manifest")
    if args.emit_release_manifest:
        figures_main_root = (
            _resolve_path(args.figures_main_root)
            if args.figures_main_root is not None
            else (run_root / "figures" / "main")
        )
        figures_supplementary_root = (
            _resolve_path(args.figures_supplementary_root)
            if args.figures_supplementary_root is not None
            else (run_root / "figures" / "supplementary")
        )
        tables_root = (
            _resolve_path(args.tables_root)
            if args.tables_root is not None
            else (run_root / "tables")
        )
        lane_manifest_paths = _resolve_lane_manifest_paths(
            args.lane_manifest, args.lane_manifests_dir
        )
        asset_source_map_path = (
            _resolve_path(args.asset_source_map) if args.asset_source_map is not None else None
        )
        manifest_output = (
            _resolve_path(args.manifest_output)
            if args.manifest_output is not None
            else (run_root / "manifests" / "paper_release_manifest.json")
        )
        manifest_payload = _emit_release_manifest(
            run_campaign_id=args.run_campaign_id or run_root.name,
            figures_main_root=figures_main_root,
            figures_supplementary_root=figures_supplementary_root,
            tables_root=tables_root,
            lane_manifest_paths=lane_manifest_paths,
            asset_source_map_path=asset_source_map_path,
            manifest_output=manifest_output,
        )
        if manifest_payload["release_status"] != "PASS":
            return 1
    print("\nAll postprocessing steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
