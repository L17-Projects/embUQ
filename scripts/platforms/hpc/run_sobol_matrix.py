#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.platforms.site_selector import resolve_hpc_site  # noqa: E402
from meso_uq.hpc_paths import default_runs_root, detect_hpc_site  # noqa: E402

_SPECS = [
    {
        "name": "compression_2.1um",
        "modality": "compression",
        "diameter": "2.1",
        "dnn_model": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BEST.pkl",
        "bnn_model": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BNN.pt",
        "script": REPO_ROOT / "emb" / "compression" / "surrogate" / "sensitivity" / "scripts" / "run_sobol_vs_disp.py",
        "axis_flag": "--n-displacements",
        "output_name": "sobol_vs_disp_2.1um.csv",
    },
    {
        "name": "compression_2.9um",
        "modality": "compression",
        "diameter": "2.9",
        "dnn_model": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BEST.pkl",
        "bnn_model": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BNN.pt",
        "script": REPO_ROOT / "emb" / "compression" / "surrogate" / "sensitivity" / "scripts" / "run_sobol_vs_disp.py",
        "axis_flag": "--n-displacements",
        "output_name": "sobol_vs_disp_2.9um.csv",
    },
    {
        "name": "compression_3.0um",
        "modality": "compression",
        "diameter": "3.0",
        "dnn_model": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BEST.pkl",
        "bnn_model": REPO_ROOT / "emb" / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BNN.pt",
        "script": REPO_ROOT / "emb" / "compression" / "surrogate" / "sensitivity" / "scripts" / "run_sobol_vs_disp.py",
        "axis_flag": "--n-displacements",
        "output_name": "sobol_vs_disp_3.0um.csv",
    },
    {
        "name": "indentation_3.2um",
        "modality": "indentation",
        "diameter": "3.2",
        "dnn_model": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BEST.pkl",
        "bnn_model": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BNN.pt",
        "script": REPO_ROOT / "emb" / "indentation" / "surrogate" / "sensitivity" / "scripts" / "run_sobol_vs_force.py",
        "axis_flag": "--n-forces",
        "output_name": "sobol_vs_force_3.2um.csv",
    },
    {
        "name": "indentation_3.4um",
        "modality": "indentation",
        "diameter": "3.4",
        "dnn_model": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BEST.pkl",
        "bnn_model": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BNN.pt",
        "script": REPO_ROOT / "emb" / "indentation" / "surrogate" / "sensitivity" / "scripts" / "run_sobol_vs_force.py",
        "axis_flag": "--n-forces",
        "output_name": "sobol_vs_force_3.4um.csv",
    },
    {
        "name": "indentation_5.8um",
        "modality": "indentation",
        "diameter": "5.8",
        "dnn_model": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BEST.pkl",
        "bnn_model": REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BNN.pt",
        "script": REPO_ROOT / "emb" / "indentation" / "surrogate" / "sensitivity" / "scripts" / "run_sobol_vs_force.py",
        "axis_flag": "--n-forces",
        "output_name": "sobol_vs_force_5.8um.csv",
    },
]


def _build_command(
    *,
    python_bin: str,
    spec: dict[str, object],
    family: str,
    output_root: Path,
    n_samples: int,
    n_axis_points: int,
    predictive_mc_samples: int,
    predictive_mc_chunk_size: int,
    device: str,
) -> tuple[list[str], Path]:
    model_path = Path(spec[f"{family}_model"])  # type: ignore[index]
    output_dir = output_root / str(spec["modality"]) / family
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(spec["output_name"])
    command = [
        python_bin,
        str(spec["script"]),
        "--model",
        str(model_path),
        "--surrogate-family",
        family,
        "--output",
        str(output_path),
        "--n-samples",
        str(n_samples),
        str(spec["axis_flag"]),
        str(n_axis_points),
        "--predictive-mc-samples",
        str(predictive_mc_samples),
        "--predictive-mc-chunk-size",
        str(predictive_mc_chunk_size),
        "--device",
        str(device),
    ]
    return command, output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Sobol sensitivity matrix across modalities/diameters/families."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument(
        "--surrogate-family",
        action="append",
        choices=["dnn", "bnn"],
        default=[],
        help="Repeat to select families. Default runs both dnn and bnn.",
    )
    parser.add_argument("--only", action="append", default=[], help="Restrict to spec names.")
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-axis-points", type=int, default=20)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--predictive-mc-chunk-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-missing-model", dest="skip_missing_model", action="store_true", default=True)
    parser.add_argument("--no-skip-missing-model", dest="skip_missing_model", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    args = parser.parse_args(argv)

    selected_specs = list(_SPECS)
    if args.only:
        allowed = set(args.only)
        selected_specs = [spec for spec in selected_specs if str(spec["name"]) in allowed]
        if not selected_specs:
            raise ValueError(f"No specs matched --only values: {sorted(allowed)}")

    families = args.surrogate_family if args.surrogate_family else ["dnn", "bnn"]
    resolved_site = resolve_hpc_site(cli_site=args.site, env=os.environ, allow_hostname=True, default="vega")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else default_runs_root(REPO_ROOT, "sobol", site=resolved_site, run_tag=args.run_tag)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for spec in selected_specs:
        for family in families:
            model_path = Path(spec[f"{family}_model"])  # type: ignore[index]
            if not model_path.exists():
                row = {
                    "name": spec["name"],
                    "modality": spec["modality"],
                    "diameter_um": spec["diameter"],
                    "surrogate_family": family,
                    "status": "skipped_missing_model",
                    "model_path": str(model_path),
                }
                report_rows.append(row)
                if not args.skip_missing_model:
                    failures.append(row)
                    if not args.continue_on_error:
                        break
                continue

            command, output_path = _build_command(
                python_bin=args.python_bin,
                spec=spec,
                family=family,
                output_root=output_root,
                n_samples=args.n_samples,
                n_axis_points=args.n_axis_points,
                predictive_mc_samples=args.predictive_mc_samples,
                predictive_mc_chunk_size=args.predictive_mc_chunk_size,
                device=args.device,
            )
            print(f"[sobol] running: {' '.join(command)}")
            proc = subprocess.run(command, cwd=str(REPO_ROOT))
            row = {
                "name": spec["name"],
                "modality": spec["modality"],
                "diameter_um": spec["diameter"],
                "surrogate_family": family,
                "status": "passed" if proc.returncode == 0 else "failed",
                "returncode": int(proc.returncode),
                "output_csv": str(output_path),
            }
            report_rows.append(row)
            if proc.returncode != 0:
                failures.append(row)
                if not args.continue_on_error:
                    break
        if failures and not args.continue_on_error:
            break

    report_df = pd.DataFrame(report_rows)
    report_csv = output_root / "sobol_matrix_report.csv"
    report_json = output_root / "sobol_matrix_report.json"
    report_df.to_csv(report_csv, index=False)
    report_json.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")
    print(f"Wrote report CSV: {report_csv}")
    print(f"Wrote report JSON: {report_json}")

    if failures:
        print(f"Encountered {len(failures)} failures.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
