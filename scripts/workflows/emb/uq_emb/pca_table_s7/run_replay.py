#!/usr/bin/env python3
"""Replay the accepted Table S7 PCA analysis from a staged case directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[4]
SOURCE_MANIFEST = SCRIPT_ROOT / "SOURCE_MANIFEST.json"
MAP_TABLE = REPO_ROOT / "papers/UQ_EMB/pca_table_s7/inputs/pca_map_6_cases.csv"
MODE_LIMIT = 300
# Accepted analytical MAP targets embedded in the six recovery-v2 selection
# receipts. They are used only by the optional independent subspace check.
ANALYTIC_MAP_FREQUENCIES_MHZ = {
    "d1": 3.1515848542769627,
    "d2": 3.2135165010943223,
    "d3": 3.0005214317087545,
    "d4": 2.2116766600634676,
    "d5": 2.0015807790290983,
    "d6": 1.2508003923783564,
}
REQUIRED_CASE_INPUTS = (
    "output/positions.xyz",
    "xyz0.xyz",
    "mesh/emb00001.off",
    "parameter/parameters-default00001.yaml",
    "parameter/parameters00001.yaml",
)
GENERATED_OUTPUTS = (
    "output/rmsfit.xyz",
    "output/ref1.xyz",
    "output/ref1_positions.txt",
    "output/eigvalues.txt",
    "output/eigvectors.txt",
    "eigvalues_new.txt",
    "eigvectors_new.txt",
    "pca_frequencies.txt",
    "individual_mode_audit_300_mode_scores.txt",
    "individual_mode_audit_300_volume_score.png",
    "individual_mode_audit_300_radial_diagnostics.png",
    "individual_mode_audit_300_combined_score.png",
    "individual_mode_audit_300_rms_volume.png",
    "individual_mode_frequencies.csv",
    "modes_pca.png",
    "breathing_subspace_mode_weights.csv",
    "provenance/breathing_subspace.json",
    "provenance/individual_breathing_mode.json",
    "provenance/pca_table_s7_replay_receipt.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preserved_sources() -> list[dict[str, str]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    verified = []
    for entry in manifest["files"]:
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"Preserved PCA source hash mismatch for {path}: "
                f"expected {entry['sha256']}, got {actual}"
            )
        verified.append({"path": entry["path"], "sha256": actual})
    return verified


def validate_case(case: Path) -> None:
    missing = [
        relative
        for relative in REQUIRED_CASE_INPUTS
        if not (case / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing PCA case inputs: " + ", ".join(missing))


def resolve_map_record(case: Path) -> tuple[dict[str, object], bool]:
    selected_map = case.parent / "selected_map_row.json"
    symbol = case.parent.name.lower()
    if selected_map.is_file():
        record = json.loads(selected_map.read_text())
        recorded_symbol = str(
            record.get("symbol", record.get("bubble_id", ""))
        ).lower()
        if recorded_symbol != symbol:
            raise ValueError(
                f"PCA case symbol mismatch: directory={symbol}, selected map={recorded_symbol}"
            )
        return record, False

    with MAP_TABLE.open(newline="") as handle:
        rows = {row["bubble_id"].lower(): row for row in csv.DictReader(handle)}
    if symbol not in rows:
        raise ValueError(f"Cannot resolve {symbol!r} from {MAP_TABLE}")
    row = rows[symbol]
    record: dict[str, object] = {
        "symbol": symbol,
        "agent": row["agent"].lower(),
        "diameter_um": float(row["diameter_um"]),
        "radius_dpd": float(row["radius_dpd"]),
        "ka_dpd": float(row["ka_dpd"]),
        "kb_dpd": float(row["kb_dpd"]),
        "mass_scale": float(row["mass_scale"]),
        "analytic_map_frequency_mhz": ANALYTIC_MAP_FREQUENCIES_MHZ[symbol],
        "analytic_reference_branch": "implemented Lim-shell vacuum l=0",
        "map_values_source": str(MAP_TABLE.relative_to(REPO_ROOT)),
        "map_values_sha256": sha256(MAP_TABLE),
    }
    return record, True


def build_commands(
    case: Path, python_bin: Path, cross_check: bool
) -> list[list[str]]:
    prefix = "individual_mode_audit_300"
    commands = [
        [
            str(python_bin),
            str(SCRIPT_ROOT / "all_analysis.py"),
            "--simnum",
            "00001",
        ],
        [str(python_bin), str(SCRIPT_ROOT / "eigenfreq2.py")],
        [
            str(python_bin),
            str(SCRIPT_ROOT / "identify_breathing_by_volume.py"),
            "--mesh",
            "mesh/emb00001.off",
            "--mean",
            "output/ref1_positions.txt",
            "--eigenvectors",
            "eigvectors_new.txt",
            "--eigenvalues",
            "eigvalues_new.txt",
            "--nmodes",
            str(MODE_LIMIT),
            "--top",
            "20",
            "--prefix",
            prefix,
        ],
        [
            str(python_bin),
            str(SCRIPT_ROOT / "select_individual_breathing_mode.py"),
            "--case",
            str(case),
            "--minimum-modes",
            str(MODE_LIMIT),
        ],
    ]
    if cross_check:
        commands.append(
            [
                str(python_bin),
                str(SCRIPT_ROOT / "breathing_subspace.py"),
                "--case",
                str(case),
                "--retained-eigenvectors",
                "eigvectors_new.txt",
                "--retained-eigenvalues",
                "eigvalues_new.txt",
            ]
        )
    return commands


def copy_prefix(source: Path, destination: Path, line_count: int) -> None:
    written = 0
    with source.open() as source_handle, destination.open("x") as destination_handle:
        for line in source_handle:
            if written == line_count:
                break
            destination_handle.write(line)
            written += 1
    if written != line_count:
        destination.unlink(missing_ok=True)
        raise ValueError(f"{source} contains {written} modes; expected at least {line_count}")


def run_command(
    command: list[str], case: Path, environment: dict[str, str]
) -> dict[str, object]:
    started = time.monotonic()
    subprocess.run(command, cwd=case, env=environment, check=True)
    return {"command": command, "elapsed_seconds": time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the accepted review2_v1 Table S7 PCA analysis. By default, "
            "validate inputs and print the execution plan without changing data."
        )
    )
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--cross-check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    case = args.case.resolve()
    python_bin = args.python_bin.resolve()
    if not python_bin.is_file():
        parser.error(f"Python executable does not exist: {python_bin}")

    validate_case(case)
    sources = verify_preserved_sources()
    map_record, map_will_be_created = resolve_map_record(case)
    commands = build_commands(case, python_bin, args.cross_check)
    plan = {
        "schema_version": "mesouq.uq_emb.pca_table_s7_replay.v1",
        "case": str(case),
        "mode_limit": MODE_LIMIT,
        "cross_check": args.cross_check,
        "source_files": sources,
        "selected_map": map_record,
        "selected_map_will_be_created": map_will_be_created,
        "commands": commands,
        "status": "ready",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    conflicts = [
        relative for relative in GENERATED_OUTPUTS if (case / relative).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite existing PCA replay outputs: " + ", ".join(conflicts)
        )

    if map_will_be_created:
        (case.parent / "selected_map_row.json").write_text(
            json.dumps(map_record, indent=2) + "\n"
        )

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    completed = []
    completed.append(run_command(commands[0], case, environment))
    copy_prefix(case / "output/eigvectors.txt", case / "eigvectors_new.txt", MODE_LIMIT)
    copy_prefix(case / "output/eigvalues.txt", case / "eigvalues_new.txt", MODE_LIMIT)
    reference_rows = case / "output/ref1.xyz"
    with reference_rows.open() as source_handle, (
        case / "output/ref1_positions.txt"
    ).open("x") as output_handle:
        for index, line in enumerate(source_handle):
            if index >= 2:
                fields = line.split()
                output_handle.write(" ".join(fields[-3:]) + "\n")
    for command in commands[1:]:
        completed.append(run_command(command, case, environment))

    selection_path = case / "provenance/individual_breathing_mode.json"
    receipt = dict(plan)
    receipt.update(
        {
            "status": "passed",
            "completed": completed,
            "selection": json.loads(selection_path.read_text())["selected"],
        }
    )
    receipt_path = case / "provenance/pca_table_s7_replay_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Wrote {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
