#!/usr/bin/env python3
"""Verify frozen UQ_EMB DNN artifacts and plan a documented refresh baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from replay_provenance import replay_receipt_provenance  # noqa: E402


PAPER_ID = "UQ_EMB"
DEPENDENCY_SET = "frozen-runtime-dependencies-202607"

CASES: tuple[dict[str, Any], ...] = (
    {
        "bubble_id": "d1",
        "agent": "definity",
        "diameter_um": 2.1,
        "modality": "compression",
        "data": "mechanical_surrogates/definity/2.1um/data/F_Delta.dat",
        "model": "mechanical_surrogates/definity/2.1um/trained/microbubble_force_BEST.pkl",
        "rows": 4200,
        "width": 128,
        "depth": 4,
        "batch_size": 128,
        "learning_rate": 5.0e-4,
    },
    {
        "bubble_id": "d2",
        "agent": "definity",
        "diameter_um": 2.9,
        "modality": "compression",
        "data": "mechanical_surrogates/definity/2.9um/data/F_Delta.dat",
        "model": "mechanical_surrogates/definity/2.9um/trained/microbubble_force_BEST.pkl",
        "rows": 4698,
        "width": 256,
        "depth": 3,
        "batch_size": 128,
        "learning_rate": 5.0e-4,
    },
    {
        "bubble_id": "d3",
        "agent": "definity",
        "diameter_um": 3.0,
        "modality": "compression",
        "data": "mechanical_surrogates/definity/3.0um/data/F_Delta.dat",
        "model": "mechanical_surrogates/definity/3.0um/trained/microbubble_force_BEST.pkl",
        "rows": 4271,
        "width": 64,
        "depth": 4,
        "batch_size": 128,
        "learning_rate": 5.0e-4,
    },
    {
        "bubble_id": "d4",
        "agent": "sonovue",
        "diameter_um": 3.2,
        "modality": "indentation",
        "data": "mechanical_surrogates/sonovue/3.2um/data/samples_all.dat",
        "model": "mechanical_surrogates/sonovue/3.2um/trained/microbubble_displacement_BEST.pkl",
        "rows": 11999,
        "width": 128,
        "depth": 3,
        "batch_size": 1024,
        "learning_rate": 5.0e-4,
    },
    {
        "bubble_id": "d5",
        "agent": "sonovue",
        "diameter_um": 3.4,
        "modality": "indentation",
        "data": "mechanical_surrogates/sonovue/3.4um/data/samples_all.dat",
        "model": "mechanical_surrogates/sonovue/3.4um/trained/microbubble_displacement_BEST.pkl",
        "rows": 25236,
        "width": 64,
        "depth": 5,
        "batch_size": 1024,
        "learning_rate": 5.0e-4,
    },
    {
        "bubble_id": "d6",
        "agent": "sonovue",
        "diameter_um": 5.8,
        "modality": "indentation",
        "data": "mechanical_surrogates/sonovue/5.8um/data/samples_all.dat",
        "model": "mechanical_surrogates/sonovue/5.8um/trained/microbubble_displacement_BEST.pkl",
        "rows": 12000,
        "width": 128,
        "depth": 3,
        "batch_size": 1024,
        "learning_rate": 5.0e-4,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hashes(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("paper_id") != PAPER_ID:
        raise ValueError(f"Unexpected paper_id in {path}")
    if payload.get("artifact_set_id") != DEPENDENCY_SET:
        raise ValueError(f"Unexpected artifact_set_id in {path}")
    if payload.get("locked") is not True:
        raise ValueError(f"Dependency manifest is not locked: {path}")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("Dependency manifest files must be a list")
    return {str(item["path"]): str(item["sha256"]) for item in files}


def _verify_file(root: Path, relative: str, expected: Mapping[str, str]) -> dict[str, Any]:
    path = root / relative
    if path.is_symlink():
        raise ValueError(f"Frozen DNN artifact cannot be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Frozen DNN artifact is missing: {path}")
    expected_hash = expected.get(relative)
    if expected_hash is None:
        raise ValueError(f"Frozen DNN artifact is absent from the locked manifest: {relative}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Frozen DNN artifact hash mismatch for {relative}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return {
        "path": str(path.resolve()),
        "relative_path": relative,
        "sha256": actual_hash,
        "size_bytes": path.stat().st_size,
    }


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _refresh_command(
    case: Mapping[str, Any],
    *,
    data_path: Path,
    repo_root: Path,
    output_root: Path,
    python_bin: str,
    seed: int,
    max_epoch: int,
) -> list[str]:
    modality = str(case["modality"])
    script = repo_root / "emb" / modality / "surrogate" / "scripts" / "emb_train.py"
    output = output_root / str(case["agent"]) / str(case["bubble_id"]) / "candidate_BEST.pkl"
    command = [
        python_bin,
        str(script.resolve()),
        str(data_path.resolve()),
        "--out",
        str(output.resolve()),
        "--report-path",
        str(output.with_suffix(".training.json").resolve()),
        "--width",
        str(case["width"]),
        "--depth",
        str(case["depth"]),
        "--batch-size",
        str(case["batch_size"]),
        "--lr",
        str(case["learning_rate"]),
        "--max-epoch",
        str(max_epoch),
        "--seed",
        str(seed),
    ]
    if modality == "indentation":
        command.extend(("--disp-source", "auto", "--rupture-ratio", "2.0"))
    return command


def audit(
    *,
    dependency_root: Path,
    manifest_path: Path,
    repo_root: Path,
    output_root: Path,
    python_bin: str,
    seed: int,
    max_epoch: int,
) -> dict[str, Any]:
    expected = _manifest_hashes(manifest_path)
    rows: list[dict[str, Any]] = []
    for case in CASES:
        data = _verify_file(dependency_root, str(case["data"]), expected)
        model = _verify_file(dependency_root, str(case["model"]), expected)
        line_count = _line_count(Path(data["path"]))
        if line_count != int(case["rows"]):
            raise ValueError(
                f"Unexpected training-table row count for {case['bubble_id']}: "
                f"expected {case['rows']}, got {line_count}"
            )
        command = _refresh_command(
            case,
            data_path=Path(data["path"]),
            repo_root=repo_root,
            output_root=output_root,
            python_bin=python_bin,
            seed=seed,
            max_epoch=max_epoch,
        )
        rows.append(
            {
                **case,
                "training_data": data,
                "accepted_model": model,
                "refresh_command": command,
                "refresh_command_shell": shlex.join(command),
            }
        )
    return {
        "schema_version": "1.0",
        "paper_id": PAPER_ID,
        "status": "PASS",
        "execution_provenance": replay_receipt_provenance(
            repo_root=repo_root,
            runner=Path(__file__),
            consumed_paths=[dependency_root],
        ),
        "accepted_artifacts": "immutable_verified",
        "exact_retraining": False,
        "retraining_limitation": (
            "The accepted random seeds and architecture-sweep receipts were not preserved. "
            "Refresh commands use the recovered accepted network shapes and explicit new seeds; "
            "they are prospective statistical-reproduction baselines, not byte-identical replays."
        ),
        "refresh_policy": {
            "seed": seed,
            "max_epoch": max_epoch,
            "site_behavior": "site-neutral command; activate the canonical MesoUQ environment on Karolina or Vega",
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify accepted UQ_EMB DNN artifacts and emit a prospective refresh plan."
    )
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--python-bin",
        required=True,
        help="Verified MesoUQ interpreter used in the prospective refresh commands.",
    )
    parser.add_argument("--seed", type=int, default=202607)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    payload = audit(
        dependency_root=args.dependency_root.resolve(),
        manifest_path=args.manifest.resolve(),
        repo_root=args.repo_root.resolve(),
        output_root=args.output_root.resolve(),
        python_bin=args.python_bin,
        seed=args.seed,
        max_epoch=args.max_epoch,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
