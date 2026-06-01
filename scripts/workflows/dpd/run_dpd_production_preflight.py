#!/usr/bin/env python3
"""Run DPD production preflight checks before launching Mirheo candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


def _configure_python_path(repo_root: Path) -> None:
    src = repo_root / "src"
    for entry in (str(repo_root), str(src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--output-root")
    source.add_argument("--candidate-manifest")
    parser.add_argument("--manifest-path")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--allow-non-scratch", action="store_true")
    parser.add_argument(
        "--scratch-root",
        action="append",
        default=[],
        help="Allowed scratch root. May be passed more than once. Defaults to MESOUQ_SCRATCH_ROOT, SCRATCH_ROOT, or /scratch.",
    )
    parser.add_argument("--min-free-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--hdf5-smoke-test", action="store_true")
    parser.add_argument("--hdf5-driver", choices=("serial", "mpio"), default="serial")
    return parser


def _root_policy_failed(payload: dict[str, object]) -> bool:
    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        return False
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("name") == "scratch_output_root" and check.get("status") == "failed":
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    _configure_python_path(repo_root)

    from meso_uq.dpd_sampling.preflight import (
        DPDProductionPreflightConfig,
        default_preflight_manifest_path,
        output_root_from_candidate_manifest,
        run_dpd_production_preflight,
        write_preflight_manifest,
    )

    candidate_manifest = Path(args.candidate_manifest).resolve() if args.candidate_manifest else None
    output_root = (
        output_root_from_candidate_manifest(candidate_manifest)
        if candidate_manifest is not None
        else Path(args.output_root).expanduser().resolve()
    )
    explicit_manifest_path = args.manifest_path is not None
    manifest_path = Path(args.manifest_path).expanduser().resolve() if explicit_manifest_path else None
    config = DPDProductionPreflightConfig(
        output_root=output_root,
        manifest_path=manifest_path,
        candidate_manifest=candidate_manifest,
        require_scratch=not args.allow_non_scratch,
        allowed_scratch_roots=tuple(Path(item).expanduser().resolve() for item in args.scratch_root),
        min_free_bytes=args.min_free_bytes,
        hdf5_smoke_test=args.hdf5_smoke_test,
        hdf5_driver=args.hdf5_driver,
    )
    payload = run_dpd_production_preflight(config)
    if explicit_manifest_path or not _root_policy_failed(payload):
        manifest_path = manifest_path or default_preflight_manifest_path(output_root)
        payload["manifest_path"] = str(manifest_path)
        write_preflight_manifest(payload, manifest_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
