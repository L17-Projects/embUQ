#!/usr/bin/env python3
"""Materialize frozen UQ_EMB HBI configs against immutable external artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


PAPER_ID = "UQ_EMB"
ACCEPTED_SET = "accepted_production_outputs_202607"
DEPENDENCY_SET = "frozen_runtime_dependencies_202607"
SUPPORTED_POPULATIONS = (10_000, 50_000)

AGENT_PATHS: dict[str, dict[str, str]] = {
    "definity": {
        "config": "definity/config/definity_accepted_production50k.yaml",
        "mechanical_experiment": "compression",
        "mechanical_data": "reference_data/mechanical/definity",
        "mechanical_surrogates": "mechanical_surrogates/definity",
        "bank": "acoustic_surrogates/frozen_banks/definity_approved_free_intercept_squared_frequency_bank.json",
        "bank_report": "acoustic_surrogates/frozen_banks/definity_approved_free_intercept_squared_frequency_bank_report.json",
        "independent_go": "acoustic_surrogates/independent_audit/definity_frozen_bank_independent_go_10k_only.json",
    },
    "sonovue": {
        "config": "sonovue/config/sonovue_liked_candidate_production50k.yaml",
        "mechanical_experiment": "indentation",
        "mechanical_data": "reference_data/mechanical/sonovue",
        "mechanical_surrogates": "mechanical_surrogates/sonovue",
        "bank": "acoustic_surrogates/frozen_banks/sonovue_approved_free_intercept_squared_frequency_bank.json",
        "bank_report": "acoustic_surrogates/frozen_banks/sonovue_approved_free_intercept_squared_frequency_bank_report.json",
        "independent_go": "acoustic_surrogates/independent_audit/sonovue_frozen_bank_independent_go_10k_only.json",
    },
}

PROMOTION_CONTRACT = "acoustic_surrogates/family_contract/promotion_contract.json"
ACOUSTIC_DATA = "reference_data/acoustic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path, artifact_set_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("paper_id") != PAPER_ID:
        raise ValueError(f"Unexpected paper_id in {path}: {payload.get('paper_id')!r}")
    if payload.get("artifact_set_id") != artifact_set_id:
        raise ValueError(
            f"Unexpected artifact_set_id in {path}: {payload.get('artifact_set_id')!r}"
        )
    if payload.get("locked") is not True:
        raise ValueError(f"Artifact manifest is not locked: {path}")
    return payload


def _manifest_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Artifact manifest files must be a list")
    return {str(entry["path"]): str(entry["sha256"]) for entry in files}


def _verified_file(root: Path, relative: str, expected: Mapping[str, str]) -> Path:
    path = root / relative
    if path.is_symlink():
        raise ValueError(f"Frozen runtime inputs cannot be symlinks: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Frozen runtime input is missing: {path}")
    expected_hash = expected.get(relative)
    if expected_hash is None:
        raise ValueError(f"Frozen runtime input is absent from its locked manifest: {relative}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Frozen runtime input hash mismatch for {relative}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return path.resolve()


def _replace(
    mapping: dict[str, Any],
    key: str,
    value: Any,
    pointer: str,
    rewrites: list[dict[str, Any]],
) -> None:
    previous = mapping.get(key)
    mapping[key] = value
    rewrites.append({"path": pointer, "from": previous, "to": value})


def rewrite_hbi_config(
    source: Mapping[str, Any],
    *,
    agent: str,
    dependency_root: Path,
    run_root: Path,
    population: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if agent not in AGENT_PATHS:
        raise ValueError(f"Unsupported UQ_EMB agent: {agent}")
    if population not in SUPPORTED_POPULATIONS:
        raise ValueError(
            f"Unsupported HBI population {population}; expected one of {SUPPORTED_POPULATIONS}"
        )

    paths = AGENT_PATHS[agent]
    config = deepcopy(dict(source))
    rewrites: list[dict[str, Any]] = []
    experiments = config.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Frozen production config must contain an experiments list")

    seen: set[str] = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise ValueError("Frozen production experiments must be mappings")
        name = str(experiment.get("name", ""))
        if name == paths["mechanical_experiment"]:
            _replace(
                experiment,
                "data_dir",
                str((dependency_root / paths["mechanical_data"]).resolve()),
                f"/experiments/{index}/data_dir",
                rewrites,
            )
            _replace(
                experiment,
                "surrogate_dir",
                str((dependency_root / paths["mechanical_surrogates"]).resolve()),
                f"/experiments/{index}/surrogate_dir",
                rewrites,
            )
            seen.add("mechanical")
        elif name == "resonance":
            _replace(
                experiment,
                "data_dir",
                str((dependency_root / ACOUSTIC_DATA).resolve()),
                f"/experiments/{index}/data_dir",
                rewrites,
            )
            seen.add("acoustic")
    if seen != {"mechanical", "acoustic"}:
        raise ValueError(f"Frozen production config has incomplete experiment coverage: {seen}")

    evaluator = config.get("resonance", {}).get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("Frozen production config must define resonance.evaluator")
    for key, relative in (
        ("artifact_path", paths["bank"]),
        ("bank_build_report_path", paths["bank_report"]),
        ("independent_go_path", paths["independent_go"]),
        ("promotion_contract_path", PROMOTION_CONTRACT),
    ):
        _replace(
            evaluator,
            key,
            str((dependency_root / relative).resolve()),
            f"/resonance/evaluator/{key}",
            rewrites,
        )

    _replace(config, "out", str(run_root.resolve()), "/out", rewrites)
    for key in ("pop_size", "hbi_pop_size", "phase3b_pop_size"):
        _replace(config, key, population, f"/{key}", rewrites)
    return config, rewrites


def materialize(
    *,
    agent: str,
    artifact_root: Path,
    manifest_root: Path,
    output_dir: Path,
    run_root: Path,
    population: int,
) -> dict[str, Any]:
    paths = AGENT_PATHS[agent]
    accepted_manifest_path = manifest_root / f"{ACCEPTED_SET}.files.json"
    dependency_manifest_path = manifest_root / f"{DEPENDENCY_SET}.files.json"
    accepted_manifest = _load_manifest(accepted_manifest_path, "accepted-production-outputs-202607")
    dependency_manifest = _load_manifest(
        dependency_manifest_path, "frozen-runtime-dependencies-202607"
    )
    accepted_hashes = _manifest_hashes(accepted_manifest)
    dependency_hashes = _manifest_hashes(dependency_manifest)
    accepted_root = artifact_root / ACCEPTED_SET
    dependency_root = artifact_root / DEPENDENCY_SET

    source_config = _verified_file(accepted_root, paths["config"], accepted_hashes)
    verified_dependencies: dict[str, str] = {}
    for relative in (
        paths["bank"],
        paths["bank_report"],
        paths["independent_go"],
        PROMOTION_CONTRACT,
    ):
        verified = _verified_file(dependency_root, relative, dependency_hashes)
        verified_dependencies[relative] = _sha256(verified)

    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"Frozen production config is not a mapping: {source_config}")
    materialized, rewrites = rewrite_hbi_config(
        source,
        agent=agent,
        dependency_root=dependency_root,
        run_root=run_root,
        population=population,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_config = output_dir / f"{agent}_hbi_{population}.yaml"
    output_config.write_text(
        yaml.safe_dump(materialized, sort_keys=False), encoding="utf-8"
    )
    receipt = {
        "schema_version": "1.0",
        "paper_id": PAPER_ID,
        "agent": agent,
        "population": population,
        "source_config": str(source_config),
        "source_config_sha256": _sha256(source_config),
        "materialized_config": str(output_config.resolve()),
        "materialized_config_sha256": _sha256(output_config),
        "artifact_root": str(artifact_root.resolve()),
        "run_root": str(run_root.resolve()),
        "verified_dependencies": verified_dependencies,
        "rewrites": rewrites,
        "preserved_configuration": (
            "All scientific priors, hyperpriors, observations, likelihood grouping, "
            "sampler controls, and evaluator hashes are preserved from the accepted config."
        ),
    }
    receipt_path = output_config.with_suffix(".materialization.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt["receipt"] = str(receipt_path.resolve())
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(AGENT_PATHS), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--population", type=int, choices=SUPPORTED_POPULATIONS, default=10_000
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = materialize(
        agent=args.agent,
        artifact_root=args.artifact_root.expanduser().resolve(),
        manifest_root=args.manifest_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        run_root=args.run_root.expanduser().resolve(),
        population=args.population,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
