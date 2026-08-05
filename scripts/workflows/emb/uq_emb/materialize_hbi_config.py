#!/usr/bin/env python3
"""Materialize frozen UQ_EMB HBI configs against immutable external artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

from replay_provenance import (  # noqa: E402
    runtime_provenance,
    verify_locked_artifact_root,
    verify_locked_manifest_members,
)


PAPER_ID = "UQ_EMB"
ACCEPTED_SET = "accepted_production_outputs_202607"
DEPENDENCY_SET = "frozen_runtime_dependencies_202607"
SUPPORTED_POPULATIONS = (10_000, 50_000)
_STAGE_SEED_PATTERNS = {
    "phase1": re.compile(r"\[Korali\] Random Seed:\s*(\d+)"),
    "phase2": re.compile(r"\[HBI\] Random Seed:\s*(\d+)"),
    "phase3b": re.compile(r"\[Phase 3b\] Random Seed for .*?:\s*(\d+)"),
}

AGENT_PATHS: dict[str, dict[str, str]] = {
    "definity": {
        "config": "definity/config/definity_accepted_production50k.yaml",
        "stage_seed_log": "definity/hbi/logs/run.stdout.log",
        "mechanical_experiment": "compression",
        "mechanical_data": "reference_data/mechanical/definity",
        "mechanical_surrogates": "mechanical_surrogates/definity",
        "bank": "acoustic_surrogates/frozen_banks/definity_approved_free_intercept_squared_frequency_bank.json",
        "bank_report": "acoustic_surrogates/frozen_banks/definity_approved_free_intercept_squared_frequency_bank_report.json",
        "independent_go": "acoustic_surrogates/independent_audit/definity_frozen_bank_independent_go_10k_only.json",
    },
    "sonovue": {
        "config": "sonovue/config/sonovue_liked_candidate_production50k.yaml",
        "stage_seed_log": "sonovue/hbi/logs/run.stdout.log",
        "mechanical_experiment": "indentation",
        "mechanical_data": "reference_data/mechanical/sonovue",
        "mechanical_surrogates": "mechanical_surrogates/sonovue",
        "bank": "acoustic_surrogates/frozen_banks/sonovue_approved_free_intercept_squared_frequency_bank.json",
        "bank_report": "acoustic_surrogates/frozen_banks/sonovue_approved_free_intercept_squared_frequency_bank_report.json",
        "independent_go": "acoustic_surrogates/independent_audit/sonovue_frozen_bank_independent_go_10k_only.json",
    },
}

PROMOTION_CONTRACT = "acoustic_surrogates/family_contract/promotion_contract.json"
POLYNOMIAL_BANK_BUILD_TOOL = (
    "acoustic_surrogates/code/freeze_approved_polynomial_banks.py"
)
PROMOTION_SOURCES = {
    "approved_coefficients": (
        "acoustic_surrogates/polynomial_fits/free_intercept_squared_frequency_fits.csv"
    ),
    "source_labels": "acoustic_surrogates/physical_labels/physical_resonance_labels.csv",
    "builder": "acoustic_surrogates/code/build_free_intercept_squared_frequency_diagnostic.py",
}
ACOUSTIC_DATA = "reference_data/acoustic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_file(path: Path, content: str) -> None:
    """Publish a fully written file without replacing an existing materialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to replace an existing materialization artifact: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _accepted_stage_seeds(
    *,
    accepted_root: Path,
    accepted_hashes: Mapping[str, str],
    seed_log_relative: str,
) -> dict[str, Any]:
    """Recover the accepted Korali base seeds from the locked production log."""
    log_path = _verified_file(accepted_root, seed_log_relative, accepted_hashes)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    seeds: dict[str, int] = {}
    for stage, pattern in _STAGE_SEED_PATTERNS.items():
        values = [int(value) for value in pattern.findall(text)]
        if not values:
            raise ValueError(
                f"Locked accepted HBI log has no {stage} Korali seed record: {log_path}"
            )
        if stage == "phase3b":
            base = min(values)
            expected = set(range(base, max(values) + 1))
            # Separate target invocations may reuse one base seed.  Older accepted
            # runs may instead increment it once per target; both forms preserve
            # the same replay base and reject gaps or unrelated seed values.
            if set(values) not in ({base}, expected):
                raise ValueError(
                    "Locked accepted HBI Phase 3b seeds must reuse one base seed or form "
                    "one contiguous sequence from that base: "
                    f"{values}"
                )
        elif len(set(values)) != 1:
            raise ValueError(
                f"Locked accepted HBI {stage} log has inconsistent Korali seeds: {values}"
            )
        if values[0] <= 0:
            raise ValueError(f"Locked accepted HBI {stage} seed must be positive: {values[0]}")
        seeds[stage] = min(values) if stage == "phase3b" else values[0]
    return {
        "accepted_stage_seeds": seeds,
        "accepted_stage_seed_log": str(log_path),
        "accepted_stage_seed_log_sha256": _sha256(log_path),
    }


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
    provenance_path_overrides: Mapping[str, str] | None = None,
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
    _replace(
        evaluator,
        "provenance_path_overrides",
        dict(provenance_path_overrides or {}),
        "/resonance/evaluator/provenance_path_overrides",
        rewrites,
    )

    _replace(config, "out", str(run_root.resolve()), "/out", rewrites)
    for key in ("pop_size", "hbi_pop_size", "phase3b_pop_size"):
        _replace(config, key, population, f"/{key}", rewrites)
    return config, rewrites


def _semantic_config_payload(
    config: Mapping[str, Any],
    *,
    agent: str,
    dependency_root: Path,
) -> dict[str, Any]:
    """Remove only relocated filesystem identity from an HBI config."""
    paths = AGENT_PATHS[agent]
    canonical = deepcopy(dict(config))
    canonical["out"] = "run://uq_emb"
    for experiment in canonical.get("experiments", []):
        if not isinstance(experiment, dict):
            continue
        if experiment.get("name") == paths["mechanical_experiment"]:
            experiment["data_dir"] = f"artifact://{paths['mechanical_data']}"
            experiment["surrogate_dir"] = f"artifact://{paths['mechanical_surrogates']}"
        elif experiment.get("name") == "resonance":
            experiment["data_dir"] = f"artifact://{ACOUSTIC_DATA}"

    evaluator = canonical.get("resonance", {}).get("evaluator", {})
    if isinstance(evaluator, dict):
        for key, relative in (
            ("artifact_path", paths["bank"]),
            ("bank_build_report_path", paths["bank_report"]),
            ("independent_go_path", paths["independent_go"]),
            ("promotion_contract_path", PROMOTION_CONTRACT),
        ):
            evaluator[key] = f"artifact://{relative}"
        overrides = evaluator.get("provenance_path_overrides")
        if isinstance(overrides, dict):
            canonical_overrides: dict[str, str] = {}
            for source, destination in overrides.items():
                destination_path = Path(str(destination)).resolve()
                try:
                    relative = destination_path.relative_to(dependency_root.resolve())
                except ValueError as exc:
                    raise ValueError(
                        "Provenance override escapes the immutable dependency root: "
                        f"{destination_path}"
                    ) from exc
                canonical_overrides[str(source)] = f"artifact://{relative.as_posix()}"
            evaluator["provenance_path_overrides"] = canonical_overrides
    return canonical


def _semantic_config_sha256(
    config: Mapping[str, Any],
    *,
    agent: str,
    dependency_root: Path,
) -> str:
    payload = _semantic_config_payload(
        config,
        agent=agent,
        dependency_root=dependency_root,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def materialize(
    *,
    agent: str,
    artifact_root: Path,
    manifest_root: Path,
    output_dir: Path,
    run_root: Path,
    population: int,
) -> dict[str, Any]:
    artifact_root = artifact_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
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
    for locked_root in (accepted_root, dependency_root):
        try:
            output_dir.relative_to(locked_root)
        except ValueError:
            continue
        raise ValueError(
            "HBI materialization output must remain outside immutable artifact roots: "
            f"output={output_dir}, locked_root={locked_root}"
        )

    source_config = _verified_file(accepted_root, paths["config"], accepted_hashes)
    accepted_source_verification = verify_locked_manifest_members(
        root=accepted_root,
        manifest_path=accepted_manifest_path,
        members=[source_config],
    )
    dependency_verification = verify_locked_artifact_root(
        root=dependency_root,
        manifest_path=dependency_manifest_path,
    )
    verified_dependencies: dict[str, str] = {}
    for relative in (
        paths["bank"],
        POLYNOMIAL_BANK_BUILD_TOOL,
        *PROMOTION_SOURCES.values(),
        paths["bank_report"],
        paths["independent_go"],
        PROMOTION_CONTRACT,
    ):
        verified = _verified_file(dependency_root, relative, dependency_hashes)
        verified_dependencies[relative] = _sha256(verified)

    bank_payload = json.loads((dependency_root / paths["bank"]).read_text(encoding="utf-8"))
    promotion_payload = json.loads(
        (dependency_root / PROMOTION_CONTRACT).read_text(encoding="utf-8")
    )
    provenance_records = {
        "build_tool": bank_payload.get("provenance", {}).get("build_tool"),
        **{
            label: promotion_payload.get(label)
            for label in PROMOTION_SOURCES
        },
    }
    provenance_relatives = {
        "build_tool": POLYNOMIAL_BANK_BUILD_TOOL,
        **PROMOTION_SOURCES,
    }
    provenance_path_overrides: dict[str, str] = {}
    for label, relative in provenance_relatives.items():
        record = provenance_records.get(label)
        if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
            raise ValueError(f"Frozen polynomial provenance has no valid {label} binding")
        if str(record["sha256"]) != dependency_hashes.get(relative):
            raise ValueError(
                f"Frozen polynomial provenance hash differs from the staged {label} dependency"
            )
        provenance_path_overrides[str(record["path"])] = str(
            (dependency_root / relative).resolve()
        )

    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"Frozen production config is not a mapping: {source_config}")
    materialized, rewrites = rewrite_hbi_config(
        source,
        agent=agent,
        dependency_root=dependency_root,
        run_root=run_root,
        population=population,
        provenance_path_overrides=provenance_path_overrides,
    )

    output_config = output_dir / f"{agent}_hbi_{population}.yaml"
    accepted_seed_provenance = _accepted_stage_seeds(
        accepted_root=accepted_root,
        accepted_hashes=accepted_hashes,
        seed_log_relative=paths["stage_seed_log"],
    )
    _write_new_file(output_config, yaml.safe_dump(materialized, sort_keys=False))
    receipt = {
        "schema_version": "1.0",
        "paper_id": PAPER_ID,
        "agent": agent,
        "population": population,
        "source_config": str(source_config),
        "source_config_sha256": _sha256(source_config),
        "materialized_config": str(output_config.resolve()),
        "materialized_config_sha256": _sha256(output_config),
        "semantic_config_sha256": _semantic_config_sha256(
            materialized,
            agent=agent,
            dependency_root=dependency_root,
        ),
        "artifact_root": str(artifact_root.resolve()),
        "run_root": str(run_root.resolve()),
        "accepted_manifest": str(accepted_manifest_path.resolve()),
        "accepted_manifest_sha256": _sha256(accepted_manifest_path),
        "dependency_manifest": str(dependency_manifest_path.resolve()),
        "dependency_manifest_sha256": _sha256(dependency_manifest_path),
        "verified_dependencies": verified_dependencies,
        "accepted_source_verification": accepted_source_verification,
        **accepted_seed_provenance,
        "dependency_verification": dependency_verification,
        "rewrites": rewrites,
        "provenance": runtime_provenance(
            repo_root=REPO_ROOT,
            site="materialization",
        ),
        "preserved_configuration": (
            "All scientific priors, hyperpriors, observations, likelihood grouping, "
            "sampler controls, and evaluator hashes are preserved from the accepted config."
        ),
    }
    receipt_path = output_config.with_suffix(".materialization.json")
    _write_new_file(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
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
