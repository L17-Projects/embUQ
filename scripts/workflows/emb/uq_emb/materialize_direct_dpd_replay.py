#!/usr/bin/env python3
"""Materialize a static, portable UQ_EMB direct-DPD replay plan.

The accepted direct-DPD outputs are immutable evidence.  This tool reads their
MAP inputs and protocol manifests, creates a fresh replay-input root, and
writes commands for the current site-neutral mechanical runner and the frozen
breathing protocols.  It never submits a scheduler job or executes Mirheo.

SonoVue acoustic re-extraction manifests retain the MAP inputs and protocol
settings but not their launch command or a source hash.  Their generated
commands are consequently marked ``candidate_provenance_incomplete`` rather
than claimed as exact replays.  Definity d2 is similarly recorded as a
near-MAP accepted result, not an exact-MAP acoustic replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "mesouq.uq_emb.direct_dpd_replay.v1"
ACCEPTED_ROOT_DEFAULT = Path(
    "/scratch/project/eu-26-17/eubrieucb/mesouq/papers/UQ_EMB/artifacts/"
    "accepted_production_outputs_202607"
)
VALID_SITES = ("karolina", "vega")
REPO_ROOT = Path(__file__).resolve().parents[4]
MECHANICAL_RUNNER = REPO_ROOT / "scripts/platforms/hpc/run_map_mirheo.py"
DEFINITY_ACOUSTIC_RUNNER = (
    REPO_ROOT / "scripts/workflows/emb/run_emb_fullfluid_ka_breathing_campaign.py"
)
SONOVUE_ACOUSTIC_RUNNER = (
    REPO_ROOT / "scripts/workflows/emb/run_emb_free_shell_breathing_protocol.py"
)
ACOUSTIC_RUNTIME_SOURCES = (
    DEFINITY_ACOUSTIC_RUNNER,
    SONOVUE_ACOUSTIC_RUNNER,
    REPO_ROOT / "scripts/workflows/emb/emb_deflate_only_analysis.py",
    REPO_ROOT / "scripts/workflows/emb/emb_particle_staging.py",
)

BUBBLES = (
    {"id": "d1", "agent": "definity", "diameter_um": 2.1, "experiment": "compression"},
    {"id": "d2", "agent": "definity", "diameter_um": 2.9, "experiment": "compression"},
    {"id": "d3", "agent": "definity", "diameter_um": 3.0, "experiment": "compression"},
    {"id": "d4", "agent": "sonovue", "diameter_um": 3.2, "experiment": "indentation"},
    {"id": "d5", "agent": "sonovue", "diameter_um": 3.4, "experiment": "indentation"},
    {"id": "d6", "agent": "sonovue", "diameter_um": 5.8, "experiment": "indentation"},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _copy_input(source: Path, destination: Path) -> dict[str, str]:
    if not source.is_file():
        raise FileNotFoundError(f"Accepted replay input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = _sha256(source)
    copied_hash = _sha256(destination)
    if source_hash != copied_hash:
        raise RuntimeError(f"Input copy hash mismatch: {source} -> {destination}")
    return {
        "accepted_path": str(source.resolve()),
        "accepted_sha256": source_hash,
        "materialized_path": str(destination.resolve()),
        "materialized_sha256": copied_hash,
    }


def _resolve_site(cli_site: str | None) -> str:
    env_site = os.environ.get("MESOUQ_SITE")
    if env_site and env_site not in VALID_SITES:
        raise ValueError(f"Unsupported MESOUQ_SITE={env_site!r}; expected {VALID_SITES}.")
    if cli_site and env_site and cli_site != env_site:
        raise ValueError(
            f"Conflicting site selectors: --site={cli_site} and MESOUQ_SITE={env_site}."
        )
    return cli_site or env_site or "karolina"


def _default_python(site: str) -> str:
    return "/usr/bin/python3.11" if site == "karolina" else "python3"


def _ensure_fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to write into a non-empty replay root: {path}. Choose a fresh output root."
        )
    path.mkdir(parents=True, exist_ok=True)


def _dataset_name(experiment: str, diameter_um: float) -> str:
    return f"{experiment}_{diameter_um:.1f}um"


def _mechanical_manifest(
    *,
    accepted_root: Path,
    bubble: dict[str, Any],
    output_root: Path,
) -> tuple[Path, list[dict[str, str]]]:
    bubble_id = str(bubble["id"])
    agent = str(bubble["agent"])
    experiment = str(bubble["experiment"])
    diameter_um = float(bubble["diameter_um"])
    dataset = _dataset_name(experiment, diameter_um)
    destination = output_root / bubble_id / "mechanical/map_phase3b/phase3b_map_manifest.json"

    if agent == "definity":
        source = (
            accepted_root
            / f"definity/direct_dpd/mechanical/{bubble_id}/map_workflow/"
            "map_phase3b/phase3b_map_manifest.json"
        )
        payload = _json(source)
        datasets = payload.get("datasets")
        if not isinstance(datasets, dict) or dataset not in datasets:
            raise ValueError(f"Accepted Definity mechanical MAP lacks {dataset}: {source}")
        frozen = dict(payload)
        frozen["datasets"] = {dataset: datasets[dataset]}
        source_hashes = [{"accepted_path": str(source.resolve()), "accepted_sha256": _sha256(source)}]
    else:
        source = accepted_root / "sonovue/direct_dpd/inputs/manifest.json"
        input_manifest = _json(source)
        cases = input_manifest.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"SonoVue direct-DPD input manifest has no cases list: {source}")
        case = next((item for item in cases if float(item["diameter_um"]) == diameter_um), None)
        if not isinstance(case, dict):
            raise ValueError(f"SonoVue direct-DPD input manifest lacks {diameter_um:.1f} um")
        map_source = accepted_root / "sonovue/direct_dpd/inputs/force_spectroscopy" / (
            f"indentation_{diameter_um:.1f}um_map.json"
        )
        map_payload = _json(map_source)
        values = dict(zip(map_payload["parameter_names"], map_payload["parameters"], strict=True))
        source_meta = map_payload.get("source")
        if not isinstance(source_meta, dict):
            raise ValueError(f"SonoVue MAP lacks source metadata: {map_source}")
        frozen = {
            "experiment": experiment,
            "model_family": "reduced-model",
            "datasets": {
                dataset: {
                    "Yt": values["Yt"],
                    "kb": values["kb"],
                    "d0": values["d0"],
                    "sigma": source_meta["sigma"],
                    "logLikelihood": source_meta["logLikelihood"],
                    "logPrior": source_meta["logPrior"],
                    "logPosterior": map_payload["logPosterior"],
                    "diameter_um": diameter_um,
                    "run_dir": source_meta["state_path"],
                    "output_csv": "",
                }
            },
        }
        source_hashes = [
            {"accepted_path": str(source.resolve()), "accepted_sha256": _sha256(source)},
            {"accepted_path": str(map_source.resolve()), "accepted_sha256": _sha256(map_source)},
        ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination, source_hashes


def _mechanical_command(
    *,
    site: str,
    python_bin: str,
    bubble: dict[str, Any],
    output_root: Path,
) -> list[str]:
    bubble_id = str(bubble["id"])
    return [
        python_bin,
        str(MECHANICAL_RUNNER),
        "--site",
        site,
        "--experiment",
        str(bubble["experiment"]),
        "--model-family",
        "reduced-model",
        "--profile",
        "production",
        "--output-dir",
        str(output_root / bubble_id / "mechanical"),
        "--python-bin",
        python_bin,
        "--n-displacements",
        "15",
        "--mpi-ranks",
        "2",
        "--max-retries",
        "0",
        "--dataset-name",
        _dataset_name(str(bubble["experiment"]), float(bubble["diameter_um"])),
    ]


def _definity_acoustic_entry(
    *, accepted_root: Path, output_root: Path, python_bin: str, bubble: dict[str, Any]
) -> dict[str, Any]:
    bubble_id = str(bubble["id"])
    if bubble_id == "d2":
        source = (
            accepted_root
            / "definity/direct_dpd/acoustic/d2_near_map_0p1482pct/setup_manifest.json"
        )
        return {
            "modality": "acoustic",
            "status": "provenance_incomplete_near_map_only",
            "command": None,
            "source_hashes": [{"accepted_path": str(source.resolve()), "accepted_sha256": _sha256(source)}],
            "provenance_gaps": [
                "Accepted d2 acoustic evidence is a 0.1482% near-MAP point, not an exact-MAP run.",
                "No accepted exact-MAP acoustic input or executable command is available.",
            ],
        }

    accepted_case = accepted_root / f"definity/direct_dpd/acoustic/{bubble_id}"
    map_source = accepted_case / "accepted_map_values.csv"
    design_source = accepted_case / "fullfluid_campaign/design/production_design.csv"
    campaign_root = output_root / bubble_id / "acoustic/fullfluid_campaign"
    map_copy = campaign_root / "inputs/accepted_map_values.csv"
    design_copy = campaign_root / "design/production_design.csv"
    sources = [_copy_input(map_source, map_copy), _copy_input(design_source, design_copy)]
    command = [
        "mpirun",
        "-n",
        "2",
        python_bin,
        str(DEFINITY_ACOUSTIC_RUNNER),
        "run",
        "--campaign-root",
        str(campaign_root),
        "--map-values",
        str(map_copy),
        "--phase",
        "production",
        "--case-index",
        "0",
        "--particle-checker-every",
        "100",
        "--particle-staging-root",
        str(output_root / "_particle_staging" / bubble_id),
    ]
    return {
        "modality": "acoustic",
        "status": "exact_protocol_command",
        "command": command,
        "source_hashes": sources,
        "provenance_gaps": [],
    }


def _sonovue_acoustic_entry(
    *, accepted_root: Path, output_root: Path, python_bin: str, bubble: dict[str, Any]
) -> dict[str, Any]:
    bubble_id = str(bubble["id"])
    diameter_um = float(bubble["diameter_um"])
    manifest_source = accepted_root / f"sonovue/direct_dpd/acoustic/{bubble_id}/manifest.json"
    manifest = _json(manifest_source)
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"Expected one accepted SonoVue acoustic result: {manifest_source}")
    result = results[0]
    protocol = result.get("setup_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"Accepted SonoVue acoustic protocol missing: {manifest_source}")
    symbol = str(result["symbol"])
    map_source = accepted_root / "sonovue/direct_dpd/inputs/breathing_frequency" / f"{symbol}.csv"
    map_copy = output_root / bubble_id / "acoustic/inputs" / f"{symbol}.csv"
    sources = [_copy_input(map_source, map_copy), {"accepted_path": str(manifest_source.resolve()), "accepted_sha256": _sha256(manifest_source)}]
    run_root = output_root / bubble_id / "acoustic/reextraction"
    command = [
        "mpirun", "-n", "2", python_bin, str(SONOVUE_ACOUSTIC_RUNNER),
        "--map-values", str(map_copy),
        "--run-root", str(run_root),
        "--symbols", symbol,
        "--seed-indices", "0",
        "--dt", str(protocol["dt"]),
        "--equil-steps", str(protocol["equil_steps"]),
        "--relax-steps", str(protocol["relax_steps"]),
        "--sample-every", str(protocol["sample_every"]),
        "--trajectory-capture", str(protocol["trajectory_capture"]),
        "--excitation-mode", str(protocol["excitation_mode"]),
        "--initial-radius-scale", str(protocol["initial_radius_scale"]),
        "--post-deflation-ramp-steps", str(protocol["post_deflation_ramp_steps"]),
        "--post-deflation-hold-steps", str(protocol["post_deflation_hold_steps"]),
        "--post-deflation-hold-update-every-steps", str(protocol["post_deflation_hold_update_every_steps"]),
        "--solvent-mode", str(protocol["solvent_mode"]),
        "--water-shell-fsi-scale", str(protocol["water_shell_fsi_scale"]),
        "--membrane-mass-scale", str(protocol["membrane_mass_scale"]),
        "--lim-mu-policy", str(protocol["lim_mu_policy"]),
        "--primary-observable", str(protocol["primary_observable"]),
        "--fit-end-dpd", str(protocol["fit_end_dpd"]),
        "--particle-dump-root", str(output_root / "_particle_staging" / bubble_id),
    ]
    return {
        "modality": "acoustic",
        "status": "candidate_provenance_incomplete",
        "command": command,
        "source_hashes": sources,
        "provenance_gaps": [
            "The accepted canonical re-extraction manifest records no launch command.",
            "The accepted canonical re-extraction manifest records no acoustic runner source hash.",
            "This command is reconstructed from retained MAP inputs and setup protocol fields; it is not claimed exact.",
        ],
    }


def _source_hashes(paths: Iterable[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required portable runtime sources are missing: " + ", ".join(missing))
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths}


def materialize_direct_dpd_replay(
    *,
    accepted_root: Path,
    output_root: Path,
    site: str,
    python_bin: str,
) -> dict[str, Any]:
    """Write a fresh replay plan and its portable, non-accepted inputs."""
    accepted_root = accepted_root.expanduser().resolve()
    if not accepted_root.is_dir():
        raise FileNotFoundError(f"Accepted direct-DPD artifact root is missing: {accepted_root}")
    output_root = output_root.expanduser().resolve()
    _ensure_fresh_output(output_root)

    entries: list[dict[str, Any]] = []
    for bubble in BUBBLES:
        bubble_id = str(bubble["id"])
        mechanical_manifest, mechanical_sources = _mechanical_manifest(
            accepted_root=accepted_root, bubble=bubble, output_root=output_root
        )
        mechanical = {
            "modality": "mechanical",
            "status": "exact_protocol_command",
            "command": _mechanical_command(
                site=site, python_bin=python_bin, bubble=bubble, output_root=output_root
            ),
            "materialized_phase3b_map_manifest": str(mechanical_manifest),
            "materialized_phase3b_map_manifest_sha256": _sha256(mechanical_manifest),
            "source_hashes": mechanical_sources,
            "provenance_gaps": [],
        }
        acoustic = (
            _definity_acoustic_entry(
                accepted_root=accepted_root, output_root=output_root, python_bin=python_bin, bubble=bubble
            )
            if bubble["agent"] == "definity"
            else _sonovue_acoustic_entry(
                accepted_root=accepted_root, output_root=output_root, python_bin=python_bin, bubble=bubble
            )
        )
        entries.append({"bubble_id": bubble_id, **bubble, "mechanical": mechanical, "acoustic": acoustic})

    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "static_dry_run_only",
        "submission": "not performed",
        "accepted_artifact_root": str(accepted_root),
        "site": site,
        "python_bin": python_bin,
        "runtime_source_hashes": _source_hashes((MECHANICAL_RUNNER, *ACOUSTIC_RUNTIME_SOURCES)),
        "bubbles": entries,
    }
    plan_path = output_root / "direct_dpd_replay_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, default=ACCEPTED_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--site", choices=VALID_SITES, default=None)
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python executable embedded in planned commands. Defaults to /usr/bin/python3.11 on Karolina.",
    )
    args = parser.parse_args(argv)
    try:
        site = _resolve_site(args.site)
        plan = materialize_direct_dpd_replay(
            accepted_root=args.accepted_root,
            output_root=args.output_root,
            site=site,
            python_bin=args.python_bin or _default_python(site),
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote static UQ_EMB direct-DPD replay plan: {args.output_root / 'direct_dpd_replay_plan.json'}")
    print(f"Planned {len(plan['bubbles']) * 2} modality entries; no commands were executed or submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
