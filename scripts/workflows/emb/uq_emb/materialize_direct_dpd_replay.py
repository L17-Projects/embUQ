#!/usr/bin/env python3
"""Materialize a static, portable UQ_EMB direct-DPD replay plan.

The accepted direct-DPD outputs are immutable evidence.  This tool reads their
MAP inputs and protocol manifests, creates a fresh replay-input root, and
writes commands for the current site-neutral mechanical runner and the frozen
breathing protocols.  It never submits a scheduler job or executes Mirheo.

Acoustic commands are reconstructed from the retained MAP rows and exact setup
protocols.  They use the byte-identical frozen breathing runner, but are not
claimed to reproduce an unrecorded shell launch wrapper.  Definity d2 is the
user-accepted 0.1482% near-MAP coordinate and is labeled accordingly.
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
ACCEPTED_ARTIFACT_SET_ID = "accepted-production-outputs-202607"
VALID_SITES = ("karolina", "vega")
REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from replay_provenance import runtime_provenance  # noqa: E402

MECHANICAL_RUNNER = REPO_ROOT / "scripts/platforms/hpc/run_map_mirheo.py"
SONOVUE_ACOUSTIC_RUNNER = (
    REPO_ROOT / "scripts/workflows/emb/run_emb_free_shell_breathing_protocol.py"
)
MECHANICAL_EVALUATORS = {
    "compression": REPO_ROOT / "propagation/scripts/evaluate_map_mirheo_optimized.py",
    "indentation": REPO_ROOT / "propagation/scripts/evaluate_map_mirheo_optimized_indentation.py",
}
ACOUSTIC_RUNTIME_SOURCES = (
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


def _load_accepted_manifest(manifest_path: Path, accepted_root: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    accepted_root = accepted_root.expanduser().resolve()
    payload = _json(manifest_path)
    if payload.get("paper_id") != "UQ_EMB":
        raise ValueError(f"Unexpected accepted-manifest paper_id: {manifest_path}")
    if payload.get("artifact_set_id") != ACCEPTED_ARTIFACT_SET_ID:
        raise ValueError(f"Unexpected accepted artifact_set_id: {manifest_path}")
    if payload.get("artifact_set_dir") != accepted_root.name:
        raise ValueError(
            "Accepted manifest artifact_set_dir does not match the accepted root: "
            f"{payload.get('artifact_set_dir')!r} != {accepted_root.name!r}"
        )
    if payload.get("locked") is not True:
        raise ValueError(f"Accepted artifact manifest is not locked: {manifest_path}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Accepted artifact manifest has no files: {manifest_path}")
    records: dict[str, dict[str, Any]] = {}
    for record in files:
        relative = str(record.get("path", ""))
        if not relative or relative in records:
            raise ValueError(f"Invalid or duplicate accepted manifest path: {relative!r}")
        records[relative] = dict(record)
    return {
        "path": manifest_path,
        "sha256": _sha256(manifest_path),
        "root": accepted_root,
        "records": records,
    }


def _accepted_source(index: dict[str, Any], path: Path) -> dict[str, str]:
    root = Path(index["root"])
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Accepted replay input escapes its artifact root: {resolved}") from exc
    record = index["records"].get(relative)
    if not isinstance(record, dict):
        raise ValueError(f"Accepted replay input is absent from the locked manifest: {relative}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Accepted replay input is missing: {resolved}")
    actual_size = resolved.stat().st_size
    if int(record.get("size_bytes", -1)) != actual_size:
        raise ValueError(f"Accepted replay input size mismatch: {relative}")
    actual_sha = _sha256(resolved)
    if str(record.get("sha256", "")) != actual_sha:
        raise ValueError(f"Accepted replay input hash mismatch: {relative}")
    return {
        "accepted_path": str(resolved),
        "accepted_sha256": actual_sha,
    }


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
    site = cli_site or env_site
    if site is None:
        raise ValueError("Missing site selector. Pass --site or set MESOUQ_SITE.")
    return site


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
    accepted_index: dict[str, Any],
    bubble: dict[str, Any],
    output_root: Path,
) -> tuple[Path, Path | None, list[dict[str, str]]]:
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
        source_record = _accepted_source(accepted_index, source)
        payload = _json(source)
        datasets = payload.get("datasets")
        if not isinstance(datasets, dict) or dataset not in datasets:
            raise ValueError(f"Accepted Definity mechanical MAP lacks {dataset}: {source}")
        frozen = dict(payload)
        frozen["datasets"] = {dataset: datasets[dataset]}
        direct_map = None
        source_hashes = [source_record]
    else:
        source = accepted_root / "sonovue/direct_dpd/inputs/manifest.json"
        source_record = _accepted_source(accepted_index, source)
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
        map_record = _accepted_source(accepted_index, map_source)
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
        source_hashes = [source_record, map_record]
        direct_map = output_root / bubble_id / "mechanical/map_json" / map_source.name
        _copy_input(map_source, direct_map)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination, direct_map, source_hashes


def _mechanical_command(
    *,
    site: str,
    python_bin: str,
    accepted_root: Path,
    accepted_index: dict[str, Any],
    bubble: dict[str, Any],
    output_root: Path,
    direct_map: Path | None,
) -> list[str]:
    bubble_id = str(bubble["id"])
    agent = str(bubble["agent"])
    experiment = str(bubble["experiment"])
    dataset = _dataset_name(experiment, float(bubble["diameter_um"]))
    if agent == "sonovue":
        if direct_map is None:
            raise ValueError(f"Missing materialized SonoVue MAP input for {bubble_id}.")
        provenance_path = _mechanical_protocol_source(accepted_root, bubble)
        _accepted_source(accepted_index, provenance_path)
        provenance = _json(provenance_path)
        protocol = provenance.get("protocol")
        if not isinstance(protocol, dict):
            raise ValueError(f"Accepted SonoVue mechanical protocol is missing: {provenance_path}")
        case_root = output_root / bubble_id / "mechanical/map_mirheo"
        return [
            "env",
            f"MESOUQ_SITE={site}",
            "mpirun",
            "--oversubscribe",
            "-x",
            "MESOUQ_SITE",
            "-n",
            str(protocol["mpi_ranks"]),
            python_bin,
            str(MECHANICAL_EVALUATORS[experiment]),
            "--map-file",
            str(direct_map),
            "--output",
            str(case_root / "results" / f"{dataset}_result.json"),
            "--n-displacements",
            str(protocol["force_grid_points"]),
            "--extend-range",
            str(protocol["force_grid_extension_fraction"]),
            "--numsteps",
            str(protocol["production_steps"]),
            "--numsteps-eq",
            str(protocol["equilibration_steps"]),
            "--retry-attempt",
            str(protocol["retry_attempt"]),
            "--scratch-root",
            str(case_root / "_scratch" / dataset),
        ]

    accepted_summary_path = _mechanical_protocol_source(accepted_root, bubble)
    _accepted_source(accepted_index, accepted_summary_path)
    accepted_summary = _json(accepted_summary_path)
    return [
        python_bin,
        str(MECHANICAL_RUNNER),
        "--site",
        site,
        "--experiment",
        experiment,
        "--model-family",
        "reduced-model",
        "--profile",
        "validation",
        "--output-dir",
        str(output_root / bubble_id / "mechanical"),
        "--python-bin",
        python_bin,
        "--n-displacements",
        str(accepted_summary["n_displacements"]),
        "--mpi-ranks",
        str(accepted_summary["mpi_ranks"]),
        "--timeout-seconds",
        str(accepted_summary["timeout_seconds"]),
        "--max-retries",
        str(accepted_summary["max_retries"]),
        "--dataset-name",
        dataset,
    ]


def _mechanical_protocol_source(accepted_root: Path, bubble: dict[str, Any]) -> Path:
    bubble_id = str(bubble["id"])
    if bubble["agent"] == "sonovue":
        return accepted_root / f"sonovue/direct_dpd/mechanical/{bubble_id}/provenance.json"
    return (
        accepted_root
        / f"definity/direct_dpd/mechanical/{bubble_id}/map_workflow/map_mirheo/"
        "map_mirheo_manifest.json"
    )


def _write_single_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def _accepted_hbi_state(
    accepted_root: Path,
    accepted_index: dict[str, Any],
    bubble: dict[str, Any],
) -> Path:
    agent = str(bubble["agent"])
    dataset = _dataset_name(str(bubble["experiment"]), float(bubble["diameter_um"]))
    path = accepted_root / agent / "hbi/results_phase_3b" / dataset / "genLatest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Accepted HBI state is missing: {path}")
    _accepted_source(accepted_index, path)
    return path


def _definity_setup_manifest(accepted_root: Path, bubble_id: str) -> Path:
    if bubble_id == "d2":
        return accepted_root / "definity/direct_dpd/acoustic/d2_near_map_0p1482pct/setup_manifest.json"
    diameter_token = {"d1": "2p1", "d3": "3p0"}[bubble_id]
    matches = sorted(
        (accepted_root / f"definity/direct_dpd/acoustic/{bubble_id}").glob(
            f"fullfluid_campaign/production/definity/definity_{diameter_token}um/"
            f"ka-index-*/definity_{diameter_token}um/seed-000/setup_manifest.json"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one accepted {bubble_id} setup manifest, found {len(matches)}.")
    return matches[0]


def _acoustic_sources(
    *, accepted_root: Path, accepted_index: dict[str, Any], bubble: dict[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    bubble_id = str(bubble["id"])
    agent = str(bubble["agent"])
    if agent == "definity":
        setup_source = _definity_setup_manifest(accepted_root, bubble_id)
        setup_record = _accepted_source(accepted_index, setup_source)
        setup = _json(setup_source)
        map_payload = setup.get("bubble")
        protocol = setup.get("protocol")
        if not isinstance(map_payload, dict) or not isinstance(protocol, dict):
            raise ValueError(f"Accepted Definity acoustic setup is incomplete: {setup_source}")
        sources = [setup_record]
        return setup_source, map_payload, protocol, sources

    manifest_source = accepted_root / f"sonovue/direct_dpd/acoustic/{bubble_id}/manifest.json"
    manifest_record = _accepted_source(accepted_index, manifest_source)
    manifest = _json(manifest_source)
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"Expected one accepted SonoVue acoustic result: {manifest_source}")
    result = results[0]
    map_payload = result.get("map")
    protocol = result.get("setup_protocol")
    if not isinstance(map_payload, dict) or not isinstance(protocol, dict):
        raise ValueError(f"Accepted SonoVue acoustic manifest is incomplete: {manifest_source}")
    sources = [manifest_record]
    return manifest_source, map_payload, protocol, sources


def _acoustic_map_row(map_payload: dict[str, Any], state_path: Path) -> dict[str, Any]:
    return {
        "agent": map_payload["agent"],
        "diameter_symbol": map_payload.get("symbol", map_payload.get("diameter_symbol")),
        "modality": map_payload["modality"],
        "dataset": map_payload["dataset"],
        "diameter_um": map_payload["diameter_um"],
        "radius_dpd": map_payload["radius_dpd"],
        "radius_source": map_payload["radius_source"],
        "source_label": map_payload["source_label"],
        "sample_index": map_payload["sample_index"],
        "sample_count": map_payload["sample_count"],
        "ka": map_payload["ka"],
        "kb": map_payload["kb"],
        "d0": map_payload["d0"],
        "sigma": map_payload["sigma"],
        "legacy_Yt": map_payload.get("legacy_Yt", map_payload.get("legacy_yt")),
        "logLikelihood": map_payload.get("logLikelihood", map_payload.get("log_likelihood")),
        "logPrior": map_payload.get("logPrior", map_payload.get("log_prior")),
        "logPosterior": map_payload.get("logPosterior", map_payload.get("log_posterior")),
        "state_path": str(state_path.resolve()),
        "map_template_state_path": str(state_path.resolve()),
    }


def _protocol_command_args(protocol: dict[str, Any]) -> list[str]:
    fit_end = protocol.get("fit_end_dpd")
    return [
        "--dt", str(protocol["dt"]),
        "--equil-steps", str(protocol["equil_steps"]),
        "--pulse-steps", str(protocol.get("pulse_steps", 0)),
        "--relax-steps", str(protocol["relax_steps"]),
        "--sample-every", str(protocol["sample_every"]),
        "--trajectory-capture", str(protocol["trajectory_capture"]),
        "--pulse-force-per-vertex", str(protocol.get("pulse_force_per_vertex", 0.001)),
        "--excitation-mode", str(protocol["excitation_mode"]),
        "--initial-radius-scale", str(protocol["initial_radius_scale"]),
        "--prestrain-placement", str(protocol.get("prestrain_placement", "post-equilibration")),
        "--post-deflation-ramp-steps", str(protocol["post_deflation_ramp_steps"]),
        "--post-deflation-hold-steps", str(protocol["post_deflation_hold_steps"]),
        "--post-deflation-hold-update-every-steps", str(protocol["post_deflation_hold_update_every_steps"]),
        "--post-deflation-hold-reset-velocities",
        "--radial-velocity-kick", str(protocol.get("radial_velocity_kick", 0.0)),
        "--radial-velocity-kick-mode", str(protocol.get("radial_velocity_kick_mode", "add")),
        "--solvent-mode", str(protocol["solvent_mode"]),
        "--water-shell-fsi-scale", str(protocol["water_shell_fsi_scale"]),
        "--water-shell-gamma-scale", str(protocol.get("water_shell_gamma_scale", 1.0)),
        "--bouncer-mode", str(protocol.get("bouncer_mode", "on")),
        "--membrane-mass-scale", str(protocol["membrane_mass_scale"]),
        "--lim-mu-policy", str(protocol["lim_mu_policy"]),
        "--primary-observable", str(protocol["primary_observable"]),
        "--box-padding-dpd", str(protocol.get("box_padding_dpd", 8.0)),
        "--transient-cut-fraction", str(protocol.get("transient_cut_fraction", 0.0)),
        "--fit-start-dpd", str(protocol.get("fit_start_dpd", 0.0)),
        "--fit-end-dpd", "-1.0" if fit_end is None else str(fit_end),
        "--particle-checker-every", str(protocol.get("particle_checker_every", 0)),
        "--pin-com",
    ]


def _acoustic_command(
    *,
    site: str,
    python_bin: str,
    state_source: Path,
    map_copy: Path,
    output_root: Path,
    bubble_id: str,
    symbol: str,
    protocol: dict[str, Any],
) -> list[str]:
    state_hash = _sha256(state_source)
    return [
        "env",
        f"MESOUQ_SITE={site}",
        f"MESOUQ_BOUND_MAP_SOURCE_PATH={state_source.resolve()}",
        f"MESOUQ_BOUND_MAP_SOURCE_SHA256={state_hash}",
        "mpirun",
        "--oversubscribe",
        "-x",
        "MESOUQ_SITE",
        "-x",
        "MESOUQ_BOUND_MAP_SOURCE_PATH",
        "-x",
        "MESOUQ_BOUND_MAP_SOURCE_SHA256",
        "-n",
        "2",
        python_bin,
        str(SONOVUE_ACOUSTIC_RUNNER),
        "--map-values",
        str(map_copy),
        "--run-root",
        str(output_root / bubble_id / "acoustic/replay"),
        "--symbols",
        symbol,
        "--seed-indices",
        "0",
        *_protocol_command_args(protocol),
        "--particle-dump-root",
        str(output_root / "_particle_staging" / bubble_id),
    ]


def _acoustic_entry(
    *,
    site: str,
    accepted_root: Path,
    accepted_index: dict[str, Any],
    output_root: Path,
    python_bin: str,
    bubble: dict[str, Any],
) -> dict[str, Any]:
    bubble_id = str(bubble["id"])
    setup_source, map_payload, protocol, sources = _acoustic_sources(
        accepted_root=accepted_root,
        accepted_index=accepted_index,
        bubble=bubble,
    )
    state_source = _accepted_hbi_state(accepted_root, accepted_index, bubble)
    sources.append(_accepted_source(accepted_index, state_source))
    symbol = str(map_payload.get("symbol", map_payload.get("diameter_symbol")))
    if not symbol or symbol == "None":
        raise ValueError(f"Accepted acoustic symbol is missing: {setup_source}")
    map_copy = output_root / bubble_id / "acoustic/inputs" / f"{symbol}.csv"
    row = _acoustic_map_row(map_payload, state_source)
    fieldnames = list(row)
    _write_single_csv_row(map_copy, fieldnames, row)
    command = _acoustic_command(
        site=site,
        python_bin=python_bin,
        state_source=state_source,
        map_copy=map_copy,
        output_root=output_root,
        bubble_id=bubble_id,
        symbol=symbol,
        protocol=protocol,
    )
    near_map = bubble_id == "d2"
    return {
        "modality": "acoustic",
        "status": (
            "user_accepted_near_map_reconstructed_exact_protocol_command"
            if near_map
            else "reconstructed_exact_protocol_command"
        ),
        "command": command,
        "materialized_map_values": str(map_copy),
        "materialized_map_values_sha256": _sha256(map_copy),
        "source_hashes": sources,
        "acceptance_notes": (
            ["d2 uses the user-accepted 17812.5 near-MAP ka, 0.1482% from the inferred MAP."]
            if near_map
            else []
        ),
        "provenance_gaps": [
            "The launch command is reconstructed from retained MAP and setup manifests; the scientific runner and protocol values are frozen."
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
    accepted_manifest: Path,
    output_root: Path,
    site: str,
    python_bin: str,
) -> dict[str, Any]:
    """Write a fresh replay plan and its portable, non-accepted inputs."""
    accepted_root = accepted_root.expanduser().resolve()
    if not accepted_root.is_dir():
        raise FileNotFoundError(f"Accepted direct-DPD artifact root is missing: {accepted_root}")
    accepted_index = _load_accepted_manifest(accepted_manifest, accepted_root)
    output_root = output_root.expanduser().resolve()
    _ensure_fresh_output(output_root)

    entries: list[dict[str, Any]] = []
    for bubble in BUBBLES:
        bubble_id = str(bubble["id"])
        mechanical_manifest, direct_map, mechanical_sources = _mechanical_manifest(
            accepted_root=accepted_root,
            accepted_index=accepted_index,
            bubble=bubble,
            output_root=output_root,
        )
        mechanical_sources.append(
            _accepted_source(
                accepted_index,
                _mechanical_protocol_source(accepted_root, bubble),
            )
        )
        mechanical = {
            "modality": "mechanical",
            "status": "exact_protocol_command",
            "command": _mechanical_command(
                site=site,
                python_bin=python_bin,
                accepted_root=accepted_root,
                accepted_index=accepted_index,
                bubble=bubble,
                output_root=output_root,
                direct_map=direct_map,
            ),
            "materialized_phase3b_map_manifest": str(mechanical_manifest),
            "materialized_phase3b_map_manifest_sha256": _sha256(mechanical_manifest),
            "source_hashes": mechanical_sources,
            "provenance_gaps": [],
        }
        acoustic = _acoustic_entry(
            site=site,
            accepted_root=accepted_root,
            accepted_index=accepted_index,
            output_root=output_root,
            python_bin=python_bin,
            bubble=bubble,
        )
        entries.append({"bubble_id": bubble_id, **bubble, "mechanical": mechanical, "acoustic": acoustic})

    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "static_dry_run_only",
        "submission": "not performed",
        "accepted_artifact_root": str(accepted_root),
        "accepted_artifact_manifest": str(Path(accepted_index["path"])),
        "accepted_artifact_manifest_sha256": str(accepted_index["sha256"]),
        "site": site,
        "python_bin": python_bin,
        "provenance": runtime_provenance(
            repo_root=REPO_ROOT,
            site=site,
            requested_python_bin=python_bin,
        ),
        "runtime_activation": {
            "required_before_execution": True,
            "site_modules": (
                "Load the Mirheo/OpenMPI/CUDA modules required by the selected site wrapper."
            ),
            "shared_activation": (
                "Set MESOUQ_SITE_RUNTIME_ROOT and MESOUQ_REPO_ROOT, source "
                "scripts/platforms/hpc/site_env.sh, call "
                f"mesouq_activate_site_env {site} \"$MESOUQ_REPO_ROOT\", then source "
                "\"$MESOUQ_SITE_RUNTIME_ROOT/gv_venv/env.sh\"."
            ),
            "reason": (
                "The verified Mirheo interpreter depends on libraries and environment variables "
                "provided by the activated site runtime."
            ),
        },
        "runtime_source_hashes": _source_hashes(
            (MECHANICAL_RUNNER, *MECHANICAL_EVALUATORS.values(), *ACOUSTIC_RUNTIME_SOURCES)
        ),
        "bubbles": entries,
    }
    plan_path = output_root / "direct_dpd_replay_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, default=ACCEPTED_ROOT_DEFAULT)
    parser.add_argument(
        "--accepted-manifest",
        type=Path,
        required=True,
        help="Locked manifest for the accepted artifact root.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--site", choices=VALID_SITES, default=None)
    parser.add_argument(
        "--python-bin",
        required=True,
        help="Verified Mirheo-capable Python executable embedded in the static replay commands.",
    )
    args = parser.parse_args(argv)
    try:
        site = _resolve_site(args.site)
        plan = materialize_direct_dpd_replay(
            accepted_root=args.accepted_root,
            accepted_manifest=args.accepted_manifest,
            output_root=args.output_root,
            site=site,
            python_bin=args.python_bin,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote static UQ_EMB direct-DPD replay plan: {args.output_root / 'direct_dpd_replay_plan.json'}")
    print(f"Planned {len(plan['bubbles']) * 2} modality entries; no commands were executed or submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
