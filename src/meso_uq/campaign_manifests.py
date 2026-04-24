from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from meso_uq.scheduler_routing import route_gpu_partition

MANIFEST_SCHEMA_VERSION = "1.0"
LANE_STAGE_ORDER = (
    "phase1",
    "phase2",
    "phase3b",
    "propagation_phase3b",
    "map_phase3b",
    "map_mirheo",
)
MANDATORY_MAIN_FIGURES = (
    "experimental_reference_curves.pdf",
    "experimental_reference_curves.png",
    "surrogate_validation.pdf",
    "surrogate_validation.png",
    "sensitivity.pdf",
    "sensitivity.png",
    "phase1_reduced_representative.pdf",
    "phase1_reduced_representative.png",
    "phase1_vs_phase3b_reduced_summary.pdf",
    "phase1_vs_phase3b_reduced_summary.png",
    "posterior_predictive_reduced_only.pdf",
    "posterior_predictive_reduced_only.png",
    "map_confirmation_reduced_representative.pdf",
    "map_confirmation_reduced_representative.png",
    "full_vs_reduced_map_overlay.pdf",
    "full_vs_reduced_map_overlay.png",
    "surrogate_group_holdout_examples.pdf",
    "surrogate_group_holdout_examples.png",
)
MANDATORY_SUPPLEMENTARY_FIGURES = (
    "map_confirmation_all.pdf",
    "map_confirmation_all.png",
    "map_confirmation_compression_2.1um.pdf",
    "map_confirmation_compression_2.1um.png",
    "map_confirmation_compression_2.9um.pdf",
    "map_confirmation_compression_2.9um.png",
    "map_confirmation_compression_3.0um.pdf",
    "map_confirmation_compression_3.0um.png",
    "map_confirmation_indentation_3.2um.pdf",
    "map_confirmation_indentation_3.2um.png",
    "map_confirmation_indentation_3.4um.pdf",
    "map_confirmation_indentation_3.4um.png",
    "map_confirmation_indentation_5.8um.pdf",
    "map_confirmation_indentation_5.8um.png",
    "full_vs_reduced_map_overlay_dpd.pdf",
    "full_vs_reduced_map_overlay_dpd.png",
)
MANDATORY_TABLES = (
    "table_crps_trusted.csv",
    "table_crps_trusted.tex",
    "table_coverage_v2.csv",
    "table_coverage_v2.tex",
    "map_parameters_reduced_phase3b.csv",
    "map_parameters_reduced_phase3b.tex",
    "map_parameter_comparison.csv",
    "map_parameter_comparison.tex",
    "map_l2_discrepancy_reduced.csv",
    "map_l2_discrepancy_reduced.tex",
    "surrogate_group_holdout_summary.csv",
    "sensitivity.csv",
    "sensitivity_bar_scores.csv",
)
REQUIRED_LANE_STAGES = LANE_STAGE_ORDER


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    try:
        process = subprocess.Popen(
            ["git", *args],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        output, _ = process.communicate()
    except OSError:
        return None
    if process.returncode != 0:
        return None
    value = (output or "").strip()
    return value or None


def load_git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    dirty_output = _run_git(repo_root, ["status", "--porcelain"])
    dirty = bool(dirty_output) if dirty_output is not None else None
    return {
        "git_commit": commit,
        "git_branch": branch,
        "dirty_worktree": dirty,
    }


def _file_entries(paths: Iterable[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in paths:
        resolved = Path(path).resolve()
        payload: dict[str, Any] = {"path": str(resolved)}
        if resolved.exists() and resolved.is_file():
            payload["sha256"] = sha256_path(resolved)
            payload["size_bytes"] = resolved.stat().st_size
        else:
            payload["sha256"] = None
            payload["missing"] = True
        entries.append(payload)
    return entries


def infer_phase2_backend(command: list[str]) -> str | None:
    if "--phase2-backend" not in command:
        return None
    return command[command.index("--phase2-backend") + 1]


def infer_gpu_job(
    *,
    step_name: str,
    command: list[str],
) -> bool:
    script_name = Path(command[1]).name if len(command) > 1 else ""
    if script_name == "run_map_mirheo.py":
        return True
    if script_name == "run_inference_stage.py":
        if "--stage" in command:
            stage = command[command.index("--stage") + 1]
        else:
            stage = step_name
        if stage == "phase2":
            return infer_phase2_backend(command) == "native-cuda"
        if "--device" in command:
            return command[command.index("--device") + 1] == "gpu"
        return False
    if script_name == "run_propagation.py" and "--device" in command:
        return command[command.index("--device") + 1] == "gpu"
    return False


def build_phase2_backend_policy(
    *,
    stage: str,
    profile: str,
    phase2_backend: str | None,
) -> dict[str, Any]:
    required_backend = "native-cuda" if stage == "phase2" and profile == "production" else None
    is_compliant: bool | None = None
    violation_reason: str | None = None
    if required_backend is not None:
        if phase2_backend is None:
            is_compliant = False
            violation_reason = "missing_phase2_backend"
        else:
            is_compliant = phase2_backend == required_backend
            if not is_compliant:
                violation_reason = (
                    f"phase2 backend '{phase2_backend}' does not satisfy required '{required_backend}'"
                )
    return {
        "required_phase2_backend": required_backend,
        "phase2_backend": phase2_backend,
        "is_compliant": is_compliant,
        "violation_reason": violation_reason,
    }


def build_partition_policy_metadata(
    *,
    is_gpu_job: bool,
    partition: str | None,
    time_limit: str | None,
) -> dict[str, Any]:
    expected_partition: str | None = None
    is_compliant: bool | None = None
    parse_error: str | None = None
    if is_gpu_job:
        if time_limit:
            try:
                expected_partition = route_gpu_partition(time_limit)
            except ValueError as exc:
                parse_error = str(exc)
        if expected_partition is not None and partition:
            is_compliant = partition == expected_partition
    return {
        "is_gpu_job": is_gpu_job,
        "policy": "strict_gpu_runtime_lt_30m_dev_else_gpu",
        "observed_partition": partition,
        "time_limit": time_limit,
        "expected_partition": expected_partition,
        "is_compliant": is_compliant,
        "parse_error": parse_error,
    }


def build_job_manifest(
    *,
    job_id: str,
    slurm_job_id: str | None,
    stage: str,
    lane: str,
    status: str,
    command: list[str],
    cwd: Path,
    start_utc: str,
    end_utc: str,
    duration_seconds: float,
    partition: str | None,
    time_limit: str | None,
    node_list: str | None,
    gpu_type: str | None,
    gpu_count: int | None,
    cpu_count: int | None,
    mem_mb: int | None,
    config_path: Path,
    logs: dict[str, str],
    backend_metadata: dict[str, Any],
    git_metadata: dict[str, Any],
    input_files: Iterable[Path],
    output_files: Iterable[Path],
) -> dict[str, Any]:
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "job_id": job_id,
        "slurm_job_id": slurm_job_id,
        "stage": stage,
        "lane": lane,
        "status": status,
        "git_commit": git_metadata.get("git_commit"),
        "git_branch": git_metadata.get("git_branch"),
        "dirty_worktree": git_metadata.get("dirty_worktree"),
        "command": command,
        "cwd": str(cwd),
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_seconds": duration_seconds,
        "partition": partition,
        "time_limit": time_limit,
        "node_list": node_list,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "cpu_count": cpu_count,
        "mem_mb": mem_mb,
        "config_path": str(config_path),
        "config_sha256": sha256_path(config_path) if config_path.exists() else None,
        "input_files": _file_entries(input_files),
        "output_files": _file_entries(output_files),
        "logs": logs,
        "backend_metadata": backend_metadata,
    }


def summarize_policy_violations(job_manifests: Iterable[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for manifest in job_manifests:
        job_id = str(manifest.get("job_id", "unknown_job"))
        backend_meta = manifest.get("backend_metadata", {})
        phase2_policy = backend_meta.get("phase2_backend_policy", {})
        partition_policy = backend_meta.get("partition_policy", {})
        if phase2_policy.get("is_compliant") is False:
            violations.append(
                f"{job_id}: phase2 backend policy violation ({phase2_policy.get('violation_reason')})"
            )
        if partition_policy.get("is_compliant") is False:
            violations.append(
                f"{job_id}: partition policy violation observed={partition_policy.get('observed_partition')} expected={partition_policy.get('expected_partition')}"
            )
    return violations


def build_lane_manifest(
    *,
    lane: str,
    selection: dict[str, str],
    steps: list[dict[str, Any]],
    artifacts: dict[str, str],
    job_manifest_paths: list[str],
    job_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses_by_stage = {step["name"]: ("passed" if step["returncode"] == 0 else "failed") for step in steps}
    ordered_statuses = [
        {"stage": stage, "status": statuses_by_stage.get(stage, "skipped")} for stage in LANE_STAGE_ORDER
    ]
    policy_violations = summarize_policy_violations(job_manifests)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "lane": lane,
        "selection": selection,
        "stage_statuses": ordered_statuses,
        "artifacts": artifacts,
        "job_manifests": job_manifest_paths,
        "policy": {
            "status": "pass" if not policy_violations else "fail",
            "violations": policy_violations,
        },
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_source_map_entry(
    key: str,
    value: object,
) -> dict[str, Any]:
    if isinstance(value, str):
        return {"source_lane": value, "source_artifacts": []}
    if isinstance(value, dict):
        source_lane = value.get("source_lane")
        source_artifacts = value.get("source_artifacts", [])
        if source_lane is not None and not isinstance(source_lane, str):
            raise ValueError(f"asset source map entry '{key}' has non-string source_lane")
        if not isinstance(source_artifacts, list):
            raise ValueError(f"asset source map entry '{key}' has non-list source_artifacts")
        return {
            "source_lane": source_lane,
            "source_artifacts": [str(item) for item in source_artifacts],
        }
    raise ValueError(
        f"asset source map entry '{key}' must be string or mapping, got {type(value).__name__}"
    )


def load_asset_source_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("asset source map root must be a JSON object")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        normalized[str(key)] = _normalize_source_map_entry(str(key), value)
    return normalized


def _lane_selection(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    selection = manifest.get("selection", {})
    if not isinstance(selection, dict):
        return None, None
    experiment = selection.get("experiment")
    model_family = selection.get("model_family")
    return (
        str(experiment) if isinstance(experiment, str) else None,
        str(model_family) if isinstance(model_family, str) else None,
    )


def _lane_artifact_paths(manifest: dict[str, Any]) -> list[str]:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return []
    return [str(value) for value in artifacts.values()]


def _tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token}


def _candidate_score(
    *,
    relative_path: str,
    experiment: str | None,
    model_family: str | None,
    artifact_paths: list[str],
) -> int:
    score = 0
    path_text = relative_path.lower()
    if "compression" in path_text:
        if experiment == "compression":
            score += 4
        else:
            return -999
    if "indentation" in path_text:
        if experiment == "indentation":
            score += 4
        else:
            return -999
    if "reduced" in path_text:
        if model_family == "reduced-model":
            score += 3
        else:
            score -= 1
    if "full_vs_reduced" not in path_text and "full" in path_text:
        if model_family == "full-model":
            score += 2

    path_tokens = _tokenize(path_text)
    artifact_tokens = set()
    for artifact_path in artifact_paths:
        artifact_tokens.update(_tokenize(artifact_path))
    score += len(path_tokens & artifact_tokens)
    return score


def derive_asset_source_map(
    *,
    required_relative_paths: Iterable[str],
    lane_manifests: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lane_records: list[dict[str, Any]] = []
    for manifest in lane_manifests:
        lane_name = manifest.get("lane")
        lane_records.append(
            {
                "lane": str(lane_name) if isinstance(lane_name, str) else None,
                "experiment": _lane_selection(manifest)[0],
                "model_family": _lane_selection(manifest)[1],
                "artifact_paths": _lane_artifact_paths(manifest),
            }
        )

    derived: dict[str, dict[str, Any]] = {}
    for relative_path in required_relative_paths:
        best_lane: str | None = None
        best_artifacts: list[str] = []
        best_score = -999
        for record in lane_records:
            score = _candidate_score(
                relative_path=relative_path,
                experiment=record["experiment"],
                model_family=record["model_family"],
                artifact_paths=record["artifact_paths"],
            )
            if score > best_score:
                best_score = score
                best_lane = record["lane"]
                best_artifacts = list(record["artifact_paths"])
            elif score == best_score and score > -999:
                # Deterministic tie-break to avoid non-reproducible manifests.
                lane_name = record["lane"]
                if lane_name is not None and (best_lane is None or lane_name < best_lane):
                    best_lane = lane_name
                    best_artifacts = list(record["artifact_paths"])
        if best_score <= -999:
            derived[relative_path] = {"source_lane": None, "source_artifacts": []}
        else:
            derived[relative_path] = {
                "source_lane": best_lane,
                "source_artifacts": best_artifacts,
            }
    return derived


def build_required_asset_entries(
    *,
    category: str,
    root: Path | None,
    required_files: Iterable[str],
    source_map: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    source_map = source_map or {}
    entries: list[dict[str, Any]] = []
    for filename in required_files:
        relative_path = f"{category}/{filename}"
        source_meta = source_map.get(relative_path, {"source_lane": None, "source_artifacts": []})
        file_path = root / filename if root is not None else None
        exists = bool(file_path and file_path.exists() and file_path.is_file())
        entries.append(
            {
                "asset_id": filename,
                "category": category,
                "relative_path": relative_path,
                "path": str(file_path.resolve()) if file_path is not None else None,
                "exists": exists,
                "sha256": (sha256_path(file_path) if exists and file_path is not None else None),
                "source_lane": source_meta.get("source_lane"),
                "source_artifacts": list(source_meta.get("source_artifacts", [])),
            }
        )
    return entries


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lane_stage_failures(lane_manifest: dict[str, Any], lane_manifest_path: Path) -> list[str]:
    failures: list[str] = []
    lane_name = str(lane_manifest.get("lane", lane_manifest_path.stem))
    stage_statuses = lane_manifest.get("stage_statuses", [])
    status_by_stage = {
        str(item.get("stage")): str(item.get("status"))
        for item in stage_statuses
        if isinstance(item, dict)
    }
    for stage in REQUIRED_LANE_STAGES:
        status = status_by_stage.get(stage, "missing")
        if status != "passed":
            failures.append(f"{lane_name}: stage '{stage}' status is '{status}'")
    return failures


def _lane_policy_failures(lane_manifest: dict[str, Any], lane_manifest_path: Path) -> list[str]:
    failures: list[str] = []
    lane_name = str(lane_manifest.get("lane", lane_manifest_path.stem))
    policy = lane_manifest.get("policy", {})
    if not isinstance(policy, dict):
        failures.append(f"{lane_name}: policy section missing or invalid")
        return failures
    if policy.get("status") == "fail":
        for violation in policy.get("violations", []):
            failures.append(f"{lane_name}: policy violation {violation}")
    return failures


def _lane_artifact_failures(lane_manifest: dict[str, Any], lane_manifest_path: Path) -> list[str]:
    failures: list[str] = []
    lane_name = str(lane_manifest.get("lane", lane_manifest_path.stem))
    artifacts = lane_manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return [f"{lane_name}: artifacts section missing or invalid"]
    for artifact_key, artifact_path in artifacts.items():
        resolved = Path(str(artifact_path))
        if not resolved.exists():
            failures.append(f"{lane_name}: missing artifact '{artifact_key}' at {resolved}")
    for job_manifest in lane_manifest.get("job_manifests", []):
        resolved = Path(str(job_manifest))
        if not resolved.exists():
            failures.append(f"{lane_name}: missing job manifest at {resolved}")
    return failures


def aggregate_lane_hard_failures(lane_manifest_paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    lane_records: list[dict[str, Any]] = []
    failures: list[str] = []
    for manifest_path in lane_manifest_paths:
        payload = _read_json(manifest_path)
        lane_records.append({"lane_manifest_path": str(manifest_path), "lane_manifest": payload})
        failures.extend(_lane_stage_failures(payload, manifest_path))
        failures.extend(_lane_policy_failures(payload, manifest_path))
        failures.extend(_lane_artifact_failures(payload, manifest_path))
    return lane_records, failures


def aggregate_missing_asset_failures(asset_entries: Iterable[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for entry in asset_entries:
        if not entry.get("exists"):
            failures.append(
                f"missing asset: {entry.get('relative_path')} expected at {entry.get('path')}"
            )
    return failures


def build_paper_release_manifest(
    *,
    run_campaign_id: str,
    generated_at_utc: str,
    lane_manifest_paths: Iterable[Path],
    figures_main_entries: list[dict[str, Any]],
    figures_supplementary_entries: list[dict[str, Any]],
    table_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    lane_records, lane_failures = aggregate_lane_hard_failures(lane_manifest_paths)
    asset_failures = (
        aggregate_missing_asset_failures(figures_main_entries)
        + aggregate_missing_asset_failures(figures_supplementary_entries)
        + aggregate_missing_asset_failures(table_entries)
    )
    hard_failures = [*lane_failures, *asset_failures]
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "run_campaign_id": run_campaign_id,
        "release_status": "PASS" if not hard_failures else "FAIL",
        "lane_manifests": [record["lane_manifest_path"] for record in lane_records],
        "lanes": [record["lane_manifest"] for record in lane_records],
        "assets": {
            "figures_main": figures_main_entries,
            "figures_supplementary": figures_supplementary_entries,
            "tables": table_entries,
        },
        "hard_failures": hard_failures,
    }
