from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from meso_uq.noise.composite import build_composite_likelihood_spec


_ALLOWED_FAMILIES = {"noise_hierarchy"}
_ALLOWED_KIND_NOISE_FAMILIES = {
    "full_hierarchy",
    "gaussian",
    "measurement_uncertainty",
    "model_discrepancy",
    "noise_hierarchy",
    "surrogate_uncertainty",
}
_PRIVATE_PATH_FRAGMENTS = ("/ceph/hpc/home/eubrieucb", "/mnt/proj", "/scratch/project/eu-26-17/eubrieucb")
_LIKELIHOOD_REQUIRED_KIND_FAMILIES = {
    "full_hierarchy",
    "gaussian",
    "measurement_uncertainty",
    "model_discrepancy",
    "surrogate_uncertainty",
}
_MODE_ORDER = (
    "legacy",
    "noise_primitives",
    "measurement_uncertainty",
    "surrogate_covariance",
    "discrepancy",
    "full_hierarchy",
    "synthetic_recovery",
    "predictive_checks",
    "emb_comparison",
)
_MODE_TO_CONFIG = {
    "legacy": {
        "stage": "M1",
        "components": ["legacy"],
        "legacy_mode": "emb_legacy",
    },
    "noise_primitives": {
        "stage": "M2",
        "components": ["additive_noise", "relative_noise", "correlated_curve_noise", "heavy_tail"],
    },
    "measurement_uncertainty": {
        "stage": "M3",
        "components": ["additive_noise", "relative_noise", "contact_alignment", "geometry"],
    },
    "surrogate_covariance": {
        "stage": "M4",
        "components": [
            "additive_noise",
            "relative_noise",
            "contact_alignment",
            "geometry",
            "surrogate_covariance",
        ],
    },
    "discrepancy": {
        "stage": "M5",
        "components": ["model_discrepancy", "total_covariance"],
    },
    "full_hierarchy": {
        "stage": "M5",
        "components": [
            "additive_noise",
            "relative_noise",
            "contact_alignment",
            "geometry",
            "surrogate_covariance",
            "model_discrepancy",
            "total_covariance",
        ],
    },
    "synthetic_recovery": {
        "stage": "M6",
        "components": ["synthetic_recovery"],
    },
    "predictive_checks": {
        "stage": "M6",
        "components": ["predictive_checks"],
    },
    "emb_comparison": {
        "stage": "M7",
        "components": ["emb_comparison", "total_covariance"],
    },
}
_VALIDATION_CONFIGS = {"synthetic_recovery", "predictive_checks", "emb_comparison"}


@dataclass(frozen=True)
class NoiseConfigValidationResult:
    path: str
    config_name: str | None
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "config_name": self.config_name,
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class NoiseModeSpec:
    mode: str
    stage: str
    components: tuple[str, ...]
    legacy_mode: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "stage": self.stage,
            "components": list(self.components),
            "legacy_mode": self.legacy_mode,
        }


@dataclass(frozen=True)
class NoiseReleaseGateResult:
    passed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "summary": dict(self.summary),
        }


def load_noise_config(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: config must be a mapping.")
    return payload


def _validate_likelihood_config(label: str, location: str, value: Any, errors: list[str]):
    if not isinstance(value, dict):
        errors.append(f"{label}: {location} must be a mapping.")
        return None
    if "components" not in value:
        errors.append(f"{label}: {location}.components must explicitly list selected likelihood components.")
    try:
        return build_composite_likelihood_spec(value)
    except ValueError as exc:
        errors.append(f"{label}: {location} is invalid: {exc}")
        return None


def _compare_likelihood_to_mode(label: str, mode: str, spec: Any, errors: list[str]) -> None:
    expected = resolve_noise_mode(mode)
    components = tuple(component.value for component in spec.components)
    if spec.stage.value != expected.stage or components != expected.components or spec.legacy_mode != expected.legacy_mode:
        errors.append(f"{label}: likelihood must resolve mode {mode!r} as stage={expected.stage}, components={list(expected.components)}, legacy_mode={expected.legacy_mode!r}.")


def validate_noise_config_document(document: Mapping[str, Any], *, source: str | Path = "<config>") -> NoiseConfigValidationResult:
    label = str(source)
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(document.get("schema_version"), (str, int)):
        errors.append(f"{label}: schema_version is required.")

    kind = document.get("kind")
    legacy_shape = kind == "noise"
    likelihood_spec = None
    if legacy_shape:
        metadata = document.get("metadata")
        spec = document.get("spec")
        if not isinstance(metadata, dict):
            errors.append(f"{label}: kind=noise config must define metadata.")
            name = None
        else:
            name = metadata.get("id")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{label}: metadata.id must be a non-empty string.")
                name = None
            if not isinstance(metadata.get("name"), str) or not metadata.get("name"):
                errors.append(f"{label}: metadata.name must be a non-empty string.")
        if not isinstance(spec, dict):
            errors.append(f"{label}: kind=noise config must define spec.")
            spec = {}
        spec_family = spec.get("family")
        if not isinstance(spec_family, str) or not spec_family:
            errors.append(f"{label}: spec.family must be a non-empty string.")
        elif spec_family not in _ALLOWED_KIND_NOISE_FAMILIES:
            errors.append(f"{label}: unsupported spec.family {spec_family!r}. Expected one of: {sorted(_ALLOWED_KIND_NOISE_FAMILIES)}.")
        likelihood = spec.get("likelihood")
        if spec_family in _LIKELIHOOD_REQUIRED_KIND_FAMILIES and likelihood is None:
            errors.append(f"{label}: spec.likelihood is required for family {spec_family!r}.")
        elif likelihood is not None:
            likelihood_spec = _validate_likelihood_config(label, "spec.likelihood", likelihood, errors)
    else:
        name = document.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{label}: name must be a non-empty string.")
            name = None
        if document.get("family") not in _ALLOWED_FAMILIES:
            errors.append(f"{label}: family must be one of {sorted(_ALLOWED_FAMILIES)}.")
        if not isinstance(document.get("milestone"), str) or not document.get("milestone"):
            errors.append(f"{label}: milestone must be a non-empty string.")
        likelihood = document.get("likelihood")
        if name in _MODE_TO_CONFIG and likelihood is None:
            errors.append(f"{label}: likelihood is required for noise hierarchy mode {name!r}.")
        elif likelihood is not None:
            likelihood_spec = _validate_likelihood_config(label, "likelihood", likelihood, errors)
        if likelihood_spec is not None and name in _MODE_TO_CONFIG:
            _compare_likelihood_to_mode(label, str(name), likelihood_spec, errors)

    if name == "legacy":
        likelihood = document.get("likelihood")
        if not isinstance(likelihood, dict):
            errors.append(f"{label}: legacy config must define likelihood.")
        else:
            if str(likelihood.get("stage", "")).upper() not in {"M0", "M1"}:
                errors.append(f"{label}: legacy likelihood.stage must be M0 or M1.")
            components = likelihood.get("components")
            if components != ["legacy"]:
                errors.append(f"{label}: legacy likelihood.components must be ['legacy'].")
            if not likelihood.get("legacy_mode"):
                errors.append(f"{label}: legacy likelihood.legacy_mode is required.")
    if name in _VALIDATION_CONFIGS:
        if not isinstance(document.get("required_scenarios"), list) or not document.get("required_scenarios"):
            errors.append(f"{label}: {name} config must list required_scenarios.")
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            errors.append(f"{label}: {name} config must define artifacts.")
    if name == "emb_comparison":
        evidence = document.get("evidence")
        if not isinstance(evidence, dict):
            errors.append(f"{label}: emb_comparison config must define evidence.")
        else:
            if evidence.get("production_claim") is True and evidence.get("class") != "production_pass":
                errors.append(f"{label}: production_claim=true requires evidence.class=production_pass.")
    for trail, value in _iter_strings(document):
        joined = ".".join(str(part) for part in trail)
        if any(fragment in value for fragment in _PRIVATE_PATH_FRAGMENTS):
            errors.append(f"{label}: {joined} contains a private HPC path literal.")
        if value.startswith("/") and not joined.endswith("forbidden_literals"):
            errors.append(f"{label}: {joined} uses an absolute path literal.")
    if name in {"gaussian_observation", "contact_alignment", "geometry_uncertainty"}:
        warnings.append(f"{label}: config is a primitive-stage example; full hierarchy behavior is covered by full_hierarchy.example.yaml.")
    return NoiseConfigValidationResult(path=label, config_name=None if name is None else str(name), passed=not errors, errors=tuple(errors), warnings=tuple(warnings))


def validate_noise_config_file(path: str | Path) -> NoiseConfigValidationResult:
    config_path = Path(path)
    return validate_noise_config_document(load_noise_config(config_path), source=config_path)


def supported_noise_modes() -> tuple[str, ...]:
    return _MODE_ORDER


def resolve_noise_mode(mode: str) -> NoiseModeSpec:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _MODE_TO_CONFIG:
        expected = ", ".join(sorted(_MODE_TO_CONFIG))
        raise ValueError(f"Unsupported noise hierarchy mode {mode!r}. Expected one of: {expected}.")
    spec = build_composite_likelihood_spec(_MODE_TO_CONFIG[normalized])
    return NoiseModeSpec(
        mode=normalized,
        stage=spec.stage.value,
        components=tuple(component.value for component in spec.components),
        legacy_mode=spec.legacy_mode,
    )


def build_noise_artifact_index(manifests: Mapping[str, str | Path]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for label, raw_path in manifests.items():
        path = Path(raw_path)
        if not path.exists():
            entries[label] = {"path": path.as_posix(), "exists": False, "all_scenarios_passed": False}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
        metrics_path = _resolve_artifact(path, artifacts.get("metrics"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path and metrics_path.exists() else {}
        entries[label] = {
            "path": path.as_posix(),
            "exists": True,
            "git_commit": payload.get("provenance", {}).get("git_commit"),
            "git_branch": payload.get("provenance", {}).get("git_branch"),
            "git_status_clean": payload.get("provenance", {}).get("git_status_clean"),
            "evidence_class": payload.get("evidence_class"),
            "production_claim": payload.get("production_claim"),
            "required_scenarios": payload.get("required_scenarios"),
            "all_scenarios_passed": metrics.get("all_scenarios_passed"),
            "scenario_gate_statuses": metrics.get("scenario_gate_statuses"),
            "commands": payload.get("commands", {}),
            "configs": payload.get("configs", payload.get("config", {})),
            "residual_risk": payload.get("residual_risk", metrics.get("residual_risk_notes", metrics.get("known_limitations"))),
            "metric_summary": _summarize_metrics(metrics),
            "artifacts": artifacts,
            "artifact_existence": _artifact_existence(artifacts),
        }
    return {"schema_version": 1, "entries": entries}


def evaluate_release_gate(
    *,
    config_results: Sequence[NoiseConfigValidationResult],
    artifact_index: Mapping[str, Any],
    gate06_manifest: Mapping[str, Any],
    merge_boundary: str,
    no_karolina_interaction: bool,
) -> NoiseReleaseGateResult:
    failures: list[str] = []
    warnings: list[str] = []
    failed_configs = [result.path for result in config_results if not result.passed]
    if failed_configs:
        failures.append(f"Noise config validation failed for: {failed_configs}.")
    required_evidence = {"synthetic_recovery", "predictive_checks", "emb_comparison"}
    entries = artifact_index.get("entries", {}) if isinstance(artifact_index, dict) else {}
    missing = sorted(required_evidence - set(entries))
    if missing:
        failures.append(f"Artifact index is missing required evidence entries: {missing}.")
    for label in sorted(required_evidence & set(entries)):
        entry = entries[label]
        if entry.get("exists") is not True:
            failures.append(f"Evidence entry {label} does not exist.")
        if entry.get("all_scenarios_passed") is not True:
            failures.append(f"Evidence entry {label} does not report all_scenarios_passed=true.")
        statuses = entry.get("scenario_gate_statuses")
        if not isinstance(statuses, dict) or not statuses:
            failures.append(f"Evidence entry {label} does not record per-scenario gate statuses.")
        elif any(status != "pass" for status in statuses.values()):
            failures.append(f"Evidence entry {label} has non-pass scenario gate statuses: {statuses}.")
        if entry.get("git_status_clean") is not True:
            failures.append(f"Evidence entry {label} does not record git_status_clean=true.")
        if not entry.get("git_commit"):
            failures.append(f"Evidence entry {label} does not record a git commit.")
        if not entry.get("commands"):
            failures.append(f"Evidence entry {label} does not record regeneration commands.")
        missing_artifacts = [name for name, exists in entry.get("artifact_existence", {}).items() if exists is not True]
        if missing_artifacts:
            failures.append(f"Evidence entry {label} has missing artifact sidecars: {missing_artifacts}.")
    if gate06_manifest.get("pass") is not True:
        failures.append("Gate06 manifest does not pass.")
    if merge_boundary not in {"merged", "human_review_required"}:
        failures.append("merge_boundary must be merged or human_review_required.")
    if merge_boundary == "human_review_required":
        warnings.append("Required PRs are at an explicitly recorded human review/merge boundary.")
    if not no_karolina_interaction:
        failures.append("Release gate requires no active Karolina worktree or running session interaction.")
    passed = not failures
    summary = {
        "config_count": len(config_results),
        "artifact_entries": sorted(entries),
        "gate06_pass": gate06_manifest.get("pass"),
        "merge_boundary": merge_boundary,
        "no_karolina_interaction": bool(no_karolina_interaction),
    }
    return NoiseReleaseGateResult(passed=passed, failures=tuple(failures), warnings=tuple(warnings), summary=summary)


def _summarize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in metrics.items():
        if key in {"scenarios", "scenario_metrics", "scenario_artifacts"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        elif key in {"scenario_gate_statuses", "thresholds"} and isinstance(value, dict):
            summary[key] = value
    return summary


def _artifact_existence(artifacts: Mapping[str, Any]) -> dict[str, bool]:
    existence: dict[str, bool] = {}
    for key, value in artifacts.items():
        if isinstance(value, str) and value:
            existence[key] = Path(value).exists()
    return existence


def _resolve_artifact(manifest_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def _iter_strings(value: Any, trail: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], str]]:
    if isinstance(value, dict):
        items: list[tuple[tuple[Any, ...], str]] = []
        for key, nested in value.items():
            items.extend(_iter_strings(nested, trail + (key,)))
        return items
    if isinstance(value, list):
        items = []
        for index, nested in enumerate(value):
            items.extend(_iter_strings(nested, trail + (index,)))
        return items
    if isinstance(value, str):
        return [(trail, value)]
    return []


__all__ = [
    "NoiseConfigValidationResult",
    "NoiseModeSpec",
    "NoiseReleaseGateResult",
    "build_noise_artifact_index",
    "evaluate_release_gate",
    "load_noise_config",
    "resolve_noise_mode",
    "supported_noise_modes",
    "validate_noise_config_document",
    "validate_noise_config_file",
]
