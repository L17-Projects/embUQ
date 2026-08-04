from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from meso_uq.config import resolve_inference_config_path
from meso_uq.config.models import EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE
from meso_uq.experiments import load_experiments
from meso_uq.inference.emb_frequency_emulator import (
    FIT_PRIMARY_LABEL_ADMISSION_POLICY,
    NONMONOTONE_FREQUENCY_POLICY,
    DpdFrequencyEmulatorBank,
    DpdPolynomialFrequencyEmulatorBank,
)
from meso_uq.inference.emb_frequency_surface import DpdFrequencySurface
from meso_uq.inference.emb_parameterization import (
    emb_elastic_stiffness_unit_n_per_m,
    ka_to_legacy_yt,
    load_emb_runtime_defaults,
)

EMB_RESONANCE_EXPERIMENT = "resonance"
EMB_RESONANCE_SCHEMA_VERSION = "meso_uq.emb_resonance_hbi.dpd.v1"
ANALYTICAL_DPD_FORWARD_MODEL = "analytical_dpd_formula"
ANALYTICAL_VACUUM_SHELL_FORWARD_MODEL = "analytical_vacuum_shell"
DPD_FREQUENCY_SURFACE_FORWARD_MODEL = "artifact_surface"
DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL = "artifact_emulator_bank"
DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL = "artifact_polynomial_bank"
SONOVUE_RESONANCE_TARGETS_MHZ = {
    2.6: 3.1,
    3.2: 2.1,
    4.0: 1.6,
}
DEFINITY_POLYNOMIAL_RESONANCE_TARGETS_MHZ = {
    1.3: 2.21,
    4.68: 1.69,
    5.18: 1.70,
    5.59: 1.49,
    6.0: 1.49,
    6.18: 1.06,
    7.08: 1.06,
    7.2: 1.50,
    7.29: 0.90,
    7.86: 0.91,
    8.67: 0.80,
    8.87: 0.90,
    10.0: 0.80,
    10.09: 0.58,
    11.2: 0.58,
}
_DIAMETER_TOLERANCE_UM = 1.0e-9
_FREQUENCY_TOLERANCE_MHZ = 1.0e-9
_MIN_RESONANCE_STD_MHZ = 1.0e-12
_INVALID_PROPOSAL_STD_MHZ = 1.0e12
_SURROGATE_SUPPORT_TOLERANCE = 1.0e-6
PCHIP_INDEPENDENT_GO_SCHEMA = "paper_week20jul.pchip_bank_independent_go.v1"
PCHIP_PARTIAL_CANDIDATE_REPORT_SCHEMA = "meso_uq.emb_frequency_emulator_bank_build.v2"
PCHIP_PARTIAL_CANDIDATE_REPORT_STATUS = "candidate_partial_support_bank_built"
REQUIRED_PCHIP_INDEPENDENT_GO_GATES = frozenset(
    {
        "artifact_integrity_passed",
        "exact_diameter_support_passed",
        "fit_primary_nonmonotone_policy_passed",
        "frequency_target_compatibility_passed",
        "hashed_source_datasets_passed",
    }
)
POLYNOMIAL_BANK_REPORT_SCHEMA = "paper_week20jul.polynomial_frequency_bank_build.v1"
POLYNOMIAL_INDEPENDENT_GO_SCHEMA = "paper_week20jul.polynomial_bank_independent_go.v1"
POLYNOMIAL_PROMOTION_SCHEMA = "paper_week20jul.polynomial_emulator_family_promotion.v1"
POLYNOMIAL_PROMOTION_STATUS = "approved_for_vectorized_10k_hbi_readiness"
POLYNOMIAL_PROMOTION_SHA256 = "8efca2d8607a9ae02da714d29aa6b20907a21867ad5f0899c8aa7132264792e2"
APPROVED_POLYNOMIAL_FAMILY = "frequency_mhz_squared_free_intercept_quadratic"
REQUIRED_POLYNOMIAL_INDEPENDENT_GO_GATES = frozenset(
    {
        "artifact_integrity_passed",
        "degree_response_contract_passed",
        "dense_finite_positive_passed",
        "exact_diameter_support_passed",
        "fixed_kb_condition_passed",
        "hashed_source_datasets_passed",
        "no_fallback_passed",
        "scientific_family_selection_approved",
    }
)


def _target_map(values: Mapping[float, float]) -> dict[float, float]:
    return {_target_diameter_key(diameter): float(frequency) for diameter, frequency in values.items()}


def _observed_target_map(observations: Mapping[float, Mapping[str, Any]]) -> dict[float, float]:
    return {
        _target_diameter_key(diameter): float(row["frequency_MHz"])
        for diameter, row in observations.items()
    }


def _require_exact_target_mapping(
    observations: Mapping[float, Mapping[str, Any]],
    expected: Mapping[float, float],
    *,
    label: str,
) -> None:
    observed = _observed_target_map(observations)
    frozen = _target_map(expected)
    if set(observed) != set(frozen) or any(
        not math.isclose(observed[diameter], frequency, rel_tol=0.0, abs_tol=_FREQUENCY_TOLERANCE_MHZ)
        for diameter, frequency in frozen.items()
    ):
        raise ValueError(
            f"{label} acoustic observations must exactly match the frozen target mapping "
            f"{dict(sorted(frozen.items()))}; observed {dict(sorted(observed.items()))}."
        )


def _expected_definity_polynomial_targets(config: Mapping[str, Any]) -> dict[float, float]:
    policy = str(
        _resonance_mapping(config).get("definity_1p3_acoustic_policy", "include")
    ).strip().lower()
    if policy == "include":
        return dict(DEFINITY_POLYNOMIAL_RESONANCE_TARGETS_MHZ)
    if policy == "exclude":
        return {
            diameter: frequency
            for diameter, frequency in DEFINITY_POLYNOMIAL_RESONANCE_TARGETS_MHZ.items()
            if not math.isclose(float(diameter), 1.3, rel_tol=0.0, abs_tol=_DIAMETER_TOLERANCE_UM)
        }
    raise ValueError(
        "resonance.definity_1p3_acoustic_policy must be 'include' or 'exclude'."
    )


def _validate_approved_polynomial_family(
    bank: DpdPolynomialFrequencyEmulatorBank,
    report: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> None:
    family_contract = promotion.get("emulator_family")
    if not isinstance(family_contract, Mapping):
        raise ValueError("Polynomial promotion contract has no emulator_family mapping.")
    if report.get("selected_family") != APPROVED_POLYNOMIAL_FAMILY:
        raise ValueError("Polynomial bank report does not select the user-approved family.")
    if report.get("family_contract") != family_contract:
        raise ValueError("Polynomial bank report family_contract differs from the approved policy.")
    if bank.conditions.get("polynomial_family_contract") != family_contract:
        raise ValueError("Polynomial bank conditions do not freeze the approved family contract.")
    if bank.conditions.get("promotion_contract_sha256") != POLYNOMIAL_PROMOTION_SHA256:
        raise ValueError("Polynomial bank is not bound to the pinned promotion contract.")
    if bank.response_encodings != ("frequency_mhz_squared",):
        raise ValueError("Approved polynomial banks must encode frequency_mhz_squared only.")
    for entry in bank.emulators:
        if entry.degree != 2 or entry.response_encoding != "frequency_mhz_squared":
            raise ValueError(
                "Approved polynomial bank entries must all be degree-2 frequency_mhz_squared."
            )
        if entry.ka_offset_dpd != 0.0 or entry.ka_scale_dpd != 1.0:
            raise ValueError("Approved polynomial bank entries must use raw ka_dpd coefficients.")
        if entry.ka_bounds_dpd[0] != 0.0:
            raise ValueError("Approved polynomial evaluation support must start at ka_dpd=0.")
        a0, a1, a2 = (float(value) for value in entry.coefficients)
        if a0 < 0.0:
            raise ValueError("Approved polynomial fitted intercept must be nonnegative.")
        derivatives = np.asarray(
            [a1 + 2.0 * a2 * bound for bound in entry.ka_bounds_dpd],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(derivatives)) or np.any(derivatives < 0.0):
            raise ValueError(
                "Approved polynomial frequency-squared response must be non-decreasing "
                f"through declared support at diameter {entry.diameter_um:g} um."
            )
        calibration = entry.validation.get("calibration_ka_bounds_dpd")
        if (
            not isinstance(calibration, Sequence)
            or isinstance(calibration, (str, bytes))
            or len(calibration) != 2
            or float(calibration[0]) <= 0.0
            or float(calibration[1]) != entry.ka_bounds_dpd[1]
        ):
            raise ValueError("Approved polynomial entry has invalid calibration-window metadata.")
        if entry.validation.get("lower_calibration_window_extrapolation_user_approved") is not True:
            raise ValueError("Approved polynomial entry does not disclose lower-window extrapolation.")


@dataclass(frozen=True)
class EmbResonanceConstants:
    rho_l_dpd: float
    p0_dpd: float
    kappa_g: float = 1.095
    kappa_g_source: str = "explicit"
    c_gas_dpd: float | None = None
    rho_gas_dpd: float | None = None
    p_gas_dpd: float | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        defaults: Mapping[str, Any],
    ) -> "EmbResonanceConstants":
        payload = payload or {}
        rho_l_dpd = _coerce_positive(
            payload.get("rho_l_dpd", payload.get("rho_l", defaults.get("rhow"))),
            "rho_l_dpd",
        )
        p0_dpd = _coerce_positive(
            payload.get("p0_dpd", payload.get("p0", defaults.get("bpress"))),
            "p0_dpd",
        )
        gas_keys = ("c_gas_dpd", "rho_gas_dpd", "p_gas_dpd")
        has_gas_inputs = any(key in payload for key in gas_keys)
        has_explicit_kappa = "kappa_g" in payload
        if has_gas_inputs and has_explicit_kappa:
            raise ValueError(
                "Resonance gas exponent is ambiguous: set either resonance.dpd_constants.kappa_g "
                "or all of c_gas_dpd/rho_gas_dpd/p_gas_dpd, not both."
            )
        if has_gas_inputs:
            missing = [key for key in gas_keys if key not in payload]
            if missing:
                raise ValueError(
                    "Resonance gas exponent from DPD gas inputs requires all of "
                    f"c_gas_dpd/rho_gas_dpd/p_gas_dpd; missing {missing}."
                )
            c_gas_dpd = _coerce_positive(payload["c_gas_dpd"], "c_gas_dpd")
            rho_gas_dpd = _coerce_positive(payload["rho_gas_dpd"], "rho_gas_dpd")
            p_gas_dpd = _coerce_positive(payload["p_gas_dpd"], "p_gas_dpd")
            kappa_g = c_gas_dpd**2 * rho_gas_dpd / p_gas_dpd
            if not math.isfinite(kappa_g) or kappa_g <= 0.0:
                raise ValueError(f"computed kappa_g must be finite and positive; got {kappa_g!r}.")
            return cls(
                rho_l_dpd=rho_l_dpd,
                p0_dpd=p0_dpd,
                kappa_g=float(kappa_g),
                kappa_g_source="gas_dpd",
                c_gas_dpd=c_gas_dpd,
                rho_gas_dpd=rho_gas_dpd,
                p_gas_dpd=p_gas_dpd,
            )
        return cls(
            rho_l_dpd=rho_l_dpd,
            p0_dpd=p0_dpd,
            kappa_g=_coerce_positive(payload.get("kappa_g", cls.kappa_g), "kappa_g"),
            kappa_g_source="explicit" if has_explicit_kappa else "default",
        )

    def to_dict(self) -> dict[str, float | str]:
        payload: dict[str, float | str] = {
            "rho_l_dpd": self.rho_l_dpd,
            "p0_dpd": self.p0_dpd,
            "kappa_g": self.kappa_g,
            "kappa_g_source": self.kappa_g_source,
        }
        if self.c_gas_dpd is not None:
            payload["c_gas_dpd"] = self.c_gas_dpd
        if self.rho_gas_dpd is not None:
            payload["rho_gas_dpd"] = self.rho_gas_dpd
        if self.p_gas_dpd is not None:
            payload["p_gas_dpd"] = self.p_gas_dpd
        return payload


@dataclass(frozen=True)
class EmbVacuumShellConstants:
    rho_shell_kg_m3: float
    shell_thickness_m: float
    poisson_ratio: float
    fscale: float
    unit_length_m: float
    stiffness_unit_n_per_m: float

    @property
    def surface_mass_kg_m2(self) -> float:
        return self.rho_shell_kg_m3 * self.shell_thickness_m

    def to_dict(self) -> dict[str, float | str]:
        return {
            "rho_shell_kg_m3": self.rho_shell_kg_m3,
            "shell_thickness_m": self.shell_thickness_m,
            "shell_thickness_nm": self.shell_thickness_m * 1.0e9,
            "poisson_ratio": self.poisson_ratio,
            "fscale": self.fscale,
            "unit_length_m": self.unit_length_m,
            "stiffness_unit_n_per_m": self.stiffness_unit_n_per_m,
            "surface_mass_kg_m2": self.surface_mass_kg_m2,
            "formula": "omega^2 = 4 * (ka_dpd * stiffness_unit_n_per_m) / (rho_shell * h * R^2)",
        }


def _coerce_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive value; got {value!r}.")
    return result


def _coerce_finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return result


def _coerce_positive_acoustic_frequency_mhz(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{name} must be a finite positive acoustic frequency in MHz; got {value!r}."
        )
    return result


def _diameter_key(value: float) -> float:
    return round(float(value), 9)


def _target_diameter_key(value: float) -> float:
    return _diameter_key(float(value))


def _resonance_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    resonance = config.get("resonance") or {}
    if not isinstance(resonance, Mapping):
        raise ValueError("resonance config must be a mapping.")
    return resonance


def resonance_agent(config: Mapping[str, Any]) -> str:
    agent = str(_resonance_mapping(config).get("agent", "sonovue")).strip().lower()
    return "sonovue" if agent == "sono_vue" else agent


def resonance_forward_model(config: Mapping[str, Any]) -> str:
    """Return the explicit resonance forward-model selection.

    Existing configurations omit this field and keep their analytical DPD
    formula.  A DPD frequency surface is opt-in because it is only valid once
    an accepted artifact has been produced on the specified support.
    """

    resonance = _resonance_mapping(config)
    evaluator = resonance.get("evaluator")
    if evaluator is None:
        return ANALYTICAL_DPD_FORWARD_MODEL
    if not isinstance(evaluator, Mapping):
        raise ValueError("resonance.evaluator must be a mapping.")
    value = str(evaluator.get("mode", ANALYTICAL_DPD_FORWARD_MODEL)).strip().lower()
    supported = {
        ANALYTICAL_DPD_FORWARD_MODEL,
        ANALYTICAL_VACUUM_SHELL_FORWARD_MODEL,
        DPD_FREQUENCY_SURFACE_FORWARD_MODEL,
        DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL,
        DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL,
    }
    if value not in supported:
        raise ValueError(
            "resonance.evaluator.mode must be one of "
            f"{sorted(supported)!r}."
        )
    return value


def _resolve_project_path(project_root: str | Path, value: Any, field_name: str) -> Path:
    if not value:
        raise ValueError(f"Missing {field_name}.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def _resolve_relocated_provenance_path(
    project_root: str | Path,
    value: Any,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve a pinned workspace artifact after a cross-site relocation."""

    recorded_path = str(value or "")
    path = Path(recorded_path).expanduser().resolve()
    if path.is_file():
        return path

    replacement = (overrides or {}).get(recorded_path)
    if replacement:
        return _resolve_project_path(
            project_root,
            replacement,
            f"resonance.evaluator.provenance_path_overrides[{recorded_path!r}]",
        )

    try:
        workspace_index = path.parts.index("workspace")
    except ValueError:
        return path

    root = Path(project_root).expanduser().resolve()
    workspace_root = next(
        (candidate for candidate in (root, *root.parents) if candidate.name == "workspace"),
        None,
    )
    if workspace_root is None:
        return path
    return workspace_root.joinpath(*path.parts[workspace_index + 1 :]).resolve()


def _normalized_sha256(value: Any, field_name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            f"{field_name} must be a 64-character hexadecimal SHA-256 digest."
        )
    return digest


@lru_cache(maxsize=None)
def _verify_immutable_artifact_sha256(path_text: str, expected_sha256: str) -> str:
    actual_sha256 = hashlib.sha256(Path(path_text).read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "DPD frequency artifact SHA-256 mismatch: "
            f"expected={expected_sha256!r}, actual={actual_sha256!r}, path={path_text}."
        )
    return actual_sha256


def _artifact_sha256(path: Path, expected_sha256: Any) -> str:
    if expected_sha256 is None:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    expected = _normalized_sha256(
        expected_sha256,
        "resonance.evaluator.artifact_sha256",
    )
    return _verify_immutable_artifact_sha256(str(path), expected)


def _load_json_mapping(path: Path, field_name: str) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must contain a mapping.")
    return payload


def _validate_external_bank_release(
    evaluator: Mapping[str, Any],
    *,
    project_root: str | Path,
    bank: DpdFrequencyEmulatorBank,
    bank_sha256: str,
) -> Mapping[str, Any] | None:
    """Validate the independent handoff required by bank-side-only artifacts."""

    fields = (
        "bank_build_report_path",
        "bank_build_report_sha256",
        "independent_go_path",
        "independent_go_sha256",
    )
    supplied = [field for field in fields if evaluator.get(field) is not None]
    bank_side_only = (
        bank.provenance.get("inference_ready") is not True
        and bank.provenance.get("bank_side_release_ready") is True
    )
    if not supplied and not bank_side_only:
        return None
    if set(supplied) != set(fields):
        raise ValueError(
            "Bank-side PCHIP release requires pinned bank_build_report and "
            "independent_go paths and SHA-256 digests."
        )

    report_path = _resolve_project_path(
        project_root,
        evaluator.get("bank_build_report_path"),
        "resonance.evaluator.bank_build_report_path",
    )
    go_path = _resolve_project_path(
        project_root,
        evaluator.get("independent_go_path"),
        "resonance.evaluator.independent_go_path",
    )
    if not report_path.is_file():
        raise FileNotFoundError(f"PCHIP bank build report not found: {report_path}")
    if not go_path.is_file():
        raise FileNotFoundError(f"Independent PCHIP GO receipt not found: {go_path}")
    report_sha256 = _artifact_sha256(
        report_path,
        evaluator.get("bank_build_report_sha256"),
    )
    report = _load_json_mapping(
        report_path,
        "resonance.evaluator.bank_build_report_path",
    )
    _artifact_sha256(go_path, evaluator.get("independent_go_sha256"))
    gate = _load_json_mapping(go_path, "resonance.evaluator.independent_go_path")
    if gate.get("schema") != PCHIP_INDEPENDENT_GO_SCHEMA:
        raise ValueError("Independent PCHIP GO receipt has an unsupported schema.")
    if gate.get("status") != "GO" or gate.get("independent_audit") is not True:
        raise ValueError("PCHIP bank has no explicit independent GO for HBI.")
    if resonance_agent({"resonance": {"agent": gate.get("agent")}}) != bank.agent:
        raise ValueError("Independent PCHIP GO agent does not match the bank artifact.")
    if gate.get("bank_artifact_sha256") != bank_sha256:
        raise ValueError("Independent PCHIP GO refers to a different bank artifact.")
    if gate.get("bank_build_report_sha256") != report_sha256:
        raise ValueError("Independent PCHIP GO refers to a different bank build report.")
    gates = gate.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != REQUIRED_PCHIP_INDEPENDENT_GO_GATES:
        raise ValueError("Independent PCHIP GO is missing the exact required gate set.")
    failed = sorted(name for name, passed in gates.items() if passed is not True)
    if failed:
        raise ValueError(
            "Independent PCHIP GO contains failed gates: " + ", ".join(failed)
        )
    partial_candidate_release = False
    if (
        bank.provenance.get("candidate_only") is True
        and bank.provenance.get("candidate_mode") == "contiguous_admitted_segments"
    ):
        if (
            report.get("schema") != PCHIP_PARTIAL_CANDIDATE_REPORT_SCHEMA
            or report.get("status") != PCHIP_PARTIAL_CANDIDATE_REPORT_STATUS
        ):
            raise ValueError(
                "Partial PCHIP candidate bank is not bound to its candidate build report."
            )
        if report.get("artifact_sha256") != bank_sha256:
            raise ValueError("Partial PCHIP candidate report refers to a different bank artifact.")
        for field in (
            "no_clipping",
            "no_extrapolation",
            "no_diameter_interpolation",
            "no_analytical_fallback",
        ):
            if report.get(field) is not True:
                raise ValueError(f"Partial PCHIP candidate report does not enforce {field}.")
        if report.get("held_diameters_um") not in ([], ()):
            raise ValueError("Partial PCHIP candidate report retains held requested diameters.")
        artifact_diameters = sorted(
            _target_diameter_key(entry.diameter_um) for entry in bank.emulators
        )
        for field in ("requested_diameters_um", "built_diameters_um"):
            values = report.get(field)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"Partial PCHIP candidate report has invalid {field}.")
            observed = sorted(
                _target_diameter_key(_coerce_positive(value, f"report.{field}"))
                for value in values
            )
            if observed != artifact_diameters:
                raise ValueError(
                    f"Partial PCHIP candidate report {field} differs from the bank artifact."
                )
        segment_gate = report.get("per_diameter_segment_gate")
        if not isinstance(segment_gate, Mapping):
            raise ValueError("Partial PCHIP candidate report has no per-diameter segment gate.")
        segment_by_diameter: dict[float, Mapping[str, Any]] = {}
        for row in segment_gate.values():
            if not isinstance(row, Mapping):
                raise ValueError("Partial PCHIP candidate segment gate contains an invalid row.")
            diameter = _target_diameter_key(
                _coerce_positive(row.get("diameter_um"), "candidate segment diameter_um")
            )
            if diameter in segment_by_diameter:
                raise ValueError("Partial PCHIP candidate segment gate contains duplicate diameters.")
            segment_by_diameter[diameter] = row
        if sorted(segment_by_diameter) != artifact_diameters:
            raise ValueError(
                "Partial PCHIP candidate segment gate differs from the bank diameters."
            )
        for entry in bank.emulators:
            row = segment_by_diameter[_target_diameter_key(entry.diameter_um)]
            if row.get("status") != "go":
                raise ValueError(
                    "Partial PCHIP candidate contains an unreleased diameter segment."
                )
            bounds = row.get("support_ka_bounds_dpd")
            if (
                not isinstance(bounds, Sequence)
                or isinstance(bounds, (str, bytes))
                or len(bounds) != 2
                or not np.allclose(
                    np.asarray(bounds, dtype=float),
                    np.asarray(entry.ka_bounds_dpd, dtype=float),
                    rtol=0.0,
                    atol=1.0e-9,
                )
            ):
                raise ValueError(
                    "Partial PCHIP candidate segment support differs from the bank entry."
                )
        if bank.validation.get("candidate_contiguous_segment_gate_passed") is not True:
            raise ValueError("Partial PCHIP candidate bank failed its contiguous-segment gate.")
        partial_candidate_release = True
    if bank_side_only:
        incomplete_entries = [
            entry.diameter_um
            for entry in bank.emulators
            if entry.provenance.get("bank_side_release_ready") is not True
        ]
        if incomplete_entries:
            raise ValueError(
                "Bank-side PCHIP artifact contains entries not released by the bank lane "
                f"at diameter(s) {incomplete_entries} um."
            )
    release = dict(gate)
    release["_partial_candidate_release"] = partial_candidate_release
    return release


@lru_cache(maxsize=None)
def _load_frequency_surface(path_text: str, artifact_sha256: str) -> DpdFrequencySurface:
    del artifact_sha256
    return DpdFrequencySurface.load(path_text)


def resolve_dpd_frequency_surface(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> DpdFrequencySurface:
    """Load and validate the explicit DPD frequency surface configured for HBI."""

    resonance = _resonance_mapping(config)
    evaluator = resonance.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("resonance.evaluator must be a mapping for artifact_surface mode.")
    path = _resolve_project_path(project_root, evaluator.get("artifact_path"), "resonance.evaluator.artifact_path")
    if not path.is_file():
        raise FileNotFoundError(f"DPD frequency surface artifact not found: {path}")
    actual_sha256 = _artifact_sha256(path, evaluator.get("artifact_sha256"))
    surface = _load_frequency_surface(str(path), actual_sha256)
    agent = resonance_agent(config)
    if surface.agent != agent:
        raise ValueError(
            "DPD frequency surface agent does not match resonance.agent: "
            f"artifact={surface.agent!r}, config={agent!r}."
        )
    expected_fixed_kb = evaluator.get("expected_fixed_kb_dpd")
    if expected_fixed_kb is not None:
        expected = _coerce_positive(expected_fixed_kb, "resonance.evaluator.expected_fixed_kb_dpd")
        actual_value = surface.conditions.get("fixed_kb_dpd")
        if actual_value is None:
            raise ValueError(
                "DPD frequency surface does not record conditions.fixed_kb_dpd "
                "required by resonance.evaluator.expected_fixed_kb_dpd."
            )
        actual = _coerce_positive(actual_value, "frequency surface conditions.fixed_kb_dpd")
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(
                "DPD frequency surface fixed kb does not match the configured expectation: "
                f"artifact={actual:g}, config={expected:g}."
            )
    return surface


@lru_cache(maxsize=None)
def _load_frequency_emulator_bank(
    path_text: str,
    artifact_sha256: str,
) -> DpdFrequencyEmulatorBank:
    del artifact_sha256
    return DpdFrequencyEmulatorBank.load(path_text)


def resolve_dpd_frequency_emulator_bank(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> DpdFrequencyEmulatorBank:
    """Load and validate the exact-diameter DPD emulator bank used by HBI."""

    resonance = _resonance_mapping(config)
    evaluator = resonance.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError(
            "resonance.evaluator must be a mapping for artifact_emulator_bank mode."
        )
    path = _resolve_project_path(
        project_root,
        evaluator.get("artifact_path"),
        "resonance.evaluator.artifact_path",
    )
    if not path.is_file():
        raise FileNotFoundError(f"DPD frequency emulator bank artifact not found: {path}")
    artifact_sha256 = evaluator.get("artifact_sha256")
    if artifact_sha256 is None:
        raise ValueError(
            "resonance.evaluator.artifact_sha256 is required for production-safe "
            "artifact_emulator_bank evaluation."
        )
    actual_sha256 = _artifact_sha256(path, artifact_sha256)
    bank = _load_frequency_emulator_bank(str(path), actual_sha256)
    agent = resonance_agent(config)
    if bank.agent != agent:
        raise ValueError(
            "DPD frequency emulator bank agent does not match resonance.agent: "
            f"artifact={bank.agent!r}, config={agent!r}."
        )
    independent_release = _validate_external_bank_release(
        evaluator,
        project_root=project_root,
        bank=bank,
        bank_sha256=actual_sha256,
    )
    if (
        bank.provenance.get("inference_ready") is not True
        and independent_release is None
    ):
        raise ValueError(
            "DPD frequency emulator bank is not inference ready; production HBI "
            "requires either an inference-ready bank or a pinned independent-GO handoff."
        )
    if bank.validation.get("label_admission_policy") != FIT_PRIMARY_LABEL_ADMISSION_POLICY:
        raise ValueError(
            "DPD frequency emulator bank is missing the required fit-primary admission gate."
        )
    if bank.validation.get("nonmonotone_frequency_policy") != NONMONOTONE_FREQUENCY_POLICY:
        raise ValueError(
            "DPD frequency emulator bank is missing the explicit nonmonotone policy."
        )
    if bank.validation.get("all_dense_grids_finite_positive") is not True:
        raise ValueError(
            "DPD frequency emulator bank is missing dense finite-positive validation."
        )
    benchmark_only = bank.provenance.get("benchmark_only") is True
    allow_benchmark = evaluator.get("allow_benchmark_artifact") is True
    if benchmark_only and not allow_benchmark:
        raise ValueError(
            "DPD frequency emulator bank is benchmark-only; explicit benchmark "
            "opt-in is forbidden in scientific HBI configurations."
        )
    if allow_benchmark and not benchmark_only:
        raise ValueError(
            "resonance.evaluator.allow_benchmark_artifact is enabled for a "
            "non-benchmark DPD frequency emulator bank."
        )
    independently_released_partial_bank = (
        independent_release is not None
        and independent_release.get("_partial_candidate_release") is True
    )
    if (
        not benchmark_only
        and bank.validation.get("production_quality_gate_passed") is not True
        and not independently_released_partial_bank
    ):
        raise ValueError(
            "DPD frequency emulator bank is missing the production coverage and "
            "holdout-error quality gate."
        )
    if not benchmark_only:
        source_datasets = bank.provenance.get("source_datasets")
        if (
            not isinstance(source_datasets, Sequence)
            or isinstance(source_datasets, (str, bytes))
            or not source_datasets
        ):
            raise ValueError(
                "Production DPD frequency emulator bank must record hashed source datasets."
            )
        for index, record in enumerate(source_datasets):
            if not isinstance(record, Mapping) or not record.get("path"):
                raise ValueError(
                    "Production DPD frequency emulator bank source dataset "
                    f"record {index} is invalid."
                )
            _normalized_sha256(
                record.get("sha256"),
                f"frequency emulator bank provenance.source_datasets[{index}].sha256",
            )
    incomplete_entries = [
        entry.diameter_um
        for entry in bank.emulators
        if entry.provenance.get("inference_ready") is not True
        and not (
            independent_release is not None
            and (
                entry.provenance.get("bank_side_release_ready") is True
                or independent_release.get("_partial_candidate_release") is True
            )
        )
    ]
    if incomplete_entries:
        raise ValueError(
            "DPD frequency emulator bank contains non-production emulator entries "
            f"at diameter(s) {incomplete_entries} um."
        )
    expected_fixed_kb = evaluator.get("expected_fixed_kb_dpd")
    if expected_fixed_kb is None:
        raise ValueError(
            "resonance.evaluator.expected_fixed_kb_dpd is required for "
            "artifact_emulator_bank evaluation."
        )
    expected = _coerce_positive(
        expected_fixed_kb,
        "resonance.evaluator.expected_fixed_kb_dpd",
    )
    actual_value = bank.conditions.get("fixed_kb_dpd")
    if actual_value is None:
        raise ValueError(
            "DPD frequency emulator bank does not record conditions.fixed_kb_dpd "
            "required by resonance.evaluator.expected_fixed_kb_dpd."
        )
    actual = _coerce_positive(
        actual_value,
        "frequency emulator bank conditions.fixed_kb_dpd",
    )
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "DPD frequency emulator bank fixed kb does not match the configured "
            f"expectation: artifact={actual:g}, config={expected:g}."
        )
    return bank


def _validate_polynomial_bank_release(
    evaluator: Mapping[str, Any],
    *,
    project_root: str | Path,
    bank: DpdPolynomialFrequencyEmulatorBank,
    bank_sha256: str,
) -> Mapping[str, Any] | None:
    benchmark_only = bank.provenance.get("benchmark_only") is True
    allow_benchmark = evaluator.get("allow_benchmark_artifact") is True
    if benchmark_only:
        if not allow_benchmark:
            raise ValueError(
                "Polynomial frequency bank is a deterministic benchmark fixture and requires "
                "explicit non-scientific opt-in."
            )
        if bank.validation.get("deterministic_fixture_gate_passed") is not True:
            raise ValueError("Polynomial benchmark fixture is missing its deterministic gate.")
        return None
    if allow_benchmark:
        raise ValueError(
            "resonance.evaluator.allow_benchmark_artifact is forbidden for a scientific bank."
        )

    if bank.provenance.get("inference_ready") is not True:
        raise ValueError("Polynomial frequency bank is not marked inference ready.")
    if bank.provenance.get("scientific_policy_status") != "audited_final":
        raise ValueError(
            "Polynomial frequency bank scientific policy is not frozen as audited_final."
        )
    if bank.validation.get("production_quality_gate_passed") is not True:
        raise ValueError("Polynomial frequency bank failed its production quality gate.")
    build_tool = bank.provenance.get("build_tool")
    if not isinstance(build_tool, Mapping):
        raise ValueError("Polynomial frequency bank has no pinned build tool.")
    provenance_path_overrides = evaluator.get("provenance_path_overrides")
    build_tool_path = _resolve_relocated_provenance_path(
        project_root,
        build_tool.get("path"),
        provenance_path_overrides,
    )
    if not build_tool_path.is_file():
        raise FileNotFoundError(f"Polynomial bank build tool not found: {build_tool_path}")
    _artifact_sha256(build_tool_path, build_tool.get("sha256"))
    incomplete_entries = [
        entry.diameter_um
        for entry in bank.emulators
        if entry.provenance.get("inference_ready") is not True
    ]
    if incomplete_entries:
        raise ValueError(
            "Polynomial frequency bank contains non-production entries at diameter(s) "
            f"{incomplete_entries} um."
        )

    fields = (
        "bank_build_report_path",
        "bank_build_report_sha256",
        "independent_go_path",
        "independent_go_sha256",
        "promotion_contract_path",
        "promotion_contract_sha256",
    )
    missing_fields = [field for field in fields if evaluator.get(field) is None]
    if missing_fields:
        raise ValueError(
            "Polynomial HBI requires pinned bank_build_report, independent_go, and "
            "promotion contract paths and SHA-256 digests; missing "
            + ", ".join(missing_fields)
            + "."
        )
    promotion_path = _resolve_project_path(
        project_root,
        evaluator.get("promotion_contract_path"),
        "resonance.evaluator.promotion_contract_path",
    )
    expected_promotion_sha256 = _normalized_sha256(
        evaluator.get("promotion_contract_sha256"),
        "resonance.evaluator.promotion_contract_sha256",
    )
    if expected_promotion_sha256 != POLYNOMIAL_PROMOTION_SHA256:
        raise ValueError("Polynomial evaluator does not pin the approved promotion contract SHA-256.")
    if not promotion_path.is_file():
        raise FileNotFoundError(f"Polynomial promotion contract not found: {promotion_path}")
    _artifact_sha256(promotion_path, expected_promotion_sha256)
    promotion = _load_json_mapping(
        promotion_path,
        "resonance.evaluator.promotion_contract_path",
    )
    if (
        promotion.get("schema") != POLYNOMIAL_PROMOTION_SCHEMA
        or promotion.get("status") != POLYNOMIAL_PROMOTION_STATUS
        or set(promotion.get("agent_scope") or ()) != {"sonovue", "definity"}
        or int(promotion.get("exact_diameter_count", -1)) != 23
    ):
        raise ValueError("Polynomial promotion contract identity or scope is invalid.")
    approved_coefficients = promotion.get("approved_coefficients")
    source_labels = promotion.get("source_labels")
    builder = promotion.get("builder")
    for label, record in (
        ("approved coefficients", approved_coefficients),
        ("source labels", source_labels),
        ("builder", builder),
    ):
        if not isinstance(record, Mapping):
            raise ValueError(f"Polynomial promotion contract has no {label} binding.")
        source_path = _resolve_relocated_provenance_path(
            project_root,
            record.get("path"),
            provenance_path_overrides,
        )
        if not source_path.is_file():
            raise FileNotFoundError(f"Polynomial promotion {label} source not found: {source_path}")
        if _artifact_sha256(source_path, record.get("sha256")) != str(record.get("sha256")):
            raise AssertionError("unreachable promotion source hash mismatch")
    sonovue_targets = promotion.get("sonovue_acoustic_targets_mhz")
    if not isinstance(sonovue_targets, Mapping) or {
        _target_diameter_key(float(diameter)): float(frequency)
        for diameter, frequency in sonovue_targets.items()
    } != _target_map(SONOVUE_RESONANCE_TARGETS_MHZ):
        raise ValueError("Promotion contract does not freeze the approved SonoVue targets.")
    if {_target_diameter_key(value) for value in promotion.get("sonovue_no_observation_diameters_um", ())} != {3.4, 5.8}:
        raise ValueError("Promotion contract SonoVue no-observation diameters are invalid.")
    definity_1p3 = promotion.get("definity_1p3_policy")
    if (
        not isinstance(definity_1p3, Mapping)
        or float(definity_1p3.get("diameter_um", -1.0)) != 1.3
        or float(definity_1p3.get("target_frequency_mhz", -1.0)) != 2.21
        or definity_1p3.get("likelihood") != "included"
        or definity_1p3.get("metadata_and_receipt_disclosure") != "required"
    ):
        raise ValueError("Promotion contract Definity 1.3 policy is invalid.")
    report_path = _resolve_project_path(
        project_root,
        evaluator.get("bank_build_report_path"),
        "resonance.evaluator.bank_build_report_path",
    )
    go_path = _resolve_project_path(
        project_root,
        evaluator.get("independent_go_path"),
        "resonance.evaluator.independent_go_path",
    )
    if not report_path.is_file():
        raise FileNotFoundError(f"Polynomial bank build report not found: {report_path}")
    if not go_path.is_file():
        raise FileNotFoundError(f"Independent polynomial GO receipt not found: {go_path}")
    report_sha256 = _artifact_sha256(
        report_path,
        evaluator.get("bank_build_report_sha256"),
    )
    report = _load_json_mapping(
        report_path,
        "resonance.evaluator.bank_build_report_path",
    )
    if report.get("schema") != POLYNOMIAL_BANK_REPORT_SCHEMA:
        raise ValueError("Polynomial bank build report has an unsupported schema.")
    if report.get("status") != "final_polynomial_bank_built":
        raise ValueError("Polynomial bank build report is not final.")
    if resonance_agent({"resonance": {"agent": report.get("agent")}}) != bank.agent:
        raise ValueError("Polynomial bank build report agent does not match the bank.")
    if report.get("bank_artifact_sha256") != bank_sha256:
        raise ValueError("Polynomial bank build report refers to a different bank artifact.")
    selected_family = str(report.get("selected_family") or "").strip()
    if not selected_family:
        raise ValueError("Polynomial bank build report has no approved selected_family.")
    if tuple(report.get("response_encodings") or ()) != bank.response_encodings:
        raise ValueError("Polynomial bank build report response encodings differ from the bank.")
    _validate_approved_polynomial_family(bank, report, promotion)
    if report.get("promotion_contract_sha256") != POLYNOMIAL_PROMOTION_SHA256:
        raise ValueError("Polynomial bank report is not bound to the promotion contract.")
    if report.get("build_tool") != build_tool:
        raise ValueError("Polynomial bank report build tool differs from the bank artifact.")
    expected_targets = (
        SONOVUE_RESONANCE_TARGETS_MHZ
        if bank.agent == "sonovue"
        else DEFINITY_POLYNOMIAL_RESONANCE_TARGETS_MHZ
    )
    reported_targets = report.get("acoustic_targets_mhz_by_diameter_um")
    if not isinstance(reported_targets, Mapping) or {
        _target_diameter_key(float(diameter)): float(frequency)
        for diameter, frequency in reported_targets.items()
    } != _target_map(expected_targets):
        raise ValueError("Polynomial bank report acoustic target mapping is not frozen correctly.")
    for field in (
        "no_clipping",
        "no_extrapolation",
        "no_diameter_interpolation",
        "no_analytical_fallback",
        "no_pchip_fallback",
    ):
        if report.get(field) is not True:
            raise ValueError(f"Polynomial bank build report does not enforce {field}.")

    _artifact_sha256(go_path, evaluator.get("independent_go_sha256"))
    gate = _load_json_mapping(go_path, "resonance.evaluator.independent_go_path")
    if gate.get("schema") != POLYNOMIAL_INDEPENDENT_GO_SCHEMA:
        raise ValueError("Independent polynomial GO receipt has an unsupported schema.")
    if gate.get("status") != "GO" or gate.get("independent_audit") is not True:
        raise ValueError("Polynomial bank has no explicit independent GO for HBI.")
    if resonance_agent({"resonance": {"agent": gate.get("agent")}}) != bank.agent:
        raise ValueError("Independent polynomial GO agent does not match the bank.")
    if gate.get("bank_artifact_sha256") != bank_sha256:
        raise ValueError("Independent polynomial GO refers to a different bank artifact.")
    if gate.get("bank_build_report_sha256") != report_sha256:
        raise ValueError("Independent polynomial GO refers to a different build report.")
    if gate.get("selected_family") != selected_family:
        raise ValueError("Independent polynomial GO selected family differs from the build report.")
    if gate.get("promotion_contract_sha256") != POLYNOMIAL_PROMOTION_SHA256:
        raise ValueError("Independent polynomial GO is not bound to the promotion contract.")
    if bank.agent == "definity" and gate.get("definity_1p3_extrapolation_disclosed") is not True:
        raise ValueError("Independent polynomial GO does not disclose Definity 1.3 extrapolation.")
    gates = gate.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != REQUIRED_POLYNOMIAL_INDEPENDENT_GO_GATES:
        raise ValueError("Independent polynomial GO is missing the exact required gate set.")
    failed = sorted(name for name, passed in gates.items() if passed is not True)
    if failed:
        raise ValueError(
            "Independent polynomial GO contains failed gates: " + ", ".join(failed)
        )
    return gate


@lru_cache(maxsize=None)
def _load_frequency_polynomial_bank(
    path_text: str,
    artifact_sha256: str,
) -> DpdPolynomialFrequencyEmulatorBank:
    del artifact_sha256
    return DpdPolynomialFrequencyEmulatorBank.load(path_text)


def resolve_dpd_frequency_polynomial_bank(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> DpdPolynomialFrequencyEmulatorBank:
    """Load the pinned, exact-diameter polynomial bank selected explicitly for HBI."""

    resonance = _resonance_mapping(config)
    evaluator = resonance.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError(
            "resonance.evaluator must be a mapping for artifact_polynomial_bank mode."
        )
    path = _resolve_project_path(
        project_root,
        evaluator.get("artifact_path"),
        "resonance.evaluator.artifact_path",
    )
    if not path.is_file():
        raise FileNotFoundError(f"DPD polynomial frequency bank artifact not found: {path}")
    if evaluator.get("artifact_sha256") is None:
        raise ValueError(
            "resonance.evaluator.artifact_sha256 is required for artifact_polynomial_bank."
        )
    actual_sha256 = _artifact_sha256(path, evaluator.get("artifact_sha256"))
    bank = _load_frequency_polynomial_bank(str(path), actual_sha256)
    agent = resonance_agent(config)
    if bank.agent != agent:
        raise ValueError(
            "DPD polynomial frequency bank agent does not match resonance.agent: "
            f"artifact={bank.agent!r}, config={agent!r}."
        )
    _validate_polynomial_bank_release(
        evaluator,
        project_root=project_root,
        bank=bank,
        bank_sha256=actual_sha256,
    )
    expected_fixed_kb = evaluator.get("expected_fixed_kb_dpd")
    if expected_fixed_kb is None:
        raise ValueError(
            "resonance.evaluator.expected_fixed_kb_dpd is required for "
            "artifact_polynomial_bank evaluation."
        )
    expected = _coerce_positive(
        expected_fixed_kb,
        "resonance.evaluator.expected_fixed_kb_dpd",
    )
    actual_value = bank.conditions.get("fixed_kb_dpd")
    if actual_value is None:
        raise ValueError("Polynomial frequency bank does not record conditions.fixed_kb_dpd.")
    actual = _coerce_positive(
        actual_value,
        "polynomial frequency bank conditions.fixed_kb_dpd",
    )
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "Polynomial frequency bank fixed kb does not match the configured expectation: "
            f"artifact={actual:g}, config={expected:g}."
        )
    return bank


def _resonance_ka_bounds(config: Mapping[str, Any], project_root: str | Path) -> tuple[float, float]:
    forward_model = resonance_forward_model(config)
    if forward_model == DPD_FREQUENCY_SURFACE_FORWARD_MODEL:
        return resolve_dpd_frequency_surface(config, project_root).ka_bounds_dpd
    if forward_model == DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL:
        return resolve_dpd_frequency_emulator_bank(config, project_root).ka_bounds_dpd
    if forward_model == DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL:
        return resolve_dpd_frequency_polynomial_bank(config, project_root).ka_bounds_dpd
    return 0.0, math.inf


def _resolve_exact_frequency_bank(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> DpdFrequencyEmulatorBank | DpdPolynomialFrequencyEmulatorBank:
    forward_model = resonance_forward_model(config)
    if forward_model == DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL:
        return resolve_dpd_frequency_emulator_bank(config, project_root)
    if forward_model == DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL:
        return resolve_dpd_frequency_polynomial_bank(config, project_root)
    raise ValueError("The selected resonance evaluator is not an exact-diameter bank.")


def resolve_resonance_runtime_defaults(config: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    resonance = _resonance_mapping(config)
    path_value = resonance.get("unit_defaults_path")
    modality_value = resonance.get("unit_defaults_modality")
    if path_value and modality_value:
        raise ValueError(
            "Resonance unit conversion is ambiguous: set only one of "
            "resonance.unit_defaults_path or resonance.unit_defaults_modality."
        )
    if path_value:
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = Path(project_root) / path
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid resonance unit defaults at {path}; expected a mapping.")
        return dict(payload)
    if not modality_value:
        raise ValueError(
            "Missing resonance unit conversion source. Set resonance.unit_defaults_modality "
            "or resonance.unit_defaults_path."
        )
    return load_emb_runtime_defaults(project_root, str(modality_value))


def resolve_resonance_constants(config: Mapping[str, Any], defaults: Mapping[str, Any]) -> EmbResonanceConstants:
    resonance = _resonance_mapping(config)
    if "physical_constants" in resonance:
        raise ValueError(
            "resonance.physical_constants is the legacy SI resonance schema. "
            "Use resonance.dpd_constants with kappa_g/rho_l_dpd/p0_dpd."
        )
    dpd_constants = resonance.get("dpd_constants") or {}
    if not isinstance(dpd_constants, Mapping):
        raise ValueError("resonance.dpd_constants must be a mapping.")
    return EmbResonanceConstants.from_mapping(dpd_constants, defaults=defaults)


def resonance_unit_time_seconds(defaults: Mapping[str, Any]) -> float:
    required = ("ul", "kbol", "t0", "rho_water", "rhow", "energyFactor")
    missing = [key for key in required if key not in defaults]
    if missing:
        raise ValueError(f"EMB resonance runtime defaults are missing DPD unit constants: {missing}")
    ul = _coerce_positive(defaults["ul"], "ul")
    kbol = _coerce_positive(defaults["kbol"], "kbol")
    t0 = _coerce_positive(defaults["t0"], "t0")
    rho_water = _coerce_positive(defaults["rho_water"], "rho_water")
    rhow = _coerce_positive(defaults["rhow"], "rhow")
    energy_factor = _coerce_positive(defaults["energyFactor"], "energyFactor")
    energy_unit = energy_factor * kbol * t0
    mass_unit = rho_water * ul**3 / rhow
    return math.sqrt(mass_unit * ul**2 / energy_unit)


def _fscale(defaults: Mapping[str, Any]) -> float:
    if "fscale" not in defaults:
        raise ValueError("EMB resonance runtime defaults are missing fscale.")
    return _coerce_positive(defaults["fscale"], "fscale")


def resolve_vacuum_shell_constants(defaults: Mapping[str, Any]) -> EmbVacuumShellConstants:
    required = ("rho_shell", "shell_th", "nu", "fscale", "ul")
    missing = [key for key in required if key not in defaults]
    if missing:
        raise ValueError(f"EMB vacuum-shell defaults are missing required constants: {missing}")
    shell_thickness = _coerce_positive(defaults["shell_th"], "shell_th")
    th_fac = _coerce_positive(defaults.get("th_fac", 1.0), "th_fac")
    poisson_ratio = _coerce_finite(defaults["nu"], "nu")
    if math.isclose(1.0 - poisson_ratio, 0.0):
        raise ValueError("nu must not make 1 - nu equal to zero for EMB vacuum-shell conversion.")
    return EmbVacuumShellConstants(
        rho_shell_kg_m3=_coerce_positive(defaults["rho_shell"], "rho_shell"),
        shell_thickness_m=shell_thickness * th_fac,
        poisson_ratio=poisson_ratio,
        fscale=_fscale(defaults),
        unit_length_m=_coerce_positive(defaults["ul"], "ul"),
        stiffness_unit_n_per_m=emb_elastic_stiffness_unit_n_per_m(
            defaults,
            label="EMB vacuum-shell",
        ),
    )


def _coerce_radius_map(payload: Mapping[str, Any] | None) -> dict[float, float]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("resonance.radius_dpd_by_diameter_um must be a mapping.")
    result: dict[float, float] = {}
    for diameter, radius in payload.items():
        result[_diameter_key(_coerce_positive(diameter, "resonance radius diameter_um"))] = _coerce_positive(
            radius,
            f"radius_dpd for diameter {diameter}",
        )
    return result


def fallback_radius_dpd_from_diameter_um(diameter_um: float) -> float:
    return _coerce_positive(diameter_um, "diameter_um") / 0.5


def resolve_resonance_radii_dpd(
    config: Mapping[str, Any],
    diameters_um: Sequence[float],
) -> np.ndarray:
    resonance = _resonance_mapping(config)
    radius_map = _coerce_radius_map(resonance.get("radius_dpd_by_diameter_um"))
    radii = []
    for diameter in diameters_um:
        key = _diameter_key(float(diameter))
        if key not in radius_map:
            radii.append(fallback_radius_dpd_from_diameter_um(float(diameter)))
            continue
        radii.append(radius_map[key])
    return np.asarray(radii, dtype=np.float64)


def resonance_radius_sources(
    config: Mapping[str, Any],
    diameters_um: Sequence[float],
) -> dict[float, str]:
    resonance = _resonance_mapping(config)
    radius_map = _coerce_radius_map(resonance.get("radius_dpd_by_diameter_um"))
    return {
        _diameter_key(float(diameter)): (
            "configured" if _diameter_key(float(diameter)) in radius_map else "fallback_diameter_um_over_0p5"
        )
        for diameter in diameters_um
    }


def emb_resonance_frequency_mhz_from_ka_dpd(
    *,
    ka_dpd: Any,
    radii_dpd: Any,
    constants: EmbResonanceConstants,
    defaults: Mapping[str, Any],
) -> np.ndarray:
    ka = np.asarray(ka_dpd, dtype=np.float64)
    radius = np.asarray(radii_dpd, dtype=np.float64)
    if np.any(~np.isfinite(ka)) or np.any(ka < 0.0):
        raise ValueError("ka_dpd must contain finite non-negative values for EMB resonance.")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radii_dpd must contain finite positive values for EMB resonance.")
    chi_dpd = ka / _fscale(defaults)
    omega_squared_dpd = (
        3.0
        / (constants.rho_l_dpd * radius**2)
        * (constants.kappa_g * constants.p0_dpd + 4.0 * chi_dpd / (3.0 * radius))
    )
    if np.any(~np.isfinite(omega_squared_dpd)) or np.any(omega_squared_dpd <= 0.0):
        raise ValueError("EMB resonance DPD formula produced non-positive omega^2 values.")
    frequency_hz = np.sqrt(omega_squared_dpd) / (2.0 * math.pi * resonance_unit_time_seconds(defaults))
    return np.asarray(frequency_hz / 1.0e6, dtype=np.float64)


def emb_resonance_ka_dpd_from_frequency_mhz(
    *,
    frequency_mhz: Any,
    radii_dpd: Any,
    constants: EmbResonanceConstants,
    defaults: Mapping[str, Any],
) -> np.ndarray:
    frequency = np.asarray(frequency_mhz, dtype=np.float64)
    radius = np.asarray(radii_dpd, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_mhz must contain finite positive values for EMB resonance.")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radii_dpd must contain finite positive values for EMB resonance.")
    omega_dpd = 2.0 * math.pi * frequency * 1.0e6 * resonance_unit_time_seconds(defaults)
    chi_dpd = (3.0 * radius / 4.0) * (
        (constants.rho_l_dpd * radius**2 * omega_dpd**2) / 3.0
        - constants.kappa_g * constants.p0_dpd
    )
    if np.any(~np.isfinite(chi_dpd)) or np.any(chi_dpd < 0.0):
        raise ValueError("EMB resonance inverse DPD formula produced negative shell elasticity.")
    return np.asarray(chi_dpd * _fscale(defaults), dtype=np.float64)


def emb_vacuum_shell_frequency_mhz_from_ka_dpd(
    *,
    ka_dpd: Any,
    radii_dpd: Any,
    defaults: Mapping[str, Any],
) -> np.ndarray:
    ka = np.asarray(ka_dpd, dtype=np.float64)
    radius = np.asarray(radii_dpd, dtype=np.float64)
    if np.any(~np.isfinite(ka)) or np.any(ka < 0.0):
        raise ValueError("ka_dpd must contain finite non-negative values for EMB vacuum-shell resonance.")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radii_dpd must contain finite positive values for EMB vacuum-shell resonance.")
    constants = resolve_vacuum_shell_constants(defaults)
    radius_m = radius * constants.unit_length_m
    implemented_area_modulus = ka * constants.stiffness_unit_n_per_m
    omega_squared_si = (
        4.0
        * implemented_area_modulus
        / (constants.surface_mass_kg_m2 * radius_m**2)
    )
    if np.any(~np.isfinite(omega_squared_si)) or np.any(omega_squared_si < 0.0):
        raise ValueError("EMB vacuum-shell formula produced invalid omega^2 values.")
    frequency_hz = np.sqrt(omega_squared_si) / (2.0 * math.pi)
    return np.asarray(frequency_hz / 1.0e6, dtype=np.float64)


def emb_vacuum_shell_ka_dpd_from_frequency_mhz(
    *,
    frequency_mhz: Any,
    radii_dpd: Any,
    defaults: Mapping[str, Any],
) -> np.ndarray:
    frequency = np.asarray(frequency_mhz, dtype=np.float64)
    radius = np.asarray(radii_dpd, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("frequency_mhz must contain finite non-negative values for EMB vacuum-shell resonance.")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radii_dpd must contain finite positive values for EMB vacuum-shell resonance.")
    constants = resolve_vacuum_shell_constants(defaults)
    radius_m = radius * constants.unit_length_m
    omega_si = 2.0 * math.pi * frequency * 1.0e6
    implemented_area_modulus = (
        omega_si**2 * constants.surface_mass_kg_m2 * radius_m**2 / 4.0
    )
    ka = implemented_area_modulus / constants.stiffness_unit_n_per_m
    if np.any(~np.isfinite(ka)) or np.any(ka < 0.0):
        raise ValueError("EMB vacuum-shell inverse formula produced invalid ka values.")
    return np.asarray(ka, dtype=np.float64)


def predict_resonance_frequencies_mhz(
    diameters_um: Sequence[float],
    *,
    ka_dpd: float,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> list[float]:
    forward_model = resonance_forward_model(config)
    if forward_model == DPD_FREQUENCY_SURFACE_FORWARD_MODEL:
        surface = resolve_dpd_frequency_surface(config, project_root)
        return surface.predict_for_diameters_mhz(float(ka_dpd), np.asarray(diameters_um, dtype=np.float64)).reshape(-1).tolist()
    if forward_model == DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL:
        bank = resolve_dpd_frequency_emulator_bank(config, project_root)
        return bank.predict_for_diameters_mhz(
            float(ka_dpd),
            np.asarray(diameters_um, dtype=np.float64),
        ).reshape(-1).tolist()
    if forward_model == DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL:
        bank = resolve_dpd_frequency_polynomial_bank(config, project_root)
        return bank.predict_for_diameters_mhz(
            float(ka_dpd),
            np.asarray(diameters_um, dtype=np.float64),
        ).reshape(-1).tolist()
    defaults = resolve_resonance_runtime_defaults(config, project_root)
    if forward_model == ANALYTICAL_VACUUM_SHELL_FORWARD_MODEL:
        radii = resolve_resonance_radii_dpd(config, diameters_um)
        predictions = emb_vacuum_shell_frequency_mhz_from_ka_dpd(
            ka_dpd=float(ka_dpd),
            radii_dpd=radii,
            defaults=defaults,
        )
        return predictions.reshape(-1).tolist()
    constants = resolve_resonance_constants(config, defaults)
    radii = resolve_resonance_radii_dpd(config, diameters_um)
    predictions = emb_resonance_frequency_mhz_from_ka_dpd(
        ka_dpd=float(ka_dpd),
        radii_dpd=radii,
        constants=constants,
        defaults=defaults,
    )
    return predictions.reshape(-1).tolist()


def predict_resonance_frequencies_mhz_batch(
    diameters_um: Sequence[float],
    *,
    ka_dpd: Any,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> np.ndarray:
    ka = np.asarray(ka_dpd, dtype=np.float64).reshape(-1, 1)
    diameters = np.asarray(diameters_um, dtype=np.float64).reshape(1, -1)
    forward_model = resonance_forward_model(config)
    if forward_model == DPD_FREQUENCY_SURFACE_FORWARD_MODEL:
        surface = resolve_dpd_frequency_surface(config, project_root)
        return surface.predict_for_diameters_mhz(ka, diameters)
    if forward_model == DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL:
        bank = resolve_dpd_frequency_emulator_bank(config, project_root)
        return bank.predict_for_diameters_mhz(ka, diameters)
    if forward_model == DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL:
        bank = resolve_dpd_frequency_polynomial_bank(config, project_root)
        return bank.predict_for_diameters_mhz(ka, diameters)
    defaults = resolve_resonance_runtime_defaults(config, project_root)
    if forward_model == ANALYTICAL_VACUUM_SHELL_FORWARD_MODEL:
        radii = resolve_resonance_radii_dpd(config, diameters.reshape(-1).tolist())
        return emb_vacuum_shell_frequency_mhz_from_ka_dpd(
            ka_dpd=ka,
            radii_dpd=radii.reshape(1, -1),
            defaults=defaults,
        )
    constants = resolve_resonance_constants(config, defaults)
    radii = resolve_resonance_radii_dpd(config, diameters.reshape(-1).tolist())
    return emb_resonance_frequency_mhz_from_ka_dpd(
        ka_dpd=ka,
        radii_dpd=radii.reshape(1, -1),
        constants=constants,
        defaults=defaults,
    )


def _valid_resonance_ka(ka_dpd: Any, config: Mapping[str, Any], project_root: str | Path) -> np.ndarray:
    ka = np.asarray(ka_dpd, dtype=np.float64)
    lower, upper = _resonance_ka_bounds(config, project_root)
    return np.isfinite(ka) & (ka >= lower) & (ka <= upper)


def _valid_nonnegative_ka(ka_dpd: Any) -> np.ndarray:
    ka = np.asarray(ka_dpd, dtype=np.float64)
    return np.isfinite(ka) & (ka >= 0.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=None)
def _load_runtime_config(config_path: str | None = None) -> dict[str, Any]:
    project_root = _repo_root()
    selected = config_path or os.getenv("HUQ_INFERENCE_CONFIG") or os.getenv("CONFIG_PATH")
    if selected:
        path = Path(selected).expanduser()
        if not path.is_absolute() and not path.exists():
            path = project_root / path
    else:
        path = resolve_inference_config_path(project_root, experiment=EMB_RESONANCE_EXPERIMENT)
    with Path(path).open("rb") as handle:
        payload = yaml.load(handle, Loader=yaml.CLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Resonance inference config must be a mapping: {path}")
    return payload


def _sample_parts(params: Any) -> tuple[float, float, float]:
    vector = np.asarray(params, dtype=np.float64).reshape(-1)
    if vector.shape == (4,):
        return float(vector[0]), float(vector[1]), float(vector[3])
    if vector.shape == (3,):
        return float(vector[0]), float(vector[1]), float(vector[2])
    raise ValueError(
        "EMB resonance expects direct parameters [ka, kb, d0, sigma] "
        f"or [ka, kb, sigma], got shape {vector.shape}."
    )


def _diameters_from_reference(reference_points: Sequence[float], diameter_um: float | None) -> list[float]:
    if len(reference_points) > 0:
        return [float(value) for value in reference_points]
    if diameter_um is None:
        raise ValueError("Resonance computation requires reference diameter points or diameter_um.")
    return [float(diameter_um)]


def _relative_resonance_standard_deviation_mhz(
    predictions_mhz: np.ndarray,
    sigmas: np.ndarray,
    *,
    valid_rows: np.ndarray | None = None,
    hard_invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    predictions = np.asarray(predictions_mhz, dtype=np.float64)
    sigma_vector = np.asarray(sigmas, dtype=np.float64).reshape(-1, 1)
    valid = np.isfinite(predictions) & np.isfinite(sigma_vector) & (sigma_vector > 0.0)
    if valid_rows is not None:
        valid = valid & np.asarray(valid_rows, dtype=bool).reshape(-1, 1)
    safe_predictions = np.where(np.isfinite(predictions), predictions, 0.0)
    safe_sigmas = np.where(np.isfinite(sigma_vector) & (sigma_vector > 0.0), sigma_vector, 1.0)
    std = safe_sigmas * np.maximum(np.abs(safe_predictions), _MIN_RESONANCE_STD_MHZ)
    result = np.where(valid, std, _INVALID_PROPOSAL_STD_MHZ)
    if hard_invalid_mask is not None:
        # Acoustic preflight requires strictly positive reference MHz values; with the
        # unsupported bank placeholder fixed at 0 MHz, Korali Bayesian/Reference Normal
        # hits +inf SSE before its post-SSE stddev floor and fails closed for this lane.
        result = np.where(np.asarray(hard_invalid_mask, dtype=bool), 0.0, result)
    return result


def _bank_predict_with_exact_support(
    bank: DpdFrequencyEmulatorBank | DpdPolynomialFrequencyEmulatorBank,
    *,
    ka_dpd: Any,
    diameters_um: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bank, DpdPolynomialFrequencyEmulatorBank):
        return bank.predict_with_support_mhz(ka_dpd, diameters_um)

    ka = np.asarray(ka_dpd, dtype=np.float64)
    diameters = np.asarray(diameters_um, dtype=np.float64)
    try:
        ka, diameters = np.broadcast_arrays(ka, diameters)
    except ValueError as error:
        raise ValueError("ka_dpd and diameter_um must be broadcast-compatible.") from error

    original_shape = ka.shape
    flat_ka = ka.reshape(-1)
    flat_diameters = diameters.reshape(-1)
    predictions = np.zeros(flat_ka.shape, dtype=np.float64)
    support_mask = np.zeros(flat_ka.shape, dtype=bool)
    finite_nonnegative_ka = np.isfinite(flat_ka) & (flat_ka >= 0.0)

    supported_diameters = np.asarray(bank.diameters_um, dtype=np.float64)
    right = np.searchsorted(supported_diameters, flat_diameters, side="left")
    right = np.clip(right, 0, supported_diameters.size - 1)
    left = np.clip(right - 1, 0, supported_diameters.size - 1)
    left_distance = np.abs(flat_diameters - supported_diameters[left])
    right_distance = np.abs(flat_diameters - supported_diameters[right])
    emulator_index = np.where(left_distance <= right_distance, left, right)
    diameter_distance = np.abs(flat_diameters - supported_diameters[emulator_index])
    diameter_supported = diameter_distance <= _DIAMETER_TOLERANCE_UM

    if np.any(diameter_supported):
        for index in np.unique(emulator_index[diameter_supported]):
            selection = diameter_supported & (emulator_index == index) & finite_nonnegative_ka
            if not np.any(selection):
                continue
            emulator = bank.emulators[int(index)]
            lower, upper = emulator.ka_bounds_dpd
            in_support = selection & (flat_ka >= lower) & (flat_ka <= upper)
            if not np.any(in_support):
                continue
            predictions[in_support] = emulator.predict_mhz(flat_ka[in_support])
            support_mask[in_support] = True

    return predictions.reshape(original_shape), support_mask.reshape(original_shape)


def _expected_observations(config: Mapping[str, Any]) -> dict[float, dict[str, Any]]:
    resonance = _resonance_mapping(config)
    agent = resonance_agent(config)
    polynomial_mode = resonance_forward_model(config) == DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL
    observations = resonance.get("observations")
    if observations is not None:
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
            raise ValueError("resonance.observations must be a list of mappings.")
        result: dict[float, dict[str, Any]] = {}
        for item in observations:
            if not isinstance(item, Mapping):
                raise ValueError("Each resonance observation must be a mapping.")
            diameter = _coerce_positive(item.get("diameter_um"), "resonance observation diameter_um")
            frequency = _coerce_positive_acoustic_frequency_mhz(
                item.get("frequency_MHz"),
                "resonance observation frequency_MHz",
            )
            key = _target_diameter_key(diameter)
            if key in result:
                raise ValueError(f"Duplicate resonance observation diameter {diameter:g} um.")
            result[key] = {
                "diameter_um": diameter,
                "frequency_MHz": frequency,
                "source_id": item.get("source_id"),
                "figure": item.get("figure"),
                "agent": item.get("agent", resonance_agent(config)),
            }
        if agent == "sonovue":
            _require_exact_target_mapping(
                result,
                SONOVUE_RESONANCE_TARGETS_MHZ,
                label="SonoVue",
            )
        elif agent == "definity" and polynomial_mode:
            _require_exact_target_mapping(
                result,
                _expected_definity_polynomial_targets(config),
                label="Definity polynomial HBI",
            )
        return result

    target_map = resonance.get("reference_targets_mhz_by_diameter_um")
    if target_map is not None:
        if not isinstance(target_map, Mapping):
            raise ValueError("resonance.reference_targets_mhz_by_diameter_um must be a mapping.")
        result = {
            _target_diameter_key(_coerce_positive(diameter, "resonance target diameter_um")): {
                "diameter_um": _coerce_positive(diameter, "resonance target diameter_um"),
                "frequency_MHz": _coerce_positive_acoustic_frequency_mhz(
                    frequency,
                    f"resonance target frequency for {diameter}",
                ),
                "source_id": None,
                "figure": None,
                "agent": resonance_agent(config),
            }
            for diameter, frequency in target_map.items()
        }
        if agent == "sonovue":
            _require_exact_target_mapping(
                result,
                SONOVUE_RESONANCE_TARGETS_MHZ,
                label="SonoVue",
            )
        elif agent == "definity" and polynomial_mode:
            _require_exact_target_mapping(
                result,
                _expected_definity_polynomial_targets(config),
                label="Definity polynomial HBI",
            )
        return result

    if agent == "sonovue":
        return {
            _target_diameter_key(diameter): {
                "diameter_um": diameter,
                "frequency_MHz": frequency,
                "source_id": "van_der_meer_2004",
                "figure": None,
                "agent": "sonovue",
            }
            for diameter, frequency in SONOVUE_RESONANCE_TARGETS_MHZ.items()
        }
    raise ValueError(
        "Resonance config must define resonance.observations or "
        "resonance.reference_targets_mhz_by_diameter_um for non-SonoVue agents."
    )


def _diameter_label(exp: Any, diameter_um: float) -> str:
    label = exp._lookup_diameter_mapping(exp.diameter_labels, diameter_um)
    if label is not None:
        return label
    text = f"{float(diameter_um):.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def _compression_training_support_for_diameter(exp: Any, diameter_um: float, project_root: Path) -> dict[str, Any]:
    label = _diameter_label(exp, diameter_um)
    data_file = exp.surrogate_dir / f"{label}um" / "data" / "F_Delta.dat"
    if not data_file.exists():
        raise FileNotFoundError(f"Compression surrogate training table not found: {data_file}")
    data = np.loadtxt(data_file, ndmin=2)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Compression surrogate training table {data_file} must have at least 3 columns.")
    yt = np.asarray(data[:, 0], dtype=np.float64)
    kb = np.asarray(data[:, 2], dtype=np.float64)
    finite = np.isfinite(yt) & np.isfinite(kb)
    if not np.any(finite):
        raise ValueError(f"Compression surrogate training table {data_file} has no finite Yt/kb rows.")
    yt = yt[finite]
    kb = kb[finite]
    ka_per_yt = 1.0 / float(ka_to_legacy_yt(1.0, modality="compression", project_root=project_root))
    ka = yt * ka_per_yt
    return {
        "diameter_um": float(diameter_um),
        "dataset_name": exp.dataset_name(diameter_um),
        "training_data_file": str(data_file),
        "n_training_curves": int(yt.shape[0]),
        "ka_per_yt": float(ka_per_yt),
        "Yt_min": float(np.min(yt)),
        "Yt_max": float(np.max(yt)),
        "ka_min": float(np.min(ka)),
        "ka_max": float(np.max(ka)),
        "kb_min": float(np.min(kb)),
        "kb_max": float(np.max(kb)),
    }


def preflight_direct_compression_surrogate_support(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root_path = Path(project_root)
    experiments = [
        exp
        for exp in load_experiments(dict(config), project_root_path)
        if exp.enabled
        and exp.name == "compression"
        and str(getattr(exp, "surrogate_parameterization", "")).lower() == "direct_ka_kb"
    ]
    if not experiments:
        return {
            "status": "skipped",
            "reason": "No enabled direct_ka_kb compression experiments.",
            "diameters": [],
        }

    global_prior_ka = config.get("prior_ka")
    global_prior_kb = config.get("prior_kb")
    if not isinstance(global_prior_ka, Sequence) or len(global_prior_ka) != 2:
        raise ValueError("Direct compression surrogate support preflight requires prior_ka: [min, max].")
    if not isinstance(global_prior_kb, Sequence) or len(global_prior_kb) != 2:
        raise ValueError("Direct compression surrogate support preflight requires prior_kb: [min, max].")
    extrapolation_policy = config.get("direct_compression_prior_extrapolation") or {}
    if not isinstance(extrapolation_policy, Mapping):
        raise ValueError("direct_compression_prior_extrapolation must be a mapping when provided.")
    allow_ka_lower = bool(extrapolation_policy.get("allow_ka_lower", False))
    allow_ka_upper = bool(extrapolation_policy.get("allow_ka_upper", False))
    allow_kb_lower = bool(extrapolation_policy.get("allow_kb_lower", False))
    allow_kb_upper = bool(extrapolation_policy.get("allow_kb_upper", False))

    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    allowed_extrapolations: list[str] = []
    common_ka = [-math.inf, math.inf]
    common_kb = [-math.inf, math.inf]
    for exp in experiments:
        for diameter in exp.diameters:
            support = _compression_training_support_for_diameter(exp, diameter, project_root_path)
            common_ka[0] = max(common_ka[0], float(support["ka_min"]))
            common_ka[1] = min(common_ka[1], float(support["ka_max"]))
            common_kb[0] = max(common_kb[0], float(support["kb_min"]))
            common_kb[1] = min(common_kb[1], float(support["kb_max"]))
            overrides = exp.phase1_prior_overrides(diameter)
            sources = exp.phase1_prior_override_sources(diameter)
            prior_ka = overrides.get("ka", list(global_prior_ka))
            prior_kb = overrides.get("kb", list(global_prior_kb))
            row = {
                **support,
                "prior_ka": [float(prior_ka[0]), float(prior_ka[1])],
                "prior_kb": [float(prior_kb[0]), float(prior_kb[1])],
                "prior_ka_source": sources.get("ka", "global"),
                "prior_kb_source": sources.get("kb", "global"),
            }
            ka_lower_outside = row["prior_ka"][0] < float(support["ka_min"]) - _SURROGATE_SUPPORT_TOLERANCE
            ka_upper_outside = row["prior_ka"][1] > float(support["ka_max"]) + _SURROGATE_SUPPORT_TOLERANCE
            kb_lower_outside = row["prior_kb"][0] < float(support["kb_min"]) - _SURROGATE_SUPPORT_TOLERANCE
            kb_upper_outside = row["prior_kb"][1] > float(support["kb_max"]) + _SURROGATE_SUPPORT_TOLERANCE
            ka_inside = not (ka_lower_outside or ka_upper_outside)
            kb_inside = not (kb_lower_outside or kb_upper_outside)
            ka_allowed = (not ka_lower_outside or allow_ka_lower) and (not ka_upper_outside or allow_ka_upper)
            kb_allowed = (not kb_lower_outside or allow_kb_lower) and (not kb_upper_outside or allow_kb_upper)
            row["prior_ka_inside_training_support"] = bool(ka_inside)
            row["prior_kb_inside_training_support"] = bool(kb_inside)
            row["prior_ka_lower_inside_training_support"] = not ka_lower_outside
            row["prior_ka_upper_inside_training_support"] = not ka_upper_outside
            row["prior_kb_lower_inside_training_support"] = not kb_lower_outside
            row["prior_kb_upper_inside_training_support"] = not kb_upper_outside
            row["prior_ka_extrapolation_allowed"] = bool((not ka_inside) and ka_allowed)
            row["prior_kb_extrapolation_allowed"] = bool((not kb_inside) and kb_allowed)
            if not ka_inside and not ka_allowed:
                violations.append(
                    f"{row['dataset_name']} prior_ka={row['prior_ka']} exceeds training ka "
                    f"[{support['ka_min']:.6g}, {support['ka_max']:.6g}]"
                )
            elif not ka_inside:
                allowed_extrapolations.append(
                    f"{row['dataset_name']} prior_ka={row['prior_ka']} exceeds training ka "
                    f"[{support['ka_min']:.6g}, {support['ka_max']:.6g}] under explicit policy"
                )
            if not kb_inside and not kb_allowed:
                violations.append(
                    f"{row['dataset_name']} prior_kb={row['prior_kb']} exceeds training kb "
                    f"[{support['kb_min']:.6g}, {support['kb_max']:.6g}]"
                )
            elif not kb_inside:
                allowed_extrapolations.append(
                    f"{row['dataset_name']} prior_kb={row['prior_kb']} exceeds training kb "
                    f"[{support['kb_min']:.6g}, {support['kb_max']:.6g}] under explicit policy"
                )
            rows.append(row)
    if violations:
        raise ValueError("Direct compression prior exceeds surrogate training support: " + "; ".join(violations))
    return {
        "status": "passed_with_prior_extrapolation" if allowed_extrapolations else "passed",
        "direct_compression_prior_extrapolation": {
            "allow_ka_lower": allow_ka_lower,
            "allow_ka_upper": allow_ka_upper,
            "allow_kb_lower": allow_kb_lower,
            "allow_kb_upper": allow_kb_upper,
            "reason": extrapolation_policy.get("reason"),
        },
        "allowed_prior_extrapolations": allowed_extrapolations,
        "common_training_support": {
            "ka": [float(common_ka[0]), float(common_ka[1])],
            "kb": [float(common_kb[0]), float(common_kb[1])],
        },
        "diameters": rows,
    }


def _indentation_training_support_for_diameter(
    exp: Any,
    diameter_um: float,
    project_root: Path,
) -> dict[str, Any]:
    label = _diameter_label(exp, diameter_um)
    data_file = exp.surrogate_dir / f"{label}um" / "data" / "samples_all.dat"
    if not data_file.exists():
        raise FileNotFoundError(f"Indentation surrogate training table not found: {data_file}")
    data = np.loadtxt(data_file, usecols=(0, 1), ndmin=2)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Indentation surrogate training table {data_file} must expose Yt and direct ka columns."
        )
    yt = np.asarray(data[:, 0], dtype=np.float64)
    stored_ka = np.asarray(data[:, 1], dtype=np.float64)
    finite = np.isfinite(yt) & np.isfinite(stored_ka)
    if not np.any(finite):
        raise ValueError(
            f"Indentation surrogate training table {data_file} has no finite Yt/ka rows."
        )
    yt = yt[finite]
    stored_ka = stored_ka[finite]
    ka_per_yt = 1.0 / float(
        ka_to_legacy_yt(1.0, modality="indentation", project_root=project_root)
    )
    converted_ka = yt * ka_per_yt
    conversion_error = float(np.max(np.abs(converted_ka - stored_ka)))
    if conversion_error > _SURROGATE_SUPPORT_TOLERANCE:
        raise ValueError(
            f"Indentation training table {data_file} has inconsistent legacy Yt/direct ka "
            f"columns (max abs error {conversion_error:.6g})."
        )
    return {
        "diameter_um": float(diameter_um),
        "dataset_name": exp.dataset_name(diameter_um),
        "training_data_file": str(data_file),
        "training_data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "n_training_curves": int(yt.shape[0]),
        "ka_per_yt": float(ka_per_yt),
        "legacy_Yt_min": float(np.min(yt)),
        "legacy_Yt_max": float(np.max(yt)),
        "direct_ka_min": float(np.min(converted_ka)),
        "direct_ka_max": float(np.max(converted_ka)),
        "stored_ka_conversion_max_abs_error": conversion_error,
    }


def preflight_indentation_surrogate_support(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate each direct-ka Phase-1 prior against its legacy indentation DNN."""

    project_root_path = Path(project_root)
    experiments = [
        exp
        for exp in load_experiments(dict(config), project_root_path)
        if exp.enabled
        and exp.name == "indentation"
        and str(getattr(exp, "surrogate_parameterization", "")).lower() == "legacy_yt_kb"
    ]
    if not experiments:
        return {
            "status": "skipped",
            "reason": "No enabled legacy_yt_kb indentation experiments.",
            "diameters": [],
        }
    global_prior_ka = config.get("prior_ka")
    if not isinstance(global_prior_ka, Sequence) or len(global_prior_ka) != 2:
        raise ValueError(
            "Indentation surrogate support preflight requires prior_ka: [min, max]."
        )

    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    for exp in experiments:
        for diameter in exp.diameters:
            support = _indentation_training_support_for_diameter(
                exp,
                diameter,
                project_root_path,
            )
            overrides = exp.phase1_prior_overrides(diameter)
            source = exp.phase1_prior_override_sources(diameter).get("ka", "global")
            prior = overrides.get("ka", list(global_prior_ka))
            prior_bounds = [float(prior[0]), float(prior[1])]
            lower_inside = (
                prior_bounds[0]
                >= float(support["direct_ka_min"]) - _SURROGATE_SUPPORT_TOLERANCE
            )
            upper_inside = (
                prior_bounds[1]
                <= float(support["direct_ka_max"]) + _SURROGATE_SUPPORT_TOLERANCE
            )
            row = {
                **support,
                "prior_ka": prior_bounds,
                "prior_ka_source": source,
                "prior_ka_inside_training_support": bool(lower_inside and upper_inside),
                "prior_ka_lower_inside_training_support": bool(lower_inside),
                "prior_ka_upper_inside_training_support": bool(upper_inside),
                "phase3b_support_source": "inherited_phase1_uniform",
            }
            if not row["prior_ka_inside_training_support"]:
                violations.append(
                    f"{row['dataset_name']} prior_ka={prior_bounds} exceeds indentation "
                    f"training ka [{support['direct_ka_min']:.6g}, "
                    f"{support['direct_ka_max']:.6g}]"
                )
            rows.append(row)
    if violations:
        raise ValueError(
            "Indentation prior exceeds surrogate training support: " + "; ".join(violations)
        )
    return {
        "status": "passed",
        "parameterization": "legacy_yt_kb",
        "hbi_parameter": "direct_ka_dpd",
        "phase1_distribution": "normalized_uniform",
        "phase3b_support_source": "inherited_phase1_uniform",
        "diameters": rows,
    }


def compute_emb_resonance(
    sample: dict[str, Any],
    reference_points: Sequence[float],
    diameter_um: float | None = None,
    device: str = "cpu",
) -> None:
    del device
    config = _load_runtime_config()
    ka, _kb, sigma = _sample_parts(sample["Parameters"])
    diameters = _diameters_from_reference(reference_points, diameter_um)
    if resonance_forward_model(config) in {
        DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL,
        DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL,
    }:
        bank = _resolve_exact_frequency_bank(config, _repo_root())
        prediction_array, support_mask = _bank_predict_with_exact_support(
            bank,
            ka_dpd=np.asarray([[ka]], dtype=np.float64),
            diameters_um=np.asarray(diameters, dtype=np.float64).reshape(1, -1),
        )
        sample["Reference Evaluations"] = prediction_array.reshape(-1).tolist()
        sample["Standard Deviation"] = _relative_resonance_standard_deviation_mhz(
            prediction_array,
            np.asarray([sigma], dtype=np.float64),
            valid_rows=np.asarray([bool(_valid_nonnegative_ka(ka))], dtype=bool),
            hard_invalid_mask=~support_mask,
        ).reshape(-1).tolist()
        return

    ka_lower, _ka_upper = _resonance_ka_bounds(config, _repo_root())
    valid_ka = bool(_valid_resonance_ka(ka, config, _repo_root()))
    predictions = predict_resonance_frequencies_mhz(
        diameters,
        ka_dpd=ka if valid_ka else ka_lower,
        config=config,
        project_root=_repo_root(),
    )
    prediction_array = np.asarray(predictions, dtype=np.float64).reshape(1, -1)
    sample["Reference Evaluations"] = prediction_array.reshape(-1).tolist()
    sample["Standard Deviation"] = _relative_resonance_standard_deviation_mhz(
        prediction_array,
        np.asarray([sigma], dtype=np.float64),
        valid_rows=np.asarray([valid_ka], dtype=bool),
    ).reshape(-1).tolist()


def compute_emb_resonance_batch(
    sample: dict[str, Any],
    reference_points: Sequence[float],
    diameter_um: float | None = None,
    device: str = "cuda",
) -> None:
    del device
    config = _load_runtime_config()
    batch = np.asarray(sample["Batch Parameters"], dtype=np.float64)
    if batch.ndim != 2 or batch.shape[1] not in {3, 4}:
        raise ValueError(
            "EMB resonance batch expects shape [batch, 4] for [ka, kb, d0, sigma] "
            f"or [batch, 3] for [ka, kb, sigma], got {batch.shape}."
        )
    diameters = _diameters_from_reference(reference_points, diameter_um)
    sigma_col = 3 if batch.shape[1] == 4 else 2
    sigmas = batch[:, sigma_col]
    raw_ka = batch[:, 0]
    if resonance_forward_model(config) in {
        DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL,
        DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL,
    }:
        bank = _resolve_exact_frequency_bank(config, _repo_root())
        predictions, support_mask = _bank_predict_with_exact_support(
            bank,
            ka_dpd=raw_ka.reshape(-1, 1),
            diameters_um=np.asarray(diameters, dtype=np.float64).reshape(1, -1),
        )
        sample["Batch Reference Evaluations"] = predictions.tolist()
        sample["Batch Standard Deviation"] = _relative_resonance_standard_deviation_mhz(
            predictions,
            sigmas,
            valid_rows=_valid_nonnegative_ka(raw_ka),
            hard_invalid_mask=~support_mask,
        ).tolist()
        return

    ka_lower, _ka_upper = _resonance_ka_bounds(config, _repo_root())
    valid_ka = _valid_resonance_ka(raw_ka, config, _repo_root())
    safe_ka = np.where(valid_ka, raw_ka, ka_lower)
    predictions = predict_resonance_frequencies_mhz_batch(
        diameters,
        ka_dpd=safe_ka,
        config=config,
        project_root=_repo_root(),
    )
    sample["Batch Reference Evaluations"] = predictions.tolist()
    sample["Batch Standard Deviation"] = _relative_resonance_standard_deviation_mhz(
        predictions,
        sigmas,
        valid_rows=valid_ka,
    ).tolist()


def preload_emb_resonance(diameter_um: float, device: str = "cpu", backend: str = "direct") -> None:
    del diameter_um, device, backend
    _load_runtime_config()


def _validate_artifact_prior_support(
    config: Mapping[str, Any],
    experiments: Sequence[Any],
    *,
    ka_bounds_dpd: tuple[float, float],
    mode: str,
    artifact_label: str,
) -> list[dict[str, Any]]:
    """Reject acoustic priors that would force artifact extrapolation."""

    lower, upper = ka_bounds_dpd
    global_prior = config.get("prior_ka")
    entries: list[tuple[str, Any]] = []
    for exp in experiments:
        if exp.name != EMB_RESONANCE_EXPERIMENT:
            continue
        for diameter in exp.diameters:
            override = exp.phase1_prior_overrides(float(diameter)).get("ka")
            entries.append(
                (
                    f"{exp.dataset_name(float(diameter))}.prior_ka",
                    override if override is not None else global_prior,
                )
            )

    validated: list[dict[str, Any]] = []
    for name, value in entries:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(f"{name} must be [minimum, maximum] for {mode} mode.")
        prior_lower = _coerce_finite(value[0], f"{name}[0]")
        prior_upper = _coerce_finite(value[1], f"{name}[1]")
        if prior_lower > prior_upper:
            raise ValueError(f"{name} must be an increasing range.")
        if prior_lower < lower or prior_upper > upper:
            raise ValueError(
                f"{name}={list(value)!r} exceeds {artifact_label} ka support "
                f"[{lower:g}, {upper:g}]."
            )
        validated.append(
            {
                "name": name,
                "bounds": [prior_lower, prior_upper],
                "evaluation_scope": "acoustic_phase1_and_phase3b",
                "phase1_distribution": "normalized_uniform",
                "phase3b_support_source": "inherited_phase1_uniform",
                "clipping_allowed": False,
                "extrapolation_allowed": False,
            }
        )
    return validated


def _validate_surface_prior_support(
    config: Mapping[str, Any],
    experiments: Sequence[Any],
    surface: DpdFrequencySurface,
) -> list[dict[str, Any]]:
    return _validate_artifact_prior_support(
        config,
        experiments,
        ka_bounds_dpd=surface.ka_bounds_dpd,
        mode=DPD_FREQUENCY_SURFACE_FORWARD_MODEL,
        artifact_label="DPD frequency-surface",
    )


def _validate_emulator_bank_prior_support(
    config: Mapping[str, Any],
    experiments: Sequence[Any],
    bank: DpdFrequencyEmulatorBank | DpdPolynomialFrequencyEmulatorBank,
) -> list[dict[str, Any]]:
    global_prior = config.get("prior_ka")
    entries: list[tuple[str, float, Any]] = []
    for exp in experiments:
        if exp.name != EMB_RESONANCE_EXPERIMENT:
            continue
        for diameter in exp.diameters:
            exact_diameter = float(diameter)
            override = exp.phase1_prior_overrides(exact_diameter).get("ka")
            entries.append(
                (
                    f"{exp.dataset_name(exact_diameter)}.prior_ka",
                    exact_diameter,
                    override if override is not None else global_prior,
                )
            )

    validated: list[dict[str, Any]] = []
    for name, diameter, value in entries:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(
                f"{name} must be [minimum, maximum] for {DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL} mode."
            )
        prior_lower = _coerce_finite(value[0], f"{name}[0]")
        prior_upper = _coerce_finite(value[1], f"{name}[1]")
        if prior_lower > prior_upper:
            raise ValueError(f"{name} must be an increasing range.")
        emulator = bank.emulator_for_diameter(diameter)
        lower, upper = emulator.ka_bounds_dpd
        overlaps_exact_support = not (prior_upper < lower or prior_lower > upper)
        if not overlaps_exact_support:
            raise ValueError(
                f"{name}={list(value)!r} does not overlap exact-diameter ka support "
                f"[{lower:g}, {upper:g}] for diameter {diameter:g} um."
            )
        validated.append(
            {
                "name": name,
                "diameter_um": diameter,
                "bounds": [prior_lower, prior_upper],
                "exact_ka_bounds_dpd": [lower, upper],
                "prior_ka_inside_exact_support": prior_lower >= lower and prior_upper <= upper,
                "prior_ka_overlaps_exact_support": overlaps_exact_support,
                "evaluation_scope": "acoustic_phase1_and_phase3b",
                "phase1_distribution": "normalized_uniform",
                "phase3b_support_source": "exact_diameter_hard_support_mask",
                "clipping_allowed": False,
                "extrapolation_allowed": False,
                "outside_exact_support_log_likelihood": "-inf",
            }
        )
    return validated


def _surface_preflight_observations(
    *,
    surface: DpdFrequencySurface,
    target_keys: Sequence[float],
    configured_targets: Mapping[float, tuple[object, float, dict[str, Any]]],
    expected_targets: Mapping[float, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build preflight evidence without using the retired analytical inverse."""

    midpoint_ka = 0.5 * sum(surface.ka_bounds_dpd)
    observations: list[dict[str, Any]] = []
    for diameter_key in target_keys:
        exp, configured_diameter, row = configured_targets[diameter_key]
        observed_diameter = _coerce_positive(row["point"], "resonance diameter_um")
        measured_mhz = _coerce_positive_acoustic_frequency_mhz(
            row["value"],
            "resonance frequency_MHz",
        )
        expected = expected_targets[diameter_key]
        radius_um = observed_diameter / 2.0
        try:
            midpoint_prediction = float(surface.predict_mhz(midpoint_ka, radius_um))
        except ValueError as error:
            raise ValueError(
                "EMB resonance observation is outside the DPD frequency-surface support: "
                f"diameter={observed_diameter:g} um, radius={radius_um:g} um."
            ) from error
        observations.append(
            {
                "dataset_name": exp.dataset_name(configured_diameter),
                "grouped_reference_data": bool(exp.grouped_reference_data),
                "agent": expected.get("agent", surface.agent),
                "source_id": expected.get("source_id"),
                "figure": expected.get("figure"),
                "diameter_um": observed_diameter,
                "configured_diameter_um": configured_diameter,
                "radius_um": radius_um,
                "radius_source": "physical_diameter_um_over_2",
                "measured_frequency_MHz": measured_mhz,
                "prediction_at_midpoint_ka_MHz": midpoint_prediction,
                "data_file": str(row["data_file"]),
            }
        )
    return observations


def _emulator_bank_preflight_observations(
    *,
    bank: DpdFrequencyEmulatorBank | DpdPolynomialFrequencyEmulatorBank,
    target_keys: Sequence[float],
    configured_targets: Mapping[float, tuple[object, float, dict[str, Any]]],
    expected_targets: Mapping[float, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require an accepted exact-diameter emulator for every observation."""

    midpoint_ka = 0.5 * sum(bank.ka_bounds_dpd)
    observations: list[dict[str, Any]] = []
    for diameter_key in target_keys:
        exp, configured_diameter, row = configured_targets[diameter_key]
        observed_diameter = _coerce_positive(row["point"], "resonance diameter_um")
        measured_mhz = _coerce_positive_acoustic_frequency_mhz(
            row["value"],
            "resonance frequency_MHz",
        )
        expected = expected_targets[diameter_key]
        try:
            emulator = bank.emulator_for_diameter(observed_diameter)
            midpoint_prediction = float(emulator.predict_mhz(midpoint_ka))
            compatible_roots = emulator.frequency_roots_ka_dpd(measured_mhz)
        except ValueError as error:
            raise ValueError(
                "EMB resonance observation has no exact supported DPD frequency emulator: "
                f"diameter={observed_diameter:g} um."
            ) from error
        if not compatible_roots:
            raise ValueError(
                "EMB resonance observation frequency is incompatible with its exact-diameter "
                f"DPD frequency emulator: diameter={observed_diameter:g} um, "
                f"frequency={measured_mhz:g} MHz."
            )
        span = emulator.ka_bounds_dpd[1] - emulator.ka_bounds_dpd[0]
        observations.append(
            {
                "dataset_name": exp.dataset_name(configured_diameter),
                "grouped_reference_data": bool(exp.grouped_reference_data),
                "agent": expected.get("agent", bank.agent),
                "source_id": expected.get("source_id"),
                "figure": expected.get("figure"),
                "diameter_um": observed_diameter,
                "configured_diameter_um": configured_diameter,
                "emulator_diameter_um": emulator.diameter_um,
                "diameter_support": "exact",
                "ka_bounds_dpd": list(emulator.ka_bounds_dpd),
                "measured_frequency_MHz": measured_mhz,
                "prediction_at_midpoint_ka_MHz": midpoint_prediction,
                "compatible_ka_roots_dpd": list(compatible_roots),
                "compatible_root_count": len(compatible_roots),
                "compatible_root_normalized_positions": [
                    (value - emulator.ka_bounds_dpd[0]) / span
                    for value in compatible_roots
                ],
                "minimum_compatible_root_boundary_distance_fraction": min(
                    min(
                        (value - emulator.ka_bounds_dpd[0]) / span,
                        (emulator.ka_bounds_dpd[1] - value) / span,
                    )
                    for value in compatible_roots
                ),
                "acoustic_frequency_compatible": True,
                "data_file": str(row["data_file"]),
            }
        )
    return observations


def _emulator_bank_target_interval_compatibility(
    *,
    bank: DpdFrequencyEmulatorBank | DpdPolynomialFrequencyEmulatorBank,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    resonance = _resonance_mapping(config)
    target_intervals = resonance.get("readiness_target_intervals_mhz_by_diameter_um")
    if target_intervals is None:
        return []
    if not isinstance(target_intervals, Mapping) or not target_intervals:
        raise ValueError(
            "resonance.readiness_target_intervals_mhz_by_diameter_um must be a non-empty mapping."
        )
    mu_bounds = config.get("hyperprior_mu_ka")
    if not isinstance(mu_bounds, Sequence) or isinstance(mu_bounds, (str, bytes)) or len(mu_bounds) != 2:
        raise ValueError("Target compatibility requires hyperprior_mu_ka=[minimum, maximum].")
    mu_lower = _coerce_finite(mu_bounds[0], "hyperprior_mu_ka[0]")
    mu_upper = _coerce_finite(mu_bounds[1], "hyperprior_mu_ka[1]")
    rows: list[dict[str, Any]] = []
    for diameter_text, frequency_bounds in sorted(
        target_intervals.items(), key=lambda item: float(item[0])
    ):
        diameter = _coerce_positive(diameter_text, "readiness target diameter_um")
        emulator = bank.emulator_for_diameter(diameter)
        branches = emulator.frequency_compatible_branches_ka_dpd(frequency_bounds)
        if not branches:
            raise ValueError(
                "Readiness acoustic target interval is incompatible with its exact-diameter "
                f"emulator: diameter={diameter:g} um, interval={frequency_bounds!r} MHz."
            )
        span = emulator.ka_bounds_dpd[1] - emulator.ka_bounds_dpd[0]
        branch_rows = [
            {
                "ka_bounds_dpd": [lower, upper],
                "normalized_bounds": [
                    (lower - emulator.ka_bounds_dpd[0]) / span,
                    (upper - emulator.ka_bounds_dpd[0]) / span,
                ],
                "hyperprior_mu_ka_fully_covers_branch": (
                    mu_lower <= lower and upper <= mu_upper
                ),
            }
            for lower, upper in branches
        ]
        rows.append(
            {
                "diameter_um": diameter,
                "frequency_bounds_mhz": [float(value) for value in frequency_bounds],
                "scientific_role": "lineage_only_superseded_analytical_formula_interval",
                "scientific_acceptance_gate": False,
                "allowed_as_acoustic_reference_data": False,
                "bank_ka_bounds_dpd": list(emulator.ka_bounds_dpd),
                "compatible_branches": branch_rows,
                "compatible_branch_count": len(branch_rows),
                "hyperprior_mu_ka_covers_at_least_one_branch": any(
                    row["hyperprior_mu_ka_fully_covers_branch"] for row in branch_rows
                ),
                "acoustic_frequency_compatible": True,
            }
        )
    return rows


def preflight_emb_resonance_config(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    if config.get("phase1_contract_mode") != EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE:
        raise ValueError(
            "EMB resonance HBI requires phase1_contract_mode="
            f"{EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE!r}."
        )
    experiments = [exp for exp in load_experiments(dict(config), Path(project_root)) if exp.enabled]
    resonance_experiments = [exp for exp in experiments if exp.name == EMB_RESONANCE_EXPERIMENT]
    if not resonance_experiments:
        raise ValueError("EMB resonance HBI config must include an enabled resonance experiment.")

    forward_model = resonance_forward_model(config)
    surface = (
        resolve_dpd_frequency_surface(config, project_root)
        if forward_model == DPD_FREQUENCY_SURFACE_FORWARD_MODEL
        else None
    )
    emulator_bank = (
        _resolve_exact_frequency_bank(config, project_root)
        if forward_model
        in {
            DPD_FREQUENCY_EMULATOR_BANK_FORWARD_MODEL,
            DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL,
        }
        else None
    )
    artifact_forward_model = surface is not None or emulator_bank is not None
    defaults = (
        None
        if artifact_forward_model
        else resolve_resonance_runtime_defaults(config, project_root)
    )
    vacuum_constants = (
        resolve_vacuum_shell_constants(defaults)
        if defaults is not None and forward_model == ANALYTICAL_VACUUM_SHELL_FORWARD_MODEL
        else None
    )
    constants = (
        None
        if defaults is None or forward_model == ANALYTICAL_VACUUM_SHELL_FORWARD_MODEL
        else resolve_resonance_constants(config, defaults)
    )
    global_prior_ka = config.get("prior_ka")
    if not isinstance(global_prior_ka, Sequence) or len(global_prior_ka) != 2:
        raise ValueError("EMB resonance HBI requires prior_ka: [min, max].")
    expected_targets = _expected_observations(config)
    configured_targets: dict[float, tuple[object, float, dict[str, Any]]] = {}
    for exp in resonance_experiments:
        for configured_diameter in exp.diameters:
            rows = exp.iter_reference_rows(float(configured_diameter))
            if not rows:
                raise ValueError(f"Resonance dataset {exp.dataset_name(configured_diameter)} has no reference rows.")
            if len(rows) != 1 and not exp.grouped_reference_data:
                raise ValueError(
                    f"Resonance dataset {exp.dataset_name(configured_diameter)} contains {len(rows)} rows. "
                    "Set grouped_reference_data: true for source-level multi-row resonance likelihoods."
                )
            for row in rows:
                diameter_value = _coerce_positive(row["point"], "resonance diameter_um")
                diameter_key = _target_diameter_key(diameter_value)
                measured_mhz = _coerce_positive_acoustic_frequency_mhz(
                    row["value"],
                    "resonance frequency_MHz",
                )
                source_diameter = float(row["source_diameter_um"])
                if not exp.grouped_reference_data and abs(diameter_value - float(configured_diameter)) > _DIAMETER_TOLERANCE_UM:
                    raise ValueError(
                        f"Resonance dataset {exp.dataset_name(configured_diameter)} has observed diameter "
                        f"{diameter_value:g} um but is configured as {float(configured_diameter):g} um."
                    )
                if exp.grouped_reference_data and abs(diameter_value - source_diameter) > _DIAMETER_TOLERANCE_UM:
                    raise ValueError(
                        f"Grouped resonance dataset {exp.dataset_name(configured_diameter)} row from "
                        f"{source_diameter:g} um has observed diameter {diameter_value:g} um."
                    )
                expected = expected_targets.get(diameter_key)
                if expected is None:
                    raise ValueError(
                        "EMB resonance HBI config contains unexpected target diameter "
                        f"{diameter_value:g} um in {exp.dataset_name(configured_diameter)}."
                    )
                expected_mhz = float(expected["frequency_MHz"])
                if abs(measured_mhz - expected_mhz) > _FREQUENCY_TOLERANCE_MHZ:
                    raise ValueError(
                        f"Resonance dataset {exp.dataset_name(configured_diameter)} has measured frequency "
                        f"{measured_mhz:g} MHz; expected {expected_mhz:g} MHz for {diameter_value:g} um."
                    )
                if diameter_key in configured_targets:
                    previous_exp, _previous_diameter, _previous_row = configured_targets[diameter_key]
                    raise ValueError(
                        "EMB resonance HBI config contains duplicate target diameter "
                        f"{diameter_value:g} um in {previous_exp.name} and {exp.name}."
                    )
                configured_targets[diameter_key] = (exp, float(configured_diameter), row)
    if set(configured_targets) != set(expected_targets):
        expected_text = ", ".join(f"{diameter:g} um" for diameter in sorted(expected_targets))
        configured_text = ", ".join(f"{diameter:g} um" for diameter in sorted(configured_targets)) or "<none>"
        raise ValueError(
            "EMB resonance HBI configured diameters do not match expected target diameters "
            f"{{{expected_text}}}; configured {{{configured_text}}}."
        )

    target_keys = sorted(configured_targets)
    compression_support = preflight_direct_compression_surrogate_support(
        config,
        project_root=project_root,
    )
    indentation_support = preflight_indentation_surrogate_support(
        config,
        project_root=project_root,
    )
    if surface is not None:
        prior_support = _validate_surface_prior_support(config, experiments, surface)
        observations = _surface_preflight_observations(
            surface=surface,
            target_keys=target_keys,
            configured_targets=configured_targets,
            expected_targets=expected_targets,
        )
        return {
            "schema_version": EMB_RESONANCE_SCHEMA_VERSION,
            "status": "passed",
            "agent": resonance_agent(config),
            "forward_model": forward_model,
            "phase1_contract_mode": str(config.get("phase1_contract_mode")),
            "frequency_surface": {
                "ka_bounds_dpd": list(surface.ka_bounds_dpd),
                "radius_bounds_um": list(surface.radius_bounds_um),
                "conditions": dict(surface.conditions),
                "provenance": dict(surface.provenance),
                "validation": dict(surface.validation),
            },
            "surface_prior_support": prior_support,
            "compression_surrogate_support": compression_support,
            "indentation_surrogate_support": indentation_support,
            "observations": observations,
        }
    if emulator_bank is not None:
        required_exact_diameters = sorted({_target_diameter_key(value) for value in target_keys})
        if not required_exact_diameters:
            raise ValueError(
                "artifact_emulator_bank mode requires configured acoustic target diameters."
            )
        artifact_diameters = sorted(
            _target_diameter_key(value) for value in emulator_bank.diameters_um
        )
        missing_exact_diameters = sorted(
            set(required_exact_diameters) - set(artifact_diameters)
        )
        if missing_exact_diameters:
            raise ValueError(
                "DPD frequency-emulator bank is missing configured acoustic exact diameters: "
                f"missing={missing_exact_diameters}, required={required_exact_diameters}, "
                f"artifact={artifact_diameters}."
            )
        prior_support = _validate_emulator_bank_prior_support(
            config,
            experiments,
            emulator_bank,
        )
        observations = _emulator_bank_preflight_observations(
            bank=emulator_bank,
            target_keys=target_keys,
            configured_targets=configured_targets,
            expected_targets=expected_targets,
        )
        target_interval_compatibility = _emulator_bank_target_interval_compatibility(
            bank=emulator_bank,
            config=config,
        )
        five_eighty_branch_diagnostics = None
        if resonance_agent(config) == "sonovue" and target_interval_compatibility:
            five_eighty_rows = [
                row
                for row in target_interval_compatibility
                if math.isclose(
                    float(row["diameter_um"]),
                    5.8,
                    rel_tol=0.0,
                    abs_tol=_DIAMETER_TOLERANCE_UM,
                )
            ]
            five_eighty = five_eighty_rows[0] if len(five_eighty_rows) == 1 else None
            if five_eighty is not None:
                five_eighty_branch_diagnostics = {
                "required": False,
                "scientific_role": "lineage_only_superseded_analytical_formula_interval",
                "scientific_acceptance_gate": False,
                "allowed_as_acoustic_reference_data": False,
                "diameter_um": 5.8,
                "compatible_branch_count": five_eighty["compatible_branch_count"],
                "compatible_branches": five_eighty["compatible_branches"],
                "postrun_required": [],
                }
        return {
            "schema_version": EMB_RESONANCE_SCHEMA_VERSION,
            "status": "passed",
            "agent": resonance_agent(config),
            "forward_model": forward_model,
            "phase1_contract_mode": str(config.get("phase1_contract_mode")),
            "frequency_emulator_bank": {
                "model_family": (
                    "low_order_polynomial"
                    if forward_model == DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL
                    else "pchip"
                ),
                "diameters_um": list(emulator_bank.diameters_um),
                "required_exact_diameters_um": required_exact_diameters,
                "common_ka_bounds_dpd": list(emulator_bank.ka_bounds_dpd),
                "exact_ka_bounds_by_diameter_um": {
                    f"{entry.diameter_um:g}": list(entry.ka_bounds_dpd)
                    for entry in emulator_bank.emulators
                },
                "conditions": dict(emulator_bank.conditions),
                "provenance": dict(emulator_bank.provenance),
                "validation": dict(emulator_bank.validation),
            },
            "emulator_bank_prior_support": prior_support,
            "compression_surrogate_support": compression_support,
            "indentation_surrogate_support": indentation_support,
            "phase2_support_contract": {
                "conditional_distribution": "unbounded_normal",
                "evaluates_acoustic_emulator": False,
                "acoustic_phase3b_support_source": "inherited_phase1_uniform",
                "mechanical_phase3b_support_source": "inherited_phase1_uniform",
            },
            "frequency_target_compatibility": target_interval_compatibility,
            "five_eighty_branch_diagnostics": five_eighty_branch_diagnostics,
            "required_postrun_metrics": {
                "support_boundary_pressure": "per evaluator and exact diameter",
                "spectroscopy_ridge_localization": "conditioned cloud versus force-only cloud",
                "experimental_acoustic_observation_coverage": "2.6=3.1, 3.2=2.1, and 4.0=1.6 MHz only",
                (
                    "exact_polynomial_propagation_support"
                    if forward_model == DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL
                    else "exact_pchip_propagation_support"
                ): "zero unsupported mechanical-posterior samples per exact diameter",
                "same_diameter_3p2_unpaired_crosscheck": "report visibly without treating as a same-bubble likelihood residual",
                "clipping_extrapolation_fallback_counts": "all must equal zero",
                "vectorized_runtime": "batched timing and zero likelihood scalar loops",
                "mandatory_paper_figure_gate": "run-specific Figure 7, Figure 8, SI, provenance, and visual audit",
            },
            "observations": observations,
        }

    assert defaults is not None
    radii_by_configured_diameter = resolve_resonance_radii_dpd(config, target_keys)
    radius_sources = resonance_radius_sources(config, target_keys)
    radius_map = {
        diameter: float(radius)
        for diameter, radius in zip(target_keys, radii_by_configured_diameter, strict=True)
    }
    observations = []
    for diameter_key in target_keys:
        exp, configured_diameter, row = configured_targets[diameter_key]
        observed_diameter = _coerce_positive(row["point"], "resonance diameter_um")
        measured_mhz = _coerce_positive_acoustic_frequency_mhz(
            row["value"],
            "resonance frequency_MHz",
        )
        expected = expected_targets[diameter_key]
        prior_overrides = exp.phase1_prior_overrides(configured_diameter)
        prior_ka = prior_overrides.get("ka", list(global_prior_ka))
        radius_dpd = radius_map[diameter_key]
        if forward_model == ANALYTICAL_VACUUM_SHELL_FORWARD_MODEL:
            solved_ka = float(
                emb_vacuum_shell_ka_dpd_from_frequency_mhz(
                    frequency_mhz=measured_mhz,
                    radii_dpd=radius_dpd,
                    defaults=defaults,
                )
            )
            predicted_mhz = float(
                emb_vacuum_shell_frequency_mhz_from_ka_dpd(
                    ka_dpd=solved_ka,
                    radii_dpd=radius_dpd,
                    defaults=defaults,
                )
            )
            chi_dpd = solved_ka / _fscale(defaults)
        else:
            assert constants is not None
            solved_ka = float(
                emb_resonance_ka_dpd_from_frequency_mhz(
                    frequency_mhz=measured_mhz,
                    radii_dpd=radius_dpd,
                    constants=constants,
                    defaults=defaults,
                )
            )
            predicted_mhz = float(
                emb_resonance_frequency_mhz_from_ka_dpd(
                    ka_dpd=solved_ka,
                    radii_dpd=radius_dpd,
                    constants=constants,
                    defaults=defaults,
                )
            )
            chi_dpd = solved_ka / _fscale(defaults)
        if not float(prior_ka[0]) <= solved_ka <= float(prior_ka[1]):
            raise ValueError(
                "EMB resonance prior_ka does not cover the DPD ka solved from "
                f"{configured_diameter:g} um resonance: ka={solved_ka:.6g}, prior_ka={prior_ka}."
            )
        observations.append(
            {
                "dataset_name": exp.dataset_name(configured_diameter),
                "grouped_reference_data": bool(exp.grouped_reference_data),
                "agent": expected.get("agent", resonance_agent(config)),
                "source_id": expected.get("source_id"),
                "figure": expected.get("figure"),
                "diameter_um": observed_diameter,
                "configured_diameter_um": configured_diameter,
                "radius_dpd": radius_dpd,
                "radius_source": radius_sources[diameter_key],
                "measured_frequency_MHz": measured_mhz,
                "ka_dpd_solved_from_frequency": solved_ka,
                "chi_dpd_solved_from_frequency": chi_dpd,
                "prediction_at_solved_ka_MHz": predicted_mhz,
                "data_file": str(row["data_file"]),
            }
        )
    return {
        "schema_version": EMB_RESONANCE_SCHEMA_VERSION,
        "status": "passed",
        "agent": resonance_agent(config),
        "forward_model": forward_model,
        "phase1_contract_mode": str(config.get("phase1_contract_mode")),
        "unit_time_seconds": resonance_unit_time_seconds(defaults),
        "fscale": _fscale(defaults),
        "constants": (
            vacuum_constants.to_dict()
            if vacuum_constants is not None
            else constants.to_dict()
        ),
        "compression_surrogate_support": compression_support,
        "indentation_surrogate_support": indentation_support,
        "observations": observations,
    }


__all__ = [
    "DPD_FREQUENCY_POLYNOMIAL_BANK_FORWARD_MODEL",
    "EMB_RESONANCE_EXPERIMENT",
    "EMB_RESONANCE_SCHEMA_VERSION",
    "SONOVUE_RESONANCE_TARGETS_MHZ",
    "EmbResonanceConstants",
    "EmbVacuumShellConstants",
    "compute_emb_resonance",
    "compute_emb_resonance_batch",
    "emb_resonance_frequency_mhz_from_ka_dpd",
    "emb_resonance_ka_dpd_from_frequency_mhz",
    "emb_vacuum_shell_frequency_mhz_from_ka_dpd",
    "emb_vacuum_shell_ka_dpd_from_frequency_mhz",
    "fallback_radius_dpd_from_diameter_um",
    "preflight_direct_compression_surrogate_support",
    "preflight_indentation_surrogate_support",
    "predict_resonance_frequencies_mhz",
    "predict_resonance_frequencies_mhz_batch",
    "preflight_emb_resonance_config",
    "preload_emb_resonance",
    "resonance_agent",
    "resonance_radius_sources",
    "resonance_unit_time_seconds",
    "resolve_dpd_frequency_emulator_bank",
    "resolve_dpd_frequency_polynomial_bank",
    "resolve_resonance_constants",
    "resolve_resonance_radii_dpd",
    "resolve_resonance_runtime_defaults",
    "resolve_vacuum_shell_constants",
]
