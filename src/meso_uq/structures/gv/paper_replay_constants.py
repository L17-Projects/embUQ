from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
import os
from pathlib import Path
from typing import Literal

import yaml

from .geometry_sources import REPO_ROOT, gv_canonical_geometry_default_source, gv_canonical_geometry_default_path
from .parameters import GV_MATERIAL_PARAMETER_NAMES, GV_PARAMETER_CONTRACT


GVPaperReplayValueStatus = Literal["paper_confirmed", "runtime_default"]

GV_PAPER_REPLAY_PROFILE_ID = "gv-paper-replay"
GV_PAPER_REPLAY_SCHEMA_VERSION = 1
GV_PAPER_REPLAY_PAPER_PDF = Path(
    os.environ.get(
        "MESOUQ_GV_PAPER_REPLAY_PAPER_PDF",
        str(Path.home() / "workspace" / "EMB_GV_DPD.pdf"),
    )
).expanduser()
GV_PAPER_REPLAY_SI_PDF = Path(
    os.environ.get(
        "MESOUQ_GV_PAPER_REPLAY_SI_PDF",
        str(Path.home() / "workspace" / "an5c02783_si_001.pdf"),
    )
).expanduser()
GV_PAPER_REPLAY_FIGURE_TARGETS = ("stretching", "buckling", "torsion", "eigenmodes")
_GEOMETRY_PARAMETER_NAMES = ("radGV", "height")
_ZERO_ALLOWED_MATERIAL_PARAMETERS = frozenset({"b1", "b2", "a3", "a4"})
_DEFAULT_SOURCE_NOTE = (
    "Current values come from canonical staged GV runtime defaults and are not paper-confirmed unless stated otherwise."
)
_AMBIGUITY_NOTES = (
    "The local paper and SI PDFs were available by path during implementation, but no local PDF text extraction tool was installed.",
    "No exact nine-parameter MesoUQ calibration tuple was unambiguously machine-extracted from the paper or SI in this change.",
    "The staged GV source defaults set a3, a4, b1, and b2 to 0.0; those coupling defaults are deterministic runtime inputs and are validated as finite non-negative values.",
    "The staged orthotropic scripts overwrite muL with mu before writing YAML outputs; this layer records that serialized runtime convention as mu_l.",
)
_LANE_MATERIAL_VALUES = {
    "torsion": {
        "ka": 40892.490643486664,
        "kb": 120.82922175289485,
        "mu": 17525.353132922857,
        "b1": 0.0,
        "b2": 0.0,
        "a3": 0.0,
        "a4": 0.0,
        "mu_l": 17525.353132922857,
        "c": 175253.53132922857,
    },
}


@dataclass(frozen=True)
class GVPaperReplayValue:
    value: float
    status: GVPaperReplayValueStatus
    source: str
    units: str
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "value": float(self.value),
            "status": self.status,
            "source": self.source,
            "units": self.units,
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class GVPaperReplayGeometry:
    radGV: GVPaperReplayValue
    height: GVPaperReplayValue

    def values(self) -> dict[str, float]:
        return {
            "radGV": float(self.radGV.value),
            "height": float(self.height.value),
        }

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            "radGV": self.radGV.to_dict(),
            "height": self.height.to_dict(),
        }


@dataclass(frozen=True)
class GVPaperReplayProvenance:
    schema_version: int
    paper_pdf_path: str
    si_pdf_path: str
    canonical_runtime_source: str
    figure_targets: tuple[str, ...]
    unit_notes: tuple[str, ...]
    convention_notes: tuple[str, ...]
    ambiguity_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "paper_pdf_path": self.paper_pdf_path,
            "si_pdf_path": self.si_pdf_path,
            "canonical_runtime_source": self.canonical_runtime_source,
            "figure_targets": list(self.figure_targets),
            "unit_notes": list(self.unit_notes),
            "convention_notes": list(self.convention_notes),
            "ambiguity_notes": list(self.ambiguity_notes),
        }


@dataclass(frozen=True)
class GVPaperReplayProfile:
    profile_id: str
    material_parameters: dict[str, GVPaperReplayValue]
    geometry: GVPaperReplayGeometry
    provenance: GVPaperReplayProvenance

    def material_values(self) -> dict[str, float]:
        return {
            name: float(self.material_parameters[name].value)
            for name in GV_MATERIAL_PARAMETER_NAMES
        }

    def material_values_for_lane(self, lane: str) -> dict[str, float]:
        values = _LANE_MATERIAL_VALUES.get(lane)
        if values is None:
            return self.material_values()
        return {name: float(values[name]) for name in GV_MATERIAL_PARAMETER_NAMES}

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "material_parameters": {
                name: self.material_parameters[name].to_dict()
                for name in GV_MATERIAL_PARAMETER_NAMES
            },
            "geometry": self.geometry.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


def _require_numeric(value: object, context: str, *, positive: bool) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric.") from exc
    if not isfinite(numeric):
        raise ValueError(f"{context} must be finite.")
    if positive and numeric <= 0.0:
        raise ValueError(f"{context} must be > 0.")
    return numeric


def _require_material_numeric(
    value: object,
    context: str,
    *,
    name: str,
    strict_positive: bool,
) -> float:
    numeric = _require_numeric(value, context, positive=False)
    if strict_positive:
        if numeric <= 0.0:
            raise ValueError(f"{context} must be > 0.")
        return numeric
    if numeric < 0.0:
        raise ValueError(f"{context} must be >= 0.")
    if numeric == 0.0 and name not in _ZERO_ALLOWED_MATERIAL_PARAMETERS:
        raise ValueError(f"{context} must be > 0.")
    return numeric


def _load_canonical_runtime_defaults() -> dict[str, float]:
    parameters_path = gv_canonical_geometry_default_path()
    with parameters_path.open("r", encoding="utf-8") as handle:
        parameters = yaml.safe_load(handle)
    if not isinstance(parameters, dict):
        raise ValueError(f"Expected YAML mapping in {parameters_path}")

    required = (
        "ul",
        "kbol",
        "t0",
        "shell_th",
        "fscale",
        "Yt",
        "Yl",
        "nu",
        "kb_fac",
        "radGV",
        "height",
        "a3",
        "a4",
        "b1",
        "b2",
    )
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(f"Canonical GV runtime defaults missing required keys: {', '.join(missing)}")

    ul = _require_numeric(parameters["ul"], "canonical ul", positive=True)
    kbol = _require_numeric(parameters["kbol"], "canonical kbol", positive=True)
    t0 = _require_numeric(parameters["t0"], "canonical t0", positive=True)
    shell_th = _require_numeric(parameters["shell_th"], "canonical shell_th", positive=True)
    fscale = _require_numeric(parameters["fscale"], "canonical fscale", positive=True)
    yt = _require_numeric(parameters["Yt"], "canonical Yt", positive=True)
    yl = _require_numeric(parameters["Yl"], "canonical Yl", positive=True)
    nu = _require_numeric(parameters["nu"], "canonical nu", positive=True)
    kb_fac = _require_numeric(parameters["kb_fac"], "canonical kb_fac", positive=True)
    ue = kbol * t0
    denominator = yl - nu**2 * yt
    if denominator <= 0.0:
        raise ValueError("Canonical GV orthotropic denominator must be > 0.")

    ka = fscale * yt * yl * shell_th * (1 + nu) / (2 * denominator) / (ue / ul**2)
    mu = fscale * yt * yl * shell_th * (1 - nu) / (2 * denominator) / (ue / ul**2)
    mu_l_pre = abs(yt * yl * (1 - nu) / (2 * denominator))
    c = (
        fscale
        * (yl**2 + yt * yl + 4 * yt * mu_l_pre * nu**2 - 4 * yl * mu_l_pre - 2 * yt * yl * nu)
        * shell_th
        / denominator
        / (ue / ul**2)
    )
    kb = kb_fac * fscale * 2.0 / sqrt(3.0) * yt * shell_th**3 / (12 * (1 - nu**2)) / ue

    return {
        "ka": ka,
        "kb": kb,
        "mu": mu,
        "b1": _require_numeric(parameters["b1"], "canonical b1", positive=False),
        "b2": _require_numeric(parameters["b2"], "canonical b2", positive=False),
        "a3": _require_numeric(parameters["a3"], "canonical a3", positive=False),
        "a4": _require_numeric(parameters["a4"], "canonical a4", positive=False),
        "mu_l": mu,
        "c": c,
        "radGV": _require_numeric(parameters["radGV"], "canonical radGV", positive=True),
        "height": _require_numeric(parameters["height"], "canonical height", positive=True),
    }


def load_gv_paper_replay_profile() -> GVPaperReplayProfile:
    runtime_defaults = _load_canonical_runtime_defaults()
    source = gv_canonical_geometry_default_source()
    material_parameters = {
        name: GVPaperReplayValue(
            value=runtime_defaults[name],
            status="runtime_default",
            source=source,
            units=GV_PARAMETER_CONTRACT.get_parameter(name).units,
            note=_DEFAULT_SOURCE_NOTE,
        )
        for name in GV_MATERIAL_PARAMETER_NAMES
    }
    geometry = GVPaperReplayGeometry(
        radGV=GVPaperReplayValue(
            value=runtime_defaults["radGV"],
            status="runtime_default",
            source=source,
            units="legacy GV generator length units",
            note=_DEFAULT_SOURCE_NOTE,
        ),
        height=GVPaperReplayValue(
            value=runtime_defaults["height"],
            status="runtime_default",
            source=source,
            units="legacy GV generator length units",
            note=_DEFAULT_SOURCE_NOTE,
        ),
    )
    provenance = GVPaperReplayProvenance(
        schema_version=GV_PAPER_REPLAY_SCHEMA_VERSION,
        paper_pdf_path=str(GV_PAPER_REPLAY_PAPER_PDF),
        si_pdf_path=str(GV_PAPER_REPLAY_SI_PDF),
        canonical_runtime_source=source,
        figure_targets=GV_PAPER_REPLAY_FIGURE_TARGETS,
        unit_notes=(
            "Material parameter units follow meso_uq.structures.gv.parameters.",
            "Geometry uses staged legacy YAML keys radGV and height.",
        ),
        convention_notes=(
            "The replay layer preserves the nine-key MesoUQ ordering ka, kb, mu, b1, b2, a3, a4, mu_l, c.",
            "Legacy runtime YAML may serialize mu_l as muL; this layer normalizes to mu_l.",
            "Paper-exact torsion uses the lane-specific parameter tuple serialized in the dropped GV torsion figure bundle.",
        ),
        ambiguity_notes=_AMBIGUITY_NOTES,
    )
    return GVPaperReplayProfile(
        profile_id=GV_PAPER_REPLAY_PROFILE_ID,
        material_parameters=material_parameters,
        geometry=geometry,
        provenance=provenance,
    )


def validate_gv_paper_replay_profile(
    profile: GVPaperReplayProfile,
    *,
    require_positive_material_parameters: bool = True,
) -> None:
    missing = [name for name in GV_MATERIAL_PARAMETER_NAMES if name not in profile.material_parameters]
    if missing:
        raise ValueError(f"GV paper replay profile missing material parameters: {', '.join(missing)}")

    extra = sorted(set(profile.material_parameters) - set(GV_MATERIAL_PARAMETER_NAMES))
    if extra:
        raise ValueError(f"GV paper replay profile has unexpected material parameters: {', '.join(extra)}")

    for name in GV_MATERIAL_PARAMETER_NAMES:
        entry = profile.material_parameters[name]
        _require_material_numeric(
            entry.value,
            f"GV paper replay material parameter '{name}'",
            name=name,
            strict_positive=require_positive_material_parameters,
        )
        if not entry.source:
            raise ValueError(f"GV paper replay material parameter '{name}' must declare a source.")
        if not entry.units:
            raise ValueError(f"GV paper replay material parameter '{name}' must declare units.")

    for name in _GEOMETRY_PARAMETER_NAMES:
        entry = getattr(profile.geometry, name)
        _require_numeric(entry.value, f"GV paper replay geometry '{name}'", positive=True)
        if not entry.source:
            raise ValueError(f"GV paper replay geometry '{name}' must declare a source.")
        if not entry.units:
            raise ValueError(f"GV paper replay geometry '{name}' must declare units.")

    provenance = profile.provenance
    if provenance.schema_version < 1:
        raise ValueError("GV paper replay provenance schema_version must be >= 1.")
    if not provenance.paper_pdf_path:
        raise ValueError("GV paper replay provenance must include paper_pdf_path.")
    if not provenance.si_pdf_path:
        raise ValueError("GV paper replay provenance must include si_pdf_path.")
    if not provenance.canonical_runtime_source:
        raise ValueError("GV paper replay provenance must include canonical_runtime_source.")
    if not provenance.figure_targets:
        raise ValueError("GV paper replay provenance must include figure_targets.")
    if not provenance.unit_notes:
        raise ValueError("GV paper replay provenance must include unit_notes.")
    if not provenance.convention_notes:
        raise ValueError("GV paper replay provenance must include convention_notes.")
    if not provenance.ambiguity_notes:
        raise ValueError("GV paper replay provenance must include ambiguity_notes.")


def canonical_runtime_source_path() -> Path:
    return REPO_ROOT / gv_canonical_geometry_default_source()
