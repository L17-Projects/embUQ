from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from meso_uq.structures.gv.paper_replay_constants import (
    GV_PAPER_REPLAY_FIGURE_TARGETS,
    GV_PAPER_REPLAY_PAPER_PDF,
    GV_PAPER_REPLAY_PROFILE_ID,
    GV_PAPER_REPLAY_SCHEMA_VERSION,
    GV_PAPER_REPLAY_SI_PDF,
    GVPaperReplayGeometry,
    GVPaperReplayProfile,
    GVPaperReplayProvenance,
    GVPaperReplayValue,
    canonical_runtime_source_path,
    load_gv_paper_replay_profile,
    validate_gv_paper_replay_profile,
)
from meso_uq.structures.gv.parameters import GV_MATERIAL_PARAMETER_NAMES


def test_load_gv_paper_replay_profile_exposes_current_runtime_defaults() -> None:
    profile = load_gv_paper_replay_profile()
    payload = profile.to_dict()

    assert profile.profile_id == GV_PAPER_REPLAY_PROFILE_ID
    assert payload["profile_id"] == GV_PAPER_REPLAY_PROFILE_ID
    assert payload["material_parameters"]["ka"]["note"]
    assert payload["geometry"]["radGV"]["value"] == 2.0
    assert tuple(profile.material_parameters) == GV_MATERIAL_PARAMETER_NAMES
    assert profile.material_values() == pytest.approx(
        {
            "ka": 31076.50001404906,
            "kb": 16.13638111016224,
            "mu": 16733.50000756487,
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
            "mu_l": 16733.50000756487,
            "c": 143430.0000648418,
        }
    )
    assert profile.geometry.values() == {"radGV": 2.0, "height": 14.28}
    assert profile.geometry.radGV.status == "runtime_default"
    assert profile.geometry.height.status == "runtime_default"
    assert all(profile.material_parameters[name].status == "runtime_default" for name in GV_MATERIAL_PARAMETER_NAMES)


def test_validate_gv_paper_replay_profile_rejects_non_positive_runtime_defaults_in_strict_mode() -> None:
    profile = load_gv_paper_replay_profile()

    with pytest.raises(ValueError, match="material parameter 'b1'"):
        validate_gv_paper_replay_profile(profile)


def test_validate_gv_paper_replay_profile_accepts_profile_when_strict_positive_disabled() -> None:
    profile = load_gv_paper_replay_profile()

    validate_gv_paper_replay_profile(profile, require_positive_material_parameters=False)


def test_validate_gv_paper_replay_profile_relaxed_mode_rejects_zero_core_parameters() -> None:
    profile = load_gv_paper_replay_profile()
    bad_parameters = dict(profile.material_parameters)
    bad_parameters["ka"] = replace(bad_parameters["ka"], value=0.0)

    with pytest.raises(ValueError, match="material parameter 'ka'"):
        validate_gv_paper_replay_profile(
            replace(profile, material_parameters=bad_parameters),
            require_positive_material_parameters=False,
        )


def test_validate_gv_paper_replay_profile_accepts_all_positive_profile() -> None:
    profile = load_gv_paper_replay_profile()
    positive_parameters = {
        name: replace(entry, value=index + 1.0)
        for index, (name, entry) in enumerate(profile.material_parameters.items())
    }
    positive_profile = replace(profile, material_parameters=positive_parameters)

    validate_gv_paper_replay_profile(positive_profile)


def test_validate_gv_paper_replay_profile_rejects_missing_keys_and_geometry() -> None:
    profile = load_gv_paper_replay_profile()
    missing_key_profile = replace(
        profile,
        material_parameters={name: value for name, value in profile.material_parameters.items() if name != "c"},
    )
    with pytest.raises(ValueError, match="missing material parameters: c"):
        validate_gv_paper_replay_profile(missing_key_profile, require_positive_material_parameters=False)

    bad_geometry = GVPaperReplayGeometry(
        radGV=replace(profile.geometry.radGV, value=0.0),
        height=profile.geometry.height,
    )
    with pytest.raises(ValueError, match="geometry 'radGV'"):
        validate_gv_paper_replay_profile(
            replace(profile, geometry=bad_geometry),
            require_positive_material_parameters=False,
        )


def test_gv_paper_replay_value_to_dict_omits_absent_note() -> None:
    entry = GVPaperReplayValue(value=1.0, status="paper_confirmed", source="paper", units="unit")

    assert entry.to_dict() == {
        "value": 1.0,
        "status": "paper_confirmed",
        "source": "paper",
        "units": "unit",
    }


def test_validate_gv_paper_replay_profile_rejects_extra_keys_and_missing_metadata() -> None:
    profile = load_gv_paper_replay_profile()
    extra_parameters = dict(profile.material_parameters)
    extra_parameters["extra"] = next(iter(profile.material_parameters.values()))
    with pytest.raises(ValueError, match="unexpected material parameters"):
        validate_gv_paper_replay_profile(
            replace(profile, material_parameters=extra_parameters),
            require_positive_material_parameters=False,
        )

    no_source = dict(profile.material_parameters)
    no_source["ka"] = replace(no_source["ka"], source="")
    with pytest.raises(ValueError, match="must declare a source"):
        validate_gv_paper_replay_profile(
            replace(profile, material_parameters=no_source),
            require_positive_material_parameters=False,
        )

    no_units = dict(profile.material_parameters)
    no_units["ka"] = replace(no_units["ka"], units="")
    with pytest.raises(ValueError, match="must declare units"):
        validate_gv_paper_replay_profile(
            replace(profile, material_parameters=no_units),
            require_positive_material_parameters=False,
        )

    bad_geometry = GVPaperReplayGeometry(
        radGV=replace(profile.geometry.radGV, source=""),
        height=profile.geometry.height,
    )
    with pytest.raises(ValueError, match="geometry 'radGV' must declare a source"):
        validate_gv_paper_replay_profile(
            replace(profile, geometry=bad_geometry),
            require_positive_material_parameters=False,
        )

    bad_geometry = GVPaperReplayGeometry(
        radGV=replace(profile.geometry.radGV, units=""),
        height=profile.geometry.height,
    )
    with pytest.raises(ValueError, match="geometry 'radGV' must declare units"):
        validate_gv_paper_replay_profile(
            replace(profile, geometry=bad_geometry),
            require_positive_material_parameters=False,
        )


def test_gv_paper_replay_provenance_schema_is_complete() -> None:
    profile = load_gv_paper_replay_profile()
    provenance = profile.provenance.to_dict()

    assert provenance["schema_version"] == GV_PAPER_REPLAY_SCHEMA_VERSION
    assert provenance["paper_pdf_path"] == str(GV_PAPER_REPLAY_PAPER_PDF)
    assert provenance["si_pdf_path"] == str(GV_PAPER_REPLAY_SI_PDF)
    assert provenance["canonical_runtime_source"] == "gv/stretching/src/parameters-default.gv.yaml"
    assert provenance["figure_targets"] == list(GV_PAPER_REPLAY_FIGURE_TARGETS)
    assert provenance["unit_notes"]
    assert provenance["convention_notes"]
    assert provenance["ambiguity_notes"]
    assert canonical_runtime_source_path().is_file()


def test_gv_paper_replay_source_pdf_defaults_are_home_relative() -> None:
    assert GV_PAPER_REPLAY_PAPER_PDF == Path(
        os.environ.get(
            "MESOUQ_GV_PAPER_REPLAY_PAPER_PDF",
            str(Path.home() / "workspace" / "EMB_GV_DPD.pdf"),
        )
    ).expanduser()
    assert GV_PAPER_REPLAY_SI_PDF == Path(
        os.environ.get(
            "MESOUQ_GV_PAPER_REPLAY_SI_PDF",
            str(Path.home() / "workspace" / "an5c02783_si_001.pdf"),
        )
    ).expanduser()


def test_validate_gv_paper_replay_profile_rejects_empty_provenance_fields() -> None:
    profile = load_gv_paper_replay_profile()
    empty_provenance = GVPaperReplayProvenance(
        schema_version=0,
        paper_pdf_path="",
        si_pdf_path="",
        canonical_runtime_source="",
        figure_targets=(),
        unit_notes=(),
        convention_notes=(),
        ambiguity_notes=(),
    )
    broken_profile = GVPaperReplayProfile(
        profile_id=profile.profile_id,
        material_parameters=profile.material_parameters,
        geometry=profile.geometry,
        provenance=empty_provenance,
    )

    with pytest.raises(ValueError, match="schema_version"):
        validate_gv_paper_replay_profile(broken_profile, require_positive_material_parameters=False)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("paper_pdf_path", "paper_pdf_path"),
        ("si_pdf_path", "si_pdf_path"),
        ("canonical_runtime_source", "canonical_runtime_source"),
        ("figure_targets", "figure_targets"),
        ("unit_notes", "unit_notes"),
        ("convention_notes", "convention_notes"),
        ("ambiguity_notes", "ambiguity_notes"),
    ],
)
def test_validate_gv_paper_replay_profile_rejects_each_empty_provenance_field(
    field: str,
    message: str,
) -> None:
    profile = load_gv_paper_replay_profile()
    replacements = {field: "" if field.endswith("_path") or field == "canonical_runtime_source" else ()}

    with pytest.raises(ValueError, match=message):
        validate_gv_paper_replay_profile(
            replace(profile, provenance=replace(profile.provenance, **replacements)),
            require_positive_material_parameters=False,
        )
