from __future__ import annotations

import json
import sys
from pathlib import Path

from math import inf

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import Candidate
from meso_uq.active_learning.constraints import (
    ActiveLearningCandidateConstraintReport,
    REASON_DUPLICATE_CANDIDATE_ID,
    REASON_INVALID_CANDIDATE,
    REASON_INVALID_EXPERIMENT,
    REASON_INVALID_FAMILY,
    REASON_INVALID_PAYLOAD_PATH,
    REASON_MISSING_GV_GEOMETRY,
    REASON_MIXED_GV_GEOMETRY,
    REASON_MISSING_REQUIRED_PARAMETER_PATH,
    REASON_NON_FINITE_NUMERIC,
    REASON_OUT_OF_RANGE,
    validate_active_learning_candidates,
)

_GV_CANONICAL_MATERIAL_PARAMETERS = {
    "ka": 1.1,
    "kb": 1.2,
    "mu": 0.9,
    "b1": 0.1,
    "b2": 0.2,
    "a3": 0.3,
    "a4": 0.4,
    "mu_l": 0.5,
    "c": 0.6,
}


def test_validate_active_learning_candidates_accepts_candidate_objects_and_mapping_shapes() -> None:
    candidates = [
        Candidate(
            candidate_id="gv-valid-object",
            parameters={
                "family": "gv",
                "gv_launch": {
                    "experiment": "stretching",
                    "geometry": {"radGV": 2.0, "height": 14.28},
                    "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                    "controls": {"tot_force": [500.0, 750.0]},
                },
            },
        ),
        {
            "candidate_id": "gv-valid-mapping",
            "parameters": {
                "family": "gv",
                "experiment": "buckling",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"buck": [0.8, 0.9]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert isinstance(report, ActiveLearningCandidateConstraintReport)
    assert [item.candidate_id for item in report.valid_candidates] == [
        "gv-valid-object",
        "gv-valid-mapping",
    ]
    assert report.rejected_candidate_ids == ()
    assert report.reason_code_counts == {}
    assert report.per_candidate_reasons == {}
    assert json.loads(json.dumps(report.as_dict())) == report.as_dict()


def test_validate_active_learning_candidates_accepts_family_inferred_from_payload_keys() -> None:
    candidates = [
        {
            "candidate_id": "gv-inferred",
            "parameters": {
                "gv_launch": {
                    "experiment": "stretching",
                    "geometry": {"radGV": 2.0, "height": 14.28},
                    "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                    "controls": {"tot_force": [1.0]},
                },
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert [item.candidate_id for item in report.valid_candidates] == ["gv-inferred"]
    assert report.rejected_candidate_ids == ()


def test_validate_active_learning_candidates_rejects_inferred_gv_launch_when_emb_only() -> None:
    candidates = [
        {
            "candidate_id": "gv-inferred-disallowed",
            "parameters": {
                "gv_launch": {
                    "experiment": "stretching",
                    "geometry": {"radGV": 2.0, "height": 14.28},
                    "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                    "controls": {"tot_force": [1.0]},
                },
            },
        },
    ]

    report = validate_active_learning_candidates(candidates, allowed_families=("emb",))

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("gv-inferred-disallowed",)
    reason = report.per_candidate_reasons["gv-inferred-disallowed"][0]
    assert reason.code == REASON_INVALID_FAMILY
    assert report.reason_code_counts[REASON_INVALID_FAMILY] == 1


def test_validate_active_learning_candidates_rejects_inferred_emb_launch_when_gv_only() -> None:
    candidates = [
        {
            "candidate_id": "emb-inferred-disallowed",
            "parameters": {
                "emb_launch": {
                    "experiment": "stretching",
                    "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                    "controls": {"tot_force": [1.0]},
                },
            },
        },
    ]

    report = validate_active_learning_candidates(candidates, allowed_families=("gv",))

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("emb-inferred-disallowed",)
    reason = report.per_candidate_reasons["emb-inferred-disallowed"][0]
    assert reason.code == REASON_INVALID_FAMILY
    assert report.reason_code_counts[REASON_INVALID_FAMILY] == 1


def test_validate_active_learning_candidates_flags_invalid_candidate_shapes() -> None:
    candidates = [
        {"candidate_id": "invalid", "parameters": "not-a-mapping"},
        {"bad": "shape"},
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("invalid", "candidate-1")
    assert report.reason_code_counts[REASON_INVALID_CANDIDATE] == 2
    assert all(
        reason.code == REASON_INVALID_CANDIDATE
        for reasons in report.per_candidate_reasons.values()
        for reason in reasons
    )


def test_validate_active_learning_candidates_flags_duplicate_candidate_ids() -> None:
    candidates = [
        {
            "candidate_id": "dupe",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
        {
            "candidate_id": "dupe",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [2.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert [item.candidate_id for item in report.valid_candidates] == ["dupe"]
    assert report.rejected_candidate_ids == ("dupe",)
    reasons = [item.code for item in report.per_candidate_reasons["dupe"]]
    assert REASON_DUPLICATE_CANDIDATE_ID in reasons
    assert report.reason_code_counts[REASON_DUPLICATE_CANDIDATE_ID] == 1


def test_validate_active_learning_candidates_rejects_duplicate_after_malformed_entry() -> None:
    candidates = [
        {
            "candidate_id": "dupe-after-malformed",
            "parameters": "not-a-mapping",
        },
        {
            "candidate_id": "dupe-after-malformed",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [2.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("dupe-after-malformed", "dupe-after-malformed")
    reasons = [item.code for item in report.per_candidate_reasons["dupe-after-malformed"]]
    assert REASON_INVALID_CANDIDATE in reasons
    assert REASON_DUPLICATE_CANDIDATE_ID in reasons
    assert report.reason_code_counts[REASON_INVALID_CANDIDATE] == 1
    assert report.reason_code_counts[REASON_DUPLICATE_CANDIDATE_ID] == 1


def test_validate_active_learning_candidates_rejects_duplicate_after_constraint_rejection() -> None:
    candidates = [
        {
            "candidate_id": "dupe-after-constraint-reject",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
            },
        },
        {
            "candidate_id": "dupe-after-constraint-reject",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [2.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == (
        "dupe-after-constraint-reject",
        "dupe-after-constraint-reject",
    )
    reasons = [item.code for item in report.per_candidate_reasons["dupe-after-constraint-reject"]]
    assert REASON_MISSING_REQUIRED_PARAMETER_PATH in reasons
    assert REASON_DUPLICATE_CANDIDATE_ID in reasons
    assert report.reason_code_counts[REASON_MISSING_REQUIRED_PARAMETER_PATH] == 1
    assert report.reason_code_counts[REASON_DUPLICATE_CANDIDATE_ID] == 1


def test_validate_active_learning_candidates_rejects_unknown_family() -> None:
    candidates = [
        {
            "candidate_id": "bad-family",
            "parameters": {
                "family": "fluid",
                "experiment": "stretching",
                "controls": {"tot_force": [1.0]},
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("bad-family",)
    reason = report.per_candidate_reasons["bad-family"][0]
    assert reason.code == REASON_INVALID_FAMILY
    assert "fluid" in reason.message


def test_validate_active_learning_candidates_rejects_missing_required_parameter_paths() -> None:
    candidates = [
        {
            "candidate_id": "missing-controls",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
            },
        },
        {
            "candidate_id": "missing-experiment",
            "parameters": {
                "family": "gv",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert len(report.valid_candidates) == 0
    assert set(report.rejected_candidate_ids) == {"missing-controls", "missing-experiment"}
    assert all(
        reason.code == REASON_MISSING_REQUIRED_PARAMETER_PATH
        for reasons in report.per_candidate_reasons.values()
        for reason in reasons
    )
    assert report.reason_code_counts[REASON_MISSING_REQUIRED_PARAMETER_PATH] == 2


def test_validate_active_learning_candidates_preserves_empty_required_parameter_paths() -> None:
    candidates = [
        {
            "candidate_id": "emb-missing-default-required-experiment",
            "parameters": {
                "family": "emb",
                "physical_parameters": {"shell_modulus": 0.5},
            },
        },
    ]

    report_without_required_checks = validate_active_learning_candidates(
        candidates,
        required_parameter_paths={},
    )
    report_with_default_required_checks = validate_active_learning_candidates(
        candidates,
        required_parameter_paths=None,
    )

    assert [item.candidate_id for item in report_without_required_checks.valid_candidates] == [
        "emb-missing-default-required-experiment",
    ]
    assert report_without_required_checks.rejected_candidate_ids == ()
    assert report_without_required_checks.reason_code_counts == {}
    assert report_with_default_required_checks.valid_candidates == ()
    assert report_with_default_required_checks.rejected_candidate_ids == (
        "emb-missing-default-required-experiment",
    )
    assert (
        report_with_default_required_checks.reason_code_counts[
            REASON_MISSING_REQUIRED_PARAMETER_PATH
        ]
        == 1
    )


def test_validate_active_learning_candidates_rejects_disallowed_experiments_and_numeric_failures() -> None:
    candidates = [
        {
            "candidate_id": "bad-experiment",
            "parameters": {
                "family": "gv",
                "experiment": "torsion",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0, 2.0]},
            },
        },
        {
            "candidate_id": "bad-numeric",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0, inf]},
                "radGV": 1.0,
                "height": 14.28,
            },
        },
        {
            "candidate_id": "out-of-range",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
                "radGV": -2.0,
                "height": 14.28,
            },
        },
    ]

    report = validate_active_learning_candidates(
        candidates,
        allowed_experiments={"stretching", "buckling"},
    )

    assert {item.candidate_id for item in report.valid_candidates} == set()
    assert set(report.rejected_candidate_ids) == {"bad-experiment", "bad-numeric", "out-of-range"}
    assert report.reason_code_counts[REASON_INVALID_EXPERIMENT] == 1
    assert report.reason_code_counts[REASON_NON_FINITE_NUMERIC] == 1
    assert report.reason_code_counts[REASON_OUT_OF_RANGE] == 1


def test_validate_active_learning_candidates_preserves_empty_numeric_range_checks() -> None:
    candidates = [
        {
            "candidate_id": "emb-non-finite-default-numeric-path",
            "parameters": {
                "family": "emb",
                "experiment": "indentation",
                "controls": {"indentation_depth": [inf]},
            },
        },
    ]

    report_without_numeric_checks = validate_active_learning_candidates(
        candidates,
        numeric_range_checks={},
    )
    report_with_default_numeric_checks = validate_active_learning_candidates(
        candidates,
        numeric_range_checks=None,
    )

    assert [item.candidate_id for item in report_without_numeric_checks.valid_candidates] == [
        "emb-non-finite-default-numeric-path",
    ]
    assert report_without_numeric_checks.rejected_candidate_ids == ()
    assert report_without_numeric_checks.reason_code_counts == {}
    assert report_with_default_numeric_checks.valid_candidates == ()
    assert report_with_default_numeric_checks.rejected_candidate_ids == (
        "emb-non-finite-default-numeric-path",
    )
    assert report_with_default_numeric_checks.reason_code_counts[REASON_NON_FINITE_NUMERIC] == 1


def test_validate_active_learning_candidates_rejects_non_numeric_path_values() -> None:
    candidates = [
        {
            "candidate_id": "non-numeric-controls",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": ["a", 1.0]},
                "radGV": 2.0,
                "height": 14.28,
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("non-numeric-controls",)
    assert report.per_candidate_reasons["non-numeric-controls"][0].code == REASON_INVALID_PAYLOAD_PATH


def test_validate_active_learning_candidates_rejects_gv_candidates_without_geometry_or_legacy_geometry_fields() -> None:
    candidates = [
        {
            "candidate_id": "missing-gv-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("missing-gv-geometry",)
    assert report.reason_code_counts[REASON_MISSING_GV_GEOMETRY] == 1
    reason = report.per_candidate_reasons["missing-gv-geometry"][0]
    assert reason.code == REASON_MISSING_GV_GEOMETRY
    assert "complete nested geometry" in reason.message


def test_validate_active_learning_candidates_rejects_gv_candidates_with_incomplete_nested_geometry() -> None:
    candidates = [
        {
            "candidate_id": "incomplete-nested-gv-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "geometry": {"radGV": 2.0},
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("incomplete-nested-gv-geometry",)
    assert report.reason_code_counts[REASON_MISSING_GV_GEOMETRY] == 1
    reason = report.per_candidate_reasons["incomplete-nested-gv-geometry"][0]
    assert reason.code == REASON_MISSING_GV_GEOMETRY


def test_validate_active_learning_candidates_rejects_gv_candidates_with_mixed_partial_nested_and_legacy_geometry() -> None:
    candidates = [
        {
            "candidate_id": "mixed-nested-legacy-gv-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "geometry": {"radGV": 2.0},
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("mixed-nested-legacy-gv-geometry",)
    assert report.reason_code_counts[REASON_MIXED_GV_GEOMETRY] == 1
    reason = report.per_candidate_reasons["mixed-nested-legacy-gv-geometry"][0]
    assert reason.code == REASON_MIXED_GV_GEOMETRY
    assert "provide exactly one geometry form" in reason.message


def test_validate_active_learning_candidates_rejects_gv_candidates_with_mixed_complete_nested_and_legacy_geometry() -> None:
    candidates = [
        {
            "candidate_id": "mixed-complete-nested-legacy-gv-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("mixed-complete-nested-legacy-gv-geometry",)
    assert report.reason_code_counts[REASON_MIXED_GV_GEOMETRY] == 1


def test_validate_active_learning_candidates_rejects_gv_candidates_with_empty_geometry_key_and_legacy_geometry() -> None:
    candidates = [
        {
            "candidate_id": "mixed-empty-geometry-and-legacy-gv-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "geometry": {},
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("mixed-empty-geometry-and-legacy-gv-geometry",)
    assert report.reason_code_counts[REASON_MIXED_GV_GEOMETRY] == 1


def test_validate_active_learning_candidates_rejects_gv_candidates_with_null_geometry_key_and_legacy_geometry() -> None:
    candidates = [
        {
            "candidate_id": "mixed-null-geometry-and-legacy-gv-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "geometry": None,
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("mixed-null-geometry-and-legacy-gv-geometry",)
    assert report.reason_code_counts[REASON_MIXED_GV_GEOMETRY] == 1


def test_validate_active_learning_candidates_accepts_gv_candidates_with_legacy_geometry_fields() -> None:
    candidates = [
        {
            "candidate_id": "legacy-gv-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert [item.candidate_id for item in report.valid_candidates] == ["legacy-gv-geometry"]
    assert report.rejected_candidate_ids == ()


def test_validate_active_learning_candidates_accepts_gv_candidates_with_complete_nested_geometry() -> None:
    candidates = [
        {
            "candidate_id": "nested-gv-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert [item.candidate_id for item in report.valid_candidates] == ["nested-gv-geometry"]
    assert report.rejected_candidate_ids == ()


def test_validate_active_learning_candidates_rejects_gv_shear_flow_by_default() -> None:
    candidates = [
        {
            "candidate_id": "gv-shear-flow",
            "parameters": {
                "family": "gv",
                "experiment": "shear_flow",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"ptan": [0.2], "afsi": 1.0, "bpress": -91.0},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("gv-shear-flow",)
    assert report.reason_code_counts[REASON_INVALID_EXPERIMENT] == 1


def test_validate_active_learning_candidates_rejects_noncanonical_gv_experiment_casing() -> None:
    candidates = [
        {
            "candidate_id": "gv-titlecase-experiment",
            "parameters": {
                "family": "gv",
                "gv_launch": {
                    "experiment": "Stretching",
                    "geometry": {"radGV": 2.0, "height": 14.28},
                    "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                    "controls": {"tot_force": [1.0]},
                },
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("gv-titlecase-experiment",)
    reason = report.per_candidate_reasons["gv-titlecase-experiment"][0]
    assert reason.code == REASON_INVALID_EXPERIMENT
    assert "Stretching" in reason.message


def test_validate_active_learning_candidates_rejects_gv_geometry_zero_for_legacy_and_nested() -> None:
    candidates = [
        {
            "candidate_id": "gv-legacy-zero-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 0.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
        {
            "candidate_id": "gv-nested-zero-geometry",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "geometry": {"radGV": 2.0, "height": 0.0},
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert set(report.rejected_candidate_ids) == {"gv-legacy-zero-geometry", "gv-nested-zero-geometry"}
    assert report.reason_code_counts[REASON_OUT_OF_RANGE] >= 2


def test_validate_active_learning_candidates_checks_gv_material_parameters_complete_and_positive() -> None:
    candidates = [
        {
            "candidate_id": "gv-partial-materials",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": {"ka": 1.0, "kb": 1.1},
                "controls": {"tot_force": [1.0]},
            },
        },
        {
            "candidate_id": "gv-complete-materials",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert [item.candidate_id for item in report.valid_candidates] == ["gv-complete-materials"]
    assert "gv-partial-materials" in report.rejected_candidate_ids
    partial_reasons = report.per_candidate_reasons["gv-partial-materials"]
    assert any(reason.code == REASON_MISSING_REQUIRED_PARAMETER_PATH for reason in partial_reasons)


def test_validate_active_learning_candidates_rejects_gv_zero_coupling_material_parameter() -> None:
    candidates = [
        {
            "candidate_id": "gv-zero-coupling",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": {**_GV_CANONICAL_MATERIAL_PARAMETERS, "b1": 0.0},
                "controls": {"tot_force": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("gv-zero-coupling",)
    assert report.reason_code_counts[REASON_OUT_OF_RANGE] == 1


def test_validate_active_learning_candidates_rejects_gv_scalar_only_controls() -> None:
    candidates = [
        {
            "candidate_id": "gv-scalar-controls",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": 1.0},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("gv-scalar-controls",)
    assert report.reason_code_counts[REASON_INVALID_PAYLOAD_PATH] == 1


def test_validate_active_learning_candidates_rejects_gv_multiple_sweep_axes() -> None:
    candidates = [
        {
            "candidate_id": "gv-multiple-sweeps",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0], "bpress": [-91.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("gv-multiple-sweeps",)
    assert report.reason_code_counts[REASON_INVALID_PAYLOAD_PATH] == 1


def test_validate_active_learning_candidates_rejects_gv_invalid_control_name() -> None:
    candidates = [
        {
            "candidate_id": "gv-invalid-control-name",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"buck": [1.0]},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert report.valid_candidates == ()
    assert report.rejected_candidate_ids == ("gv-invalid-control-name",)
    assert report.reason_code_counts[REASON_INVALID_PAYLOAD_PATH] == 1


def test_validate_active_learning_candidates_accepts_valid_gv_handoff_controls_and_materials() -> None:
    candidates = [
        {
            "candidate_id": "gv-handoff-valid",
            "parameters": {
                "family": "gv",
                "experiment": "stretching",
                "radGV": 2.0,
                "height": 14.28,
                "material_parameters": _GV_CANONICAL_MATERIAL_PARAMETERS,
                "controls": {"tot_force": [1.0], "bpress": -91.0},
            },
        },
    ]

    report = validate_active_learning_candidates(candidates)

    assert [item.candidate_id for item in report.valid_candidates] == ["gv-handoff-valid"]
    assert report.rejected_candidate_ids == ()
