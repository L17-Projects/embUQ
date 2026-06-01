from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES,
    EMB_34UM_DNN_CAUSAL_B1_VALUE,
    EMB_34UM_DNN_CAUSAL_B2_VALUE,
    EMB_34UM_DNN_CAUSAL_A3_VALUE,
    EMB_34UM_DNN_CAUSAL_A4_VALUE,
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_EXPERIMENT,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
    EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
    EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
    EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
    EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC,
    EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET,
    EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE,
    EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT,
    EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION,
    EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND,
    EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
    EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD,
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_STEP_SIZE,
    EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
    EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
    dnn_causal_curves_per_replicate,
    dnn_causal_total_dpd_curve_count,
    dnn_causal_training_curve_count,
    dnn_causal_total_dpd_curve_count_with_pilot,
    validate_dnn_causal_force_grid,
    validate_dnn_causal_lhs_source,
    validate_dnn_causal_protocol_manifest,
    validate_dnn_causal_protocol_payload,
    validate_dnn_causal_replicate_count,
    validate_dnn_causal_selector_backend,
    validate_dnn_causal_timing_canary,
    validate_dnn_causal_cycle_count,
)


def test_protocol_defaults_encode_dnn_causal_contract() -> None:
    payload = validate_dnn_causal_protocol_payload(
        {"ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE, "timing_canary": {"required": True}}
    )
    assert payload.cycle_count == EMB_34UM_DNN_CAUSAL_MIN_CYCLES
    assert payload.primary_replicate_count == EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT
    assert payload.max_replicate_count == EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT
    assert payload.active_replicate_count == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert payload.replicate_count == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert payload.force_grid == EMB_34UM_DNN_CAUSAL_FORCE_GRID
    assert payload.lhs_source == EMB_34UM_DNN_CAUSAL_LHS_SOURCE
    assert payload.selector_backend == EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND
    assert payload.uncertainty_source == EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE
    assert payload.ensemble_size == EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    assert payload.timing_canary == {
        "required": True,
        "point_count": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
    }
    manifest = payload.as_manifest()
    assert manifest["active_variables"] == list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES)
    assert manifest["bounds"]["radp"] == list(EMB_34UM_DNN_CAUSAL_BOUNDS["radp"])
    assert manifest["bounds"]["shell_th"] == list(EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"])
    assert manifest["retry_limit_default"] == EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT
    assert manifest["pilot_size"] == EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    assert manifest["primary_statistic"] == EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC
    assert manifest["primary_statistic_target"] == EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET
    assert manifest["final_ci_upper_bound_threshold"] == EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD


def test_protocol_default_bounds_and_fixed_coefficients() -> None:
    assert EMB_34UM_DNN_CAUSAL_BOUNDS["ka"] == (1e3, 1e5)
    assert EMB_34UM_DNN_CAUSAL_BOUNDS["kb"] == (2e3, 2e4)
    assert EMB_34UM_DNN_CAUSAL_BOUNDS["radp"] == (6.45, 6.60)
    assert EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"] == (3.75e-9, 4.0e-9)
    assert EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT == 0
    assert EMB_34UM_DNN_CAUSAL_EXPERIMENT == "indentation"
    assert EMB_34UM_DNN_CAUSAL_B1_VALUE == 0.0
    assert EMB_34UM_DNN_CAUSAL_B2_VALUE == 0.0
    assert EMB_34UM_DNN_CAUSAL_A3_VALUE == 0.0
    assert EMB_34UM_DNN_CAUSAL_A4_VALUE == 0.0
    assert EMB_34UM_DNN_CAUSAL_BPRESS_VALUE == -91.0
    assert EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES == ("ka", "kb", "radp", "shell_th")
    assert EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE == 200
    assert EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE == 300
    assert EMB_34UM_DNN_CAUSAL_STEP_SIZE == 100


def test_validate_force_grid_rejects_15_points() -> None:
    with pytest.raises(ValueError, match="must contain 8 points"):
        validate_dnn_causal_force_grid(tuple(range(15)))


def test_validate_force_grid_rejects_wrong_spacing() -> None:
    wrong_grid = tuple(i * 600.0 for i in range(8))
    with pytest.raises(ValueError, match="exactly 8 evenly spaced points"):
        validate_dnn_causal_force_grid(wrong_grid)


def test_validate_replicate_count_rejects_outside_accepted_bounds() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_dnn_causal_replicate_count(0)
    with pytest.raises(ValueError, match=r"\[1, 5\]"):
        validate_dnn_causal_replicate_count(6)


def test_validate_lhs_source_rejects_non_fresh_sources() -> None:
    with pytest.raises(ValueError, match="fresh DPD"):
        validate_dnn_causal_lhs_source("samples_all.dat")
    with pytest.raises(ValueError, match="fresh DPD"):
        validate_dnn_causal_lhs_source("historical")
    with pytest.raises(ValueError, match="fresh DPD"):
        validate_dnn_causal_lhs_source("replay")


def test_validate_selector_backend_rejects_lightweight_tokens() -> None:
    with pytest.raises(ValueError, match="lightweight"):
        validate_dnn_causal_selector_backend("fixed_architecture_dnn_lightweight")
    with pytest.raises(ValueError, match="lightweight"):
        validate_dnn_causal_selector_backend("random-feature")


def test_validate_ensemble_size_is_fixed_10_for_dnn_protocol() -> None:
    payload = {
        "ensemble_size": 10,
        "timing_canary": {"required": True},
        "replicate_count": EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
        "force_grid": EMB_34UM_DNN_CAUSAL_FORCE_GRID,
        "lhs_source": EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
        "selector_backend": EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND,
    }
    protocol = validate_dnn_causal_protocol_payload(payload)
    assert protocol.ensemble_size == EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE


def test_validate_protocol_payload_rejects_wrong_ensemble_size() -> None:
    with pytest.raises(ValueError, match="fixed at 10"):
        validate_dnn_causal_protocol_payload(
            {
                "ensemble_size": 9,
                "timing_canary": {"required": True},
                "replicate_count": EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
                "force_grid": EMB_34UM_DNN_CAUSAL_FORCE_GRID,
                "lhs_source": EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
                "selector_backend": EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND,
            }
        )


def test_validate_timing_canary_is_required_and_cannot_be_downgraded_without_reason() -> None:
    with pytest.raises(ValueError, match="must include timing_canary"):
        validate_dnn_causal_protocol_payload({"ensemble_size": 10})

    with pytest.raises(ValueError, match="override_reason"):
        validate_dnn_causal_timing_canary({"required": True, "point_count": 5})


def test_validate_protocol_payload_rejects_wrong_protocol_shape() -> None:
    # D2-style payload keys should still resolve to the D4 contract that includes
    # ka/kb/radp/shell_th, not to a legacy 2-variable interpretation.
    manifest = validate_dnn_causal_protocol_manifest(
        {
            "schema_version": EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION,
            "family": "emb",
            "experiment": "indentation",
            "active_variables": ["ka", "kb"],
            "bounds": {
                "ka": list(EMB_34UM_DNN_CAUSAL_BOUNDS["ka"]),
                "kb": list(EMB_34UM_DNN_CAUSAL_BOUNDS["kb"]),
            },
            "primary_replicate_count": EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
            "max_replicate_count": EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
            "active_replicate_count": EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT,
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "cycle_count": EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
            "ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
            "timing_canary": {"required": True},
            "lhs_source": EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
            "selector_backend": EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND,
        }
    )
    assert manifest["replicate_count"] == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["primary_replicate_count"] == EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT
    assert manifest["max_replicate_count"] == EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT
    assert manifest["active_replicate_count"] == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["ensemble_size"] == EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    assert manifest["schema_version"] == EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION
    assert manifest["active_variables"] == list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES)
    assert manifest["bounds"]["radp"] == list(EMB_34UM_DNN_CAUSAL_BOUNDS["radp"])
    assert manifest["bounds"]["shell_th"] == list(EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"])


def test_validate_protocol_payload_accepts_escalation_replicate_count_five() -> None:
    protocol = validate_dnn_causal_protocol_payload(
        {
            "ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
            "timing_canary": {"required": True},
            "active_replicate_count": EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
        }
    )
    assert protocol.active_replicate_count == EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT


def test_total_dpd_curve_count_scales_with_active_replicates() -> None:
    cycles = EMB_34UM_DNN_CAUSAL_MIN_CYCLES
    per_replicate = EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE + (2 * cycles * EMB_34UM_DNN_CAUSAL_STEP_SIZE)
    assert dnn_causal_total_dpd_curve_count(cycle_count=cycles, active_replicate_count=3) == (
        EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE + 3 * per_replicate
    )
    assert dnn_causal_total_dpd_curve_count(cycle_count=cycles, active_replicate_count=5) == (
        EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE + 5 * per_replicate
    )


def test_dnn_causal_curve_helpers_include_3900_plus_pilot() -> None:
    cycles = EMB_34UM_DNN_CAUSAL_MIN_CYCLES
    assert dnn_causal_curves_per_replicate(cycle_count=cycles) == EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE + 2 * EMB_34UM_DNN_CAUSAL_STEP_SIZE * cycles
    assert dnn_causal_training_curve_count(cycle_count=cycles) == 1200
    assert dnn_causal_total_dpd_curve_count(cycle_count=cycles, active_replicate_count=3) == 3900
    assert dnn_causal_total_dpd_curve_count_with_pilot(cycle_count=cycles, active_replicate_count=3) == (
        3900 + EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    )


def test_validate_protocol_payload_rejects_missing_ensemble_or_timing_fields() -> None:
    with pytest.raises(ValueError, match="must include ensemble_size"):
        validate_dnn_causal_protocol_payload({"timing_canary": {"required": True}})
    with pytest.raises(ValueError, match="must include timing_canary"):
        validate_dnn_causal_protocol_payload({"ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE})


def test_validate_cycle_count_bounds() -> None:
    assert validate_dnn_causal_cycle_count(EMB_34UM_DNN_CAUSAL_MIN_CYCLES) == EMB_34UM_DNN_CAUSAL_MIN_CYCLES
    assert validate_dnn_causal_cycle_count(EMB_34UM_DNN_CAUSAL_MAX_CYCLES) == EMB_34UM_DNN_CAUSAL_MAX_CYCLES
    with pytest.raises(ValueError, match="at least"):
        validate_dnn_causal_cycle_count(EMB_34UM_DNN_CAUSAL_MIN_CYCLES - 1)
    with pytest.raises(ValueError, match="at most"):
        validate_dnn_causal_cycle_count(EMB_34UM_DNN_CAUSAL_MAX_CYCLES + 1)
