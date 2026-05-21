from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_B1_VALUE,
    EMB_34UM_DNN_CAUSAL_B2_VALUE,
    EMB_34UM_DNN_CAUSAL_A3_VALUE,
    EMB_34UM_DNN_CAUSAL_A4_VALUE,
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_EXPERIMENT,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
    EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
    EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
    EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE,
    EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION,
    EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND,
    EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
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


def test_protocol_default_bounds_and_fixed_coefficients() -> None:
    assert EMB_34UM_DNN_CAUSAL_BOUNDS["ka"] == (1e2, 6e5)
    assert EMB_34UM_DNN_CAUSAL_BOUNDS["kb"] == (400.0, 70000.0)
    assert EMB_34UM_DNN_CAUSAL_EXPERIMENT == "indentation"
    assert EMB_34UM_DNN_CAUSAL_B1_VALUE == 0.0
    assert EMB_34UM_DNN_CAUSAL_B2_VALUE == 0.0
    assert EMB_34UM_DNN_CAUSAL_A3_VALUE == 0.0
    assert EMB_34UM_DNN_CAUSAL_A4_VALUE == 0.0


def test_validate_force_grid_rejects_15_points() -> None:
    with pytest.raises(ValueError, match="must contain 8 points"):
        validate_dnn_causal_force_grid(tuple(range(15)))


def test_validate_force_grid_rejects_wrong_spacing() -> None:
    wrong_grid = tuple(i * 600.0 for i in range(8))
    with pytest.raises(ValueError, match="exactly 8 evenly spaced points"):
        validate_dnn_causal_force_grid(wrong_grid)


def test_validate_replicate_count_rejects_not_five() -> None:
    with pytest.raises(ValueError, match="is fixed at 5"):
        validate_dnn_causal_replicate_count(4)


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
        "replicate_count": 5,
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
                "replicate_count": 5,
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
            "replicate_count": EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT,
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "cycle_count": EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
            "ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
            "timing_canary": {"required": True},
            "lhs_source": EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
            "selector_backend": EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND,
        }
    )
    assert manifest["replicate_count"] == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["ensemble_size"] == EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    assert manifest["schema_version"] == EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION


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
