from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_acquisition import (  # noqa: E402
    EMB_34UM_DNN_D4_CANDIDATE_POOL_DEFAULT_REQUESTED_SIZE,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_design import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    build_emb_34um_dnn_causal_validation_campaign_manifest,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES,
    EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_FORCE_COUNT,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID_POLICY,
    EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
    EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
    EMB_34UM_DNN_CAUSAL_B1_VALUE,
    EMB_34UM_DNN_CAUSAL_B2_VALUE,
    EMB_34UM_DNN_CAUSAL_A3_VALUE,
    EMB_34UM_DNN_CAUSAL_A4_VALUE,
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_FRESH_ONLY,
    EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE,
    EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD,
    EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT,
    EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC,
    EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET,
    EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION,
    EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
    EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
    EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
    EMB_34UM_DNN_CAUSAL_STEP_SIZE,
    EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
    EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
    dnn_causal_total_dpd_curve_count,
    dnn_causal_total_dpd_curve_count_with_pilot,
    is_dnn_causal_low_corner_excluded,
)


def _load_script_module() -> Any:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "prepare_emb_34um_dnn_causal_validation.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_emb_34um_dnn_causal_validation", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_force_grid(path: Path) -> None:
    force_grid = " ".join(str(value) for value in EMB_34UM_DNN_CAUSAL_FORCE_GRID)
    path.write_text(f"0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 {force_grid}\n", encoding="utf-8")


def test_dnn_causal_manifest_counts_and_provenance() -> None:
    manifest = build_emb_34um_dnn_causal_validation_campaign_manifest(
        timestamp="20260521_120000",
        run_id_prefix="emb-34um-dnn-causal-test",
        campaign_root=Path("/tmp") / "emb-34um-dnn-causal-test",
        scratch_root=Path("/tmp") / "emb-34um-dnn-causal-test",
        vault_root_timestamp=Path("/tmp") / "vault" / "emb-34um-dnn-causal-test",
        force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    )

    assert manifest["policy"]["primary_replicate_count"] == EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT
    assert manifest["policy"]["max_replicate_count"] == EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT
    assert manifest["policy"]["active_replicate_count"] == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["policy"]["replicate_count"] == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["policy"]["replicates"] == [1, 2, 3]
    assert manifest["policy"]["min_cycles"] == EMB_34UM_DNN_CAUSAL_MIN_CYCLES
    assert manifest["policy"]["max_cycles"] == EMB_34UM_DNN_CAUSAL_MAX_CYCLES
    assert manifest["policy"]["shared_initial_size"] == EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE
    assert manifest["policy"]["unseen_test_size"] == EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE
    assert manifest["policy"]["step_size"] == EMB_34UM_DNN_CAUSAL_STEP_SIZE
    assert manifest["policy"]["force_grid"] == list(EMB_34UM_DNN_CAUSAL_FORCE_GRID)
    assert manifest["policy"]["force_count"] == EMB_34UM_DNN_CAUSAL_FORCE_COUNT
    assert manifest["policy"]["force_min"] == 0.0
    assert manifest["policy"]["force_max"] == 5000.0
    assert manifest["policy"]["force_grid_policy"] == EMB_34UM_DNN_CAUSAL_FORCE_GRID_POLICY
    assert manifest["policy"]["active_variables"] == list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES)
    assert manifest["policy"]["bounds"]["radp"] == list(EMB_34UM_DNN_CAUSAL_BOUNDS["radp"])
    assert manifest["policy"]["bounds"]["shell_th"] == list(EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"])
    assert manifest["policy"]["parameter_space"] == EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE
    assert manifest["policy"]["pilot_size"] == EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    assert manifest["policy"]["fresh_only"] == EMB_34UM_DNN_CAUSAL_FRESH_ONLY
    assert manifest["policy"]["primary_statistic"] == EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC
    assert manifest["policy"]["primary_statistic_target"] == EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET
    assert (
        manifest["policy"]["final_ci_upper_bound_threshold"]
        == EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD
    )
    assert manifest["policy"]["min_final_relative_improvement"] == EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT
    assert manifest["policy"]["total_dpd_curve_count"] == dnn_causal_total_dpd_curve_count(
        cycle_count=EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
        active_replicate_count=EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT,
    )
    assert manifest["policy"]["total_dpd_curve_count_with_pilot"] == dnn_causal_total_dpd_curve_count_with_pilot(
        cycle_count=EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
        active_replicate_count=EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT,
    )
    assert manifest["policy"]["ensemble_size"] == EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    assert manifest["policy"]["dpd_walltime_target_default"] == EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT
    assert manifest["policy"]["dpd_walltime_target"] == EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT
    assert manifest["policy"]["low_corner_exclusion"] == EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION
    assert manifest["policy"]["lhs_source"] == EMB_34UM_DNN_CAUSAL_LHS_SOURCE
    assert manifest["policy"]["test_set_source"] == EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE

    assert len(manifest["unseen_test"]["candidate_records"]) == EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE
    assert len(manifest["pilot"]["candidate_records"]) == EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    assert manifest["pilot"]["selection_source"] == "fresh_dpd_pilot"
    assert manifest["pilot"]["selection_mode"] == "dnn_causal_fresh_only"
    assert manifest["pilot"]["batch_root"].endswith("/pilot")
    assert manifest["policy"]["unseen_test_start_index"] > 0
    assert len(manifest["seeds"]) == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["command_inventory"]["count"] == 35
    assert manifest["command_inventory"]["entries"][0]["execution_mode"] == "render-only"
    assert manifest["command_inventory"]["entries"][0]["array"] == "0-199%30"
    assert manifest["command_inventory"]["entries"][0]["command"].endswith(
        "scripts/platforms/karolina/sbatch/emb_34um_dnn_causal_validation_array.sbatch"
    )
    batch_roots = {
        (entry["replica"], entry["mode"]): entry["batch_root"]
        for entry in manifest["command_inventory"]["entries"]
    }
    assert batch_roots[(1, "al-step-01")].endswith("/replica-001/al-step-01")
    assert batch_roots[(1, "lhs-step-01")].endswith("/replica-001/lhs-step-01")
    assert "al-step-01-01" not in json.dumps(manifest["command_inventory"])
    assert "lhs-step-01-01" not in json.dumps(manifest["command_inventory"])
    assert manifest["provenance"]["ensemble_size"] == EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    assert manifest["provenance"]["max_replicate_count"] == EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT
    assert manifest["provenance"]["active_replicate_count"] == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["provenance"]["force_grid_signature"] == manifest["policy"]["force_grid_signature"]
    assert manifest["provenance"]["unseen_test_start_index"] == manifest["policy"]["unseen_test_start_index"]
    assert manifest["provenance"]["active_variables"] == list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES)
    assert manifest["provenance"]["bounds"]["radp"] == list(EMB_34UM_DNN_CAUSAL_BOUNDS["radp"])
    assert manifest["provenance"]["bounds"]["shell_th"] == list(EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"])
    assert manifest["policy"]["fixed_coefficients"] == {
        "b1": EMB_34UM_DNN_CAUSAL_B1_VALUE,
        "b2": EMB_34UM_DNN_CAUSAL_B2_VALUE,
        "a3": EMB_34UM_DNN_CAUSAL_A3_VALUE,
        "a4": EMB_34UM_DNN_CAUSAL_A4_VALUE,
    }
    assert manifest["policy"]["fixed_controls"] == {"bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE}
    assert manifest["provenance"]["fixed_coefficients"] == manifest["policy"]["fixed_coefficients"]
    assert manifest["provenance"]["fixed_controls"] == manifest["policy"]["fixed_controls"]
    assert manifest["provenance"]["pilot_size"] == EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    assert manifest["provenance"]["primary_statistic"] == EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC
    assert manifest["provenance"]["primary_statistic_target"] == EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET
    assert (
        manifest["provenance"]["final_ci_upper_bound_threshold"]
        == EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD
    )
    assert (
        manifest["provenance"]["min_final_relative_improvement"]
        == EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT
    )

    selection_sources = set()
    for record in manifest["unseen_test"]["candidate_records"]:
        selection_sources.add(record["selection_source"])
        assert set(("ka", "kb", "radp", "shell_th")).issubset(record)
        request_params = record["request_payload"]["parameters"]
        assert set(("ka", "kb", "radp", "shell_th", "b1", "b2", "a3", "a4", "bpress")).issubset(
            request_params
        )
        assert request_params["radp"] == record["radp"]
        assert request_params["shell_th"] == record["shell_th"]
        assert request_params["b1"] == EMB_34UM_DNN_CAUSAL_B1_VALUE
        assert request_params["b2"] == EMB_34UM_DNN_CAUSAL_B2_VALUE
        assert request_params["a3"] == EMB_34UM_DNN_CAUSAL_A3_VALUE
        assert request_params["a4"] == EMB_34UM_DNN_CAUSAL_A4_VALUE
        assert request_params["bpress"] == EMB_34UM_DNN_CAUSAL_BPRESS_VALUE
    for record in manifest["pilot"]["candidate_records"]:
        selection_sources.add(record["selection_source"])
        assert set(("ka", "kb", "radp", "shell_th")).issubset(record)
        assert record["selection_source"] == "fresh_dpd_pilot"
        assert record["selection_mode"] == "dnn_causal_fresh_only"
        assert record["fresh_only"] is True
        assert record["selection_mode_metadata"]["stage"] == "pilot"
        assert record["selection_mode_metadata"]["method_label"] == "fresh_dpd_pilot"
        assert record["selection_mode_metadata"]["selection_source"] == "fresh_dpd_pilot"
        request_params = record["request_payload"]["parameters"]
        assert request_params["radp"] == record["radp"]
        assert request_params["shell_th"] == record["shell_th"]
        assert request_params["b1"] == EMB_34UM_DNN_CAUSAL_B1_VALUE
        assert request_params["b2"] == EMB_34UM_DNN_CAUSAL_B2_VALUE
        assert request_params["a3"] == EMB_34UM_DNN_CAUSAL_A3_VALUE
        assert request_params["a4"] == EMB_34UM_DNN_CAUSAL_A4_VALUE
        assert request_params["bpress"] == EMB_34UM_DNN_CAUSAL_BPRESS_VALUE
    for seed_payload in manifest["seeds"]:
        assert len(seed_payload["shared_initial"]["candidate_records"]) == EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE
        assert len(seed_payload["al_steps"]) == EMB_34UM_DNN_CAUSAL_MIN_CYCLES
        assert len(seed_payload["lhs_steps"]) == EMB_34UM_DNN_CAUSAL_MIN_CYCLES
        for record in seed_payload["shared_initial"]["candidate_records"]:
            assert set(("ka", "kb", "radp", "shell_th")).issubset(record)
            shared_request_params = record["request_payload"]["parameters"]
            assert shared_request_params["radp"] == record["radp"]
            assert shared_request_params["shell_th"] == record["shell_th"]
            assert shared_request_params["bpress"] == EMB_34UM_DNN_CAUSAL_BPRESS_VALUE
        for step_payload in seed_payload["al_steps"]:
            assert step_payload["candidate_count"] == EMB_34UM_DNN_CAUSAL_STEP_SIZE
            assert not step_payload["candidate_records"]
            al_payload = step_payload["selection_payload"]
            assert al_payload["candidate_pool_size"] == EMB_34UM_DNN_D4_CANDIDATE_POOL_DEFAULT_REQUESTED_SIZE
            assert al_payload["candidate_pool_generation"]["generator_type"] in {"sobol", "stratified", "stratified_fallback"}
            assert al_payload["candidate_pool_generation"]["candidate_pool_generator"] == "auto"
            candidate_pool_generation = al_payload["candidate_pool_generation"]
            assert candidate_pool_generation["requested_size"] == candidate_pool_generation["actual_size"]
            assert candidate_pool_generation["seed"] == al_payload["selection_seed"]
            assert candidate_pool_generation["dimension_names"] == ["ka", "kb", "radp", "shell_th"]
            candidate_pool_bounds = candidate_pool_generation["bounds"]
            assert set(candidate_pool_bounds) == {"ka", "kb", "radp", "shell_th"}
            for key in candidate_pool_bounds:
                assert candidate_pool_bounds[key] == list(EMB_34UM_DNN_CAUSAL_BOUNDS[key])
        for step_payload in seed_payload["lhs_steps"]:
            assert len(step_payload["candidate_records"]) == EMB_34UM_DNN_CAUSAL_STEP_SIZE
            selection_sources.update(record["selection_source"] for record in step_payload["candidate_records"])
            for record in step_payload["candidate_records"]:
                assert set(("ka", "kb", "radp", "shell_th")).issubset(record)
                step_request_params = record["request_payload"]["parameters"]
                assert step_request_params["radp"] == record["radp"]
                assert step_request_params["shell_th"] == record["shell_th"]
                assert step_request_params["bpress"] == EMB_34UM_DNN_CAUSAL_BPRESS_VALUE

    assert selection_sources == {
        EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
        EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
        "fresh_dpd_pilot",
    }
    assert "samples_all" not in json.dumps(manifest)


def test_dnn_causal_manifest_supports_escalation_to_five_active_replicates() -> None:
    manifest = build_emb_34um_dnn_causal_validation_campaign_manifest(
        timestamp="20260521_120000",
        run_id_prefix="emb-34um-dnn-causal-test",
        campaign_root=Path("/tmp") / "emb-34um-dnn-causal-test",
        scratch_root=Path("/tmp") / "emb-34um-dnn-causal-test",
        vault_root_timestamp=Path("/tmp") / "vault" / "emb-34um-dnn-causal-test",
        force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
        active_replicate_count=EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
    )

    assert manifest["policy"]["active_replicate_count"] == EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT
    assert manifest["policy"]["replicates"] == [1, 2, 3, 4, 5]
    assert len(manifest["seeds"]) == EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT
    assert manifest["command_inventory"]["count"] == 57
    assert len(manifest["command_inventory"]["entries"]) == 57


def test_dnn_causal_unseen_test_does_not_overlap_training_design_points() -> None:
    manifest = build_emb_34um_dnn_causal_validation_campaign_manifest(
        timestamp="20260521_120000",
        run_id_prefix="emb-34um-dnn-causal-test",
        campaign_root=Path("/tmp") / "emb-34um-dnn-causal-test",
        scratch_root=Path("/tmp") / "emb-34um-dnn-causal-test",
        vault_root_timestamp=Path("/tmp") / "vault" / "emb-34um-dnn-causal-test",
        force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    )

    unseen = {
        (round(record["ka"], 12), round(record["kb"], 12), round(record["radp"], 12), round(record["shell_th"], 12))
        for record in manifest["unseen_test"]["candidate_records"]
    }
    training_points = set()
    for seed_payload in manifest["seeds"]:
        training_points.update(
            (
                round(record["ka"], 12),
                round(record["kb"], 12),
                round(record["radp"], 12),
                round(record["shell_th"], 12),
            )
            for record in seed_payload["shared_initial"]["candidate_records"]
        )
        for step_payload in seed_payload["lhs_steps"]:
            training_points.update(
                (
                    round(record["ka"], 12),
                    round(record["kb"], 12),
                    round(record["radp"], 12),
                    round(record["shell_th"], 12),
                )
                for record in step_payload["candidate_records"]
            )

    assert unseen
    assert training_points
    assert unseen.isdisjoint(training_points)


def test_dnn_causal_manifest_avoids_low_ka_low_kb_timeout_corner() -> None:
    assert is_dnn_causal_low_corner_excluded(189.530, 453.441)
    assert is_dnn_causal_low_corner_excluded(136.956, 736.554)
    assert is_dnn_causal_low_corner_excluded(3162.2776601683795, 3556.558820077846)
    assert is_dnn_causal_low_corner_excluded(
        1272.145949837055,
        9011.372292672091,
        6.489854540732141,
        3.9170402720459855e-9,
    )
    assert is_dnn_causal_low_corner_excluded(
        22432.81994559314,
        2436.648543412221,
        6.584751551356739,
        3.844815260086293e-9,
    )
    assert not is_dnn_causal_low_corner_excluded(
        1358.7312744799053,
        9561.356860131997,
        6.506801087377002,
        3.95994097350211e-9,
    )
    assert not is_dnn_causal_low_corner_excluded(
        18904.735,
        2095.531,
        6.571402,
        3.82656498446e-9,
    )
    assert not is_dnn_causal_low_corner_excluded(
        26619.331,
        2833.294,
        6.598102,
        3.86306553571e-9,
    )
    assert not is_dnn_causal_low_corner_excluded(5000.0, 3556.558820077846)
    assert not is_dnn_causal_low_corner_excluded(3162.2776601683795, 5000.0)

    manifest = build_emb_34um_dnn_causal_validation_campaign_manifest(
        timestamp="20260521_120000",
        run_id_prefix="emb-34um-dnn-causal-test",
        campaign_root=Path("/tmp") / "emb-34um-dnn-causal-test",
        scratch_root=Path("/tmp") / "emb-34um-dnn-causal-test",
        vault_root_timestamp=Path("/tmp") / "vault" / "emb-34um-dnn-causal-test",
        force_grid=EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    )

    assert manifest["policy"]["exclusion_policy"] == EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION

    records = list(manifest["unseen_test"]["candidate_records"])
    for seed_payload in manifest["seeds"]:
        records.extend(seed_payload["shared_initial"]["candidate_records"])
        for step_payload in seed_payload["lhs_steps"]:
            records.extend(step_payload["candidate_records"])

    assert records
    assert not any(
        is_dnn_causal_low_corner_excluded(
            record["ka"],
            record["kb"],
            record["radp"],
            record["shell_th"],
        )
        for record in records
    )
    assert all("source_index" in record for record in records)


def test_prepare_script_writes_manifest_and_inventory_without_submission(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    timestamp = "20260521_130000"
    rendered_batch_ids: list[str] = []

    def fake_render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        rendered_batch_ids.append(batch_id)
        for candidate in candidates:
            parameters = candidate.parameters
            fingerprint = parameters.get("runtime_fingerprint", {})
            assert {"ka", "kb", "radp", "shell_th", "bpress"}.issubset(parameters)
            assert fingerprint["radp"] == parameters["radp"]
            assert fingerprint["shell_th"] == parameters["shell_th"]
            assert fingerprint["bpress"] == parameters["bpress"]
        return {
            "batch_root": str(batch_root),
            "batch_id": batch_id,
            "run_id": run_id,
            "candidate_count": len(candidates),
            "submission_expected": {
                "render_only": True,
                "submission_commands_empty": True,
                "submitted": False,
            },
            "rendered_candidate_manifests": [],
        }

    monkeypatch.setattr(module, "_render_batch", fake_render_batch)

    result = module.prepare_emb_34um_dnn_causal_validation(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=None,
        walltime="00:12:00",
        active_replicate_count=EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
        max_replicate_count=EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
        concurrent_jobs=30,
        retry_limit=0,
        run_id_prefix="emb-34um-dnn-causal-test",
        cycle_count=EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
        include_plot_requirements=False,
    )

    manifest_path = Path(result["manifest_path"])
    command_inventory_path = Path(result["command_inventory_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command_inventory = json.loads(command_inventory_path.read_text(encoding="utf-8"))

    assert manifest_path.is_file()
    assert command_inventory_path.is_file()
    assert manifest["force_grid_source"] == "protocol_default"
    assert command_inventory["count"] == 35
    assert manifest["command_inventory"]["count"] == 35
    assert len(manifest["runtime_batches"]) == 35
    assert all(batch["submission_expected"]["render_only"] is True for batch in manifest["runtime_batches"])
    assert all(batch["submission_expected"]["submitted"] is False for batch in manifest["runtime_batches"])
    assert "pilot" in rendered_batch_ids
    assert manifest["validation_plot"]["required"] is False
    assert manifest["command_inventory"]["entries"][0]["execution_mode"] == "render-only"
    assert manifest["command_inventory"]["entries"][0]["array"] == "0-199%30"
    assert "samples_all" not in json.dumps(manifest["command_inventory"])


def test_prepare_script_writes_multiplot_coverage_requirements_and_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    timestamp = "20260521_140000"

    def fake_render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        for candidate in candidates:
            parameters = candidate.parameters
            fingerprint = parameters.get("runtime_fingerprint", {})
            assert {"ka", "kb", "radp", "shell_th", "bpress"}.issubset(parameters)
            assert fingerprint["radp"] == parameters["radp"]
            assert fingerprint["shell_th"] == parameters["shell_th"]
            assert fingerprint["bpress"] == parameters["bpress"]
        return {
            "batch_root": str(batch_root),
            "batch_id": batch_id,
            "run_id": run_id,
            "candidate_count": len(candidates),
            "submission_expected": {
                "render_only": True,
                "submission_commands_empty": True,
                "submitted": False,
            },
            "rendered_candidate_manifests": [],
        }

    monkeypatch.setattr(module, "_render_batch", fake_render_batch)

    result = module.prepare_emb_34um_dnn_causal_validation(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=None,
        walltime="00:12:00",
        active_replicate_count=EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
        max_replicate_count=EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
        concurrent_jobs=30,
        retry_limit=0,
        run_id_prefix="emb-34um-dnn-causal-test",
        cycle_count=EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
        include_plot_requirements=True,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    sidecar_path = Path(manifest["validation_plot"]["coverage_plot_sidecar"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    expected_modes = [
        "log10(ka)-vs-log10(kb)",
        "log10(ka)-vs-log10(radp)",
        "log10(kb)-vs-log10(shell_th)",
        "log10(radp)-vs-log10(shell_th)",
    ]
    expected_point_counts = {
        "pilot": len(manifest["pilot"]["candidate_records"]),
        "unseen_test": len(manifest["unseen_test"]["candidate_records"]),
        "shared_initial": sum(
            len(seed_payload["shared_initial"]["candidate_records"]) for seed_payload in manifest["seeds"]
        ),
        "lhs": sum(
            len(step_payload["candidate_records"])
            for seed_payload in manifest["seeds"]
            for step_payload in seed_payload["lhs_steps"]
        ),
        "al_placeholders": sum(
            int(step_payload.get("candidate_count", 0))
            for seed_payload in manifest["seeds"]
            for step_payload in seed_payload["al_steps"]
        ),
    }

    assert manifest["validation_plot"]["required"] is True
    assert manifest["validation_plot"]["projection_count"] == 4
    assert manifest["validation_plot"]["projection_modes"] == expected_modes
    assert manifest["validation_plot"]["projection_point_counts"] == expected_point_counts
    assert sidecar["projection_count"] == 4
    assert [entry["mode"] for entry in sidecar["projection_modes"]] == expected_modes
    assert [entry["point_counts"] for entry in sidecar["projection_modes"]] == [expected_point_counts] * 4
    assert sidecar["point_counts"] == expected_point_counts
    assert sidecar["projection_modes"][1]["x_label"] == "log10(ka)"
    assert sidecar["projection_modes"][1]["y_label"] == "log10(radp)"
    assert sidecar["projection_modes"][2]["x_label"] == "log10(kb)"
    assert sidecar["projection_modes"][2]["y_label"] == "log10(shell_th)"
    assert sidecar["projection_modes"][3]["x_label"] == "log10(radp)"
    assert sidecar["projection_modes"][3]["y_label"] == "log10(shell_th)"
    assert Path(manifest["validation_plot"]["coverage_plot"]).is_file()
    assert sidecar_path.is_file()


def test_prepare_script_reads_rendered_request_schema_version(tmp_path: Path) -> None:
    module = _load_script_module()
    manifest_path = tmp_path / "dpd_sampling_candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rendered_payload": {
                    "request_payload": {
                        "schema_version": module.EMB_34UM_DPD_SCHEMA_VERSION,
                    },
                },
                "normalized_payload": {"schema_version": "legacy-fallback"},
            }
        ),
        encoding="utf-8",
    )

    assert module._ensure_request_payload_schema_version(manifest_path) == module.EMB_34UM_DPD_SCHEMA_VERSION


def test_dnn_causal_validation_sbatch_supports_execute_and_runner_invocation() -> None:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "platforms"
        / "karolina"
        / "sbatch"
        / "emb_34um_dnn_causal_validation_array.sbatch"
    )
    text = script_path.read_text(encoding="utf-8")

    assert 'source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"' in text
    assert "mesouq_activate_site_env karolina \"${REPO_ROOT}\"" in text
    assert "MESOUQ_SITE_RUNTIME_ROOT must be set before using this Karolina sbatch script." in text
    assert 'EXECUTION_MODE="${EXECUTION_MODE:-execute}"' in text
    assert 'RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/workflows/emb/active_learning/run_emb_34um_final_gate_candidate.py}"' in text
    assert '"${EXECUTION_MODE}" == "render-only"' in text
    assert "BATCH_SUMMARY=\"${BATCH_DIR}/emb_34um_batch_summary.json\"" in text
    assert "select_candidate_manifest" in text
    assert "\"${PYTHON_EXECUTABLE}\" \"${RUNNER_SCRIPT}\" \\" in text
    assert "--candidate-manifest \"${CANDIDATE_MANIFEST}\"" in text
    assert 'OMPI_MCA_btl="${MESOUQ_KAROLINA_OMPI_MCA_BTL:-self,vader,tcp}"' in text
    assert '"OMPI_MCA_btl=${OMPI_MCA_btl}"' in text
    assert "srun --ntasks=2 --kill-on-bad-exit=1" in text
    assert "--mark-failed" in text
