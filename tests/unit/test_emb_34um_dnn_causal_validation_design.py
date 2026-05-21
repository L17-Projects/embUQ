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

from meso_uq.active_learning.emb_34um_dnn_causal_validation_design import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    build_emb_34um_dnn_causal_validation_campaign_manifest,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
    EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
    EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
    EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_REPLICATES,
    EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
    EMB_34UM_DNN_CAUSAL_STEP_SIZE,
    EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
    EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
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

    assert manifest["policy"]["replicate_count"] == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["policy"]["replicates"] == list(EMB_34UM_DNN_CAUSAL_REPLICATES)
    assert manifest["policy"]["min_cycles"] == EMB_34UM_DNN_CAUSAL_MIN_CYCLES
    assert manifest["policy"]["max_cycles"] == EMB_34UM_DNN_CAUSAL_MAX_CYCLES
    assert manifest["policy"]["shared_initial_size"] == EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE
    assert manifest["policy"]["unseen_test_size"] == EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE
    assert manifest["policy"]["step_size"] == EMB_34UM_DNN_CAUSAL_STEP_SIZE
    assert manifest["policy"]["force_grid"] == list(EMB_34UM_DNN_CAUSAL_FORCE_GRID)
    assert len(manifest["policy"]["force_grid"]) == 8
    assert manifest["policy"]["ensemble_size"] == EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    assert manifest["policy"]["lhs_source"] == EMB_34UM_DNN_CAUSAL_LHS_SOURCE
    assert manifest["policy"]["test_set_source"] == EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE

    assert len(manifest["unseen_test"]["candidate_records"]) == EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE
    assert len(manifest["seeds"]) == EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT
    assert manifest["command_inventory"]["count"] == 56
    assert manifest["command_inventory"]["entries"][0]["execution_mode"] == "render-only"
    assert manifest["command_inventory"]["entries"][0]["array"] == "0-99%30"
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
    assert manifest["provenance"]["force_grid_signature"] == manifest["policy"]["force_grid_signature"]

    selection_sources = set()
    for record in manifest["unseen_test"]["candidate_records"]:
        selection_sources.add(record["selection_source"])
    for seed_payload in manifest["seeds"]:
        assert len(seed_payload["shared_initial"]["candidate_records"]) == EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE
        assert len(seed_payload["al_steps"]) == EMB_34UM_DNN_CAUSAL_MIN_CYCLES
        assert len(seed_payload["lhs_steps"]) == EMB_34UM_DNN_CAUSAL_MIN_CYCLES
        for step_payload in seed_payload["al_steps"]:
            assert step_payload["candidate_count"] == EMB_34UM_DNN_CAUSAL_STEP_SIZE
            assert not step_payload["candidate_records"]
        for step_payload in seed_payload["lhs_steps"]:
            assert len(step_payload["candidate_records"]) == EMB_34UM_DNN_CAUSAL_STEP_SIZE
            selection_sources.update(record["selection_source"] for record in step_payload["candidate_records"])

    assert selection_sources == {EMB_34UM_DNN_CAUSAL_LHS_SOURCE, EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE}
    assert "samples_all" not in json.dumps(manifest)


def test_prepare_script_writes_manifest_and_inventory_without_submission(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    timestamp = "20260521_130000"

    def fake_render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
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
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
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
    assert command_inventory["count"] == 56
    assert manifest["command_inventory"]["count"] == 56
    assert len(manifest["runtime_batches"]) == 56
    assert all(batch["submission_expected"]["render_only"] is True for batch in manifest["runtime_batches"])
    assert all(batch["submission_expected"]["submitted"] is False for batch in manifest["runtime_batches"])
    assert manifest["validation_plot"]["required"] is False
    assert manifest["command_inventory"]["entries"][0]["execution_mode"] == "render-only"
    assert manifest["command_inventory"]["entries"][0]["array"] == "0-99%30"
    assert "samples_all" not in json.dumps(manifest["command_inventory"])


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

    assert "source scripts/platforms/hpc/site_env.sh" in text
    assert "mesouq_activate_site_env karolina \"${REPO_ROOT}\"" in text
    assert "MESOUQ_SITE_RUNTIME_ROOT must be set before using this Karolina sbatch script." in text
    assert 'EXECUTION_MODE="${EXECUTION_MODE:-execute}"' in text
    assert 'RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/workflows/emb/active_learning/run_emb_34um_final_gate_candidate.py}"' in text
    assert '"${EXECUTION_MODE}" == "render-only"' in text
    assert "BATCH_SUMMARY=\"${BATCH_DIR}/emb_34um_batch_summary.json\"" in text
    assert "select_candidate_manifest" in text
    assert "\"${PYTHON_EXECUTABLE}\" \"${RUNNER_SCRIPT}\" \\" in text
    assert "--candidate-manifest \"${CANDIDATE_MANIFEST}\"" in text
    assert "srun --ntasks=2" in text
    assert "--mark-failed" in text
