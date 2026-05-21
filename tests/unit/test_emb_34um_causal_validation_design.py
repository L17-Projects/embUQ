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

from meso_uq.active_learning.emb_34um_causal_validation_design import (
    EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
    EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS,
    EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS,
    EMB_34UM_CAUSAL_VALIDATION_REQUIRED_CURVES_PER_SEED,
    EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION,
    EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
    EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE,
    EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE,
    build_emb_34um_causal_validation_campaign_manifest,
)  # noqa: E402


def _load_script_module() -> Any:
    script_path = REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "prepare_emb_34um_causal_validation.py"
    spec = importlib.util.spec_from_file_location("prepare_emb_34um_causal_validation", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_script_bootstraps_repo_source_before_package_imports() -> None:
    script_path = REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "prepare_emb_34um_causal_validation.py"
    source = script_path.read_text(encoding="utf-8")

    bootstrap_index = source.index("sys.path.insert(0, str(_SRC_ROOT))")
    package_import_index = source.index("from meso_uq.active_learning")

    assert bootstrap_index < package_import_index


def _simple_force_grid() -> tuple[float, ...]:
    return (1.0, 2.0, 5.0, 10.0)


def _collect_candidate_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed_payload in manifest["seeds"]:
        records.extend(seed_payload["shared_initial"]["candidate_records"])
        records.extend(seed_payload["validation"]["candidate_records"])
        for batch in seed_payload["al_steps"]:
            records.extend(batch["candidate_records"])
        for batch in seed_payload["lhs_steps"]:
            records.extend(batch["candidate_records"])
    return records


def test_causal_manifest_enumerates_required_counts_default() -> None:
    manifest = build_emb_34um_causal_validation_campaign_manifest(
        timestamp="20260520_120000",
        run_id_prefix="emb-34um-causal-test",
        campaign_root=Path("/tmp") / "emb-34um-causal-test",
        scratch_root=Path("/tmp") / "emb-34um-causal-test",
        vault_root_timestamp=Path("/tmp") / "vault" / "emb-34um-causal-test",
        force_grid=_simple_force_grid(),
    )

    assert manifest["schema_version"] == EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION
    assert manifest["policy"]["required_seed_count"] == len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS)
    assert manifest["policy"]["max_seed_count"] == EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS
    assert manifest["policy"]["step_count"] == EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS
    assert manifest["policy"]["step_size"] == EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE
    assert manifest["policy"]["shared_size"] == EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE
    assert manifest["policy"]["validation_size"] == EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE
    assert "static_preplanned" in manifest
    assert "adaptive_al_placeholders" in manifest

    assert manifest["statistical_gate"]["required_curves_per_seed"] == EMB_34UM_CAUSAL_VALIDATION_REQUIRED_CURVES_PER_SEED
    assert manifest["statistical_gate"]["required_primary_seed_curves"] == 3 * EMB_34UM_CAUSAL_VALIDATION_REQUIRED_CURVES_PER_SEED

    records = _collect_candidate_records(manifest)
    static_records_per_seed = (
        EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE
        + EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE
        + EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS * EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE
    )
    assert len(records) == 3 * static_records_per_seed

    for seed_payload in manifest["seeds"]:
        assert seed_payload["seed"] in EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS
        assert seed_payload["required_seed"] is True
        assert len(seed_payload["shared_initial"]["candidate_records"]) == EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE
        assert len(seed_payload["validation"]["candidate_records"]) == EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE
        assert len(seed_payload["al_steps"]) == EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS
        assert len(seed_payload["lhs_steps"]) == EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS
        assert all(len(batch["candidate_records"]) == 0 for batch in seed_payload["al_steps"])
        assert all(batch["candidate_count"] == EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE for batch in seed_payload["al_steps"])
        assert all(batch["select_render_action"].endswith("-select-render") for batch in seed_payload["al_steps"])
        assert all(batch["expected_batch_summary_path"].endswith("emb_34um_batch_summary.json") for batch in seed_payload["al_steps"])
        assert all(len(batch["candidate_records"]) == EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE for batch in seed_payload["lhs_steps"])
        assert (
            len(seed_payload["shared_initial"]["candidate_records"])
            + len(seed_payload["validation"]["candidate_records"])
            + sum(len(batch["candidate_records"]) for batch in seed_payload["al_steps"])
            + sum(len(batch["candidate_records"]) for batch in seed_payload["lhs_steps"])
            == static_records_per_seed
        )

    placeholder_roots = len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS) * EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS
    assert len(manifest["expected_output_roots"]["scratch"]) == (3 * static_records_per_seed) + placeholder_roots
    assert len(manifest["expected_output_roots"]["vault"]) == (3 * static_records_per_seed) + placeholder_roots
    assert len(set(manifest["expected_output_roots"]["scratch"])) == len(manifest["expected_output_roots"]["scratch"])
    assert len(set(manifest["expected_output_roots"]["vault"])) == len(manifest["expected_output_roots"]["vault"])

    families = {item["family"] for item in records}
    experiments = {item["experiment"] for item in records}
    assert families == {"emb"}
    assert experiments == {"indentation"}

    assert manifest["command_inventory"]["count"] == 36
    command_modes = {command["mode"] for command in manifest["command_inventory"]["entries"]}
    assert "shared_initial" in command_modes
    assert "validation" in command_modes
    assert "al-step-01" in command_modes
    assert "lhs-step-05" in command_modes
    for command in manifest["command_inventory"]["entries"]:
        assert command["execution_mode"] == "render-only"
        assert command["candidate_count"] == 100
        assert command["array"].endswith(f"%{manifest['command_inventory']['concurrent_jobs']}")
        assert "BATCH_DIR_OVERRIDE=" in command["command"]

    assert all(
        step["candidate_records"] == []
        for step in manifest["adaptive_al_placeholders"]["al_steps"]
    )


def test_causal_manifest_preserves_serial_array_throttle(tmp_path: Path) -> None:
    manifest = build_emb_34um_causal_validation_campaign_manifest(
        timestamp="20260520_120000",
        run_id_prefix="emb-34um-causal-test",
        campaign_root=tmp_path / "campaign",
        scratch_root=tmp_path,
        vault_root_timestamp=tmp_path / "vault",
        force_grid=_simple_force_grid(),
        concurrent_jobs=1,
    )

    for command in manifest["command_inventory"]["entries"]:
        assert command["array"].endswith("%1")
        assert "--array=0-99%1" in command["command"]


def test_optional_five_seed_manifest_has_expected_total_curves() -> None:
    manifest = build_emb_34um_causal_validation_campaign_manifest(
        timestamp="20260520_120001",
        run_id_prefix="emb-34um-causal-test",
        campaign_root=Path("/tmp") / "emb-34um-causal-test-5",
        scratch_root=Path("/tmp") / "emb-34um-causal-test-5",
        vault_root_timestamp=Path("/tmp") / "vault" / "emb-34um-causal-test-5",
        force_grid=_simple_force_grid(),
        seed_count=5,
    )

    assert len(manifest["seeds"]) == 5
    static_records_per_seed = (
        EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE
        + EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE
        + EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS * EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE
    )
    assert len(_collect_candidate_records(manifest)) == 5 * static_records_per_seed


def test_causal_manifest_prevents_family_or_experiment_mix() -> None:
    manifest = build_emb_34um_causal_validation_campaign_manifest(
        timestamp="20260520_120002",
        run_id_prefix="emb-34um-causal-test",
        campaign_root=Path("/tmp") / "emb-34um-causal-test-mix",
        scratch_root=Path("/tmp") / "emb-34um-causal-test-mix",
        vault_root_timestamp=Path("/tmp") / "vault" / "emb-34um-causal-test-mix",
        force_grid=_simple_force_grid(),
    )

    records = _collect_candidate_records(manifest)
    assert {item["family"] for item in records} == {"emb"}
    assert {item["experiment"] for item in records} == {"indentation"}
    assert len(manifest["command_inventory"]["entries"]) == 36


def test_causal_manifest_does_not_precompute_adaptive_al_candidate_records() -> None:
    manifest = build_emb_34um_causal_validation_campaign_manifest(
        timestamp="20260520_120003",
        run_id_prefix="emb-34um-causal-test",
        campaign_root=Path("/tmp") / "emb-34um-causal-test-adaptive",
        scratch_root=Path("/tmp") / "emb-34um-causal-test-adaptive",
        vault_root_timestamp=Path("/tmp") / "vault" / "emb-34um-causal-test-adaptive",
        force_grid=_simple_force_grid(),
    )
    for seed_payload in manifest["seeds"]:
        for al_step in seed_payload["al_steps"]:
            assert al_step["candidate_count"] == EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE
            assert al_step["candidate_records"] == []


def test_prepare_script_renders_static_stage_batch_summaries(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    timestamp = "20260520_130000"
    force_grid = tmp_path / "samples_all_custom.dat"
    force_grid.write_text("0 0 0 0 0 0 0 0 1 2 3 4 5 6\n", encoding="utf-8")
    scratch_root = tmp_path / "scratch"
    vault_root = tmp_path / "vault"
    campaign_root = scratch_root / timestamp
    vault_root_timestamp = vault_root / timestamp

    def _record(stage: str, candidate_id: str, *, step: int) -> dict[str, Any]:
        output_root = campaign_root / "seed-001" / stage / "emb" / candidate_id
        vault_output_root = vault_root_timestamp / output_root.relative_to(campaign_root)
        return {
            "candidate_id": candidate_id,
            "ka": 1000.0,
            "kb": 1200.0,
            "seed": 1,
            "step": step,
            "selection_seed": 101,
            "selection_mode": "causal_fresh_only",
            "method": stage,
            "force_grid_signature": "test-force-grid",
            "output_root": str(output_root),
            "vault_output_root": str(vault_output_root),
        }

    fake_manifest = {
        "schema_version": EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION,
        "policy": {"required_seed_count": len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS)},
        "command_inventory": {"entries": [], "count": 0, "execution_mode": "render-only"},
        "validation_plot": {"required": True},
        "seeds": [
            {
                "seed": 1,
                "shared_initial": {"candidate_records": [_record("shared_initial", "shared-001", step=0)]},
                "al_steps": [{"candidate_records": [], "candidate_count": EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE}],
                "lhs_steps": [{"candidate_records": [_record("lhs-step-01", "lhs-001", step=1)]}],
                "validation": {"candidate_records": [_record("validation", "validation-001", step=0)]},
            }
        ],
    }

    monkeypatch.setattr(
        module,
        "build_emb_34um_causal_validation_campaign_manifest",
        lambda **_: fake_manifest,
    )

    result = module.prepare_emb_34um_causal_validation(
        timestamp=timestamp,
        scratch_root=scratch_root,
        vault_root=vault_root,
        force_grid_path=force_grid,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-test",
        step_count=EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS,
        seed_count=len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS),
        include_plot_requirements=True,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    command_inventory_path = Path(result["command_inventory_path"])
    command_inventory = json.loads(command_inventory_path.read_text(encoding="utf-8"))
    assert manifest["seeds"][0]["seed"] == 1
    assert manifest["command_inventory"]["count"] == command_inventory["count"]
    assert command_inventory_path.is_file()
    assert manifest["validation_plot"]["required"] is True
    coverage_plot = Path(manifest["validation_plot"]["coverage_plot"])
    coverage_sidecar = Path(manifest["validation_plot"]["coverage_plot_sidecar"])
    assert coverage_plot.is_file()
    assert coverage_plot.stat().st_size > 0
    assert coverage_sidecar.is_file()
    coverage_payload = json.loads(coverage_sidecar.read_text(encoding="utf-8"))
    assert coverage_payload["status"] in {"rendered", "fallback_png"}
    assert coverage_payload["static_point_count"] >= 3
    assert manifest["policy"]["required_seed_count"] == len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS)

    static_stages = ("shared_initial", "validation", "lhs-step-01")
    for stage in static_stages:
        summary_path = campaign_root / "seed-001" / stage / "emb_34um_batch_summary.json"
        assert summary_path.is_file()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["rendered_candidate_manifests"]
        assert summary["expected_output_roots"]
        assert Path(summary["manifest_path"]).name == "dpd_sampling_gate_manifest.json"
        assert (summary_path.parent / "dpd_sampling_batch_manifest.json").is_file()
        assert summary["submission_expected"]["render_only"] is True
        assert summary["submission_expected"]["submission_commands_empty"] is True
        assert summary["submission_expected"]["submitted"] is False
        for manifest_path in summary["rendered_candidate_manifests"]:
            candidate_manifest = Path(manifest_path)
            assert candidate_manifest.is_file()
            assert candidate_manifest.name == "dpd_sampling_candidate_manifest.json"

    assert not (campaign_root / "seed-001" / "al-step-01" / "emb_34um_batch_summary.json").exists()


def test_sbatch_wrapper_supports_legacy_and_causal_modes() -> None:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "platforms"
        / "karolina"
        / "sbatch"
        / "emb_34um_active_learning_array.sbatch"
    )
    text = script_path.read_text(encoding="utf-8")

    assert "full|canary|lhs|benchmark|validation|shared_initial|al-step-[0-9][0-9]|lhs-step-[0-9][0-9]" in text
    assert "if [[ -n \"${BATCH_DIR_OVERRIDE}\" ]]; then" in text
    assert "BATCH_DIR=\"${BATCH_DIR_OVERRIDE}\"" in text
    assert "BATCH_SUMMARY=\"${BATCH_DIR}/emb_34um_batch_summary.json\"" in text
