from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _prepare_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    script_path = repo_root / "scripts" / "workflows" / "emb" / "active_learning" / "prepare_emb_34um_final_gate.py"
    spec = importlib.util.spec_from_file_location("prepare_emb_34um_final_gate", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_custom_force_grid(path: Path) -> None:
    path.write_text("0 0 0 0 0 0 0 0 1 2 3 10 20 30\n", encoding="utf-8")


def test_prepare_emb_34um_final_gate_manifests_full_and_canary_campaigns(tmp_path: Path) -> None:
    module = _prepare_module()
    timestamp = "20260501_120000"
    force_grid_path = tmp_path / "samples_all_custom.dat"
    _write_custom_force_grid(force_grid_path)
    force_grid = module._load_force_grid(force_grid_path)
    expected_canary_forces = list(module._canonical_canary_force_points(force_grid, 3))
    result = module.prepare_emb_34um_final_gate(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid_path,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-final-gate-test",
        canary_force_count=3,
        skip_vault_copy=False,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    campaign_root = tmp_path / "scratch" / timestamp
    vault_target = tmp_path / "vault" / timestamp

    assert manifest["campaign_root"] == str(campaign_root)
    assert manifest["vault_root_timestamp"] == str(vault_target)
    assert manifest["scratch_root"] == str(tmp_path / "scratch")
    assert manifest["vault_root"] == str(tmp_path / "vault")
    assert manifest["platform"] == "karolina"
    assert manifest["schema_version"] == module.FULL_GATE_SCHEMA_VERSION

    resources = manifest["expected_resources"]
    assert resources["platform"] == "karolina"
    assert resources["concurrent_jobs"] == 30
    assert resources["retry_limit"] == 3
    assert resources["gpu_count"] == 1

    full_gate = manifest["full_gate"]
    canary_gate = manifest["canary_gate"]
    assert full_gate["candidate_count"] == 90
    assert canary_gate["candidate_count"] == 1
    assert manifest["canary"]["force_point_count"] == 3

    assert full_gate["scheduler_boundary"]["submission"]["submitted"] is False
    assert canary_gate["scheduler_boundary"]["submission"]["submitted"] is False
    assert full_gate["scheduler_boundary"]["submission"]["submission_commands"] == []
    assert canary_gate["scheduler_boundary"]["submission"]["submission_commands"] == []
    assert full_gate["submission_expected"]["submitted"] is False
    assert canary_gate["submission_expected"]["submitted"] is False
    assert full_gate["submission_expected"]["submission_commands"] == []
    assert canary_gate["submission_expected"]["submission_commands"] == []
    assert manifest["acceptance"]["pass"] is True

    assert full_gate["candidate_lineage"][0]["candidate_hash"] is not None
    assert len(full_gate["candidate_lineage"]) == 90
    assert len(canary_gate["candidate_lineage"]) == 1
    assert full_gate["lineage_fingerprint"] != canary_gate["lineage_fingerprint"]

    assert manifest["commands"]["full"]["mode"] == "full"
    assert manifest["commands"]["canary"]["mode"] == "canary"
    assert manifest["commands"]["full"]["array"] == "0-89%30"
    assert manifest["commands"]["canary"]["array"] == "0-0"
    assert manifest["commands"]["full"]["execution_mode"] == "execute"
    assert "--array=0-89%30" in manifest["commands"]["full"]["command"]
    assert "--array=0-0" in manifest["commands"]["canary"]["command"]
    assert "MODE=full" in manifest["commands"]["full"]["command"]
    assert "MODE=canary" in manifest["commands"]["canary"]["command"]
    assert "EXECUTION_MODE=execute" in manifest["commands"]["full"]["command"]
    assert manifest["commands"]["full"]["script"].endswith("emb_34um_active_learning_array.sbatch")
    assert manifest["commands"]["canary"]["script"].endswith("emb_34um_active_learning_array.sbatch")

    assert manifest["acceptance"]["criteria"]["full_count_is_90"] is True
    assert manifest["acceptance"]["criteria"]["canary_count_is_1"] is True
    assert manifest["acceptance"]["criteria"]["canary_points_is_3"] is True
    assert manifest["full_gate"]["request_payload_schema_versions"] == [module.EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION] * 90
    assert manifest["canary_gate"]["request_payload_schema_versions"] == [module.EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION]

    assert campaign_root.is_dir()
    assert vault_target.is_dir()
    assert (campaign_root / "full_gate" / "emb_34um_batch_summary.json").is_file()
    assert (campaign_root / "canary" / "emb_34um_batch_summary.json").is_file()

    full_manifest_path = Path(full_gate["manifest_path"])
    canary_manifest_path = Path(canary_gate["manifest_path"])
    assert full_manifest_path.exists()
    assert canary_manifest_path.exists()
    assert full_manifest_path.name == "dpd_sampling_gate_manifest.json"
    assert canary_manifest_path.name == "dpd_sampling_gate_manifest.json"
    assert Path(full_gate["report_path"]).name == "dpd_sampling_gate_report.json"
    assert Path(canary_gate["report_path"]).name == "dpd_sampling_gate_report.json"

    full_gate_payload = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    canary_gate_payload = json.loads(canary_manifest_path.read_text(encoding="utf-8"))
    assert full_gate_payload["candidate_count"] == 90
    assert canary_gate_payload["candidate_count"] == 1
    assert full_gate_payload["candidate_records"][0]["experiment"] == "indentation"
    assert canary_gate_payload["candidate_records"][0]["experiment"] == "indentation"
    assert manifest["canary"]["force_point_count"] == 3
    assert manifest["canary"]["force_points"] == expected_canary_forces

    assert len(full_gate_payload["rendered_manifest_paths"]) == 92
    assert len(canary_gate_payload["rendered_manifest_paths"]) == 3
    full_rendered_candidate_manifests = full_gate.get("rendered_candidate_manifests", [])
    canary_rendered_candidate_manifests = canary_gate.get("rendered_candidate_manifests", [])
    assert len(full_rendered_candidate_manifests) == 90
    assert len(canary_rendered_candidate_manifests) == 1
    for rendered_path in full_rendered_candidate_manifests:
        rendered = json.loads((Path(rendered_path)).read_text(encoding="utf-8"))
        request_payload = rendered["rendered_payload"]["request_payload"]
        assert request_payload["schema_version"] == module.EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION
        assert request_payload["force_grid"] == list(force_grid)
        assert rendered["family"] == "emb"
        assert rendered["rendered_payload"]["family"] == "emb"
        assert rendered["normalized_payload"]["experiment"] == "indentation"
        normalized_parameters = rendered["normalized_payload"]["parameters"]
        assert "ka" in normalized_parameters
        assert "kb" in normalized_parameters
        assert "Yt" not in normalized_parameters
    for rendered_path in canary_rendered_candidate_manifests:
        rendered = json.loads(Path(rendered_path).read_text(encoding="utf-8"))
        request_payload = rendered["rendered_payload"]["request_payload"]
        assert request_payload["schema_version"] == module.EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION
        assert request_payload["force_grid"] == expected_canary_forces
        assert rendered["family"] == "emb"
        assert rendered["rendered_payload"]["family"] == "emb"
        assert rendered["normalized_payload"]["experiment"] == "indentation"
        assert rendered["normalized_payload"]["canary"]["enabled"] is True


def test_prepare_emb_34um_final_gate_rejects_non_default_canary_count(tmp_path: Path) -> None:
    module = _prepare_module()
    repo_root = Path(__file__).resolve().parents[2]

    try:
        module.prepare_emb_34um_final_gate(
            timestamp="20260501_130000",
            scratch_root=tmp_path / "scratch",
            vault_root=tmp_path / "vault",
            force_grid_path=repo_root / module.DEFAULT_FORCE_GRID_DATA,
            walltime="00:30:00",
            concurrent_jobs=30,
            retry_limit=3,
            run_id_prefix="emb-34um-final-gate-test",
            canary_force_count=5,
            skip_vault_copy=True,
        )
    except ValueError as exc:
        assert "fixed at 3" in str(exc)
    else:
        raise AssertionError("Expected non-default canary force count to fail.")
