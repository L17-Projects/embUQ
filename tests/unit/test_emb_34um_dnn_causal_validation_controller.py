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


def _load_controller_module() -> Any:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "run_emb_34um_dnn_causal_validation_controller.py"
    )
    spec = importlib.util.spec_from_file_location("run_emb_34um_dnn_causal_validation_controller", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_design_manifest(campaign_root: Path, module: Any) -> Path:
    entries = [
        {
            "mode": "unseen_test",
            "replica": 0,
            "cycle": 0,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "unseen_test" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 ...",
        },
        {
            "mode": "shared_initial",
            "replica": 1,
            "cycle": 0,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-001" / "shared_initial" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 ...",
        },
        {
            "mode": "al-step-01",
            "replica": 1,
            "cycle": 1,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-001" / "al-step-01" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 ...",
        },
        {
            "mode": "lhs-step-01",
            "replica": 1,
            "cycle": 1,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-001" / "lhs-step-01" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 ...",
        },
    ]
    design_manifest = {
        "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation.v1",
        "campaign_root": str(campaign_root),
        "command_inventory": {
            "count": len(entries),
            "walltime": "00:30:00",
            "concurrent_jobs": 30,
            "retry_limit": 3,
            "entries": entries,
        },
    }
    manifest_path = campaign_root / module.EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(design_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def test_controller_builds_render_only_manifest_and_stage_order(tmp_path: Path) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "campaign"
    manifest_path = _write_design_manifest(campaign_root, module)

    result = module.build_emb_34um_dnn_causal_validation_controller(
        campaign_root=campaign_root,
        design_manifest_path=manifest_path,
    )

    output_path = Path(result["manifest_path"])
    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.is_file()
    assert manifest["schema_version"] == module.CONTROLLER_SCHEMA_VERSION
    assert manifest["execution_mode"] == "render-only"
    assert manifest["render_only"] is True
    assert manifest["command_inventory"]["count"] == 4
    assert manifest["stage_order"] == [
        "unseen_test",
        "shared_initial",
        "al-step-01",
        "lhs-step-01",
    ]
    assert len(manifest["parallel_arrays"]) == 4
    assert all(entry["execution_mode"] == "render-only" for entry in manifest["parallel_arrays"])
    assert all(entry["array"] == "0-99%30" for entry in manifest["parallel_arrays"])
    assert manifest["submission"]["submitted"] is False
    assert manifest["submission"]["submission_commands"] == []


def test_controller_defaults_to_render_only_no_submission_commands(tmp_path: Path) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "campaign"
    manifest_path = _write_design_manifest(campaign_root, module)

    result = module.build_emb_34um_dnn_causal_validation_controller(
        campaign_root=campaign_root,
        design_manifest_path=manifest_path,
        execution_mode="render-only",
    )
    manifest = result["manifest"]

    assert manifest["render_only"] is True
    assert manifest["submission"]["submitted"] is False
    assert not manifest["submission"]["submission_commands"]
    assert manifest["design_command_inventory_path"].endswith(
        "emb_34um_dnn_causal_validation_command_inventory.json"
    )
