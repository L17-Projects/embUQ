from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "workflows" / "gv" / "prepare_eigenmodes_domain_rank_canaries.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mesouq_test_gv_eigenmodes_canary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_eigenmodes_domain_rank_canaries_karolina_defaults(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "canaries"

    rc = module.main(
        [
            "--campaign-id",
            "unit-canary",
            "--output-root",
            str(output_root),
            "--include-4gpu",
        ]
    )

    manifest_path = output_root / "gv_eigenmodes_domain_rank_canary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert manifest["site"] == "karolina"
    assert manifest["submitted"] is False
    assert [case["domain_ranks"] for case in manifest["cases"]] == ["1,1,1", "2,1,1", "2,2,1"]
    assert [case["mpi_ranks"] for case in manifest["cases"]] == [2, 4, 8]
    assert manifest["acceptance_criteria"]["mode_window"]["raw_head_low_frequency_modes_not_used"] is True

    for case in manifest["cases"]:
        assert case["site"] == "karolina"
        expected_replay_id = "unit-canary-" + case["case_id"]
        assert case["paper_replay_campaign_id"] == expected_replay_id
        assert case["output_root"].endswith(f"_runs/gv/figure_replay/{expected_replay_id}")
        script = Path(case["scheduler_script"])
        text = script.read_text(encoding="utf-8")
        assert "sbatch" not in "\n".join(line for line in text.splitlines() if not line.startswith("#SBATCH"))
        assert "MESOUQ_GV_EIGENMODES_PROFILE=canary" in text
        assert f"MESOUQ_GV_EIGENMODES_DOMAIN_RANKS={case['domain_ranks']}" in text
        assert f"--campaign-id {expected_replay_id}" in text
        assert f"_runs/gv/figure_replay/{expected_replay_id}" in text
        assert "--eigenmodes-profile canary" in text
        assert "mesouq_activate_site_env karolina" in text
        assert '#SBATCH --account=eu-26-17' in text
        assert "#SBATCH --partition=qgpu" in text
        assert f"#SBATCH --gpus={case['gpu_count']}" in text
        expected_script_command = f"sbatch {shlex.quote(str(script))}"
        assert case["submission_command"] == expected_script_command
        assert case["submission_commands"] == [expected_script_command]


def test_prepare_eigenmodes_domain_rank_canaries_vega_site(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "canaries-vega"

    rc = module.main(
        [
            "--campaign-id",
            "unit-canary-vega",
            "--output-root",
            str(output_root),
            "--site",
            "vega",
        ]
    )

    manifest_path = output_root / "gv_eigenmodes_domain_rank_canary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert manifest["site"] == "vega"
    assert manifest["submitted"] is False
    assert [case["domain_ranks"] for case in manifest["cases"]] == ["1,1,1", "2,1,1"]
    assert [case["site"] for case in manifest["cases"]] == ["vega", "vega"]

    for case in manifest["cases"]:
        expected_replay_id = "unit-canary-vega-" + case["case_id"]
        assert case["paper_replay_campaign_id"] == expected_replay_id
        assert case["output_root"].endswith(f"_runs/gv/figure_replay/{expected_replay_id}")
        script = Path(case["scheduler_script"])
        text = script.read_text(encoding="utf-8")
        assert "sbatch" not in "\n".join(line for line in text.splitlines() if not line.startswith("#SBATCH"))
        assert "#SBATCH --partition=gpu" in text
        assert f"#SBATCH --gres=gpu:{case['gpu_count']}" in text
        assert "#SBATCH --exclude=gn10" in text
        assert "module purge" in text
        assert "mesouq_activate_site_env vega" in text
        expected_script_command = f"sbatch {shlex.quote(str(script))}"
        assert case["submission_command"] == expected_script_command
        assert case["submission_commands"] == [expected_script_command]
