from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_controller_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    script_path = (
        repo_root
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "run_emb_34um_active_learning_controller.py"
    )
    spec = importlib.util.spec_from_file_location("run_emb_34um_active_learning_controller", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _disable_plot_generation(monkeypatch: pytest.MonkeyPatch, *, plot_root: Path) -> None:
    import meso_uq.dpd_sampling.boundary as boundary

    def _fake_render_validation_plot(*_: object, **__: object):
        plot_root.mkdir(parents=True, exist_ok=True)
        plot = plot_root / "dpd_sampling_validation_plot.png"
        sidecar = plot_root / "dpd_sampling_validation_plot.png.json"
        plot.write_bytes(b"")
        sidecar.write_text("{}", encoding="utf-8")
        return plot, sidecar

    monkeypatch.setattr(boundary, "_render_validation_plot", _fake_render_validation_plot)


def _write_custom_force_grid(path: Path) -> None:
    path.write_text("0 0 0 0 0 0 0 0 1 2 3 10 20 30\n", encoding="utf-8")


def test_emb_34um_active_learning_controller_dry_run_manifest_stage_order_and_traceability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _disable_plot_generation(monkeypatch, plot_root=tmp_path / "plots")
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_custom_force_grid(force_grid)

    result = module.build_emb_34um_active_learning_controller(
        timestamp="20260501_140000",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-final-gate-test",
        skip_vault_copy=True,
        dry_run=True,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    expected_stage_order = [
        "canary",
        "benchmark",
        "al_round_1_submit",
        "al_round_1_retrain_model_select",
        "al_round_1_score",
        "al_round_2_submit",
        "al_round_2_retrain_model_select",
        "al_round_2_score",
        "al_round_3_submit",
        "al_round_3_retrain_model_select",
        "al_round_3_score",
        "al_vs_lhs_validation",
        "final_gate_report",
    ]

    assert manifest["schema_version"] == module.CONTROLLER_SCHEMA_VERSION
    assert manifest["stage_order"] == expected_stage_order
    assert manifest["acceptance"]["pass"] is True
    assert manifest["linear_traceability"]["project"] == "Active Learning Engine"
    assert manifest["linear_traceability"]["engine"] == "Active Learning Engine"
    assert manifest["linear_traceability"]["issue"] == "MES-210"

    command_texts = []
    stage_names = []
    for stage in manifest["stages"]:
        stage_names.append(stage["name"])
        assert "commands" in stage and stage["commands"]
        for command in stage["commands"]:
            command_texts.append(command["command"])
            assert "EXECUTION_MODE=render-only" in command["command"]
        assert "expected_output_roots" in stage and isinstance(stage["expected_output_roots"], list)

    assert stage_names == expected_stage_order
    assert any("validate_emb_34um_al_vs_lhs.py" in item for item in command_texts)
    assert any("write_active_learning_final_gate_artifacts" in item for item in command_texts)


def test_emb_34um_active_learning_controller_dry_run_does_not_submit_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _disable_plot_generation(monkeypatch, plot_root=tmp_path / "plots")
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_custom_force_grid(force_grid)

    called = []

    def _fake_run(*_: object, **__: object) -> None:
        called.append(True)
        return None

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = module.main(
        [
            "--timestamp",
            "20260501_140001",
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--vault-root",
            str(tmp_path / "vault"),
            "--force-grid",
            str(force_grid),
            "--skip-vault-copy",
            "--dry-run",
            "--run-commands",
        ]
    )
    assert exit_code == 0
    assert called == []


def test_emb_34um_active_learning_controller_schedules_al_submit_30_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _disable_plot_generation(monkeypatch, plot_root=tmp_path / "plots")
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_custom_force_grid(force_grid)

    result = module.build_emb_34um_active_learning_controller(
        timestamp="20260501_140002",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-final-gate-test",
        skip_vault_copy=True,
        dry_run=True,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    submit_commands = {
        stage["name"]: stage["commands"][0]["command"] for stage in manifest["stages"] if stage["name"].endswith("_submit")
    }
    assert "--array=0-29%30" in submit_commands["al_round_1_submit"]
    assert "--array=30-59%30" in submit_commands["al_round_2_submit"]
    assert "--array=60-89%30" in submit_commands["al_round_3_submit"]


def test_emb_34um_active_learning_controller_keeps_final_validation_and_report_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _disable_plot_generation(monkeypatch, plot_root=tmp_path / "plots")
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_custom_force_grid(force_grid)

    result = module.build_emb_34um_active_learning_controller(
        timestamp="20260501_140003",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-final-gate-test",
        skip_vault_copy=True,
        dry_run=True,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    command_inventory = manifest["command_inventory"]["stages"]
    inventory = {item["name"]: item["commands"] for item in command_inventory}

    assert any("validate_emb_34um_al_vs_lhs.py" in command for command in inventory["al_vs_lhs_validation"])
    assert any("write_active_learning_final_gate_artifacts" in command for command in inventory["final_gate_report"])
