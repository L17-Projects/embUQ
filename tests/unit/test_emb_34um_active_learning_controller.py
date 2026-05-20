from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

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


def _synthetic_curve(*, ka: float, kb: float, force_grid: tuple[float, ...]) -> list[float]:
    return [round((ka * 1.0e-4) + (kb * 2.0e-5) + force * 0.015, 8) for force in force_grid]


def _write_completed_result(
    root: Path,
    *,
    candidate_id: str,
    ka: float,
    kb: float,
    force_grid: tuple[float, ...],
    runtime_seconds: float,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "emb_34um_result.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "parameter_names": ["ka", "kb"],
                "parameters": [ka, kb],
                "force_grid": list(force_grid),
                "vertical_diameter": _synthetic_curve(ka=ka, kb=kb, force_grid=force_grid),
                "runtime_seconds": runtime_seconds,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "emb_34um_runtime_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "runtime_seconds": runtime_seconds,
                "retry_count": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_completed_roots(
    roots: list[Path],
    *,
    prefix: str,
    ka_base: float,
    kb_base: float,
    force_grid: tuple[float, ...],
) -> list[str]:
    candidate_ids = []
    for order, root in enumerate(roots, start=1):
        candidate_id = f"{prefix}-c{order:03d}"
        candidate_ids.append(candidate_id)
        _write_completed_result(
            root,
            candidate_id=candidate_id,
            ka=ka_base + order * 10.0,
            kb=kb_base + order * 5.0,
            force_grid=force_grid,
            runtime_seconds=20.0 + order,
        )
    return candidate_ids


def _write_adaptive_round(
    *,
    campaign_root: Path,
    round_index: int,
    force_grid: tuple[float, ...],
) -> None:
    batch_root = campaign_root / f"adaptive_round_{round_index:02d}"
    output_roots = []
    rendered_manifests = []
    selected_candidates = []
    for order in range(1, 31):
        candidate_id = f"adaptive-r{round_index:02d}-c{order:03d}"
        output_root = batch_root / candidate_id
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        source = "ensemble_disagreement_diversity" if order <= 24 else "exploration"
        ka = 3000.0 + round_index * 100.0 + order * 7.0
        kb = 1400.0 + round_index * 50.0 + order * 3.0
        _write_completed_result(
            output_root,
            candidate_id=candidate_id,
            ka=ka,
            kb=kb,
            force_grid=force_grid,
            runtime_seconds=40.0 + round_index + order,
        )
        manifest_path.write_text(
            json.dumps({"candidate_id": candidate_id, "output_root": str(output_root)}, sort_keys=True),
            encoding="utf-8",
        )
        output_roots.append(str(output_root))
        rendered_manifests.append(str(manifest_path))
        selected_candidates.append(
            {
                "candidate_id": candidate_id,
                "round": round_index,
                "order": order,
                "source": source,
                "ka": ka,
                "kb": kb,
                "acquisition_score": round(1.0 + round_index * 0.1 + order * 0.01, 6),
                "ensemble_disagreement": round(0.5 + order * 0.005, 6),
            }
        )

    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "emb_34um_batch_summary.json").write_text(
        json.dumps(
            {
                "batch_root": str(batch_root),
                "candidate_count": 30,
                "expected_output_roots": output_roots,
                "rendered_candidate_manifests": rendered_manifests,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (batch_root / "selection_manifest.json").write_text(
        json.dumps(
            {
                "round": round_index,
                "selection_policy": "synthetic test selection",
                "selected_candidates": selected_candidates,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_ingestion_report(
    *,
    campaign_root: Path,
    completed_round3_count: int,
) -> None:
    records: list[dict[str, object]] = []
    for round_index in (1, 2, 3):
        completed_count = 30 if round_index < 3 else completed_round3_count
        for order in range(1, 31):
            completed = order <= completed_count
            records.append(
                {
                    "gate": "full_gate",
                    "round": round_index,
                    "candidate_id": f"emb-34um-final-gate-full-r0{round_index}-c{order:03d}",
                    "status": "completed" if completed else "failed",
                    "quarantined": bool(round_index == 3 and not completed),
                }
            )
    report_path = campaign_root / "ingestion_report" / "emb_34um_final_gate_ingestion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"records": records}, indent=2, sort_keys=True), encoding="utf-8")


def _fake_surrogate_report(
    records: object,
    *,
    validation_records: tuple[dict[str, Any], ...],
    **_: object,
) -> dict[str, Any]:
    record_count = len(tuple(records)) if not isinstance(records, tuple) else len(records)
    references = [list(record["reference_curve"]) for record in validation_records]
    predicted = [
        [round(float(value) * (1.0 + record_count * 1.0e-5), 8) for value in reference]
        for reference in references
    ]
    residuals = [round(0.1 + record_count * 0.001 + index * 0.0001, 8) for index, _ in enumerate(references)]
    return {
        "median_curve_rel_l2_pct": residuals[len(residuals) // 2] if residuals else 0.0,
        "mean_curve_rel_l2_pct": sum(residuals) / len(residuals) if residuals else 0.0,
        "max_curve_rel_l2_pct": max(residuals) if residuals else 0.0,
        "residuals": residuals,
        "predicted_curves": predicted,
        "reference_curves": references,
        "model_selection": {
            "rerun": True,
            "architecture": "synthetic_linear",
            "backend": "test",
            "notes": f"synthetic records={record_count}",
        },
    }


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
        "lhs_submit",
        "validation_submit",
        "al_round_1_submit",
        "al_round_2_select_render",
        "al_round_2_submit",
        "al_round_3_select_render",
        "al_round_3_submit",
        "final_ingestion",
        "build_al_vs_lhs_rows",
        "al_vs_lhs_validation",
        "final_gate_report",
    ]

    assert manifest["schema_version"] == module.CONTROLLER_SCHEMA_VERSION
    assert manifest["stage_order"] == expected_stage_order
    assert manifest["acceptance"]["pass"] is True
    assert manifest["production_readiness"]["status"] == "ready"
    assert manifest["acceptance"]["criteria"]["round_2_fresh_surrogate_selection_implemented"] is True
    assert manifest["acceptance"]["criteria"]["round_3_fresh_surrogate_selection_implemented"] is True
    assert manifest["linear_traceability"]["project"] == "Active Learning Engine"
    assert manifest["linear_traceability"]["engine"] == "Active Learning Engine"
    assert manifest["linear_traceability"]["issue"] == "MES-210"

    command_texts = []
    stage_names = []
    for stage in manifest["stages"]:
        stage_names.append(stage["name"])
        assert "commands" in stage
        for command in stage["commands"]:
            command_texts.append(command["command"])
            assert "EXECUTION_MODE=render-only" in command["command"]
        assert "expected_output_roots" in stage and isinstance(stage["expected_output_roots"], list)

    assert stage_names == expected_stage_order
    assert any("validate_emb_34um_al_vs_lhs.py" in item for item in command_texts)
    assert any("--adaptive-acquisition-available" in item for item in command_texts)
    assert any("--acquisition-engine dnn_ensemble_disagreement_diversity" in item for item in command_texts)
    assert any("--runtime-rows" in item for item in command_texts)
    assert any("--stage-action final-report" in item for item in command_texts)
    assert any("--stage-action select-render-round" in item for item in command_texts)

    stages = {stage["name"]: stage for stage in manifest["stages"]}
    assert stages["al_round_2_select_render"]["status"] == "planned"
    assert stages["al_round_2_select_render"]["commands"]
    assert stages["al_round_2_submit"]["status"] == "planned"
    assert stages["al_round_2_submit"]["commands"]
    assert stages["al_round_3_select_render"]["status"] == "planned"
    assert stages["al_round_3_submit"]["commands"]
    assert len(stages["al_vs_lhs_validation"]["expected_output_roots"]) >= 12


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
        stage["name"]: stage["commands"][0]["command"]
        for stage in manifest["stages"]
        if stage["name"].endswith("_submit") and stage["commands"]
    }
    assert "--array=0-29%30" in submit_commands["al_round_1_submit"]
    assert "--array=0-89%30" in submit_commands["lhs_submit"]
    assert "--array=0-29%30" in submit_commands["validation_submit"]
    assert "--array=0-29%30" in submit_commands["al_round_2_submit"]
    assert "BATCH_DIR_OVERRIDE=" in submit_commands["al_round_2_submit"]
    assert "--array=0-29%30" in submit_commands["al_round_3_submit"]
    assert "BATCH_DIR_OVERRIDE=" in submit_commands["al_round_3_submit"]


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
    assert any("--stage-action final-report" in command for command in inventory["final_gate_report"])
    assert any("--stage-action select-render-round" in command for command in inventory["al_round_2_select_render"])
    assert any("--stage-action select-render-round" in command for command in inventory["al_round_3_select_render"])


def test_emb_34um_active_learning_controller_builds_final_evidence_from_completed_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _disable_plot_generation(monkeypatch, plot_root=tmp_path / "plots")
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_custom_force_grid(force_grid)

    result = module.build_emb_34um_active_learning_controller(
        timestamp="20260501_140004",
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
    controller_manifest = result["manifest"]
    campaign_manifest_path = Path(controller_manifest["prepare_manifest_path"])
    campaign_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
    campaign_root = Path(campaign_manifest["campaign_root"])
    synthetic_force_grid = (0.0, 0.5, 1.0)

    _write_completed_roots(
        [Path(path) for path in campaign_manifest["full_gate"]["expected_output_roots"][:30]],
        prefix="al-r01",
        ka_base=1000.0,
        kb_base=500.0,
        force_grid=synthetic_force_grid,
    )
    _write_adaptive_round(campaign_root=campaign_root, round_index=2, force_grid=synthetic_force_grid)
    _write_adaptive_round(campaign_root=campaign_root, round_index=3, force_grid=synthetic_force_grid)
    round3_summary_path = campaign_root / "adaptive_round_03" / "emb_34um_batch_summary.json"
    round3_summary = json.loads(round3_summary_path.read_text(encoding="utf-8"))
    round3_summary["expected_output_roots"] = round3_summary["expected_output_roots"][:-1]
    round3_summary["rendered_candidate_manifests"] = round3_summary["rendered_candidate_manifests"][:-1]
    round3_summary_path.write_text(json.dumps(round3_summary, indent=2, sort_keys=True), encoding="utf-8")
    quarantined_root = campaign_root / "adaptive_round_03" / "adaptive-r03-c030"
    for filename in ("emb_34um_result.json", "emb_34um_runtime_status.json"):
        path = quarantined_root / filename
        if path.is_file():
            path.unlink()
    _write_completed_roots(
        [Path(path) for path in campaign_manifest["lhs_gate"]["expected_output_roots"]],
        prefix="lhs",
        ka_base=2000.0,
        kb_base=800.0,
        force_grid=synthetic_force_grid,
    )
    _write_completed_roots(
        [Path(path) for path in campaign_manifest["validation_gate"]["expected_output_roots"]],
        prefix="validation",
        ka_base=4000.0,
        kb_base=1200.0,
        force_grid=synthetic_force_grid,
    )

    import meso_uq.active_learning.emb_34um_final_gate_surrogate as surrogate

    monkeypatch.setattr(surrogate, "train_emb_34um_surrogate_ensemble", _fake_surrogate_report)

    exit_code = module.main(["--stage-action", "build-evidence", "--campaign-manifest", str(campaign_manifest_path)])

    assert exit_code == 0
    rows_path = campaign_root / "al_vs_lhs_rows.json"
    rounds_path = campaign_root / "round_payloads.json"
    runtime_rows_path = campaign_root / "runtime_rows.json"
    assert rows_path.is_file()
    assert rounds_path.is_file()
    assert runtime_rows_path.is_file()

    curve_rows = json.loads(rows_path.read_text(encoding="utf-8"))["curve_rows"]
    round_payloads = json.loads(rounds_path.read_text(encoding="utf-8"))["rounds"]
    runtime_rows = json.loads(runtime_rows_path.read_text(encoding="utf-8"))["runtime_rows"]
    validation_count = len(campaign_manifest["validation_gate"]["expected_output_roots"])

    assert len(curve_rows) == 3 * 2 * validation_count
    for round_index in (1, 2, 3):
        assert sum(1 for row in curve_rows if row["strategy"] == "al" and row["round"] == round_index) == validation_count
        assert sum(1 for row in curve_rows if row["strategy"] == "lhs" and row["round"] == round_index) == validation_count

    assert [item["round"] for item in round_payloads] == [1, 2, 3]
    for payload in round_payloads:
        assert payload["al_curve_count"] == 30
        assert payload["lhs_curve_count"] == 90
        assert len(payload["ka_kb_coverage"]) == 30
    for payload in round_payloads[1:]:
        assert payload["acquisition_scores"]
        assert payload["selected_candidate_scores"] == payload["acquisition_scores"]
        assert all(float(score) > 0.0 for score in payload["acquisition_scores"])

    assert len(runtime_rows) == 89 + 90 + validation_count
    assert all(row["status"] == "completed" for row in runtime_rows)
    assert all(row["runtime_status_path"] for row in runtime_rows)
    assert all(row["candidate_id"] for row in curve_rows if row["strategy"] in {"al", "lhs"})
    assert all(row["runtime_join_key"] == row["candidate_id"] for row in runtime_rows)
    assert all(row["runtime_join_key"] for row in curve_rows if row["strategy"] in {"al", "lhs"})

    runtime_map = {row["runtime_join_key"]: row["runtime_seconds"] for row in runtime_rows}
    for row in curve_rows:
        if row["strategy"] == "al":
            assert row["runtime_join_key"] in runtime_map
        if row["strategy"] == "lhs":
            assert row["runtime_join_key"] in runtime_map
    for round_index in (1, 2, 3):
        lhs_ids = [
            row["candidate_id"]
            for row in curve_rows
            if row["strategy"] == "lhs" and int(row["round"]) == round_index
        ]
        expected_ids = [
            f"lhs-c{order:03d}"
            for order in range((round_index - 1) * validation_count + 1, round_index * validation_count + 1)
        ]
        assert lhs_ids == expected_ids

    import meso_uq.active_learning.emb_34um_al_vs_lhs_validation as validation

    validation_manifest, _ = validation.build_emb_34um_al_vs_lhs_validation_report(
        curve_rows=curve_rows,
        runtime_rows=runtime_rows,
    )
    assert validation_manifest["runtime_curve_count"] == len(curve_rows)
    assert {
        row["runtime_join_key"]: row.get("runtime_seconds")
        for row in validation_manifest["curve_records"]
        if row["strategy"] in {"al", "lhs"}
    } == {
        row["runtime_join_key"]: runtime_map[row["runtime_join_key"]]
        for row in curve_rows
        if row["strategy"] in {"al", "lhs"}
    }


def test_emb_34um_active_learning_controller_builds_final_evidence_with_quarantined_curve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _disable_plot_generation(monkeypatch, plot_root=tmp_path / "plots")
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_custom_force_grid(force_grid)

    result = module.build_emb_34um_active_learning_controller(
        timestamp="20260501_140005",
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
    controller_manifest = result["manifest"]
    campaign_manifest_path = Path(controller_manifest["prepare_manifest_path"])
    campaign_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
    campaign_root = Path(campaign_manifest["campaign_root"])
    synthetic_force_grid = (0.0, 0.5, 1.0)

    _write_completed_roots(
        [Path(path) for path in campaign_manifest["full_gate"]["expected_output_roots"][:30]],
        prefix="al-r01",
        ka_base=1000.0,
        kb_base=500.0,
        force_grid=synthetic_force_grid,
    )
    _write_adaptive_round(campaign_root=campaign_root, round_index=2, force_grid=synthetic_force_grid)
    _write_adaptive_round(campaign_root=campaign_root, round_index=3, force_grid=synthetic_force_grid)
    failed_root = campaign_root / "adaptive_round_03" / "adaptive-r03-c005"
    (failed_root / "emb_34um_result.json").unlink()
    (failed_root / "emb_34um_runtime_status.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "retry_count": 3,
                "retry_limit": 3,
                "runtime_seconds": 7200.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    ingestion_dir = campaign_root / "ingestion_report"
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    (ingestion_dir / module.EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME).write_text(
        json.dumps(
            {
                "records": [
                    {
                        "candidate_id": "adaptive-r03-c005",
                        "gate": "full_gate",
                        "round": 3,
                        "status": "failed",
                        "quarantined": True,
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_completed_roots(
        [Path(path) for path in campaign_manifest["lhs_gate"]["expected_output_roots"]],
        prefix="lhs",
        ka_base=2000.0,
        kb_base=800.0,
        force_grid=synthetic_force_grid,
    )
    _write_completed_roots(
        [Path(path) for path in campaign_manifest["validation_gate"]["expected_output_roots"]],
        prefix="validation",
        ka_base=4000.0,
        kb_base=1200.0,
        force_grid=synthetic_force_grid,
    )

    import meso_uq.active_learning.emb_34um_final_gate_surrogate as surrogate

    monkeypatch.setattr(surrogate, "train_emb_34um_surrogate_ensemble", _fake_surrogate_report)

    exit_code = module.main(["--stage-action", "build-evidence", "--campaign-manifest", str(campaign_manifest_path)])

    assert exit_code == 0
    round_payloads = json.loads((campaign_root / "round_payloads.json").read_text(encoding="utf-8"))["rounds"]
    runtime_rows = json.loads((campaign_root / "runtime_rows.json").read_text(encoding="utf-8"))["runtime_rows"]

    round_three = round_payloads[2]
    assert round_three["al_curve_count"] == 30
    assert round_three["al_completed_curve_count"] == 29
    assert round_three["al_training_curve_count"] == 89
    assert round_three["lhs_training_curve_count"] == 89
    assert len(round_three["ka_kb_coverage"]) == 30
    assert round_three["failure_counts"] == {"failed": 1}
    assert round_three["quarantine_counts"] == {"failed": 1}
    assert len(runtime_rows) == 89 + 90 + len(campaign_manifest["validation_gate"]["expected_output_roots"])
