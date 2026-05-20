from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


def _disable_plot_generation(monkeypatch: pytest.MonkeyPatch, *, plot_root: Path) -> None:
    import meso_uq.dpd_sampling.boundary as boundary

    def _fake_render_validation_plot(*_: object, **__: object) -> tuple[Path, Path]:
        plot_root.mkdir(parents=True, exist_ok=True)
        plot = plot_root / "dpd_sampling_validation_plot.png"
        sidecar = plot_root / "dpd_sampling_validation_plot.png.json"
        plot.write_bytes(b"")
        sidecar.write_text("{}", encoding="utf-8")
        return plot, sidecar

    monkeypatch.setattr(boundary, "_render_validation_plot", _fake_render_validation_plot)


def _write_custom_force_grid(path: Path) -> None:
    path.write_text("0 0 0 0 0 0 0 0 1 2 3 10 20 30\n", encoding="utf-8")


def test_prepare_emb_34um_final_gate_manifests_full_lhs_benchmark_validation_campaigns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _prepare_module()
    _disable_plot_generation(monkeypatch, plot_root=tmp_path / "plot_stubs")
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
        lhs_candidate_count=90,
        validation_target_count=30,
        dnn_ensemble_target_size=10,
        candidate_pool_size=100,
        skip_vault_copy=False,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    campaign_root = tmp_path / "scratch" / timestamp
    vault_target = tmp_path / "vault" / timestamp

    assert manifest["campaign_root"] == str(campaign_root)
    assert manifest["platform"] == "karolina"
    assert manifest["schema_version"] == module.FULL_GATE_SCHEMA_VERSION

    full_gate = manifest["full_gate"]
    canary_gate = manifest["canary_gate"]
    lhs_gate = manifest["lhs_gate"]
    benchmark_gate = manifest["benchmark_gate"]
    validation_gate = manifest["validation_gate"]

    assert full_gate["candidate_count"] == 90
    assert canary_gate["candidate_count"] == 1
    assert lhs_gate["candidate_count"] == 90
    assert benchmark_gate["candidate_count"] == 3
    assert validation_gate["candidate_count"] == 30

    for gate in (full_gate, canary_gate, lhs_gate, benchmark_gate, validation_gate):
        assert gate["submission_expected"]["submission_commands"] == []
        assert gate["submission_expected"]["render_only"] is True
        assert gate["submission_expected"]["submitted"] is False
        assert gate["scheduler_boundary"]["submission"]["submitted"] is False

    assert manifest["acceptance"]["pass"] is True
    assert manifest["acceptance"]["criteria"]["full_count_is_90"] is True
    assert manifest["acceptance"]["criteria"]["canary_count_is_1"] is True
    assert manifest["acceptance"]["criteria"]["lhs_count_is_90"] is True
    assert manifest["acceptance"]["criteria"]["benchmark_count_is_3"] is True
    assert manifest["acceptance"]["criteria"]["validation_count_is_30"] is True

    assert manifest["commands"]["full"]["mode"] == "full"
    assert manifest["commands"]["canary"]["mode"] == "canary"
    assert manifest["commands"]["lhs"]["mode"] == module.LHS_GATE_NAME
    assert manifest["commands"]["benchmark"]["mode"] == module.BENCHMARK_GATE_NAME
    assert manifest["commands"]["validation"]["mode"] == module.VALIDATION_GATE_NAME
    assert manifest["commands"]["full"]["execution_mode"] == module.EXECUTION_MODE_RENDER_ONLY
    assert manifest["commands"]["canary"]["execution_mode"] == module.EXECUTION_MODE_RENDER_ONLY
    assert manifest["commands"]["lhs"]["execution_mode"] == module.EXECUTION_MODE_RENDER_ONLY
    assert manifest["commands"]["benchmark"]["execution_mode"] == module.EXECUTION_MODE_RENDER_ONLY
    assert manifest["commands"]["validation"]["execution_mode"] == module.EXECUTION_MODE_RENDER_ONLY
    assert manifest["commands"]["full"]["array"] == "0-89%30"
    assert manifest["commands"]["canary"]["array"] == "0-0"
    assert manifest["commands"]["lhs"]["array"] == "0-89%30"
    assert manifest["commands"]["benchmark"]["array"] == "0-2%30"
    assert manifest["commands"]["validation"]["array"] == "0-29%30"

    for value in manifest["commands"].values():
        assert value["script"].endswith("emb_34um_active_learning_array.sbatch")
        assert "EXECUTION_MODE=render-only" in value["command"]

    assert manifest["policy"]["candidate_pool_size"] == 100
    assert manifest["policy"]["accepted_curves"]["rounds"] == 3
    assert manifest["policy"]["accepted_curves"]["per_round"] == 30
    assert manifest["policy"]["accepted_curves"]["total"] == 90
    assert manifest["policy"]["lhs_comparator_curves"] == 90
    assert manifest["policy"]["validation_target_curves"] == 30
    assert manifest["policy"]["benchmark_curves"] == 3
    assert manifest["policy"]["dnn_ensemble_target_size"] == 10
    assert manifest["policy"]["sampling"]["parameter_space"] == "log10"
    assert manifest["policy"]["sampling"]["bounds"]["ka"] == [1e2, 6e5]
    assert manifest["policy"]["sampling"]["bounds"]["kb"] == [400.0, 70000.0]
    assert manifest["policy"]["failure_policy"]["replacement_mode"] == "next_candidate"
    assert manifest["policy"]["adaptive_selection"]["round_1_source"] == "initial_sobol_maximin"
    assert manifest["policy"]["adaptive_selection"]["exploration_count"] == 6
    assert manifest["policy"]["adaptive_selection"]["acquisition_count"] == 24
    assert manifest["policy"]["adaptive_selection"]["posterior_aware"] is False
    assert manifest["linear_traceability"]["project"] == "Active Learning Engine"
    assert manifest["linear_traceability"]["engine"] == "Active Learning Engine"
    assert manifest["linear_traceability"]["issue"] == "MES-210"
    assert len(manifest["full_design"]["round_manifests"]) == 3
    assert manifest["full_design"]["round_manifests"][0]["selected_source_distribution"] == {
        "initial_sobol_maximin": 30
    }
    assert manifest["full_design"]["round_manifests"][1]["selected_source_distribution"] == {
        "exploration": 6,
        "ensemble_disagreement_diversity": 24,
    }

    assert len(manifest["runtime_artifacts"]["benchmark"]) == 3
    assert len(manifest["runtime_artifacts"]["validation"]) == 30
    assert (campaign_root / "full_gate" / "emb_34um_batch_summary.json").is_file()
    assert (campaign_root / "canary" / "emb_34um_batch_summary.json").is_file()
    assert (campaign_root / "lhs_gate" / "emb_34um_batch_summary.json").is_file()
    assert (campaign_root / "benchmark" / "emb_34um_batch_summary.json").is_file()
    assert (campaign_root / "validation" / "emb_34um_batch_summary.json").is_file()
    assert vault_target.is_dir()

    for payload in manifest["runtime_artifacts"].values():
        for artifact in payload:
            assert artifact.endswith("emb_34um_runtime_status.json")

    benchmark_labels = tuple(
        json.loads(Path(path).read_text(encoding="utf-8"))["candidate_id"].split("-")[-1]
        for path in benchmark_gate["rendered_candidate_manifests"]
    )
    assert set(benchmark_labels) == set(module.BENCHMARK_LABELS)

    full_gate_manifest = json.loads(Path(full_gate["manifest_path"]).read_text(encoding="utf-8"))
    for candidate_record in full_gate_manifest["candidate_records"]:
        assert candidate_record["experiment"] == "indentation"
        assert candidate_record["family"] == "emb"
        assert "candidate_id" in candidate_record
    for rendered_path in full_gate["rendered_candidate_manifests"][:3]:
        rendered = json.loads(Path(rendered_path).read_text(encoding="utf-8"))
        request_payload = rendered["rendered_payload"]["request_payload"]
        assert request_payload["schema_version"] == module.EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION
        assert request_payload["force_grid"] == list(force_grid)
        normalized_parameters = rendered["normalized_payload"]["parameters"]
        assert "ka" in normalized_parameters
        assert "kb" in normalized_parameters
        assert "Yt" not in normalized_parameters

    canary_rendered = json.loads(Path(canary_gate["rendered_candidate_manifests"][0]).read_text(encoding="utf-8"))
    assert canary_rendered["rendered_payload"]["request_payload"]["force_grid"] == expected_canary_forces
    assert canary_rendered["normalized_payload"]["canary"]["enabled"] is True


def test_prepare_emb_34um_final_gate_rejects_non_default_counts(tmp_path: Path) -> None:
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
            lhs_candidate_count=90,
            validation_target_count=30,
            dnn_ensemble_target_size=10,
            candidate_pool_size=100,
            skip_vault_copy=True,
        )
    except ValueError as exc:
        assert "fixed at 3" in str(exc)
    else:
        raise AssertionError("Expected non-default canary force count to fail.")

    try:
        module.prepare_emb_34um_final_gate(
            timestamp="20260501_140000",
            scratch_root=tmp_path / "scratch",
            vault_root=tmp_path / "vault",
            force_grid_path=repo_root / module.DEFAULT_FORCE_GRID_DATA,
            walltime="00:30:00",
            concurrent_jobs=30,
            retry_limit=3,
            run_id_prefix="emb-34um-final-gate-test",
            canary_force_count=3,
            lhs_candidate_count=10,
            validation_target_count=30,
            dnn_ensemble_target_size=10,
            candidate_pool_size=100,
            skip_vault_copy=True,
        )
    except ValueError as exc:
        assert "lhs_candidate_count is fixed" in str(exc)
    else:
        raise AssertionError("Expected invalid lhs_candidate_count to fail.")
