from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

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
        / "run_emb_34um_causal_validation_controller.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_emb_34um_causal_validation_controller", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _install_fast_prepare(module)
    return module


def _install_fast_prepare(module: object) -> None:
    def candidate_record(*, seed: int, mode: str, index: int) -> dict[str, object]:
        return {
            "candidate_id": f"fast-seed{seed:03d}-{mode}-c{index:03d}",
            "candidate_hash": f"h-{seed:03d}-{mode}-{index:03d}",
            "family": "emb",
            "experiment": "indentation",
            "ka": 100.0 + float(seed * 10 + index),
            "kb": 400.0 + float(seed * 10 + index),
            "force_grid": [10.0, 20.0, 30.0],
        }

    def stage_candidate_count(mode: str) -> int:
        if mode == "validation":
            return int(module.EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE)
        if mode == "shared_initial":
            return int(module.EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE)
        return int(module.EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE)

    def write_batch_summary(*, campaign_root: Path, seed: int, mode: str, candidate_count: int) -> None:
        stage_root = campaign_root / f"seed-{seed:03d}" / mode
        summary_path = stage_root / module.BATCH_SUMMARY_FILENAME
        candidate_ids = [f"c{index:03d}" for index in range(1, candidate_count + 1)]
        _write_batch_summary_with_candidate_roots(summary_path, candidate_ids)

    def command_entry(
        *,
        campaign_root: Path,
        seed: int,
        mode: str,
        candidate_count: int,
        walltime: str,
        concurrent_jobs: int,
        retry_limit: int,
    ) -> dict[str, object]:
        stage_root = campaign_root / f"seed-{seed:03d}" / mode
        array_max = candidate_count - 1
        command = (
            f"sbatch --parsable --time={walltime} --array=0-{array_max}%{concurrent_jobs} "
            f"--export=EXECUTION_MODE=render-only,BATCH_DIR_OVERRIDE={stage_root},SEED={seed},"
            f"MODE={mode},CONCURRENT_JOBS={concurrent_jobs},RETRY_LIMIT={retry_limit} "
            "scripts/platforms/karolina/sbatch/emb_34um_active_learning_array.sbatch"
        )
        return {
            "mode": mode,
            "seed": seed,
            "candidate_count": candidate_count,
            "command": command,
            "batch_summary_path": str(stage_root / module.BATCH_SUMMARY_FILENAME),
            "batch_manifest_path": str(stage_root / module.BATCH_MANIFEST_FILENAME),
        }

    def fake_prepare_emb_34um_causal_validation(
        *,
        timestamp,
        scratch_root,
        vault_root,
        force_grid_path,
        walltime,
        concurrent_jobs,
        retry_limit,
        run_id_prefix,
        step_count,
        seed_count,
        include_plot_requirements,
    ):
        campaign_root = Path(scratch_root) / str(timestamp)
        campaign_root.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, object]] = []
        seeds: list[dict[str, object]] = []
        for seed in range(1, int(seed_count) + 1):
            shared_count = stage_candidate_count("shared_initial")
            validation_count = stage_candidate_count("validation")
            write_batch_summary(campaign_root=campaign_root, seed=seed, mode="shared_initial", candidate_count=shared_count)
            write_batch_summary(campaign_root=campaign_root, seed=seed, mode="validation", candidate_count=validation_count)

            seed_payload: dict[str, object] = {
                "seed": seed,
                "shared_initial": {
                    "candidate_records": [
                        candidate_record(seed=seed, mode="shared-initial", index=index)
                        for index in range(1, shared_count + 1)
                    ]
                },
                "validation": {
                    "candidate_records": [
                        candidate_record(seed=seed, mode="validation", index=index)
                        for index in range(1, validation_count + 1)
                    ]
                },
                "al_steps": [],
                "lhs_steps": [],
            }

            for mode in ("shared_initial", "validation"):
                entries.append(
                    command_entry(
                        campaign_root=campaign_root,
                        seed=seed,
                        mode=mode,
                        candidate_count=stage_candidate_count(mode),
                        walltime=str(walltime),
                        concurrent_jobs=int(concurrent_jobs),
                        retry_limit=int(retry_limit),
                    )
                )

            for step in range(1, int(step_count) + 1):
                al_mode = f"al-step-{step:02d}"
                lhs_mode = f"lhs-step-{step:02d}"
                step_count_target = stage_candidate_count(al_mode)
                prior_prereqs = [campaign_root / f"seed-{seed:03d}" / "shared_initial" / module.BATCH_SUMMARY_FILENAME]
                prior_prereqs.extend(
                    campaign_root / f"seed-{seed:03d}" / f"al-step-{previous:02d}" / module.BATCH_SUMMARY_FILENAME
                    for previous in range(1, step)
                )
                seed_payload["al_steps"].append(
                    {
                        "step": step,
                        "candidate_count": step_count_target,
                        "candidate_pool_size": 500,
                        "acquisition_count": 80,
                        "exploration_count": 20,
                        "selection_seed": seed * 10_000 + step,
                        "selection_prerequisites": [str(path) for path in prior_prereqs],
                    }
                )
                seed_payload["lhs_steps"].append(
                    {
                        "step": step,
                        "candidate_records": [
                            candidate_record(seed=seed, mode=lhs_mode, index=index)
                            for index in range(1, step_count_target + 1)
                        ],
                    }
                )
                for mode in (al_mode, lhs_mode):
                    entries.append(
                        command_entry(
                            campaign_root=campaign_root,
                            seed=seed,
                            mode=mode,
                            candidate_count=step_count_target,
                            walltime=str(walltime),
                            concurrent_jobs=int(concurrent_jobs),
                            retry_limit=int(retry_limit),
                        )
                    )
            seeds.append(seed_payload)

        command_inventory = {
            "entries": entries,
            "count": len(entries),
            "walltime": str(walltime),
            "concurrent_jobs": int(concurrent_jobs),
            "retry_limit": int(retry_limit),
        }
        manifest = {
            "schema_version": module._DESIGN_SCHEMA_VERSION,
            "timestamp": str(timestamp),
            "campaign_root": str(campaign_root),
            "scratch_root": str(scratch_root),
            "vault_root_timestamp": str(Path(vault_root) / str(timestamp)),
            "run_id_prefix": str(run_id_prefix),
            "force_grid_path": str(force_grid_path),
            "sampling": {"parameter_space": "log10"},
            "seeds": seeds,
            "command_inventory": command_inventory,
        }
        manifest_path = campaign_root / module.EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME
        inventory_path = campaign_root / module.EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
        inventory_path.write_text(json.dumps(command_inventory, sort_keys=True, indent=2), encoding="utf-8")
        return {"manifest": manifest, "manifest_path": manifest_path, "command_inventory_path": inventory_path}

    module._prepare.prepare_emb_34um_causal_validation = fake_prepare_emb_34um_causal_validation


def _write_force_grid(path: Path) -> None:
    path.write_text("0 0 0 0 0 0 0 0 1 2 3 10 20 30\n", encoding="utf-8")


def _patch_fast_adaptive_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from meso_uq.active_learning import emb_34um_final_gate_design as design_module
    from meso_uq.active_learning import emb_34um_final_gate_surrogate as surrogate_module

    def fake_train_emb_34um_surrogate_ensemble(*, records, validation_records, seeds, architecture_names):
        return {
            "fit": object(),
            "model_selection": {
                "selected_architecture": "test-linear",
                "candidate_count": len(records),
                "validation_count": len(validation_records),
            },
        }

    def fake_build_emb_34um_final_gate_design_round(
        *,
        run_id,
        round_index,
        seed,
        existing_points,
        candidate_pool_size,
        candidate_prefix,
        use_log_space,
    ):
        round_pool = [
            SimpleNamespace(
                candidate_id=f"{candidate_prefix}-pool-{index:03d}",
                parameters={"ka": 100.0 + float(index), "kb": 400.0 + float(index)},
            )
            for index in range(1, int(candidate_pool_size) + 1)
        ]
        return SimpleNamespace(round_pool=round_pool)

    def fake_score_emb_34um_candidate_pool(fit, *, candidate_records, force_grid, existing_points):
        scored = []
        total = len(candidate_records)
        for index, row in enumerate(candidate_records, start=1):
            scored.append(
                {
                    **row,
                    "acquisition_score": float(total - index + 1),
                    "ensemble_disagreement": 1.0 / float(index),
                }
            )
        return scored

    monkeypatch.setattr(surrogate_module, "train_emb_34um_surrogate_ensemble", fake_train_emb_34um_surrogate_ensemble)
    monkeypatch.setattr(surrogate_module, "score_emb_34um_candidate_pool", fake_score_emb_34um_candidate_pool)
    monkeypatch.setattr(design_module, "build_emb_34um_final_gate_design_round", fake_build_emb_34um_final_gate_design_round)


def _build_ingest_audit(path: Path, *, clean: bool = False, shortfall_count: int = 0) -> None:
    payload = {
        "status": "clean" if clean else "dirty",
        "passed": bool(clean),
        "shortfall_count": shortfall_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _materialize_completed_batch_outputs(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_output_roots = summary.get("expected_output_roots", [])
    assert isinstance(expected_output_roots, list)
    assert expected_output_roots
    for index, output_root in enumerate(expected_output_roots, start=1):
        root = Path(str(output_root))
        root.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "candidate_id": root.name,
            "parameter_names": ["ka", "kb"],
            "parameters": [1000.0 + float(index), 500.0 + float(index)],
            "force_grid": [10.0, 20.0, 30.0],
            "vertical_diameter": [1.0, 1.1, 1.2],
        }
        (root / "emb_34um_result.json").write_text(json.dumps(result_payload), encoding="utf-8")


def _write_batch_summary(path: Path, *, candidate_count: int = 3) -> None:
    payload = {
        "status": "selection_rendered",
        "candidate_count": candidate_count,
        "expected_output_roots": [str(path.parent / "emb" / f"c{index:03d}") for index in range(1, candidate_count + 1)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_batch_summary_with_candidate_roots(path: Path, candidate_ids: Sequence[str]) -> None:
    payload = {
        "status": "selection_rendered",
        "candidate_count": len(candidate_ids),
        "expected_output_roots": [str(path.parent / "emb" / candidate_id) for candidate_id in candidate_ids],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _materialize_replacement_outputs(*, replacement_batch_summary_path: Path) -> None:
    summary = json.loads(replacement_batch_summary_path.read_text(encoding="utf-8"))
    replacement_records = summary.get("replacement_records", [])
    assert isinstance(replacement_records, list)
    assert replacement_records
    for index, record in enumerate(replacement_records, start=1):
        root = Path(str(record["output_root"]))
        root.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "candidate_id": str(record["replacement_candidate_id"]),
            "parameter_names": ["ka", "kb"],
            "parameters": [float(record["ka"]), float(record["kb"])],
            "force_grid": [10.0, 20.0, 30.0],
            "vertical_diameter": [1.0, 1.1, 1.2 + index * 0.01],
        }
        (root / "emb_34um_result.json").write_text(json.dumps(result_payload), encoding="utf-8")


def _write_design_manifest_for_replacement_testing(
    campaign_root: Path,
    *,
    manifest_filename: str = "emb_34um_causal_validation_manifest.json",
    run_id_prefix: str = "emb-34um-causal-replace",
    walltime: str = "00:30:00",
    concurrent_jobs: int = 11,
    retry_limit: int = 2,
) -> None:
    manifest_path = campaign_root / manifest_filename
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "seeds": [],
                "run_id_prefix": run_id_prefix,
                "vault_root_timestamp": str(campaign_root / "vault"),
                "command_inventory": {
                    "walltime": walltime,
                    "concurrent_jobs": concurrent_jobs,
                    "retry_limit": retry_limit,
                },
                "runtime_fingerprint": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_lhs_design_manifest_for_replacement_testing(
    campaign_root: Path,
    *,
    seed: int = 1,
    step_count: int = 5,
    candidates_per_step: int = 4,
    run_id_prefix: str = "emb-34um-causal-lhs-replace",
    walltime: str = "00:30:00",
    concurrent_jobs: int = 30,
    retry_limit: int = 3,
) -> None:
    lhs_steps: list[dict[str, object]] = []
    for step in range(1, step_count + 1):
        records: list[dict[str, Any]] = []
        for index in range(1, candidates_per_step + 1):
            records.append(
                {
                    "candidate_id": f"{run_id_prefix}-seed{seed:03d}-lhs-step{step:02d}-c{index:03d}",
                    "candidate_hash": f"h-{seed:03d}-{step:02d}-{index:03d}",
                    "family": "emb",
                    "experiment": "indentation",
                    "ka": float(100.0 + index + step),
                    "kb": float(200.0 + index + step),
                    "force_grid": [10.0, 20.0, 30.0],
                }
            )
        lhs_steps.append({"step": step, "candidate_records": records})

    payload = {
        "seeds": [
            {
                "seed": seed,
                "lhs_steps": lhs_steps,
                "shared_initial": {"candidate_records": []},
                "al_steps": [],
                "validation": {"candidate_records": []},
            }
        ],
        "run_id_prefix": run_id_prefix,
        "vault_root_timestamp": str(campaign_root / "vault"),
        "command_inventory": {
            "walltime": walltime,
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
        },
        "runtime_fingerprint": {},
    }

    manifest_path = campaign_root / "emb_34um_causal_validation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_selection_manifest_with_reserve(
    stage_root: Path,
    reserve_records: list[dict[str, object]],
    *,
    selection_manifest_filename: str = "emb_34um_causal_validation_selection_manifest.json",
) -> Path:
    payload = {
        "status": "selection_rendered",
        "candidate_reserve": reserve_records,
    }
    manifest_path = stage_root / selection_manifest_filename
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _configure_al_step01_with_replacement_records(
    *,
    campaign_root: Path,
    module: object,
    seed: int,
    missing_candidate_ids: Sequence[str],
    materialize_replacement_outputs: bool = True,
) -> dict[str, str]:
    stage_root = campaign_root / f"seed-{seed:03d}" / "al-step-01"
    stage_root.mkdir(parents=True, exist_ok=True)

    candidate_ids = [f"c{i:03d}" for i in range(1, 101)]
    step_summary = stage_root / module.BATCH_SUMMARY_FILENAME
    _write_batch_summary_with_candidate_roots(step_summary, candidate_ids=candidate_ids)
    _materialize_completed_batch_outputs(summary_path=step_summary)
    for candidate_id in missing_candidate_ids:
        (stage_root / "emb" / candidate_id / "emb_34um_result.json").unlink()

    reserve_records: list[dict[str, Any]] = []
    for index, candidate_index in enumerate(range(1, len(missing_candidate_ids) + 1), start=1):
        reserve_records.append(
            {
                "candidate_id": f"reserve-{seed:03d}-{index:03d}",
                "candidate_pool_id": f"pool-reserve-{seed:03d}-{index:03d}",
                "candidate_hash": f"h-{seed:03d}-{index:03d}",
                "family": "emb",
                "experiment": "indentation",
                "ka": 1000.0 + candidate_index,
                "kb": 500.0 + candidate_index,
                "force_grid": [10.0, 20.0, 30.0],
                "selection_seed": 1000 + seed * 100 + index,
                "selection_order": index,
                "selection_pool_rank": 100 + index,
                "order": index,
                "acquisition_score": 0.95 / index,
                "ensemble_disagreement": 0.2 / index,
            }
        )

    _write_selection_manifest_with_reserve(stage_root=stage_root, reserve_records=reserve_records)
    (stage_root / module.SELECTION_BATCH_SUMMARY_FILENAME).write_text(
        json.dumps(
            {
                "status": "selection_rendered",
                "seed": seed,
                "step": 1,
                "candidate_count": 100,
                "batch_summary_path": str(step_summary),
            }
        ),
        encoding="utf-8",
    )

    replacement_results = module._build_replacement_candidates_from_reserve(
        campaign_root=campaign_root,
        seed=seed,
        step=1,
        failed_candidate_ids=tuple(missing_candidate_ids),
    )

    if materialize_replacement_outputs:
        _materialize_replacement_outputs(
            replacement_batch_summary_path=Path(replacement_results["replacement_batch_summary_path"])
        )
    return replacement_results


def test_causal_controller_stage_vocabulary_and_order(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    result = module.build_emb_34um_causal_validation_controller(
        timestamp="20260501_100001",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-vocab",
        execute=False,
        dry_run=True,
    )
    manifest = result["manifest"]

    expected = [
        "render_protocol",
        "shared_initial",
        "validation",
        "al-step-01-select-render",
        "al-step-01-submit",
        "lhs-step-01-submit",
        "al-step-02-select-render",
        "al-step-02-submit",
        "lhs-step-02-submit",
        "al-step-03-select-render",
        "al-step-03-submit",
        "lhs-step-03-submit",
        "al-step-04-select-render",
        "al-step-04-submit",
        "lhs-step-04-submit",
        "al-step-05-select-render",
        "al-step-05-submit",
        "lhs-step-05-submit",
        "ingest",
        "analyze",
        "escalation",
    ]

    assert manifest["stage_order"] == expected
    assert manifest["stage_order"][:3] == ["render_protocol", "shared_initial", "validation"]
    assert manifest["stage_order"].index("al-step-01-select-render") < manifest["stage_order"].index("al-step-01-submit")
    assert manifest["stage_order"].index("al-step-01-submit") < manifest["stage_order"].index("lhs-step-01-submit")
    assert set(manifest["linear_traceability"]["issues"]) >= {f"MES-{i}" for i in range(219, 223)}
    assert manifest["command_inventory"]["stages"][0]["name"] == "render_protocol"


def test_causal_controller_dispatch_alias_mapping_and_guard(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)
    campaign_root = tmp_path / "scratch" / "20260501_100002"
    campaign_root.mkdir(parents=True, exist_ok=True)

    assert module._coerce_stage_action("render_protocol") == "render_protocol"
    assert module._coerce_stage_action("shared") == "shared_initial"
    assert module._coerce_stage_action("al_01") == "al-step-01-submit"
    assert module._coerce_stage_action("lhs-step-04") == "lhs-step-04-submit"
    assert module._coerce_stage_action("al-step-02-select-render") == "al-step-02-select-render"
    assert module._coerce_stage_action("al_06") == "al-step-06-submit"
    assert module._coerce_stage_action("al-step-06-select-render", step_count=6) == "al-step-06-select-render"
    assert module._coerce_stage_action("escalate") == "escalation"
    with pytest.raises(ValueError, match="exceeds configured step-count"):
        module._coerce_stage_action("al-step-06-submit", step_count=5)

    called: list[str] = []

    def fake_dispatch(*, action: str, campaign_root: Path, args) -> list[str]:
        called.append(action)
        return ["noop://command"]

    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(module, "_build_dispatcher", fake_dispatch)
        module.main(
            [
                "--stage-action",
                "shared",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
        )
        module.main(
            [
                "--stage-action",
                "al_01",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
        )
        module.main(
            [
                "--stage-action",
                "lhs_04",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
        )
        module.main(
            [
                "--stage-action",
                "al-step-06-select-render",
                "--step-count",
                "6",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
        )

    assert called == ["shared_initial", "al-step-01-submit", "lhs-step-04-submit", "al-step-06-select-render"]

    with pytest.raises(ValueError, match="--execute"):
        module.main(
            [
                "--stage-action",
                "shared",
                "--campaign-root",
                str(campaign_root),
            ]
        )


def test_causal_controller_stage_dispatch_validates_root_and_execution_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "scratch" / "20260501_100003"
    campaign_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="--campaign-root"):
        module.main(["--stage-action", "shared", "--execute"])
    with pytest.raises(ValueError, match="--campaign-root"):
        module.main(["--stage-action", "shared", "--campaign-root", "   ", "--execute"])

    dispatched: list[tuple[str, Path]] = []
    runs: list[tuple[str, bool, bool]] = []

    def fake_dispatch(*, action: str, campaign_root: Path, args) -> list[str]:
        dispatched.append((action, campaign_root))
        return ["noop://command"]

    def fake_run(command: str, *, shell: bool, check: bool) -> None:
        runs.append((command, shell, check))

    monkeypatch.setattr(module, "_build_dispatcher", fake_dispatch)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert (
        module.main(
            [
                "--stage-action",
                "shared",
                "--campaign-root",
                str(campaign_root),
                "--execution-mode",
                "execute",
                "--run-commands",
            ]
        )
        == 0
    )

    assert dispatched == [("shared_initial", campaign_root)]
    assert runs == [("noop://command", True, True)]


def test_causal_controller_default_render_only_and_no_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    calls: list[tuple[str, bool, bool]] = []

    def fake_run(command: str, *, shell: bool, check: bool) -> None:
        calls.append((command, shell, check))
        return None  # pragma: no cover

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code = module.main(
        [
            "--timestamp",
            "20260501_100003",
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--vault-root",
            str(tmp_path / "vault"),
            "--force-grid",
            str(force_grid),
            "--run-commands",
        ]
    )
    assert exit_code == 0
    assert calls == []

    manifest = json.loads(
        (tmp_path / "scratch" / "20260501_100003" / module.CONTROLLER_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    stage_commands = [command for stage in manifest["command_inventory"]["stages"] for command in stage["commands"]]
    assert all("EXECUTION_MODE=render-only" in command for command in stage_commands)
    assert not any("EXECUTION_MODE=execute" in command for command in stage_commands)


def test_causal_controller_execute_inventory_and_execute_guard(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    campaign_root = tmp_path / "scratch" / "20260501_100004"
    campaign_root.mkdir(parents=True, exist_ok=True)

    _build_ingest_audit(campaign_root / "ingest" / module.INGEST_REPORT_FILENAME, clean=True, shortfall_count=0)

    result = module.build_emb_34um_causal_validation_controller(
        timestamp="20260501_100004",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-execute",
        execute=True,
        dry_run=True,
    )

    command_inventory = {item["name"]: item["commands"] for item in result["manifest"]["command_inventory"]["stages"]}
    for stage_name in ("shared_initial", "lhs-step-05-submit", "analyze"):
        assert command_inventory[stage_name]
        assert all("EXECUTION_MODE=execute" in command for command in command_inventory[stage_name])

    assert command_inventory["al-step-01-submit"] == []

    with pytest.raises(ValueError, match="--execute"):
        module.main(
            [
                "--stage-action",
                "analyze",
                "--campaign-root",
                str(campaign_root),
            ]
        )

    _build_ingest_audit(campaign_root / "ingest" / module.INGEST_REPORT_FILENAME, clean=False)
    with pytest.raises(ValueError, match="requires a clean ingestion audit"):
        module.main(
            [
                "--stage-action",
                "analyze",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
        )

    _build_ingest_audit(campaign_root / "ingest" / module.INGEST_REPORT_FILENAME, clean=True, shortfall_count=0)
    assert module.main(
        [
            "--stage-action",
            "analyze",
            "--campaign-root",
            str(campaign_root),
            "--execute",
        ]
    ) == 0
    assert module.main(
        [
            "--stage-action",
            "analyze",
            "--campaign-root",
            str(campaign_root),
            "--execution-mode",
            "execute",
        ]
    ) == 0


def test_causal_controller_manifest_to_command_integrity(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    result = module.build_emb_34um_causal_validation_controller(
        timestamp="20260501_100005",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=5,
        step_count=5,
        walltime="00:45:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-integrity",
        execute=False,
        dry_run=True,
    )

    stages = {stage["name"]: stage for stage in result["manifest"]["stages"]}
    for name in ("shared_initial", "validation", "lhs-step-05-submit"):
        assert len(stages[name]["commands"]) == 5
        for cmd in (entry["command"] for entry in stages[name]["commands"]):
            assert "--batch-candidate-count 100" in cmd
            assert "--array=" in cmd
            assert "--time=00:45:00" in cmd
            assert "CONCURRENT_JOBS=30" in cmd
            assert "RETRY_LIMIT=3" in cmd

    assert len(stages["al-step-01-select-render"]["commands"]) == 1
    assert "al-step-01-select-render" in stages["al-step-01-select-render"]["commands"][0]["command"]
    assert stages["al-step-01-submit"]["status"] == "blocked"
    assert stages["al-step-01-submit"]["commands"] == []
    assert stages["al-step-01-submit"]["blockers"] == ["selection_output_batch_summary_required_before_submit"]
    assert any(
        path.endswith("dpd_sampling_batch_manifest.json")
        for path in stages["shared_initial"]["expected_output_roots"]
    )
    assert not any(
        path.endswith("emb_34um_causal_validation_batch_manifest.json")
        for path in stages["shared_initial"]["expected_output_roots"]
    )

    target = result["manifest"]["target_curve_counts"]
    assert target["requested_total_curves"] == 5 * target["per_seed"]["per_seed_total"]
    assert target["requested_total_curves"] == 5 * module.EMB_34UM_CAUSAL_VALIDATION_REQUIRED_CURVES_PER_SEED


def test_causal_controller_target_curve_counts_use_configured_step_count(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    result = module.build_emb_34um_causal_validation_controller(
        timestamp="20260501_100006",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=6,
        walltime="00:45:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-integrity",
        execute=False,
        dry_run=True,
    )

    target = result["manifest"]["target_curve_counts"]
    assert target["per_seed"]["per_seed_total"] == 100 + 100 + 6 * 100 * 2
    assert target["requested_total_curves"] == 3 * target["per_seed"]["per_seed_total"]
    assert target["three_seed_target_curves"] == 3 * target["per_seed"]["per_seed_total"]
    assert target["five_seed_target_curves"] == 5 * target["per_seed"]["per_seed_total"]


def test_causal_controller_rewrites_submit_walltime_from_campaign_inventory() -> None:
    module = _load_controller_module()
    command = (
        "sbatch --parsable --time=00:30:00 --array=0-99%30 "
        "--export=EXECUTION_MODE=render-only scripts/platforms/karolina/sbatch/emb_34um_active_learning_array.sbatch"
    )

    rewritten = module._build_stage_command(
        {"candidate_count": 100, "command": command},
        execution_mode="execute",
        walltime="01:00:00",
    )

    assert "--time=01:00:00" in rewritten
    assert "--time=00:30:00" not in rewritten
    assert "EXECUTION_MODE=execute" in rewritten
    assert "--batch-candidate-count 100" in rewritten


def test_causal_controller_ingest_command_targets_real_script(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)
    timestamp = "20260501_100008"
    campaign_root = tmp_path / "scratch" / timestamp

    result = module.build_emb_34um_causal_validation_controller(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-ingest",
        execute=False,
        dry_run=True,
    )
    stage_commands = {
        item["name"]: item["commands"][0]["command"]
        for item in result["manifest"]["stages"]
        if item["name"] == "ingest"
    }
    ingest_command = stage_commands["ingest"]
    assert "python -c" not in ingest_command
    assert "ingest_emb_34um_causal_validation.py" in ingest_command

    parts = shlex.split(ingest_command)
    assert "--campaign-manifest" in parts
    assert str(campaign_root / module.EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME) in parts
    assert "--output-root" in parts
    assert str(campaign_root / "ingest") in parts
    assert "--replacement-manifest-output" in parts
    assert str(
        campaign_root / "ingest" / module.INGEST_REPLACEMENT_PLAN_FILENAME
    ) in parts


def test_causal_controller_select_render_emits_manifest_and_unblocks_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _patch_fast_adaptive_selection(monkeypatch)
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)
    timestamp = "20260501_100009"
    campaign_root = tmp_path / "scratch" / timestamp

    initial = module.build_emb_34um_causal_validation_controller(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-select",
        execute=False,
        dry_run=True,
    )
    initial_stages = {stage["name"]: stage for stage in initial["manifest"]["stages"]}
    assert initial_stages["al-step-01-submit"]["status"] == "blocked"

    with pytest.raises(ValueError, match="requires completed prior curves"):
        module.main(
            [
                "--stage-action",
                "al-step-01-select-render",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
        )

    for seed in range(1, 4):
        shared_summary = campaign_root / f"seed-{seed:03d}" / "shared_initial" / module.BATCH_SUMMARY_FILENAME
        _materialize_completed_batch_outputs(shared_summary)

    assert module.main(
        [
            "--stage-action",
            "al-step-01-select-render",
            "--campaign-root",
            str(campaign_root),
            "--execute",
        ]
    ) == 0

    for seed in range(1, 4):
        stage_root = campaign_root / f"seed-{seed:03d}" / "al-step-01"
        selection_manifest = stage_root / module.SELECTION_MANIFEST_FILENAME
        selection_summary = stage_root / module.SELECTION_BATCH_SUMMARY_FILENAME
        batch_summary = stage_root / module.BATCH_SUMMARY_FILENAME
        assert selection_manifest.is_file()
        assert selection_summary.is_file()
        assert batch_summary.is_file()
        payload = json.loads(selection_manifest.read_text(encoding="utf-8"))
        assert payload["candidate_count"] == 100
        assert len(payload["candidate_records"]) == 100
        assert len(payload["candidate_reserve"]) == 400
        assert payload["candidate_reserve_count"] == 400
        assert payload["training_count"] == 100
        assert payload["candidate_pool_count"] == 500
        assert payload["selected_count"] == 100
        assert payload["selection_policy"]["acquisition_count"] == 80
        assert payload["selection_policy"]["exploration_count"] == 20
        assert payload["selection_policy"]["acquisition_policy"] == "dnn_ensemble_disagreement_plus_diversity"
        assert payload["selection_policy"]["exploration_policy"] == "deterministic_pool_exploration"
        assert payload["status"] == "selection_rendered"
        assert isinstance(payload["model_selection"], dict)
        assert payload["blockers"] == []
        assert {item["candidate_pool_id"] for item in payload["candidate_records"]}.isdisjoint(
            {item["candidate_pool_id"] for item in payload["candidate_reserve"]}
        )

        summary_payload = json.loads(selection_summary.read_text(encoding="utf-8"))
        assert summary_payload["status"] == "selection_rendered"
        assert summary_payload["candidate_count"] == 100
        assert summary_payload["training_count"] == 100
        assert summary_payload["candidate_pool_count"] == 500
        assert summary_payload["selected_count"] == 100
        assert summary_payload["candidate_reserve_count"] == 400
        assert summary_payload["acquisition_count"] == 80
        assert summary_payload["exploration_count"] == 20
        assert summary_payload["acquisition_policy"] == "dnn_ensemble_disagreement_plus_diversity"
        assert summary_payload["exploration_policy"] == "deterministic_pool_exploration"
        assert isinstance(summary_payload["model_selection"], dict)
        assert summary_payload["batch_summary_path"] == str(batch_summary)

    updated = module.build_emb_34um_causal_validation_controller(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-select",
        execute=False,
        dry_run=True,
    )
    updated_stages = {stage["name"]: stage for stage in updated["manifest"]["stages"]}
    assert updated_stages["al-step-01-submit"]["status"] == "planned"
    assert len(updated_stages["al-step-01-submit"]["commands"]) == 3


def test_causal_controller_adaptive_selection_uses_replacement_records_for_prior_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _patch_fast_adaptive_selection(monkeypatch)
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    timestamp = "20260501_200001"
    campaign_root = tmp_path / "scratch" / timestamp

    module.build_emb_34um_causal_validation_controller(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-replacement-prior",
        execute=False,
        dry_run=True,
    )

    for seed in range(1, 4):
        shared_summary = campaign_root / f"seed-{seed:03d}" / "shared_initial" / module.BATCH_SUMMARY_FILENAME
        _write_batch_summary(shared_summary, candidate_count=100)
        _materialize_completed_batch_outputs(shared_summary)
        _configure_al_step01_with_replacement_records(
            campaign_root=campaign_root,
            module=module,
            seed=seed,
            missing_candidate_ids=("c001", "c002"),
            materialize_replacement_outputs=True,
        )

    assert module.main(
        [
            "--stage-action",
            "al-step-02-select-render",
            "--campaign-root",
            str(campaign_root),
            "--execute",
        ]
    ) == 0

    for seed in range(1, 4):
        selection_summary = json.loads(
            (campaign_root / f"seed-{seed:03d}" / "al-step-02" / module.SELECTION_BATCH_SUMMARY_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert selection_summary["status"] == "selection_rendered"
        assert selection_summary["training_count"] == 200
        assert selection_summary["training_record_count"] == 200


def test_causal_controller_prior_step_missing_main_records_are_skipped_when_replaced(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    timestamp = "20260501_200002"
    campaign_root = tmp_path / "scratch" / timestamp

    module.build_emb_34um_causal_validation_controller(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-replacement-direct",
        execute=False,
        dry_run=True,
    )

    shared_summary = campaign_root / "seed-001" / "shared_initial" / module.BATCH_SUMMARY_FILENAME
    _write_batch_summary(shared_summary, candidate_count=100)
    _materialize_completed_batch_outputs(shared_summary)
    replacement_info = _configure_al_step01_with_replacement_records(
        campaign_root=campaign_root,
        module=module,
        seed=1,
        missing_candidate_ids=("c001", "c002"),
        materialize_replacement_outputs=True,
    )

    prior_records = module._completed_records_from_batch_summary(
        summary_path=campaign_root / "seed-001" / "al-step-01" / module.BATCH_SUMMARY_FILENAME,
        strategy="al",
        step=1,
        include_replacement_records=True,
    )
    assert len(prior_records) == 100
    ids = {record["candidate_id"] for record in prior_records}
    assert "c001" not in ids and "c002" not in ids

    replacement_manifest = json.loads(Path(replacement_info["replacement_manifest_path"]).read_text(encoding="utf-8"))
    replacement_ids = {str(row["replacement_candidate_id"]) for row in replacement_manifest["replacement_records"]}
    assert replacement_ids <= ids
    assert all(item in ids for item in replacement_ids)
    assert any(record.get("is_replacement") is True for record in prior_records)


def test_causal_controller_incomplete_prior_replacement_results_fail_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()
    _patch_fast_adaptive_selection(monkeypatch)
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    timestamp = "20260501_200003"
    campaign_root = tmp_path / "scratch" / timestamp

    module.build_emb_34um_causal_validation_controller(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-replacement-incomplete",
        execute=False,
        dry_run=True,
    )

    shared_summary = campaign_root / "seed-001" / "shared_initial" / module.BATCH_SUMMARY_FILENAME
    _write_batch_summary(shared_summary, candidate_count=100)
    _materialize_completed_batch_outputs(shared_summary)
    _configure_al_step01_with_replacement_records(
        campaign_root=campaign_root,
        module=module,
        seed=1,
        missing_candidate_ids=("c001", "c002"),
        materialize_replacement_outputs=False,
    )
    for seed in (2, 3):
        shared_summary = campaign_root / f"seed-{seed:03d}" / "shared_initial" / module.BATCH_SUMMARY_FILENAME
        _write_batch_summary(shared_summary, candidate_count=100)
        _materialize_completed_batch_outputs(shared_summary)
        _configure_al_step01_with_replacement_records(
            campaign_root=campaign_root,
            module=module,
            seed=seed,
            missing_candidate_ids=("c001", "c002"),
            materialize_replacement_outputs=True,
        )

    with pytest.raises(ValueError, match="Replacement result is missing"):
        module.main(
            [
                "--stage-action",
                "al-step-02-select-render",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
        )


def test_causal_controller_selection_policy_generates_deterministic_reserve_payload(tmp_path: Path) -> None:
    module = _load_controller_module()

    manifest = {
        "run_id_prefix": "emb-34um-causal-reserve",
        "vault_root_timestamp": tmp_path / "vault",
    }
    campaign_root = tmp_path / "scratch" / "deterministic-selection"
    placeholder = {"selection_seed": 1234, "acquisition_count": 2, "exploration_count": 1}
    scored_rows = [
        {
            "candidate_id": "pool-c001",
            "ka": 10.0,
            "kb": 1.0,
            "acquisition_score": 5.0,
            "ensemble_disagreement": 0.2,
        },
        {
            "candidate_id": "pool-c002",
            "ka": 8.0,
            "kb": 2.0,
            "acquisition_score": 4.0,
            "ensemble_disagreement": 0.9,
        },
        {
            "candidate_id": "pool-c003",
            "ka": 20.0,
            "kb": 2.0,
            "acquisition_score": 3.0,
            "ensemble_disagreement": 0.1,
        },
        {
            "candidate_id": "pool-c004",
            "ka": 5.0,
            "kb": 10.0,
            "acquisition_score": 2.0,
            "ensemble_disagreement": 0.8,
        },
        {
            "candidate_id": "pool-c005",
            "ka": 6.0,
            "kb": 6.0,
            "acquisition_score": 1.0,
            "ensemble_disagreement": 0.7,
        },
    ]
    force_grid = (10.0, 20.0, 30.0)

    _, selected_one, reserve_one = module._selection_candidates_from_scored_rows(
        manifest=manifest,
        placeholder=placeholder,
        scored_rows=scored_rows,
        campaign_root=campaign_root,
        seed=1,
        step=1,
        force_grid=force_grid,
    )
    _, selected_two, reserve_two = module._selection_candidates_from_scored_rows(
        manifest=manifest,
        placeholder=placeholder,
        scored_rows=scored_rows,
        campaign_root=campaign_root,
        seed=1,
        step=1,
        force_grid=force_grid,
    )

    assert [row["selection_source"] for row in selected_one[:2]] == ["ensemble_disagreement_diversity"] * 2
    assert selected_one[2]["selection_source"] == "exploration"
    assert [row["candidate_pool_id"] for row in selected_one] == ["pool-c001", "pool-c002", "pool-c004"]
    assert [row["candidate_pool_id"] for row in reserve_one] == ["pool-c003", "pool-c005"]
    assert reserve_one == reserve_two
    assert selected_one == selected_two
    assert {row["candidate_pool_id"] for row in selected_one}.isdisjoint(
        {row["candidate_pool_id"] for row in reserve_one}
    )
    assert len(reserve_one) == 2
    assert all(
        {
            "candidate_id",
            "candidate_pool_id",
            "selection_source",
            "selection_mode",
            "acquisition_score",
            "ensemble_disagreement",
            "selection_order",
        }
        <= row.keys()
        for row in reserve_one
    )


def test_causal_controller_replacement_manifest_uses_next_unused_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "scratch" / "replace-step-01"
    stage_root = campaign_root / "seed-001" / "al-step-01"
    _write_design_manifest_for_replacement_testing(
        campaign_root=campaign_root,
        manifest_filename=module.EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    )
    _write_selection_manifest_with_reserve(
        stage_root=stage_root,
        selection_manifest_filename=module.SELECTION_MANIFEST_FILENAME,
        reserve_records=[
            {
                "candidate_id": "reserve-001",
                "candidate_pool_id": "pool-reserve-001",
                "candidate_hash": "h-001",
                "family": "emb",
                "experiment": "indentation",
                "ka": 11.0,
                "kb": 22.0,
                "force_grid": [10.0, 20.0, 30.0],
                "selection_seed": 1001,
                "selection_order": 1,
                "selection_pool_rank": 101,
                "order": 1,
                "acquisition_score": 0.95,
                "ensemble_disagreement": 0.2,
            },
            {
                "candidate_id": "reserve-002",
                "candidate_pool_id": "pool-reserve-002",
                "candidate_hash": "h-002",
                "family": "emb",
                "experiment": "indentation",
                "ka": 12.0,
                "kb": 24.0,
                "force_grid": [10.0, 20.0, 30.0],
                "selection_seed": 1002,
                "selection_order": 2,
                "selection_pool_rank": 102,
                "order": 2,
                "acquisition_score": 0.85,
                "ensemble_disagreement": 0.1,
            },
        ],
    )

    calls: list[dict[str, Any]] = []

    def fake_render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        calls.append(
            {
                "candidate_ids": tuple(candidate.candidate_id for candidate in candidates),
                "batch_root": str(batch_root),
                "run_id": run_id,
                "batch_id": batch_id,
                "walltime": walltime,
                "concurrent_jobs": concurrent_jobs,
                "retry_limit": retry_limit,
            }
        )
        return {
            "candidate_count": len(candidates),
            "expected_output_roots": [str(batch_root / "emb" / candidate.candidate_id) for candidate in candidates],
            "rendered_candidate_manifests": [str(batch_root / "emb" / candidate.candidate_id / "manifest.json") for candidate in candidates],
        }

    monkeypatch.setattr(module._prepare, "_render_batch", fake_render_batch)

    first = module._build_replacement_candidates_from_reserve(
        campaign_root=campaign_root,
        seed=1,
        step=1,
        failed_candidate_ids=("failed-al-001",),
    )
    assert calls[-1]["candidate_ids"] == ("reserve-001",)
    first_summary = json.loads(Path(first["replacement_batch_summary_path"]).read_text(encoding="utf-8"))
    assert first_summary["replacement_records"][0]["replacement_candidate_id"] == "reserve-001"
    first_summary = json.loads(Path(first["replacement_manifest_path"]).read_text(encoding="utf-8"))
    assert first_summary["replacement_records"][0]["replacement_candidate_id"] == "reserve-001"

    second = module._build_replacement_candidates_from_reserve(
        campaign_root=campaign_root,
        seed=1,
        step=1,
        failed_candidate_ids=("failed-al-002",),
    )
    assert calls[-1]["candidate_ids"] == ("reserve-002",)
    second_summary = json.loads(Path(second["replacement_batch_summary_path"]).read_text(encoding="utf-8"))
    assert second_summary["replacement_records"][0]["replacement_candidate_id"] == "reserve-002"

    manifest = json.loads(Path(second["replacement_manifest_path"]).read_text(encoding="utf-8"))
    assert [row["replacement_candidate_id"] for row in manifest["replacement_records"]] == ["reserve-001", "reserve-002"]


def test_causal_controller_replacement_path_raises_when_no_reserve_remains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "scratch" / "replace-step-02"
    stage_root = campaign_root / "seed-001" / "al-step-01"

    _write_design_manifest_for_replacement_testing(
        campaign_root=campaign_root,
        manifest_filename=module.EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    )
    _write_selection_manifest_with_reserve(
        stage_root=stage_root,
        selection_manifest_filename=module.SELECTION_MANIFEST_FILENAME,
        reserve_records=[
            {
                "candidate_id": "reserve-001",
                "candidate_pool_id": "pool-reserve-001",
                "candidate_hash": "h-001",
                "family": "emb",
                "experiment": "indentation",
                "ka": 11.0,
                "kb": 22.0,
                "force_grid": [10.0, 20.0, 30.0],
                "selection_seed": 1001,
                "selection_order": 1,
                "selection_pool_rank": 101,
            }
        ],
    )

    def fake_render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        return {
            "candidate_count": len(candidates),
            "expected_output_roots": [str(batch_root / "emb" / candidate.candidate_id) for candidate in candidates],
            "rendered_candidate_manifests": [str(batch_root / "emb" / candidate.candidate_id / "manifest.json") for candidate in candidates],
        }

    monkeypatch.setattr(module._prepare, "_render_batch", fake_render_batch)

    assert module._build_replacement_candidates_from_reserve(
        campaign_root=campaign_root,
        seed=1,
        step=1,
        failed_candidate_ids=("failed-al-001",),
    )["replacement_records"]

    with pytest.raises(ValueError, match="No unused reserve candidates remain"):
        module._build_replacement_candidates_from_reserve(
            campaign_root=campaign_root,
            seed=1,
            step=1,
            failed_candidate_ids=("failed-al-002",),
        )


def test_causal_controller_lhs_replacement_manifest_records_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "scratch" / "lhs-replace-metadata"
    _write_lhs_design_manifest_for_replacement_testing(
        campaign_root=campaign_root,
        step_count=3,
        candidates_per_step=4,
        run_id_prefix="emb-34um-causal-lhs-test",
    )

    calls: list[dict[str, str]] = []

    def fake_render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        calls.append(
            {
                "candidate_ids": tuple(candidate.candidate_id for candidate in candidates),
                "batch_root": str(batch_root),
                "run_id": run_id,
            }
        )
        return {
            "candidate_count": len(candidates),
            "expected_output_roots": [str(batch_root / "emb" / candidate.candidate_id) for candidate in candidates],
            "rendered_candidate_manifests": [str(batch_root / "emb" / candidate.candidate_id / "manifest.json") for candidate in candidates],
        }

    monkeypatch.setattr(module._prepare, "_render_batch", fake_render_batch)

    result = module._build_lhs_replacement_candidates_from_mode(
        campaign_root=campaign_root,
        seed=1,
        step=1,
        failed_candidate_ids=("emb-34um-causal-lhs-test-seed001-lhs-step01-c001",),
        mode="next-lhs",
    )

    manifest = json.loads(Path(result["replacement_manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["campaign_root"] == str(campaign_root)
    assert manifest["seed"] == 1
    assert manifest["step"] == 1
    assert manifest["stage"] == "lhs-step-01"
    assert manifest["status"] == "replacement_rendered"
    assert manifest["replacement_policy"] == "next-lhs"
    assert manifest["selection_manifest_path"] == str(
        campaign_root / "seed-001" / "lhs-step-01" / module.SELECTION_MANIFEST_FILENAME
    )
    assert manifest["replacement_count"] == 1
    assert len(manifest["replacement_records"]) == 1
    record = manifest["replacement_records"][0]
    assert record["seed"] == 1
    assert record["step"] == 1
    assert record["failed_candidate_id"] == "emb-34um-causal-lhs-test-seed001-lhs-step01-c001"
    assert record["replacement_candidate_id"] == "emb-34um-causal-lhs-test-seed001-lhs-step01-c005"
    assert isinstance(record["ka"], float)
    assert isinstance(record["kb"], float)
    assert record["replacement_policy"] == "next-lhs"
    assert record["output_root"] == str(
        campaign_root / "seed-001" / "lhs-step-01" / "replacement" / "batch-001" / "emb" / "emb-34um-causal-lhs-test-seed001-lhs-step01-c005"
    )
    assert record["vault_output_root"] == str(
        campaign_root / "vault" / "seed-001" / "lhs-step-01" / "replacement" / "batch-001" / "emb" / "emb-34um-causal-lhs-test-seed001-lhs-step01-c005"
    )
    assert isinstance(record["replacement_candidate_hash"], str)
    assert len(record["replacement_candidate_hash"]) == 16

    assert calls[-1] == {
        "candidate_ids": ("emb-34um-causal-lhs-test-seed001-lhs-step01-c005",),
        "batch_root": str(campaign_root / "seed-001" / "lhs-step-01" / "replacement" / "batch-001"),
        "run_id": "emb-34um-causal-lhs-test-seed001-lhs-step01-replacement-001",
    }

    batch_summary = json.loads(Path(result["replacement_batch_summary_path"]).read_text(encoding="utf-8"))
    assert batch_summary["replacement_batch_root"] == str(
        campaign_root / "seed-001" / "lhs-step-01" / "replacement" / "batch-001"
    )
    assert batch_summary["replacement_batch_index"] == 1
    assert batch_summary["replacement_policy"] == "next-lhs"
    assert batch_summary["replacement_records"] == [record]
    assert batch_summary["replacement_record_count"] == 1


def test_causal_controller_lhs_replacement_repeatable_across_campaign_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_controller_module()

    def fake_render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        return {
            "candidate_count": len(candidates),
            "expected_output_roots": [str(batch_root / "emb" / candidate.candidate_id) for candidate in candidates],
            "rendered_candidate_manifests": [str(batch_root / "emb" / candidate.candidate_id / "manifest.json") for candidate in candidates],
        }

    monkeypatch.setattr(module._prepare, "_render_batch", fake_render_batch)

    campaign_roots = []
    run_id_prefix = "emb-34um-causal-lhs-repeat"
    failed_candidate_id = f"{run_id_prefix}-seed001-lhs-step01-c001"
    for suffix in ("one", "two"):
        campaign_root = tmp_path / f"scratch-{suffix}" / "run"
        _write_lhs_design_manifest_for_replacement_testing(
            campaign_root=campaign_root,
            step_count=4,
            candidates_per_step=4,
            run_id_prefix=run_id_prefix,
        )
        campaign_roots.append(campaign_root)

    next_lhs_records: list[list[dict[str, Any]]] = []
    for campaign_root in campaign_roots:
        result = module._build_lhs_replacement_candidates_from_mode(
            campaign_root=campaign_root,
            seed=1,
            step=1,
            failed_candidate_ids=(failed_candidate_id,),
            mode="next-lhs",
        )
        next_lhs_records.append(json.loads(Path(result["replacement_batch_summary_path"]).read_text(encoding="utf-8"))["replacement_records"])

    rerun_campaign_roots = []
    for suffix in ("one", "two"):
        campaign_root = tmp_path / f"scratch-rerun-{suffix}" / "run"
        _write_lhs_design_manifest_for_replacement_testing(
            campaign_root=campaign_root,
            step_count=4,
            candidates_per_step=4,
            run_id_prefix=run_id_prefix,
        )
        rerun_campaign_roots.append(campaign_root)

    rerun_records: list[list[dict[str, Any]]] = []
    for campaign_root in rerun_campaign_roots:
        result = module._build_lhs_replacement_candidates_from_mode(
            campaign_root=campaign_root,
            seed=1,
            step=1,
            failed_candidate_ids=(failed_candidate_id,),
            mode="rerun-original",
        )
        rerun_records.append(json.loads(Path(result["replacement_batch_summary_path"]).read_text(encoding="utf-8"))["replacement_records"])

    next_lhs_output_suffix = (
        "seed-001/lhs-step-01/replacement/batch-001/emb/"
        "emb-34um-causal-lhs-repeat-seed001-lhs-step01-c005"
    )
    assert next_lhs_records[0][0]["replacement_candidate_id"] == next_lhs_records[1][0]["replacement_candidate_id"]
    assert next_lhs_records[0][0]["ka"] == next_lhs_records[1][0]["ka"]
    assert next_lhs_records[0][0]["kb"] == next_lhs_records[1][0]["kb"]
    assert next_lhs_records[0][0]["replacement_policy"] == next_lhs_records[1][0]["replacement_policy"]
    assert next_lhs_records[0][0]["failed_candidate_id"] == next_lhs_records[1][0]["failed_candidate_id"]
    assert next_lhs_records[0][0]["seed"] == next_lhs_records[1][0]["seed"]
    assert next_lhs_records[0][0]["step"] == next_lhs_records[1][0]["step"]
    assert next_lhs_records[0][0]["output_root"].endswith(next_lhs_output_suffix)
    assert next_lhs_records[1][0]["output_root"].endswith(next_lhs_output_suffix)
    assert next_lhs_records[0][0]["output_root"] != next_lhs_records[1][0]["output_root"]
    assert next_lhs_records[0][0]["vault_output_root"] != next_lhs_records[1][0]["vault_output_root"]

    rerun_output_suffix = (
        "seed-001/lhs-step-01/replacement/batch-001/emb/"
        "emb-34um-causal-lhs-repeat-seed001-lhs-step01-c001"
    )

    assert rerun_records[0][0]["replacement_candidate_id"] == rerun_records[1][0]["replacement_candidate_id"] == failed_candidate_id
    assert rerun_records[0][0]["ka"] == rerun_records[1][0]["ka"]
    assert rerun_records[0][0]["kb"] == rerun_records[1][0]["kb"]
    assert rerun_records[0][0]["replacement_policy"] == rerun_records[1][0]["replacement_policy"] == "rerun-original"
    assert rerun_records[0][0]["failed_candidate_id"] == rerun_records[1][0]["failed_candidate_id"]
    assert rerun_records[0][0]["seed"] == rerun_records[1][0]["seed"]
    assert rerun_records[0][0]["step"] == rerun_records[1][0]["step"]
    assert rerun_records[0][0]["output_root"].endswith(rerun_output_suffix)
    assert rerun_records[1][0]["output_root"].endswith(rerun_output_suffix)


def test_causal_controller_lhs_next_replacement_can_supersede_timed_out_exact_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "scratch" / "lhs-next-after-rerun"
    run_id_prefix = "emb-34um-causal-lhs-next-after-rerun"
    failed_candidate_id = f"{run_id_prefix}-seed001-lhs-step01-c001"
    _write_lhs_design_manifest_for_replacement_testing(
        campaign_root=campaign_root,
        step_count=4,
        candidates_per_step=4,
        run_id_prefix=run_id_prefix,
    )

    def fake_render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        return {
            "candidate_count": len(candidates),
            "expected_output_roots": [str(batch_root / "emb" / candidate.candidate_id) for candidate in candidates],
            "rendered_candidate_manifests": [str(batch_root / "emb" / candidate.candidate_id / "manifest.json") for candidate in candidates],
        }

    monkeypatch.setattr(module._prepare, "_render_batch", fake_render_batch)

    first = module._build_lhs_replacement_candidates_from_mode(
        campaign_root=campaign_root,
        seed=1,
        step=1,
        failed_candidate_ids=(failed_candidate_id,),
        mode="rerun-original",
    )
    second = module._build_lhs_replacement_candidates_from_mode(
        campaign_root=campaign_root,
        seed=1,
        step=1,
        failed_candidate_ids=(failed_candidate_id,),
        mode="next-lhs",
    )

    first_summary = json.loads(Path(first["replacement_batch_summary_path"]).read_text(encoding="utf-8"))
    second_summary = json.loads(Path(second["replacement_batch_summary_path"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(second["replacement_manifest_path"]).read_text(encoding="utf-8"))

    assert first_summary["replacement_batch_index"] == 1
    assert first_summary["replacement_records"][0]["replacement_candidate_id"] == failed_candidate_id
    assert second_summary["replacement_batch_index"] == 2
    assert second_summary["replacement_records"][0]["replacement_policy"] == "next-lhs"
    assert second_summary["replacement_records"][0]["failed_candidate_id"] == failed_candidate_id
    assert second_summary["replacement_records"][0]["replacement_candidate_id"] == (
        f"{run_id_prefix}-seed001-lhs-step01-c006"
    )
    assert manifest["replacement_count"] == 2
    assert [record["replacement_policy"] for record in manifest["replacement_records"]] == [
        "rerun-original",
        "next-lhs",
    ]


def test_causal_controller_al_submit_stage_dispatch_falls_back_to_design_when_controller_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)
    timestamp = "20260501_100010"
    campaign_root = tmp_path / "scratch" / timestamp

    initial = module.build_emb_34um_causal_validation_controller(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-submit-fallback",
        execute=False,
        dry_run=True,
    )
    stages = {stage["name"]: stage for stage in initial["manifest"]["stages"]}
    assert stages["al-step-01-submit"]["status"] == "blocked"
    assert stages["al-step-01-submit"]["commands"] == []

    for seed in range(1, 4):
        _write_batch_summary(
            campaign_root / f"seed-{seed:03d}" / "al-step-01" / module.BATCH_SUMMARY_FILENAME,
            candidate_count=3,
        )

    calls: list[str] = []

    def fake_run(command: str, *, shell: bool, check: bool) -> None:
        calls.append(command)
        return None  # pragma: no cover

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert (
        module.main(
            [
                "--stage-action",
                "al_01",
                "--campaign-root",
                str(campaign_root),
                "--execute",
                "--run-commands",
            ]
        )
        == 0
    )
    assert len(calls) == 3
    assert all("sbatch --parsable" in call for call in calls)
    assert all("EXECUTION_MODE=execute" in call for call in calls)


def test_causal_controller_al_submit_stage_dispatch_requires_batch_summaries(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)
    timestamp = "20260501_100011"
    campaign_root = tmp_path / "scratch" / timestamp

    module.build_emb_34um_causal_validation_controller(
        timestamp=timestamp,
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-submit-gated",
        execute=False,
        dry_run=True,
    )

    for seed in range(1, 3):
        _write_batch_summary(
            campaign_root / f"seed-{seed:03d}" / "al-step-01" / module.BATCH_SUMMARY_FILENAME,
            candidate_count=3,
        )

    with pytest.raises(ValueError, match="rendered batch summaries.*seed-003"):
        module.main(
            [
                "--stage-action",
                "al-step-01-submit",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
        )


def test_causal_controller_ingest_audit_gates(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    result = module.build_emb_34um_causal_validation_controller(
        timestamp="20260501_100006",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-audit",
        execute=False,
        dry_run=True,
    )
    by_name = {stage["name"]: stage for stage in result["manifest"]["stages"]}
    assert by_name["analyze"]["status"] == "blocked"
    assert by_name["escalation"]["status"] == "blocked"

    _build_ingest_audit(tmp_path / "scratch" / "20260501_100006" / "ingest" / module.INGEST_REPORT_FILENAME, clean=True)
    result_with_audit = module.build_emb_34um_causal_validation_controller(
        timestamp="20260501_100006",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=5,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-audit",
        execute=False,
        dry_run=True,
    )
    updated = {stage["name"]: stage for stage in result_with_audit["manifest"]["stages"]}
    assert updated["analyze"]["status"] == "planned"
    analyze_command = updated["analyze"]["commands"][0]["command"]
    analyze_parts = shlex.split(analyze_command)
    assert "--ingestion-manifest" in analyze_parts
    assert "--rows" not in analyze_parts
    analyze_outputs = result_with_audit["manifest"]["expected_output_roots"]["analyze"]
    assert any(path.endswith("emb_34um_causal_validation_report.json") for path in analyze_outputs)
    assert any(path.endswith("emb_34um_causal_validation_summary.csv") for path in analyze_outputs)
    assert any(path.endswith("emb_34um_causal_validation_added_samples_by_step.png") for path in analyze_outputs)
    assert not any("analysis_manifest" in path for path in analyze_outputs)
    assert updated["escalation"]["status"] == "blocked"
    assert updated["escalation"]["blockers"] == ["escalation_not_required_when_seed_count_is_max"]


def test_causal_controller_no_public_terms_in_artifacts(tmp_path: Path) -> None:
    module = _load_controller_module()
    force_grid = tmp_path / "samples_all_custom.dat"
    _write_force_grid(force_grid)

    result = module.build_emb_34um_causal_validation_controller(
        timestamp="20260501_100007",
        scratch_root=tmp_path / "scratch",
        vault_root=tmp_path / "vault",
        force_grid_path=force_grid,
        seed_count=3,
        step_count=5,
        walltime="00:30:00",
        concurrent_jobs=30,
        retry_limit=3,
        run_id_prefix="emb-34um-causal-public",
        execute=False,
        dry_run=True,
    )

    outputs: list[str] = []
    for value in result["manifest"]["expected_output_roots"].values():
        outputs.extend(value)
    blocked_terms = (
        "".join(chr(value) for value in (97, 115, 115, 105, 115, 116, 97, 110, 116)),
        "".join(chr(value) for value in (97, 103, 101, 110, 116)),
        "".join(chr(value) for value in (109, 111, 100, 101, 108)),
    )

    for text in outputs:
        lowered = text.lower()
        for term in blocked_terms:
            assert term not in lowered

    command_text = [command for item in result["manifest"]["command_inventory"]["stages"] for command in item["commands"]]
    for text in command_text:
        lowered = text.lower()
        for term in blocked_terms:
            assert term not in lowered
