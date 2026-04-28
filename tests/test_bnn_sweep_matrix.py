from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _parse_arg(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def _assert_runner_import_without_torch(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root}{':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''}"
    script = f"""
import builtins
import importlib.util

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.', 1)[0] == 'torch':
        raise ModuleNotFoundError('blocked torch import')
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
spec = importlib.util.spec_from_file_location('runner_import_test', {str(path)!r})
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _make_spec(tmp_path: Path, name: str) -> dict[str, str]:
    data = tmp_path / f"{name}.dat"
    dnn = tmp_path / f"{name}.pkl"
    script = tmp_path / f"{name}_train.py"
    for path in (data, dnn, script):
        path.write_text("placeholder", encoding="utf-8")
    return {
        "name": name,
        "modality": "compression",
        "diameter_um": "2.1",
        "data": str(data),
        "dnn_artifact": str(dnn),
        "bnn_artifact": str(tmp_path / f"{name}.pt"),
        "dnn_train_script": str(script),
        "dnn_multi_arch_script": str(script),
        "bnn_train_script": str(script),
        "group_holdout_script": str(script),
    }


def _write_dnn_selection(
    seed_dir: Path,
    *,
    artifact_path: Path,
    architecture: str,
) -> Path:
    seed_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_path.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"val_loss": 0.1}), encoding="utf-8")
    (seed_dir / "selection.json").write_text(
        json.dumps(
            {
                "best_architecture": architecture,
                "best_artifact_path": str(artifact_path),
                "best_report_path": str(report_path),
            }
        ),
        encoding="utf-8",
    )
    return seed_dir / "selection.json"


def test_bnn_sweep_runner_selects_top_architecture_from_full_stage1_grid(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_main_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    called: list[list[str]] = []

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        called.append(command)
        out_path = Path(_parse_arg(command, "--out"))
        report_path = Path(_parse_arg(command, "--report-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("artifact", encoding="utf-8")
        metric = 0.5
        if "stage1__w32_d2__prior0.5" in out_path.stem:
            metric = 0.20
        elif "stage1__w32_d2__prior1" in out_path.stem:
            metric = 0.40
        elif "stage1__w64_d2__prior0.5" in out_path.stem:
            metric = 0.35
        elif "stage1__w64_d2__prior1" in out_path.stem:
            metric = 0.30
        elif "stage2__w32_d2" in out_path.stem:
            metric = 0.11
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"training": {"best_val_rmse": metric, "final_val_rmse": metric + 0.01}}),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "101",
            "--architectures",
            "w32_d2,w64_d2",
            "--stage1-prior-scales",
            "0.5,1.0",
            "--stage1-obs-noise-prior-scales",
            "1.0",
            "--stage1-lrs",
            "0.001",
            "--top-k",
            "1",
            "--prior-scales",
            "0.5",
            "--obs-noise-prior-scales",
            "0.1",
            "--lrs",
            "0.001",
        ]
    )
    assert rc == 0
    selection_path = tmp_path / "out" / "spec_a" / "seed_101" / "selection.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    assert payload["top_architectures"] == ["w32_d2"]
    assert payload["best_candidate_metric"] == pytest.approx(0.11)
    stage1_stems = {Path(_parse_arg(command, "--out")).stem for command in called if "stage1__" in " ".join(command)}
    assert stage1_stems == {
        "stage1__w32_d2__prior0.5__obs1__lr0.001",
        "stage1__w32_d2__prior1__obs1__lr0.001",
        "stage1__w64_d2__prior0.5__obs1__lr0.001",
        "stage1__w64_d2__prior1__obs1__lr0.001",
    }
    stage2_stems = [Path(_parse_arg(command, "--out")).stem for command in called if "stage2__" in " ".join(command)]
    assert stage2_stems == ["stage2__w32_d2__prior0.5__obs0.1__lr0.001"]


def test_bnn_sweep_runner_resume_skips_completed_stage1_grid_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_resume_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    seed_root = tmp_path / "out" / "spec_a" / "seed_20260317"
    artifact_path, report_path = module._candidate_paths(
        seed_root,
        "stage1__w32_d2__prior0.5__obs1__lr0.001",
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("artifact", encoding="utf-8")
    report_path.write_text(
        json.dumps({"training": {"best_val_rmse": 0.15, "final_val_rmse": 0.16}}),
        encoding="utf-8",
    )

    called: list[list[str]] = []

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        called.append(command)
        out_path = Path(_parse_arg(command, "--out"))
        candidate_report = Path(_parse_arg(command, "--report-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("artifact", encoding="utf-8")
        metric = 0.4
        if "stage1__w32_d2__prior1" in out_path.stem:
            metric = 0.25
        elif "stage1__w64_d2__prior0.5" in out_path.stem:
            metric = 0.30
        elif "stage1__w64_d2__prior1" in out_path.stem:
            metric = 0.35
        elif "stage2__w32_d2" in out_path.stem:
            metric = 0.10
        candidate_report.parent.mkdir(parents=True, exist_ok=True)
        candidate_report.write_text(
            json.dumps({"training": {"best_val_rmse": metric, "final_val_rmse": metric + 0.01}}),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "20260317",
            "--architectures",
            "w32_d2,w64_d2",
            "--stage1-prior-scales",
            "0.5,1.0",
            "--stage1-obs-noise-prior-scales",
            "1.0",
            "--stage1-lrs",
            "0.001",
            "--top-k",
            "1",
            "--prior-scales",
            "0.5",
            "--obs-noise-prior-scales",
            "0.1",
            "--lrs",
            "0.001",
        ]
    )
    assert rc == 0
    stage1_stems = {Path(_parse_arg(command, "--out")).stem for command in called if "stage1__" in " ".join(command)}
    assert stage1_stems == {
        "stage1__w32_d2__prior1__obs1__lr0.001",
        "stage1__w64_d2__prior0.5__obs1__lr0.001",
        "stage1__w64_d2__prior1__obs1__lr0.001",
    }
    assert "stage1__w32_d2__prior0.5__obs1__lr0.001" not in stage1_stems
    selection_path = tmp_path / "out" / "spec_a" / "seed_20260317" / "selection.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    assert payload["top_architectures"] == ["w32_d2"]
    assert payload["best_candidate_metric"] == pytest.approx(0.10)


def test_bnn_sweep_runner_dnn_root_overrides_catalog_reference_for_explicit_architectures(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_dnn_root_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    dnn_root = tmp_path / "dnn_root"
    selected_dnn = tmp_path / "seeded_dnn.pkl"
    selected_dnn.write_text("seeded-artifact", encoding="utf-8")
    selection_path = _write_dnn_selection(
        dnn_root / spec["name"] / "seed_101",
        artifact_path=selected_dnn,
        architecture="w64_d2",
    )

    called: list[list[str]] = []

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        called.append(command)
        out_path = Path(_parse_arg(command, "--out"))
        report_path = Path(_parse_arg(command, "--report-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("artifact", encoding="utf-8")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"training": {"best_val_rmse": 0.2, "final_val_rmse": 0.21}}),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--dnn-root",
            str(dnn_root),
            "--seed",
            "101",
            "--architectures",
            "w32_d2",
            "--top-k",
            "1",
            "--prior-scales",
            "0.5",
            "--obs-noise-prior-scales",
            "0.1",
            "--lrs",
            "0.001",
        ]
    )
    assert rc == 0
    assert called
    assert {_parse_arg(command, "--dnn-reference") for command in called} == {str(selected_dnn)}

    payload = json.loads((tmp_path / "out" / "spec_a" / "seed_101" / "selection.json").read_text(encoding="utf-8"))
    assert payload["architecture_source"] == "explicit"
    assert payload["candidate_architectures"] == ["w32_d2"]
    assert payload["dnn_reference_path"] == str(selected_dnn)
    assert payload["dnn_selection_path"] == str(selection_path)
    assert payload["dnn_selected_architecture"] == "w64_d2"


def test_bnn_sweep_runner_dnn_selection_architecture_mode_uses_seeded_winner(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_dnn_arch_source_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [spec])

    dnn_root = tmp_path / "dnn_root"
    selected_dnn = tmp_path / "seeded_dnn.pkl"
    selected_dnn.write_text("seeded-artifact", encoding="utf-8")
    _write_dnn_selection(
        dnn_root / spec["name"] / "seed_101",
        artifact_path=selected_dnn,
        architecture="w64_d2",
    )

    called: list[list[str]] = []

    def fake_run(command, cwd, check):  # noqa: ANN001
        del cwd, check
        called.append(command)
        out_path = Path(_parse_arg(command, "--out"))
        report_path = Path(_parse_arg(command, "--report-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("artifact", encoding="utf-8")
        metric = 0.18 if "stage2__w64_d2" in out_path.stem else 0.22
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"training": {"best_val_rmse": metric, "final_val_rmse": metric + 0.01}}),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--dnn-root",
            str(dnn_root),
            "--architecture-source",
            "dnn-selection",
            "--seed",
            "101",
            "--top-k",
            "1",
            "--prior-scales",
            "0.5",
            "--obs-noise-prior-scales",
            "0.1",
            "--lrs",
            "0.001",
        ]
    )
    assert rc == 0
    assert {_parse_arg(command, "--dnn-reference") for command in called} == {str(selected_dnn)}
    stage_stems = {Path(_parse_arg(command, "--out")).stem for command in called}
    assert stage_stems == {
        "stage1__w64_d2__prior1__obs1__lr0.001",
        "stage2__w64_d2__prior0.5__obs0.1__lr0.001",
    }

    payload = json.loads((tmp_path / "out" / "spec_a" / "seed_101" / "selection.json").read_text(encoding="utf-8"))
    assert payload["architecture_source"] == "dnn-selection"
    assert payload["candidate_architectures"] == ["w64_d2"]
    assert payload["top_architectures"] == ["w64_d2"]
    assert payload["dnn_selected_architecture"] == "w64_d2"


def test_bnn_sweep_runner_import_does_not_require_torch() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _assert_runner_import_without_torch(repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py")


def test_bnn_sweep_runner_stage1_grid_parsing_defaults_and_resume_helpers(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_parse_test",
    )

    with pytest.raises(ValueError, match="non-empty"):
        module._parse_architecture_names(" , ")
    with pytest.raises(ValueError, match="Unknown BNN architecture"):
        module._parse_architecture_names("unknown_arch")
    with pytest.raises(ValueError, match="Float grid must be non-empty"):
        module._parse_float_list(" , ")
    assert module._resolve_stage1_grid(
        grid_text=None,
        scalar_value=None,
        default_text="1.0",
        option_name="--stage1-prior-scales",
        legacy_option_name="--stage1-prior-scale",
    ) == [1.0]
    assert module._resolve_stage1_grid(
        grid_text="0.5, 1.0",
        scalar_value=None,
        default_text="1.0",
        option_name="--stage1-prior-scales",
        legacy_option_name="--stage1-prior-scale",
    ) == [0.5, 1.0]
    assert module._resolve_stage1_grid(
        grid_text=None,
        scalar_value=0.25,
        default_text="1.0",
        option_name="--stage1-prior-scales",
        legacy_option_name="--stage1-prior-scale",
    ) == [0.25]
    with pytest.raises(ValueError, match="Use either --stage1-prior-scales or --stage1-prior-scale"):
        module._resolve_stage1_grid(
            grid_text="0.5,1.0",
            scalar_value=1.0,
            default_text="1.0",
            option_name="--stage1-prior-scales",
            legacy_option_name="--stage1-prior-scale",
        )
    assert module._resolve_seeds([]) == list(module.DEFAULT_SEEDS)
    assert module._iter_hyperparameter_grid(
        prior_scales=[0.5, 1.0],
        obs_noise_prior_scales=[0.1],
        lrs=[1e-3, 2e-3],
    ) == [
        (0.5, 0.1, 1e-3),
        (0.5, 0.1, 2e-3),
        (1.0, 0.1, 1e-3),
        (1.0, 0.1, 2e-3),
    ]

    seed_root = tmp_path / "seed"
    artifact_path, report_path = module._candidate_paths(seed_root, "stage1__w32_d2__prior1__obs1__lr0.001")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("artifact", encoding="utf-8")
    report_path.write_text("{broken json", encoding="utf-8")
    assert module._candidate_completed(report_path, artifact_path) is False


def test_bnn_sweep_runner_rejects_unknown_only_filter(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_only_filter_test",
    )
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [_make_spec(tmp_path, "spec_a")])

    with pytest.raises(ValueError, match="No EMB dataset specs matched"):
        module.main(["--output-root", str(tmp_path / "out"), "--only", "missing"])


def test_bnn_sweep_runner_rejects_dnn_selection_architecture_mode_without_dnn_root(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_missing_dnn_root_test",
    )
    monkeypatch.setattr(module, "resolve_emb_dataset_specs", lambda _root: [_make_spec(tmp_path, "spec_a")])

    with pytest.raises(ValueError, match="requires --dnn-root"):
        module.main(
            [
                "--output-root",
                str(tmp_path / "out"),
                "--seed",
                "101",
                "--architecture-source",
                "dnn-selection",
            ]
        )


def test_bnn_sweep_runner_build_command_honors_require_parity_flag(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "karolina" / "run_bnn_sweep_matrix.py",
        "run_bnn_sweep_matrix_build_command_test",
    )
    spec = _make_spec(tmp_path, "spec_a")
    command, _artifact_path, _report_path = module._build_command(
        python_bin="python3",
        spec=spec,
        seed_root=tmp_path / "seed",
        dnn_reference_path=None,
        stage="stage1",
        arch_name="w32_d2",
        width=32,
        depth=2,
        prior_scale=1.0,
        obs_noise_prior_scale=1.0,
        batch_size=16,
        lr=1e-3,
        max_steps=10,
        eval_every=1,
        predictive_mc_samples=4,
        max_walltime_seconds=60,
        seed=101,
        parity_tol=1.2,
        require_parity=True,
        device="cpu",
    )
    assert "--no-require-parity" not in command


def test_hpc_bnn_sweep_wrapper_dispatches_to_selected_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_bnn_sweep_matrix.py"),
        "hpc_bnn_sweep_dispatch_test",
    )
    monkeypatch.setenv("HPC_SITE", "karolina")
    captured: list[list[str]] = []

    def _fake_call(cmd):  # noqa: ANN001
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    rc = module.main(["--seed", "123"])
    assert rc == 0
    assert captured
    assert sys.executable in captured[0][0]
    assert "scripts/platforms/karolina/run_bnn_sweep_matrix.py" in " ".join(captured[0])


def test_hpc_bnn_sweep_wrapper_rejects_unknown_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_bnn_sweep_matrix.py"),
        "hpc_bnn_sweep_invalid_site_test",
    )
    monkeypatch.setenv("HPC_SITE", "unknown")
    with pytest.raises(SystemExit, match="Unsupported HPC_SITE"):
        module.main([])
