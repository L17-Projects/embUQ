from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_runtime_paths(module, monkeypatch, tmp_path: Path, *, create_tinytex: bool = False) -> Path:  # noqa: ANN001
    tinytex_bin = tmp_path / "tinytex_bin"
    if create_tinytex:
        tinytex_bin.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "DEFAULT_TEXDEPS_DIR", tmp_path / "missing_texdeps")
    monkeypatch.setattr(
        module,
        "get_runtime_paths",
        lambda _repo_root: types.SimpleNamespace(
            tinytex_bin_dir=tinytex_bin,
            korali_site_packages=tmp_path / "korali_site_packages",
        ),
    )
    monkeypatch.setattr(module, "build_runtime_pythonpath", lambda *args, **kwargs: "stub-pythonpath")
    return tinytex_bin


def _write_staging_report(path: Path, *, group_holdout_root: Path, sobol_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "group_holdout_root": str(group_holdout_root),
                "sobol_root": str(sobol_root),
            }
        ),
        encoding="utf-8",
    )


def _write_main_outputs(module, generated_root: Path) -> None:  # noqa: ANN001
    generated_root.mkdir(parents=True, exist_ok=True)
    figures_root = generated_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    for name in module.MANDATORY_MAIN_FIGURES:
        (figures_root / name).write_text("main\n", encoding="utf-8")

    figure_tables = {
        "map_l2_discrepancy_reduced.csv",
        "map_l2_discrepancy_reduced.tex",
    }
    for name in module._required_tables(include_supplementary=False):
        target_root = figures_root if name in figure_tables else generated_root
        (target_root / name).write_text("table\n", encoding="utf-8")


def _write_supplementary_outputs(module, generated_root: Path) -> None:  # noqa: ANN001
    supp_root = generated_root / "supplementary"
    supp_root.mkdir(parents=True, exist_ok=True)
    for name in module.MANDATORY_SUPPLEMENTARY_FIGURES:
        (supp_root / name).write_text("supp\n", encoding="utf-8")
    for name in module.SUPPLEMENTARY_ONLY_TABLES:
        (supp_root / name).write_text("table\n", encoding="utf-8")


def _write_exported_outputs(module, paper_data_root: Path) -> None:  # noqa: ANN001
    figures_main = paper_data_root / "figures" / "main"
    figures_supp = paper_data_root / "figures" / "supplementary"
    tables_root = paper_data_root / "tables"
    figures_main.mkdir(parents=True, exist_ok=True)
    figures_supp.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    for name in module.MANDATORY_MAIN_FIGURES:
        (figures_main / name).write_text("stale-main\n", encoding="utf-8")
    for name in module.MANDATORY_SUPPLEMENTARY_FIGURES:
        (figures_supp / name).write_text("stale-supp\n", encoding="utf-8")
    for name in module.MANDATORY_TABLES:
        (tables_root / name).write_text("stale-table\n", encoding="utf-8")


def test_run_exact_uqdpd_asset_port_infers_single_campaign_and_emits_report(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_success_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp1"
    campaign_root.mkdir(parents=True, exist_ok=True)

    tinytex_bin = _stub_runtime_paths(module, monkeypatch, tmp_path, create_tinytex=True)

    calls: list[dict[str, object]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check
        command = list(command)
        calls.append({"command": command, "env": dict(env) if env is not None else None})
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            assert command[0] == "python3"
            assert command[command.index("--python-bin") + 1] == "python3"
            assert command[command.index("--device") + 1] == "cuda"
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            _write_staging_report(
                staging_root / "dnn_figure_input_staging_report.json",
                group_holdout_root=staging_root / "group_holdout",
                sobol_root=staging_root / "sobol",
            )
        elif script_name == "uqdpd_generate_reduced_story_assets.py":
            assert env is not None
            assert command[0] == "python3"
            assert env["PYTHON_BIN"] == "python3"
            assert env["HUQ_PAPER_DISABLE_TEX"] == "1"
            assert env["PATH"].split(os.pathsep)[0] == str(tinytex_bin)
            _write_main_outputs(module, Path(env["MESOUQ_PAPER_STAGE_ROOT"]) / "generated")
        elif script_name == "uqdpd_generate_supplementary_map_figures.py":
            assert env is not None
            assert command[0] == "python3"
            assert env["PYTHON_BIN"] == "python3"
            assert env["HUQ_PAPER_DISABLE_TEX"] == "1"
            _write_supplementary_outputs(module, Path(env["MESOUQ_PAPER_STAGE_ROOT"]) / "generated")
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root), "--python-bin", "python3"])
    assert rc == 0

    assert [Path(call["command"][1]).name for call in calls] == [
        "stage_dnn_figure_inputs.py",
        "uqdpd_generate_reduced_story_assets.py",
        "uqdpd_generate_supplementary_map_figures.py",
    ]
    assert "--campaign-id" in calls[0]["command"]
    assert calls[0]["command"][calls[0]["command"].index("--campaign-id") + 1] == "camp1"

    report_path = campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["campaign_id"] == "camp1"
    assert report["python_bin"] == "python3"
    assert report["tex"]["disable_tex"] is True
    assert not report["hard_failures"]

    assert (paper_data_root / "figures" / "main" / module.MANDATORY_MAIN_FIGURES[0]).exists()
    assert (
        paper_data_root / "figures" / "supplementary" / module.MANDATORY_SUPPLEMENTARY_FIGURES[0]
    ).exists()
    assert (paper_data_root / "tables" / "map_parameter_comparison.csv").exists()


def test_run_exact_uqdpd_asset_port_skip_supplementary_validates_partial_mode(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_skip_supp_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp2"
    campaign_root.mkdir(parents=True, exist_ok=True)

    _stub_runtime_paths(module, monkeypatch, tmp_path)

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            assert command[command.index("--device") + 1] == "cpu"
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            _write_staging_report(
                staging_root / "dnn_figure_input_staging_report.json",
                group_holdout_root=staging_root / "group_holdout",
                sobol_root=staging_root / "sobol",
            )
        elif script_name == "uqdpd_generate_reduced_story_assets.py":
            assert env is not None
            _write_main_outputs(module, Path(env["MESOUQ_PAPER_STAGE_ROOT"]) / "generated")
        elif script_name == "uqdpd_generate_supplementary_map_figures.py":
            raise AssertionError("supplementary generator should not run when --skip-supplementary is set")
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--paper-data-root",
            str(paper_data_root),
            "--campaign-id",
            "camp2",
            "--staging-device",
            "cpu",
            "--skip-supplementary",
        ]
    )
    assert rc == 0

    report_path = campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["include_supplementary"] is False
    assert "figures_supplementary" not in report["required_assets"]
    assert "map_parameter_comparison.csv" not in {
        entry["asset_id"] for entry in report["required_assets"]["tables"]
    }


def test_run_exact_uqdpd_asset_port_fails_when_required_assets_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_missing_assets_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp3"
    campaign_root.mkdir(parents=True, exist_ok=True)

    _stub_runtime_paths(module, monkeypatch, tmp_path)

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check, env
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            _write_staging_report(
                staging_root / "dnn_figure_input_staging_report.json",
                group_holdout_root=staging_root / "group_holdout",
                sobol_root=staging_root / "sobol",
            )
        elif script_name == "uqdpd_generate_reduced_story_assets.py":
            generated_root = campaign_root / "paper_exact_stage" / "generated" / "figures"
            generated_root.mkdir(parents=True, exist_ok=True)
            (generated_root / module.MANDATORY_MAIN_FIGURES[0]).write_text("partial\n", encoding="utf-8")
        elif script_name == "uqdpd_generate_supplementary_map_figures.py":
            pass
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root), "--campaign-id", "camp3"])
    assert rc == 1

    report_path = campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["hard_failures"]


def test_run_exact_uqdpd_asset_port_does_not_accept_stale_exported_assets(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_stale_exports_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp3_stale"
    campaign_root.mkdir(parents=True, exist_ok=True)
    _write_exported_outputs(module, paper_data_root)

    _stub_runtime_paths(module, monkeypatch, tmp_path)

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check, env
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            _write_staging_report(
                staging_root / "dnn_figure_input_staging_report.json",
                group_holdout_root=staging_root / "group_holdout",
                sobol_root=staging_root / "sobol",
            )
        elif script_name == "uqdpd_generate_reduced_story_assets.py":
            generated_figures = campaign_root / "paper_exact_stage" / "generated" / "figures"
            generated_figures.mkdir(parents=True, exist_ok=True)
            (generated_figures / module.MANDATORY_MAIN_FIGURES[0]).write_text("fresh\n", encoding="utf-8")
        elif script_name == "uqdpd_generate_supplementary_map_figures.py":
            pass
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root), "--campaign-id", "camp3_stale"])
    assert rc == 1

    report_path = campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    missing_name = module.MANDATORY_MAIN_FIGURES[1]
    missing_entry = next(
        entry for entry in report["required_assets"]["figures_main"] if entry["asset_id"] == missing_name
    )
    assert missing_entry["exists"] is False
    assert missing_entry["path"] == str(
        (campaign_root / "paper_exact_stage" / "generated" / "figures" / missing_name).resolve()
    )
    assert (paper_data_root / "figures" / "main" / missing_name).exists()


def test_run_exact_uqdpd_asset_port_force_cleans_stage_root_and_forwards_flag(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_force_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp4"
    stage_root = campaign_root / "paper_exact_stage"
    stale_marker = stage_root / "stale.txt"
    stale_marker.parent.mkdir(parents=True, exist_ok=True)
    stale_marker.write_text("stale\n", encoding="utf-8")
    _stub_runtime_paths(module, monkeypatch, tmp_path)

    commands: list[list[str]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check, env
        command = list(command)
        commands.append(command)
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            _write_staging_report(
                staging_root / "dnn_figure_input_staging_report.json",
                group_holdout_root=staging_root / "group_holdout",
                sobol_root=staging_root / "sobol",
            )
        elif script_name == "uqdpd_generate_reduced_story_assets.py":
            _write_main_outputs(module, campaign_root / "paper_exact_stage" / "generated")
        elif script_name == "uqdpd_generate_supplementary_map_figures.py":
            _write_supplementary_outputs(module, campaign_root / "paper_exact_stage" / "generated")
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root), "--campaign-id", "camp4", "--force"])
    assert rc == 0
    assert not stale_marker.exists()
    assert "--force" in commands[0]


def test_run_exact_uqdpd_asset_port_fails_when_staging_step_fails(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_stage_fail_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp5"
    campaign_root.mkdir(parents=True, exist_ok=True)
    _stub_runtime_paths(module, monkeypatch, tmp_path)

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del command, cwd, text, capture_output, check, env
        return _Result(returncode=2, stderr="boom\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root), "--campaign-id", "camp5"])
    assert rc == 1
    report = json.loads(
        (campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["steps"][0]["status"] == "failed"


def test_run_exact_uqdpd_asset_port_fails_when_staging_report_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_missing_staging_report_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp6"
    campaign_root.mkdir(parents=True, exist_ok=True)
    _stub_runtime_paths(module, monkeypatch, tmp_path)

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del command, cwd, text, capture_output, check, env
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root), "--campaign-id", "camp6"])
    assert rc == 1
    report = json.loads(
        (campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["steps"][0]["missing_outputs"]


def test_run_exact_uqdpd_asset_port_fails_when_staging_report_status_is_failed(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_failed_staging_status_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp7"
    campaign_root.mkdir(parents=True, exist_ok=True)
    _stub_runtime_paths(module, monkeypatch, tmp_path)

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check, env
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            report_path = staging_root / "dnn_figure_input_staging_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "group_holdout_root": str(staging_root / "group_holdout"),
                        "sobol_root": str(staging_root / "sobol"),
                    }
                ),
                encoding="utf-8",
            )
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root), "--campaign-id", "camp7"])
    assert rc == 1
    report = json.loads(
        (campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["steps"][0]["staging_status"] == "failed"


def test_run_exact_uqdpd_asset_port_fails_when_main_step_fails(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_main_fail_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp8"
    campaign_root.mkdir(parents=True, exist_ok=True)
    _stub_runtime_paths(module, monkeypatch, tmp_path)

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            _write_staging_report(
                staging_root / "dnn_figure_input_staging_report.json",
                group_holdout_root=staging_root / "group_holdout",
                sobol_root=staging_root / "sobol",
            )
            return _Result(returncode=0, stdout="ok\n")
        if script_name == "uqdpd_generate_reduced_story_assets.py":
            return _Result(returncode=3, stderr="main failed\n")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root), "--campaign-id", "camp8"])
    assert rc == 1
    report = json.loads(
        (campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["steps"][1]["status"] == "failed"


def test_run_exact_uqdpd_asset_port_fails_when_supplementary_step_fails(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_supp_fail_test",
    )

    paper_data_root = tmp_path / "paper_data"
    campaign_root = paper_data_root / "runs" / "camp9"
    campaign_root.mkdir(parents=True, exist_ok=True)
    texdeps_dir = tmp_path / "texdeps"
    texdeps_dir.mkdir(parents=True, exist_ok=True)
    _stub_runtime_paths(module, monkeypatch, tmp_path)

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            _write_staging_report(
                staging_root / "dnn_figure_input_staging_report.json",
                group_holdout_root=staging_root / "group_holdout",
                sobol_root=staging_root / "sobol",
            )
            return _Result(returncode=0, stdout="ok\n")
        if script_name == "uqdpd_generate_reduced_story_assets.py":
            assert env is not None
            assert env["MESOUQ_PAPER_TEXDEPS_DIR"] == str(texdeps_dir.resolve())
            _write_main_outputs(module, Path(env["MESOUQ_PAPER_STAGE_ROOT"]) / "generated")
            return _Result(returncode=0, stdout="ok\n")
        if script_name == "uqdpd_generate_supplementary_map_figures.py":
            return _Result(returncode=4, stderr="supp failed\n")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--paper-data-root",
            str(paper_data_root),
            "--campaign-id",
            "camp9",
            "--texdeps-dir",
            str(texdeps_dir),
        ]
    )
    assert rc == 1
    report = json.loads(
        (campaign_root / "paper_exact_stage" / "run_exact_uqdpd_asset_port.report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["steps"][2]["status"] == "failed"


def test_exact_wrapper_helper_paths_and_tex_resolution(monkeypatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py",
        "run_exact_uqdpd_asset_port_helpers_test",
    )

    missing_candidates = [tmp_path / "missing_python_a", tmp_path / "missing_python_b"]
    monkeypatch.setattr(module, "DEFAULT_PYTHON_CANDIDATES", missing_candidates)
    assert module._default_python_bin() == sys.executable
    assert module._resolve_python_bin("python3") == "python3"
    assert module._resolve_python_bin(str(tmp_path / "venv" / "bin" / "python")) == str(
        (tmp_path / "venv" / "bin" / "python").resolve()
    )

    paper_data_root = tmp_path / "paper_data"
    with pytest.raises(ValueError, match="does not exist"):
        module._resolve_campaign_id(paper_data_root, None)

    runs_root = paper_data_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No campaigns found"):
        module._resolve_campaign_id(paper_data_root, None)

    (runs_root / "campA").mkdir()
    with pytest.raises(ValueError, match="Campaign 'missing' not found"):
        module._resolve_campaign_id(paper_data_root, "missing")

    (runs_root / "campB").mkdir()
    with pytest.raises(ValueError, match="Multiple campaigns found"):
        module._resolve_campaign_id(paper_data_root, None)

    disabled = module._resolve_tex_config(
        types.SimpleNamespace(disable_tex=True, texdeps_dir=None)
    )
    assert disabled["disable_tex"] is True

    explicit_texdeps = tmp_path / "texdeps_explicit"
    explicit_texdeps.mkdir(parents=True, exist_ok=True)
    explicit = module._resolve_tex_config(
        types.SimpleNamespace(disable_tex=False, texdeps_dir=str(explicit_texdeps))
    )
    assert explicit["texdeps_dir"] == explicit_texdeps.resolve()

    with pytest.raises(ValueError, match="--texdeps-dir does not exist"):
        module._resolve_tex_config(
            types.SimpleNamespace(disable_tex=False, texdeps_dir=str(tmp_path / "missing_texdeps"))
        )

    env_texdeps = tmp_path / "texdeps_env"
    env_texdeps.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MESOUQ_PAPER_TEXDEPS_DIR", str(env_texdeps))
    env_based = module._resolve_tex_config(
        types.SimpleNamespace(disable_tex=False, texdeps_dir=None)
    )
    assert env_based["texdeps_dir"] == env_texdeps.resolve()

    monkeypatch.setenv("MESOUQ_PAPER_TEXDEPS_DIR", str(tmp_path / "missing_env_texdeps"))
    with pytest.raises(ValueError, match="points to a missing path"):
        module._resolve_tex_config(
            types.SimpleNamespace(disable_tex=False, texdeps_dir=None)
        )

    monkeypatch.delenv("MESOUQ_PAPER_TEXDEPS_DIR", raising=False)
    default_texdeps = tmp_path / "texdeps_default"
    default_texdeps.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "DEFAULT_TEXDEPS_DIR", default_texdeps)
    default = module._resolve_tex_config(
        types.SimpleNamespace(disable_tex=False, texdeps_dir=None)
    )
    assert default["texdeps_dir"] == default_texdeps.resolve()


def test_supplementary_map_figures_supports_tex_toggle(monkeypatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "papers" / "huq_emb" / "uqdpd_generate_supplementary_map_figures.py"
    texdeps_dir = tmp_path / "texdeps"
    texdeps_dir.mkdir(parents=True, exist_ok=True)

    previous_texdeps = os.environ.get("MESOUQ_PAPER_TEXDEPS_DIR")
    previous_disable = os.environ.get("HUQ_PAPER_DISABLE_TEX")

    os.environ["MESOUQ_PAPER_TEXDEPS_DIR"] = str(texdeps_dir)
    os.environ.pop("HUQ_PAPER_DISABLE_TEX", None)
    try:
        enabled_module = _load_module(script_path, "uqdpd_supplementary_tex_enabled_test")
        enabled_module.configure_matplotlib()
        assert enabled_module.mpl.rcParams["text.usetex"] is True
    finally:
        enabled_module.mpl.rcdefaults()

    os.environ["MESOUQ_PAPER_TEXDEPS_DIR"] = str(texdeps_dir)
    os.environ["HUQ_PAPER_DISABLE_TEX"] = "1"
    try:
        disabled_module = _load_module(script_path, "uqdpd_supplementary_tex_disabled_test")
        disabled_module.configure_matplotlib()
        assert disabled_module.mpl.rcParams["text.usetex"] is False
    finally:
        disabled_module.mpl.rcdefaults()
        if previous_texdeps is None:
            os.environ.pop("MESOUQ_PAPER_TEXDEPS_DIR", None)
        else:
            os.environ["MESOUQ_PAPER_TEXDEPS_DIR"] = previous_texdeps
        if previous_disable is None:
            os.environ.pop("HUQ_PAPER_DISABLE_TEX", None)
        else:
            os.environ["HUQ_PAPER_DISABLE_TEX"] = previous_disable
