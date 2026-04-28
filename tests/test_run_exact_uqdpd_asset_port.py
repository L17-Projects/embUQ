from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path


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

    monkeypatch.setattr(module, "DEFAULT_TEXDEPS_DIR", tmp_path / "missing_texdeps")
    monkeypatch.setattr(
        module,
        "get_vega_paths",
        lambda _repo_root: types.SimpleNamespace(
            tinytex_bin_dir=tmp_path / "tinytex_bin",
            korali_site_packages=tmp_path / "korali_site_packages",
        ),
    )
    monkeypatch.setattr(module, "build_runtime_pythonpath", lambda *args, **kwargs: "stub-pythonpath")

    calls: list[dict[str, object]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check
        command = list(command)
        calls.append({"command": command, "env": dict(env) if env is not None else None})
        script_name = Path(command[1]).name
        if script_name == "stage_dnn_figure_inputs.py":
            staging_root = campaign_root / "postprocess_graph" / "dnn_figure_input_staging"
            _write_staging_report(
                staging_root / "dnn_figure_input_staging_report.json",
                group_holdout_root=staging_root / "group_holdout",
                sobol_root=staging_root / "sobol",
            )
        elif script_name == "uqdpd_generate_reduced_story_assets.py":
            assert env is not None
            assert env["HUQ_PAPER_DISABLE_TEX"] == "1"
            _write_main_outputs(module, Path(env["MESOUQ_PAPER_STAGE_ROOT"]) / "generated")
        elif script_name == "uqdpd_generate_supplementary_map_figures.py":
            assert env is not None
            assert env["HUQ_PAPER_DISABLE_TEX"] == "1"
            _write_supplementary_outputs(module, Path(env["MESOUQ_PAPER_STAGE_ROOT"]) / "generated")
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--paper-data-root", str(paper_data_root)])
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

    monkeypatch.setattr(module, "DEFAULT_TEXDEPS_DIR", tmp_path / "missing_texdeps")
    monkeypatch.setattr(
        module,
        "get_vega_paths",
        lambda _repo_root: types.SimpleNamespace(
            tinytex_bin_dir=tmp_path / "tinytex_bin",
            korali_site_packages=tmp_path / "korali_site_packages",
        ),
    )
    monkeypatch.setattr(module, "build_runtime_pythonpath", lambda *args, **kwargs: "stub-pythonpath")

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

    monkeypatch.setattr(module, "DEFAULT_TEXDEPS_DIR", tmp_path / "missing_texdeps")
    monkeypatch.setattr(
        module,
        "get_vega_paths",
        lambda _repo_root: types.SimpleNamespace(
            tinytex_bin_dir=tmp_path / "tinytex_bin",
            korali_site_packages=tmp_path / "korali_site_packages",
        ),
    )
    monkeypatch.setattr(module, "build_runtime_pythonpath", lambda *args, **kwargs: "stub-pythonpath")

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
