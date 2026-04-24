from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _touch_outputs(module, roots: dict[str, Path], specs: list[dict[str, str]]) -> None:
    for path in module._expected_holdout_outputs(roots["group_holdout_root"], specs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for path in module._expected_sobol_outputs(roots["sobol_root"], specs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("axis,parameter,index_type,value\n0.0,Yt,ST,0.1\n", encoding="utf-8")
    for path in module._expected_summary_outputs(roots):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("summary\n", encoding="utf-8")


def test_stage_dnn_figure_inputs_runs_expected_commands(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "stage_dnn_figure_inputs.py",
        "stage_dnn_figure_inputs_test",
    )

    paper_data_root = tmp_path / "paper_data"
    roots = module._staging_roots(
        paper_data_root=paper_data_root,
        campaign_id="camp1",
        staging_dirname="dnn_figure_input_staging",
    )
    specs = module._dataset_specs()
    calls: list[list[str]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        del cwd, text, capture_output, check
        calls.append(list(command))
        script_name = Path(command[1]).name
        if script_name == "run_surrogate_group_holdout.py":
            for path in module._expected_holdout_outputs(Path(_arg_value(command, "--output-root")), specs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
        elif script_name == "run_sobol_matrix.py":
            for path in module._expected_sobol_outputs(Path(_arg_value(command, "--output-root")), specs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("axis,parameter,index_type,value\n0.0,Yt,ST,0.1\n", encoding="utf-8")
        elif script_name == "generate_surrogate_holdout_l2_figure.py":
            out_dir = Path(_arg_value(command, "--output-dir"))
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "surrogate_holdout_l2_summary.csv",
                "surrogate_holdout_l2_comparison.png",
                "surrogate_holdout_l2_comparison.pdf",
            ):
                (out_dir / name).write_text("holdout", encoding="utf-8")
        elif script_name == "generate_surrogate_sensitivity_figure.py":
            out_dir = Path(_arg_value(command, "--output-dir"))
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "surrogate_sensitivity_summary.csv",
                "surrogate_sensitivity_comparison.png",
                "surrogate_sensitivity_comparison.pdf",
            ):
                (out_dir / name).write_text("sensitivity", encoding="utf-8")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--paper-data-root",
            str(paper_data_root),
            "--campaign-id",
            "camp1",
        ]
    )
    assert rc == 0

    assert [Path(call[1]).name for call in calls] == [
        "run_surrogate_group_holdout.py",
        "run_sobol_matrix.py",
        "generate_surrogate_holdout_l2_figure.py",
        "generate_surrogate_sensitivity_figure.py",
    ]

    holdout_cmd = calls[0]
    assert _arg_value(holdout_cmd, "--output-root") == str(roots["group_holdout_root"])
    assert _arg_value(holdout_cmd, "--site") == "vega"
    assert holdout_cmd.count("--surrogate-family") == 1
    assert _arg_value(holdout_cmd, "--surrogate-family") == "dnn"
    assert holdout_cmd.count("--only") == 6

    sobol_cmd = calls[1]
    assert _arg_value(sobol_cmd, "--output-root") == str(roots["sobol_root"])
    assert _arg_value(sobol_cmd, "--surrogate-family") == "dnn"
    assert sobol_cmd.count("--only") == 6

    holdout_summary_cmd = calls[2]
    assert _arg_value(holdout_summary_cmd, "--input-root") == str(roots["group_holdout_root"])
    assert _arg_value(holdout_summary_cmd, "--output-dir") == str(roots["holdout_figures_root"])

    sensitivity_summary_cmd = calls[3]
    assert _arg_value(sensitivity_summary_cmd, "--input-root") == str(roots["sobol_root"])
    assert _arg_value(sensitivity_summary_cmd, "--output-dir") == str(roots["sensitivity_figures_root"])

    report = json.loads(roots["manifest_path"].read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["selected_datasets"] == [spec["name"] for spec in specs]
    assert Path(report["staging_root"]) == roots["staging_root"]


def test_stage_dnn_figure_inputs_skips_existing_outputs(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "stage_dnn_figure_inputs.py",
        "stage_dnn_figure_inputs_skip_test",
    )

    paper_data_root = tmp_path / "paper_data"
    roots = module._staging_roots(
        paper_data_root=paper_data_root,
        campaign_id="camp2",
        staging_dirname="dnn_figure_input_staging",
    )
    specs = module._dataset_specs()
    _touch_outputs(module, roots, specs)

    def fail_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("subprocess.run should not be called when staging outputs already exist")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    rc = module.main(
        [
            "--paper-data-root",
            str(paper_data_root),
            "--campaign-id",
            "camp2",
        ]
    )
    assert rc == 0

    report = json.loads(roots["manifest_path"].read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert [step["status"] for step in report["steps"]] == [
        "skipped_existing",
        "skipped_existing",
        "skipped_existing",
        "skipped_existing",
    ]


def test_stage_dnn_figure_inputs_force_reruns_existing_outputs(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "stage_dnn_figure_inputs.py",
        "stage_dnn_figure_inputs_force_test",
    )

    paper_data_root = tmp_path / "paper_data"
    roots = module._staging_roots(
        paper_data_root=paper_data_root,
        campaign_id="camp3",
        staging_dirname="dnn_figure_input_staging",
    )
    specs = module._dataset_specs()
    _touch_outputs(module, roots, specs)
    calls: list[list[str]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        del cwd, text, capture_output, check
        calls.append(list(command))
        script_name = Path(command[1]).name
        if script_name == "run_surrogate_group_holdout.py":
            for path in module._expected_holdout_outputs(Path(_arg_value(command, "--output-root")), specs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
        elif script_name == "run_sobol_matrix.py":
            for path in module._expected_sobol_outputs(Path(_arg_value(command, "--output-root")), specs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("axis,parameter,index_type,value\n0.0,Yt,ST,0.1\n", encoding="utf-8")
        elif script_name == "generate_surrogate_holdout_l2_figure.py":
            out_dir = Path(_arg_value(command, "--output-dir"))
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "surrogate_holdout_l2_summary.csv",
                "surrogate_holdout_l2_comparison.png",
                "surrogate_holdout_l2_comparison.pdf",
            ):
                (out_dir / name).write_text("holdout", encoding="utf-8")
        elif script_name == "generate_surrogate_sensitivity_figure.py":
            out_dir = Path(_arg_value(command, "--output-dir"))
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "surrogate_sensitivity_summary.csv",
                "surrogate_sensitivity_comparison.png",
                "surrogate_sensitivity_comparison.pdf",
            ):
                (out_dir / name).write_text("sensitivity", encoding="utf-8")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--paper-data-root",
            str(paper_data_root),
            "--campaign-id",
            "camp3",
            "--force",
        ]
    )
    assert rc == 0
    assert len(calls) == 4
