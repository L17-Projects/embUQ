import importlib.util
import json
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _selection_dataset(command: list[str]) -> str:
    experiment = _arg_value(command, "--experiment")
    if experiment == "compression":
        return "compression_2.1um"
    return "indentation_2.1um"


def test_workflow_matrix_runner_writes_machine_readable_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "vega" / "run_workflow_matrix.py", "workflow_matrix_test")

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        script_name = Path(command[1]).name
        output_root = Path(_arg_value(command, "--output-dir"))
        dataset_name = _selection_dataset(command)

        if script_name == "extract_map.py":
            stage = _arg_value(command, "--stage")
            manifest_root = output_root / ("map_phase1" if stage == "phase1" else "map_phase3b")
            manifest_root.mkdir(parents=True, exist_ok=True)
            manifest_name = "phase1_map_manifest.json" if stage == "phase1" else "phase3b_map_manifest.json"
            manifest_path = manifest_root / manifest_name
            manifest_path.write_text(
                json.dumps(
                    {
                        "datasets": {
                            dataset_name: {
                                "output_csv": str(manifest_root / f"{dataset_name}.csv"),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (manifest_root / f"{dataset_name}.csv").write_text("parameter,value\nYt,1.0\n", encoding="utf-8")
        elif script_name == "run_propagation.py":
            summary_csv = output_root / "propagation_phase3b" / dataset_name / "summary.csv"
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            summary_csv.write_text("x,mean\n0.0,0.0\n", encoding="utf-8")

        return _Result(0, stdout=f"ran {script_name}\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    matrix_root = tmp_path / "matrix"
    rc = module.main(
        [
            "--experiments",
            "compression",
            "--model-families",
            "full-model",
            "reduced-model",
            "--profiles",
            "validation",
            "--output-root",
            str(matrix_root),
            "--phase2-cpu-ranks",
            "4",
        ]
    )

    assert rc == 0
    report_path = matrix_root / "workflow_matrix_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert [entry["selection"] for entry in report["selections"]] == [
        "compression:full-model:validation",
        "compression:reduced-model:validation",
    ]
    for entry in report["selections"]:
        assert [step["name"] for step in entry["steps"]] == [
            "phase1",
            "map_phase1",
            "phase2",
            "phase3b",
            "propagation_phase3b",
            "map_phase3b",
        ]
        assert Path(entry["summary_path"]).exists()
        assert Path(entry["artifacts"]["phase1_map_manifest"]).exists()
        assert Path(entry["artifacts"]["phase3b_map_manifest"]).exists()


def test_workflow_matrix_runner_accepts_explicit_override_selector(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "vega" / "run_workflow_matrix.py", "workflow_matrix_override_test")

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        return _Result(0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    override_path = tmp_path / "override.yaml"
    override_path.write_text("enabled_experiments: []\n", encoding="utf-8")

    matrix_root = tmp_path / "matrix"
    rc = module.main(
        [
            "--selection",
            "compression:full-model:validation",
            "--output-root",
            str(matrix_root),
            "--config-override",
            f"compression:full-model:validation={override_path}",
            "--skip-phase1-map",
            "--skip-phase3b-propagation",
            "--skip-phase3b-map",
        ]
    )

    assert rc == 0
    report = json.loads((matrix_root / "workflow_matrix_report.json").read_text(encoding="utf-8"))
    assert report["selections"][0]["config"] == str(override_path.resolve())


def test_validation_matrix_wrapper_delegates_to_workflow_matrix_with_validation_profile(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "vega" / "run_validation_matrix.py", "validation_matrix_wrapper_test")
    captured = {}

    def fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--experiments",
            "compression",
            "--model-families",
            "full-model",
            "--output-root",
            str(tmp_path / "validation"),
            "--phase2-cpu-ranks",
            "4",
            "--python-bin",
            "python",
        ]
    )

    assert rc == 0
    assert captured["cwd"] == str(repo_root)
    assert captured["command"][0] == "python"
    assert captured["command"][1] == str(repo_root / "scripts" / "vega" / "run_workflow_matrix.py")
    assert "--profiles" in captured["command"]
    assert captured["command"][captured["command"].index("--profiles") + 1] == "validation"
    assert str(tmp_path / "validation") in captured["command"]
