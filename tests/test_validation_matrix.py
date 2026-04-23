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
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_workflow_matrix.py", "workflow_matrix_test"
    )

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        script_name = Path(command[1]).name
        output_root = Path(_arg_value(command, "--output-dir"))
        dataset_name = _selection_dataset(command)

        if script_name == "extract_map.py":
            stage = _arg_value(command, "--stage")
            manifest_root = output_root / ("map_phase1" if stage == "phase1" else "map_phase3b")
            manifest_root.mkdir(parents=True, exist_ok=True)
            manifest_name = (
                "phase1_map_manifest.json" if stage == "phase1" else "phase3b_map_manifest.json"
            )
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
            (manifest_root / f"{dataset_name}.csv").write_text(
                "parameter,value\nYt,1.0\n", encoding="utf-8"
            )
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

    assert rc == 1
    report_path = matrix_root / "workflow_matrix_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["release_status"] == "FAIL"
    assert Path(report["manifests"]["paper_release_manifest"]).exists()
    assert any("missing asset:" in item for item in report["hard_failures"])
    assert [entry["selection"] for entry in report["selections"]] == [
        "compression:full-model:validation",
        "compression:reduced-model:validation",
    ]
    for entry in report["selections"]:
        assert [step["name"] for step in entry["steps"]] == [
            "phase1",
            "phase2",
            "phase3b",
            "propagation_phase3b",
            "map_phase3b",
        ]
        assert Path(entry["summary_path"]).exists()
        assert Path(entry["artifacts"]["phase3b_map_manifest"]).exists()
        assert Path(entry["lane_manifest"]).exists()
        assert len(entry["job_manifests"]) == len(entry["steps"])
        for manifest_path in entry["job_manifests"]:
            assert Path(manifest_path).exists()
        map_job_manifest_path = next(
            Path(path) for path in entry["job_manifests"] if "__map_phase3b__" in path
        )
        map_job_manifest = json.loads(map_job_manifest_path.read_text(encoding="utf-8"))
        map_output_paths = {item["path"] for item in map_job_manifest["output_files"]}
        assert any(path.endswith("phase3b_map_manifest.json") for path in map_output_paths)
        assert any(path.endswith(".csv") for path in map_output_paths)


def test_workflow_matrix_emits_policy_metadata_in_job_and_lane_manifests(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_workflow_matrix.py",
        "workflow_matrix_policy_manifest_test",
    )

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        script_name = Path(command[1]).name
        output_root = Path(_arg_value(command, "--output-dir"))
        dataset_name = _selection_dataset(command)
        if script_name == "extract_map.py":
            manifest_root = output_root / "map_phase3b"
            manifest_root.mkdir(parents=True, exist_ok=True)
            (manifest_root / "phase3b_map_manifest.json").write_text(
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
            (manifest_root / f"{dataset_name}.csv").write_text(
                "parameter,value\nYt,1.0\n", encoding="utf-8"
            )
        return _Result(0, stdout=f"ran {script_name}\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_PARTITION", "dev")
    monkeypatch.setenv("SLURM_TIMELIMIT", "00:20:00")
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "1")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "64000")

    matrix_root = tmp_path / "matrix"
    rc = module.main(
        [
            "--selection",
            "compression:full-model:production",
            "--output-root",
            str(matrix_root),
        ]
    )
    assert rc == 1

    report = json.loads((matrix_root / "workflow_matrix_report.json").read_text(encoding="utf-8"))
    selection_report = report["selections"][0]

    phase2_job_manifest_path = next(
        Path(path) for path in selection_report["job_manifests"] if "__phase2__" in path
    )
    phase2_job_manifest = json.loads(phase2_job_manifest_path.read_text(encoding="utf-8"))
    assert phase2_job_manifest["backend_metadata"]["phase2_backend"] == "native-cuda"
    assert (
        phase2_job_manifest["backend_metadata"]["phase2_backend_policy"]["is_compliant"] is True
    )

    phase1_job_manifest_path = next(
        Path(path) for path in selection_report["job_manifests"] if "__phase1__" in path
    )
    phase1_job_manifest = json.loads(phase1_job_manifest_path.read_text(encoding="utf-8"))
    partition_policy = phase1_job_manifest["backend_metadata"]["partition_policy"]
    assert partition_policy["expected_partition"] == "dev"
    assert partition_policy["is_compliant"] is True

    lane_manifest_path = Path(selection_report["lane_manifest"])
    lane_manifest = json.loads(lane_manifest_path.read_text(encoding="utf-8"))
    status_by_stage = {entry["stage"]: entry["status"] for entry in lane_manifest["stage_statuses"]}
    assert status_by_stage["phase1"] == "passed"
    assert status_by_stage["map_mirheo"] == "skipped"
    assert lane_manifest["policy"]["status"] == "pass"


def test_workflow_matrix_writes_paper_release_manifest_with_assets_and_mapping(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_workflow_matrix.py",
        "workflow_matrix_release_manifest_test",
    )

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        script_name = Path(command[1]).name
        output_root = Path(_arg_value(command, "--output-dir"))
        dataset_name = _selection_dataset(command)

        if script_name == "extract_map.py":
            stage = _arg_value(command, "--stage")
            manifest_root = output_root / ("map_phase1" if stage == "phase1" else "map_phase3b")
            manifest_root.mkdir(parents=True, exist_ok=True)
            manifest_name = (
                "phase1_map_manifest.json" if stage == "phase1" else "phase3b_map_manifest.json"
            )
            (manifest_root / manifest_name).write_text(
                json.dumps(
                    {"datasets": {dataset_name: {"output_csv": str(manifest_root / "out.csv")}}}
                ),
                encoding="utf-8",
            )
            (manifest_root / "out.csv").write_text("x,y\n0,1\n", encoding="utf-8")
        elif script_name == "run_map_mirheo.py":
            manifest_root = output_root / "map_mirheo"
            manifest_root.mkdir(parents=True, exist_ok=True)
            (manifest_root / "map_mirheo_manifest.json").write_text(
                json.dumps({"status": "ok"}), encoding="utf-8"
            )
        elif script_name == "run_propagation.py":
            propagation_root = output_root / "propagation_phase3b"
            propagation_root.mkdir(parents=True, exist_ok=True)
            (propagation_root / "summary.csv").write_text("x,mean\n0.0,0.0\n", encoding="utf-8")

        return _Result(0, stdout=f"ran {script_name}\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")
    monkeypatch.setenv("SLURM_TIMELIMIT", "00:45:00")
    monkeypatch.setenv("SLURM_JOB_ID", "900001")

    figures_main_root = tmp_path / "figures" / "main"
    figures_supp_root = tmp_path / "figures" / "supplementary"
    tables_root = tmp_path / "tables"
    figures_main_root.mkdir(parents=True, exist_ok=True)
    figures_supp_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    for name in module.MANDATORY_MAIN_FIGURES:
        (figures_main_root / name).write_text("main", encoding="utf-8")
    for name in module.MANDATORY_SUPPLEMENTARY_FIGURES:
        (figures_supp_root / name).write_text("supp", encoding="utf-8")
    for name in module.MANDATORY_TABLES:
        (tables_root / name).write_text("table", encoding="utf-8")

    source_map_path = tmp_path / "asset_source_map.json"
    source_map_path.write_text(
        json.dumps(
            {
                "figures/main/experimental_reference_curves.pdf": {
                    "source_lane": "compression:full-model:production",
                    "source_artifacts": ["runs/compression/full-model/production/phase3b"],
                }
            }
        ),
        encoding="utf-8",
    )

    matrix_root = tmp_path / "matrix"
    rc = module.main(
        [
            "--selection",
            "compression:full-model:production",
            "--output-root",
            str(matrix_root),
            "--run-phase1-map",
            "--run-map-mirheo",
            "--figures-main-root",
            str(figures_main_root),
            "--figures-supplementary-root",
            str(figures_supp_root),
            "--tables-root",
            str(tables_root),
            "--asset-source-map",
            str(source_map_path),
        ]
    )
    assert rc == 0

    report = json.loads((matrix_root / "workflow_matrix_report.json").read_text(encoding="utf-8"))
    assert report["release_status"] == "PASS"
    assert report["hard_failures"] == []

    paper_manifest = json.loads(
        Path(report["manifests"]["paper_release_manifest"]).read_text(encoding="utf-8")
    )
    assert paper_manifest["release_status"] == "PASS"
    mapped = next(
        item
        for item in paper_manifest["assets"]["figures_main"]
        if item["asset_id"] == "experimental_reference_curves.pdf"
    )
    assert mapped["source_lane"] == "compression:full-model:production"


def test_workflow_matrix_runner_accepts_explicit_override_selector(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_workflow_matrix.py", "workflow_matrix_override_test"
    )

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
            "--skip-phase3b-propagation",
            "--skip-phase3b-map",
        ]
    )

    assert rc == 1
    report = json.loads((matrix_root / "workflow_matrix_report.json").read_text(encoding="utf-8"))
    assert report["selections"][0]["config"] == str(override_path.resolve())


def test_workflow_matrix_forwards_requested_devices(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_workflow_matrix.py", "workflow_matrix_device_test"
    )
    captured: list[list[str]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        captured.append(command)
        return _Result(0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    matrix_root = tmp_path / "matrix"
    rc = module.main(
        [
            "--selection",
            "compression:full-model:validation",
            "--output-root",
            str(matrix_root),
            "--inference-device",
            "gpu",
            "--propagation-device",
            "gpu",
            "--skip-phase3b-map",
        ]
    )

    assert rc == 1
    phase1_command = next(
        cmd for cmd in captured if "--stage" in cmd and cmd[cmd.index("--stage") + 1] == "phase1"
    )
    phase3b_command = next(
        cmd for cmd in captured if "--stage" in cmd and cmd[cmd.index("--stage") + 1] == "phase3b"
    )
    propagation_command = next(cmd for cmd in captured if Path(cmd[1]).name == "run_propagation.py")
    assert "--device" in phase1_command
    assert phase1_command[phase1_command.index("--device") + 1] == "gpu"
    assert "--device" in phase3b_command
    assert phase3b_command[phase3b_command.index("--device") + 1] == "gpu"
    assert "--device" in propagation_command
    assert propagation_command[propagation_command.index("--device") + 1] == "gpu"


def test_workflow_matrix_phase2_backend_defaults_by_profile(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_workflow_matrix.py", "workflow_matrix_phase2_backend_defaults_test"
    )
    captured: list[list[str]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        captured.append(command)
        return _Result(0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    matrix_root = tmp_path / "matrix"
    rc = module.main(
        [
            "--selection",
            "compression:full-model:production",
            "--selection",
            "compression:full-model:validation",
            "--output-root",
            str(matrix_root),
            "--skip-phase3b-map",
        ]
    )

    assert rc == 1
    phase2_commands = [
        cmd for cmd in captured if "--stage" in cmd and cmd[cmd.index("--stage") + 1] == "phase2"
    ]
    assert len(phase2_commands) == 2
    by_profile = {_arg_value(cmd, "--profile"): _arg_value(cmd, "--phase2-backend") for cmd in phase2_commands}
    assert by_profile["production"] == "native-cuda"
    assert by_profile["validation"] == "cpu-mpi"


def test_workflow_matrix_allow_release_fail_keeps_zero_exit_for_command_success(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_workflow_matrix.py",
        "workflow_matrix_allow_release_fail_test",
    )

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        del cwd, text, capture_output, check
        script_name = Path(command[1]).name
        output_root = Path(_arg_value(command, "--output-dir"))
        if script_name == "extract_map.py":
            manifest_root = output_root / "map_phase3b"
            manifest_root.mkdir(parents=True, exist_ok=True)
            (manifest_root / "phase3b_map_manifest.json").write_text(
                json.dumps({"datasets": {}}), encoding="utf-8"
            )
        return _Result(0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    matrix_root = tmp_path / "matrix"
    rc = module.main(
        [
            "--selection",
            "compression:full-model:production",
            "--output-root",
            str(matrix_root),
            "--allow-release-fail",
        ]
    )
    assert rc == 0
    report = json.loads((matrix_root / "workflow_matrix_report.json").read_text(encoding="utf-8"))
    assert report["release_status"] == "FAIL"


def test_workflow_matrix_skip_release_manifest_marks_workflow_only_scope(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_workflow_matrix.py",
        "workflow_matrix_skip_release_manifest_test",
    )

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        del command, cwd, text, capture_output, check
        return _Result(0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    matrix_root = tmp_path / "matrix"
    rc = module.main(
        [
            "--selection",
            "compression:full-model:production",
            "--output-root",
            str(matrix_root),
            "--skip-release-manifest",
        ]
    )
    assert rc == 0
    report = json.loads((matrix_root / "workflow_matrix_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["release_scope"] == "workflow-only"
    assert report["release_status"] == "SKIPPED"
    assert report["hard_failures"] == []
    assert report["manifests"]["paper_release_manifest"] is None


def test_validation_matrix_wrapper_delegates_to_workflow_matrix_with_validation_profile(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_validation_matrix.py",
        "validation_matrix_wrapper_test",
    )
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


def test_validation_matrix_wrapper_resolves_relative_output_root_from_repo_root(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_validation_matrix.py",
        "validation_matrix_wrapper_relative_output_test",
    )
    captured = {}

    def fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    rc = module.main(
        [
            "--selection",
            "compression:full-model:validation",
            "--output-root",
            "tmp_validation",
            "--python-bin",
            "python",
        ]
    )

    assert rc == 0
    assert captured["cwd"] == str(repo_root)
    assert _arg_value(captured["command"], "--output-root") == str(
        (repo_root / "tmp_validation").resolve()
    )
