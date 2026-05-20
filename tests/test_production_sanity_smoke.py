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


def test_production_sanity_command_writes_override_configs_and_machine_readable_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_production_sanity.py",
        "run_production_sanity_test",
    )
    captured = {}

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        captured["command"] = command
        matrix_root = Path(command[command.index("--output-root") + 1])
        matrix_root.mkdir(parents=True, exist_ok=True)
        (matrix_root / "workflow_matrix_report.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "selections": [
                        {"selection": "compression:full-model:production"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return _Result(returncode=0, stdout="matrix ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "load_korali_build_state",
        lambda repo_root: {"status": "detected", "build_options": {"native_cuda_batch": False}},
    )
    monkeypatch.setattr(
        module,
        "render_production_sanity_plots",
        lambda repo_root, **kwargs: {
            "compression:full-model:production": {
                "phase2": {"korali_plot": str(output_root / "plots" / "phase2.png")}
            }
        },
    )

    output_root = tmp_path / "production_sanity"
    rc = module.main(["--output-root", str(output_root), "--python-bin", "python"])

    assert rc == 0
    assert "--run-phase1-map" not in captured["command"]
    assert "compression:full-model:production" in captured["command"]
    assert "--skip-release-manifest" in captured["command"]
    assert captured["command"][captured["command"].index("--phase2-cpu-ranks") + 1] == "1"
    assert "--phase2-backend" not in captured["command"]

    config_path = output_root / "configs" / "compression__full-model__production.yaml"
    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert "pop_size: 10" in config_text
    assert "phase3b_max_gen: 1" in config_text
    assert "map_n_displacements: 1" in config_text

    report_path = output_root / "production_sanity_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["korali"]["build_options"]["native_cuda_batch"] is False
    assert report["selections"][0]["selection"] == "compression:full-model:production"
    assert report["phase2_backend"] == "native-cuda"
    assert report["phase2_cpu_ranks"] == 1
    assert Path(report["matrix"]["stdout_log"]).exists()
    assert "compression:full-model:production" in report["plots"]


def test_production_sanity_rejects_native_cuda_multi_rank(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_production_sanity.py",
        "run_production_sanity_rank_guard_test",
    )

    try:
        module.main(["--output-root", str(tmp_path / "out"), "--phase2-cpu-ranks", "2"])
    except ValueError as exc:
        assert "native-cuda Phase 2 requires --phase2-cpu-ranks 1" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected native-cuda rank guard to fail")


def test_production_sanity_allows_cpu_mpi_multi_rank(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_production_sanity.py",
        "run_production_sanity_cpu_mpi_test",
    )
    captured = {}

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        captured["command"] = command
        matrix_root = Path(command[command.index("--output-root") + 1])
        matrix_root.mkdir(parents=True, exist_ok=True)
        (matrix_root / "workflow_matrix_report.json").write_text(
            json.dumps({"status": "passed", "selections": []}),
            encoding="utf-8",
        )
        return _Result(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "load_korali_build_state", lambda repo_root: {"status": "unknown"})
    monkeypatch.setattr(module, "render_production_sanity_plots", lambda *args, **kwargs: {})

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--phase2-backend",
            "cpu-mpi",
            "--phase2-cpu-ranks",
            "4",
        ]
    )

    assert rc == 0
    assert captured["command"][captured["command"].index("--phase2-backend") + 1] == "cpu-mpi"
    assert captured["command"][captured["command"].index("--phase2-cpu-ranks") + 1] == "4"
