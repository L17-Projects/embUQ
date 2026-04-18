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


def test_local_workstation_runner_invokes_matrix_with_gpu_devices_and_writes_report(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_test",
    )
    captured = {}

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        if len(command) >= 2 and command[1] == "-c":
            return _Result(returncode=0, stdout="", stderr="")
        captured["command"] = command
        captured["env"] = env
        matrix_root = Path(command[command.index("--output-root") + 1])
        matrix_root.mkdir(parents=True, exist_ok=True)
        (matrix_root / "workflow_matrix_report.json").write_text(
            json.dumps({"status": "passed"}),
            encoding="utf-8",
        )
        return _Result(returncode=0, stdout="matrix ok\n", stderr="")

    def fake_render(repo_root, selections, sanity_configs, matrix_root, output_root):
        overlays = {}
        for selection in selections:
            key = module.selection_key(selection)
            dataset = f"{selection.experiment}_2.1um"
            prop_plot = output_root / "plots" / key / "prop.png"
            map_plot = output_root / "plots" / key / "map.png"
            prop_plot.parent.mkdir(parents=True, exist_ok=True)
            prop_plot.write_text("plot", encoding="utf-8")
            map_plot.write_text("plot", encoding="utf-8")
            overlays[key] = {
                "datasets": {
                    dataset: {
                        "propagation_phase3b": {"plot": str(prop_plot)},
                        "map_phase3b": {"plot": str(map_plot)},
                    }
                }
            }
        return overlays

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "render_production_sanity_plots", fake_render)
    monkeypatch.setattr(
        module, "_build_runtime_env", lambda _python_bin: ({"PYTHONPATH": "x"}, ["note"])
    )

    output_root = tmp_path / "o369"
    rc = module.main(
        [
            "--output-root",
            str(output_root),
            "--python-bin",
            "python",
            "--inference-device",
            "gpu",
            "--propagation-device",
            "gpu",
            "--phase2-cpu-ranks",
            "2",
        ]
    )

    assert rc == 0
    assert captured["command"][0] == "python"
    assert "--inference-device" in captured["command"]
    assert captured["command"][captured["command"].index("--inference-device") + 1] == "gpu"
    assert "--propagation-device" in captured["command"]
    assert captured["command"][captured["command"].index("--propagation-device") + 1] == "gpu"
    assert "--selection" in captured["command"]
    assert captured["command"].count("--selection") == 4
    assert captured["env"] == {"PYTHONPATH": "x"}

    report = json.loads((output_root / "local_validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["phase2_native_cuda_supported"] is False
    assert report["device_contract"]["inference_requested"] == "gpu"
    assert report["device_contract"]["inference_effective"] == "gpu"
    assert report["device_contract"]["propagation_requested"] == "gpu"
    assert report["device_contract"]["propagation_effective"] == "gpu"
    assert report["compatibility_warnings"] == []
    assert report["runtime_notes"] == ["note"]
    assert not report["missing_outputs"]
