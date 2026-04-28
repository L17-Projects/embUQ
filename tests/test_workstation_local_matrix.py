import importlib.util
import json
from pathlib import Path

import pytest
import yaml


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
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
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
    assert report["phase2_backend_contract"] == "dual_backend"
    assert report["phase2_backend_default_for_profile"] == "cpu-mpi"
    assert report["phase2_backend_effective"] == "cpu-mpi"
    assert report["device_contract"]["inference_requested"] == "gpu"
    assert report["device_contract"]["inference_effective"] == "gpu"
    assert report["device_contract"]["propagation_requested"] == "gpu"
    assert report["device_contract"]["propagation_effective"] == "gpu"
    assert report["compatibility_warnings"] == []
    assert report["runtime_notes"] == ["note"]
    assert not report["missing_outputs"]


def test_prepend_pythonpath_handles_empty_duplicate_and_prepend(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_prepend",
    )

    env = {}
    first = tmp_path / "first"
    second = tmp_path / "second"

    module._prepend_pythonpath(env, first)
    assert env["PYTHONPATH"] == str(first)

    module._prepend_pythonpath(env, first)
    assert env["PYTHONPATH"] == str(first)

    module._prepend_pythonpath(env, second)
    assert env["PYTHONPATH"] == f"{second}:{first}"


def test_resolve_path_and_discover_repo_local_korali_site(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_paths",
    )

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    assert module._resolve_path("relative/output") == (tmp_path / "relative" / "output").resolve()

    site_a = tmp_path / "_vega" / "korali" / "install" / "lib" / "python3.10" / "site-packages"
    site_b = tmp_path / "_vega" / "korali" / "install" / "lib" / "python3.11" / "site-packages"
    site_a.mkdir(parents=True, exist_ok=True)
    site_b.mkdir(parents=True, exist_ok=True)
    assert module._discover_repo_local_korali_site() == site_b

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path / "missing-root")
    assert module._discover_repo_local_korali_site() is None


def test_probe_korali_engine_reports_success_and_failure(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_probe",
    )

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: _Result(returncode=0, stdout="", stderr=""),
    )
    ok, error = module._probe_korali_engine("python", {})
    assert ok is True
    assert error == ""

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: _Result(returncode=1, stdout="", stderr="engine failed"),
    )
    ok, error = module._probe_korali_engine("python", {})
    assert ok is False
    assert error == "engine failed"


def test_resolve_selections_supports_explicit_and_all_lanes(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_resolve",
    )

    explicit = module._resolve_selections(["compression:full-model:validation"], all_lanes=False)
    assert len(explicit) == 1
    assert explicit[0].experiment == "compression"

    expanded = module._resolve_selections([], all_lanes=True)
    assert len(expanded) == 4


def test_resolve_selections_rejects_non_validation_profile():
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_non_validation",
    )

    with pytest.raises(ValueError):
        module._resolve_selections(["compression:full-model:production"], all_lanes=False)


def test_build_runtime_env_handles_probe_outcomes(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_runtime_env",
    )

    monkeypatch.setattr(module, "_discover_repo_local_korali_site", lambda: tmp_path / "korali")
    monkeypatch.setattr(module, "_probe_korali_engine", lambda python_bin, env: (True, ""))
    env_ok, notes_ok = module._build_runtime_env("python")
    assert "Using repo-local Korali runtime path" in notes_ok[0]
    assert str(tmp_path / "korali") in env_ok.get("PYTHONPATH", "")

    monkeypatch.setattr(module, "_discover_repo_local_korali_site", lambda: None)
    monkeypatch.setattr(
        module,
        "_probe_korali_engine",
        lambda python_bin, env: (False, "plain failure"),
    )
    env_fail, notes_fail = module._build_runtime_env("python")
    assert any(
        "not found under _vega/korali/install/lib/python*/site-packages" in note
        for note in notes_fail
    )
    assert any(
        "Korali probe failed before workflow run: plain failure" in note for note in notes_fail
    )
    assert isinstance(env_fail, dict)


def test_build_runtime_env_retries_user_site_on_mpi4py_error(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_runtime_retry",
    )

    user_site = tmp_path / "user-site"
    user_site.mkdir(parents=True)
    monkeypatch.setattr(module, "_discover_repo_local_korali_site", lambda: None)
    monkeypatch.setattr(module.site, "getusersitepackages", lambda: str(user_site))

    calls = {"count": 0}

    def _probe(python_bin, env):
        calls["count"] += 1
        if calls["count"] == 1:
            return False, "Could not load mpi4py API."
        return True, ""

    monkeypatch.setattr(module, "_probe_korali_engine", _probe)
    env, notes = module._build_runtime_env("python")
    assert calls["count"] == 2
    assert str(user_site) in env.get("PYTHONPATH", "")
    assert any("Prepended user-site packages" in note for note in notes)


def test_build_runtime_env_records_retry_failure(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_runtime_retry_fail",
    )

    user_site = tmp_path / "user-site"
    user_site.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "_discover_repo_local_korali_site", lambda: None)
    monkeypatch.setattr(module.site, "getusersitepackages", lambda: str(user_site))

    calls = {"count": 0}

    def _probe(python_bin, env):
        calls["count"] += 1
        if calls["count"] == 1:
            return False, "Could not load mpi4py API."
        return False, "retry failed"

    monkeypatch.setattr(module, "_probe_korali_engine", _probe)
    env, notes = module._build_runtime_env("python")
    assert calls["count"] == 2
    assert str(user_site) in env.get("PYTHONPATH", "")
    assert any("Retry after user-site prepend still failed: retry failed" in note for note in notes)


def test_derive_validation_smoke_config_rejects_non_mapping_yaml(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_bad_yaml",
    )

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("- not-a-mapping\n", encoding="utf-8")
    monkeypatch.setattr(
        module, "resolve_workflow_config_path", lambda repo_root, selection: bad_config
    )

    selection = module.parse_selection("compression:full-model:validation")
    with pytest.raises(ValueError):
        module._derive_validation_smoke_config(selection, tmp_path / "out", {"pop_size": 8})


def test_derive_validation_smoke_config_ignores_unknown_overrides(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_override_filter",
    )

    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump({"description": "base", "pop_size": 16}, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "resolve_workflow_config_path",
        lambda _repo_root, _selection: base_config,
    )

    selection = module.parse_selection("compression:full-model:validation")
    derived_path = module._derive_validation_smoke_config(
        selection,
        tmp_path / "out",
        {"pop_size": 8, "unknown_override": 99},
    )
    payload = yaml.safe_load(derived_path.read_text(encoding="utf-8"))
    assert payload["pop_size"] == 8
    assert "unknown_override" not in payload


def test_collect_and_validate_overlay_paths_reports_missing(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_overlays",
    )

    overlays = module._collect_overlay_paths(
        {
            "compression:full-model:validation": {
                "datasets": {
                    "compression_2.1um": {
                        "propagation_phase3b": {"plot": str(tmp_path / "missing_prop.png")},
                        "map_phase3b": {},
                    }
                }
            }
        }
    )
    missing = module._validate_overlay_outputs(overlays)
    assert any("missing map_vs_reference overlays" in entry for entry in missing)
    assert any("missing file" in entry for entry in missing)


def test_main_rejects_invalid_phase2_cpu_ranks(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_invalid_ranks",
    )

    with pytest.raises(ValueError):
        module.main(["--output-root", str(tmp_path / "out"), "--phase2-cpu-ranks", "0"])


def test_main_returns_matrix_returncode_on_subprocess_failure(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_subprocess_fail",
    )

    monkeypatch.setattr(
        module, "_build_runtime_env", lambda _python_bin: ({"PYTHONPATH": "x"}, ["note"])
    )
    monkeypatch.setattr(module, "render_production_sanity_plots", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: _Result(returncode=7, stdout="bad", stderr="fail"),
    )

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "o369_fail"),
            "--python-bin",
            "python",
            "--selection",
            "compression:full-model:validation",
        ]
    )
    assert rc == 7


def test_main_returns_one_when_overlays_are_missing(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_missing_outputs",
    )

    monkeypatch.setattr(
        module, "_build_runtime_env", lambda _python_bin: ({"PYTHONPATH": "x"}, ["note"])
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: _Result(returncode=0))
    monkeypatch.setattr(
        module,
        "render_production_sanity_plots",
        lambda *args, **kwargs: {
            "compression:full-model:validation": {
                "datasets": {
                    "compression_2.1um": {
                        "propagation_phase3b": {},
                        "map_phase3b": {},
                    }
                }
            }
        },
    )

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "o369_missing"),
            "--python-bin",
            "python",
            "--selection",
            "compression:full-model:validation",
        ]
    )
    assert rc == 1


def test_main_forwards_continue_on_error_flag(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "workstation" / "run_local_validation_matrix.py",
        "run_local_validation_matrix_continue_flag",
    )

    captured = {}

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        captured["command"] = command
        return _Result(returncode=7, stdout="bad", stderr="fail")

    monkeypatch.setattr(
        module, "_build_runtime_env", lambda _python_bin: ({"PYTHONPATH": "x"}, ["note"])
    )
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "render_production_sanity_plots", lambda *args, **kwargs: {})

    rc = module.main(
        [
            "--output-root",
            str(tmp_path / "o369_continue"),
            "--python-bin",
            "python",
            "--selection",
            "compression:full-model:validation",
            "--continue-on-error",
        ]
    )

    assert rc == 7
    assert "--continue-on-error" in captured["command"]
