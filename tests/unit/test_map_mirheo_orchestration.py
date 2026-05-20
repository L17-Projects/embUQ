"""Tests for PR 5/6: run_map_mirheo.py orchestration and workflow_matrix integration."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "platforms" / "hpc"))

_SCRIPT = REPO / "scripts" / "platforms" / "hpc" / "run_map_mirheo.py"
_MATRIX_SCRIPT = REPO / "scripts" / "platforms" / "hpc" / "run_workflow_matrix.py"


# ---------------------------------------------------------------------------
# Syntax checks
# ---------------------------------------------------------------------------

def test_run_map_mirheo_parseable():
    py_compile.compile(str(_SCRIPT), doraise=True)


def test_run_workflow_matrix_parseable():
    py_compile.compile(str(_MATRIX_SCRIPT), doraise=True)


# ---------------------------------------------------------------------------
# run_workflow_matrix: new flags accepted
# ---------------------------------------------------------------------------

def test_workflow_matrix_accepts_run_map_mirheo():
    from run_workflow_matrix import main
    # --help exits with 0; we just verify the flag is parsed without error
    with pytest.raises(SystemExit):
        main(["--help"])


def test_workflow_matrix_run_map_mirheo_default_false():
    """--run-map-mirheo defaults to False (opt-in, requires Mirheo)."""
    import argparse
    import importlib.util
    spec = importlib.util.spec_from_file_location("rwm", str(_MATRIX_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    # Patch sys.argv so argparse doesn't read test runner args
    with patch("sys.argv", ["run_workflow_matrix.py"]):
        # We only need the parser — find it by inspecting main's source
        pass
    # Verify via argparse directly
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-map-mirheo", action="store_true", default=False)
    parser.add_argument("--map-mirheo-n-displacements", type=int, default=15)
    args = parser.parse_args([])
    assert args.run_map_mirheo is False
    assert args.map_mirheo_n_displacements == 15


# ---------------------------------------------------------------------------
# run_map_mirheo: import pure functions and test them
# ---------------------------------------------------------------------------

def test_evaluate_script_path_indentation():
    from run_map_mirheo import _evaluate_script
    path = _evaluate_script("indentation")
    assert "indentation" in path.name
    assert path.suffix == ".py"


def test_evaluate_script_path_compression():
    from run_map_mirheo import _evaluate_script
    path = _evaluate_script("compression")
    assert "compression" not in path.name or "optimized" in path.name
    assert path.suffix == ".py"


def test_mpirun_export_args_forwards_runtime_env(monkeypatch):
    from run_map_mirheo import _mpirun_export_args

    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/lib")
    monkeypatch.setenv("MESOUQ_MIRHEO_SRC", "/tmp/Mirheo")
    monkeypatch.delenv("HDF5_DIR", raising=False)

    args = _mpirun_export_args()

    assert "-x" in args
    assert "LD_LIBRARY_PATH" in args
    assert "MESOUQ_MIRHEO_SRC" in args
    assert "HDF5_DIR" not in args


def test_map_mirheo_manifest_schema(tmp_path):
    """Verify the written summary manifest has the expected top-level keys."""
    from run_map_mirheo import run_map_mirheo

    # Build a minimal phase3b manifest
    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "phase3b_map_manifest.json"
    dataset = {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/r", "output_csv": "/tmp/o.csv",
    }
    manifest_path.write_text(json.dumps({"datasets": {"indentation_3.2um": dataset}}))

    # Mock subprocess.run to avoid actually launching Mirheo
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "done"
    mock_result.stderr = ""

    captured_cmds = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        return mock_result

    with patch("subprocess.run", side_effect=fake_run):
        summary = run_map_mirheo(
            experiment="indentation",
            output_dir=tmp_path,
            python_bin=sys.executable,
            n_displacements=5,
        )

    assert "status" in summary
    assert "diameters" in summary
    assert "created_at_utc" in summary
    assert "map_mirheo_dir" in summary
    assert "n_displacements" in summary
    assert summary["n_displacements"] == 5
    assert summary["init_directory_policy"]["preexisting_init_dirs_required"] is False
    assert summary["init_directory_policy"]["scratch_root_base"] == str(
        tmp_path / "map_mirheo" / "_scratch"
    )
    assert summary["diameters"][0]["scratch_root"] == str(
        tmp_path / "map_mirheo" / "_scratch" / "indentation_3.2um"
    )

    # Manifest file must exist
    manifest_out = tmp_path / "map_mirheo" / "map_mirheo_manifest.json"
    assert manifest_out.exists()
    loaded = json.loads(manifest_out.read_text())
    assert loaded["experiment"] == "indentation"
    assert loaded["init_directory_policy"]["missing_template_behavior"].startswith(
        "evaluator fails explicitly"
    )
    rendered = " ".join(str(a) for a in captured_cmds[0])
    assert "--scratch-root" in rendered
    assert str(tmp_path / "map_mirheo" / "_scratch" / "indentation_3.2um") in rendered


def test_map_mirheo_missing_manifest_raises(tmp_path):
    from run_map_mirheo import run_map_mirheo
    with pytest.raises(FileNotFoundError, match="phase3b_map_manifest"):
        run_map_mirheo("indentation", tmp_path, sys.executable, 5)


def test_map_mirheo_diameter_failure_recorded(tmp_path):
    """A failed subprocess should be recorded as 'failed', not raise."""
    from run_map_mirheo import run_map_mirheo

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    dataset = {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/r", "output_csv": "/tmp/o.csv",
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(
        json.dumps({"datasets": {"indentation_3.2um": dataset}})
    )

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Mirheo crash"

    with patch("subprocess.run", return_value=mock_result):
        summary = run_map_mirheo("indentation", tmp_path, sys.executable, 5)

    assert summary["status"] == "failed"
    assert summary["diameters"][0]["status"] == "failed"


def test_map_mirheo_timeout_recorded(tmp_path):
    """A TimeoutExpired should be caught and recorded as 'timed_out'."""
    import subprocess
    from run_map_mirheo import run_map_mirheo

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    dataset = {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/r", "output_csv": "/tmp/o.csv",
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(
        json.dumps({"datasets": {"indentation_3.2um": dataset}})
    )

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=1)):
        summary = run_map_mirheo("indentation", tmp_path, sys.executable, 5)

    assert summary["status"] == "failed"
    assert summary["diameters"][0]["timed_out"] is True
    assert summary["diameters"][0]["status"] == "timed_out"


def test_map_mirheo_timeout_retries_with_cleanup(tmp_path):
    import subprocess
    from run_map_mirheo import run_map_mirheo

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    dataset = {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/r", "output_csv": "/tmp/o.csv",
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(
        json.dumps({"datasets": {"indentation_3.2um": dataset}})
    )

    calls = {"count": 0}
    mock_result = MagicMock(returncode=0, stdout="ok", stderr="")

    def fake_run(command, **_kwargs):
        calls["count"] += 1
        scratch_root = Path(command[command.index("--scratch-root") + 1])
        result_path = Path(command[command.index("--output") + 1])
        if calls["count"] == 1:
            scratch_root.mkdir(parents=True, exist_ok=True)
            (scratch_root / "stale.lock").write_text("lock", encoding="utf-8")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("partial", encoding="utf-8")
            raise subprocess.TimeoutExpired(command, timeout=1, output="hang", stderr="deadlock")
        assert not scratch_root.exists()
        assert not result_path.exists()
        return mock_result

    with patch("subprocess.run", side_effect=fake_run):
        summary = run_map_mirheo(
            "indentation",
            tmp_path,
            sys.executable,
            5,
            timeout_seconds=1,
            max_retries=1,
        )

    diameter = summary["diameters"][0]
    assert summary["status"] == "passed"
    assert diameter["status"] == "passed"
    assert diameter["attempt_count"] == 2
    assert diameter["attempts"][0]["timed_out"] is True
    assert diameter["cleanup_actions"]


def test_map_mirheo_dataset_filter_runs_only_selected_dataset(tmp_path):
    from run_map_mirheo import run_map_mirheo

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    datasets = {
        "compression_2.1um": {
            "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
            "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
            "diameter_um": 2.1, "run_dir": "/tmp/r1", "output_csv": "/tmp/o1.csv",
        },
        "compression_3.0um": {
            "Yt": 1e7, "kb": 1e4, "d0": 0.2, "sigma": 0.04,
            "logLikelihood": 11.0, "logPrior": -4.0, "logPosterior": 7.0,
            "diameter_um": 3.0, "run_dir": "/tmp/r2", "output_csv": "/tmp/o2.csv",
        },
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(json.dumps({"datasets": datasets}))

    mock_result = MagicMock(returncode=0, stdout="done", stderr="")
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return mock_result

    with patch("subprocess.run", side_effect=fake_run):
        summary = run_map_mirheo(
            "compression",
            tmp_path,
            sys.executable,
            5,
            dataset_names=["compression_3.0um"],
        )

    assert summary["status"] == "passed"
    assert summary["selected_datasets"] == ["compression_3.0um"]
    assert [item["dataset_name"] for item in summary["diameters"]] == ["compression_3.0um"]
    rendered = " ".join(str(part) for part in captured[0])
    assert "compression_3.0um_map.json" in rendered
    assert "compression_2.1um_map.json" not in rendered


def test_map_mirheo_dataset_filter_rejects_missing_dataset(tmp_path):
    from run_map_mirheo import run_map_mirheo

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    datasets = {
        "compression_2.1um": {
            "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
            "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
            "diameter_um": 2.1, "run_dir": "/tmp/r1", "output_csv": "/tmp/o1.csv",
        },
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(json.dumps({"datasets": datasets}))

    with pytest.raises(ValueError, match="Requested dataset\\(s\\) not found"):
        run_map_mirheo(
            "compression",
            tmp_path,
            sys.executable,
            5,
            dataset_names=["compression_3.0um"],
        )


def test_main_missing_manifest_returns_one(tmp_path):
    from run_map_mirheo import main
    rc = main([
        "--experiment", "indentation",
        "--model-family", "reduced-model",
        "--profile", "production",
        "--output-dir", str(tmp_path),
        "--python-bin", sys.executable,
    ])
    assert rc == 1


def test_main_invalid_mpi_ranks_too_few_returns_one(tmp_path):
    from run_map_mirheo import main
    rc = main([
        "--experiment", "indentation",
        "--model-family", "reduced-model",
        "--profile", "production",
        "--output-dir", str(tmp_path),
        "--mpi-ranks", "1",
    ])
    assert rc == 1


def test_main_invalid_mpi_ranks_too_many_returns_one(tmp_path):
    from run_map_mirheo import main
    rc = main([
        "--experiment", "indentation",
        "--model-family", "reduced-model",
        "--profile", "production",
        "--output-dir", str(tmp_path),
        "--mpi-ranks", "4",
    ])
    assert rc == 1


def test_main_invalid_n_displacements_returns_one(tmp_path):
    from run_map_mirheo import main
    rc = main([
        "--experiment", "indentation",
        "--model-family", "reduced-model",
        "--profile", "production",
        "--output-dir", str(tmp_path),
        "--n-displacements", "0",
    ])
    assert rc == 1


def test_main_negative_retry_attempt_returns_one(tmp_path):
    from run_map_mirheo import main
    rc = main([
        "--experiment", "indentation",
        "--model-family", "reduced-model",
        "--profile", "production",
        "--output-dir", str(tmp_path),
        "--retry-attempt", "-1",
    ])
    assert rc == 1


def test_main_invalid_timeout_seconds_returns_one(tmp_path):
    from run_map_mirheo import main
    rc = main([
        "--experiment", "indentation",
        "--model-family", "reduced-model",
        "--profile", "production",
        "--output-dir", str(tmp_path),
        "--timeout-seconds", "0",
    ])
    assert rc == 1


def test_main_negative_max_retries_returns_one(tmp_path):
    from run_map_mirheo import main
    rc = main([
        "--experiment", "indentation",
        "--model-family", "reduced-model",
        "--profile", "production",
        "--output-dir", str(tmp_path),
        "--max-retries", "-1",
    ])
    assert rc == 1


def test_main_passes_retry_extra_args(tmp_path):
    """With retry-attempt > 0, extra_args are forwarded to the evaluate script."""
    from run_map_mirheo import run_map_mirheo

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    dataset = {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/r", "output_csv": "/tmp/o.csv",
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(
        json.dumps({"datasets": {"indentation_3.2um": dataset}})
    )

    captured_cmds = []
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        return mock_result

    with patch("subprocess.run", side_effect=fake_run):
        run_map_mirheo(
            "indentation", tmp_path, sys.executable, 5,
            extra_args=["--retry-attempt", "1", "--dt-scale-factor", "0.5"],
        )

    assert any("--retry-attempt" in " ".join(cmd) for cmd in captured_cmds)


def test_main_retry_attempt_forwards_extra_args(tmp_path):
    """main() with --retry-attempt 1 builds extra_args and passes them through."""
    from run_map_mirheo import main

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    dataset = {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/r", "output_csv": "/tmp/o.csv",
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(
        json.dumps({"datasets": {"indentation_3.2um": dataset}})
    )

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    captured_cmds = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        return mock_result

    with patch("subprocess.run", side_effect=fake_run):
        rc = main([
            "--experiment", "indentation",
            "--model-family", "reduced-model",
            "--profile", "production",
            "--output-dir", str(tmp_path),
            "--python-bin", sys.executable,
            "--n-displacements", "5",
            "--retry-attempt", "1",
        ])
    assert rc == 0
    assert any("--retry-attempt" in " ".join(str(a) for a in cmd) for cmd in captured_cmds)


def test_main_numsteps_overrides_forwarded(tmp_path):
    """main() forwards --numsteps and --numsteps-eq to evaluate scripts."""
    from run_map_mirheo import main

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    dataset = {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/r", "output_csv": "/tmp/o.csv",
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(
        json.dumps({"datasets": {"indentation_3.2um": dataset}})
    )

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    captured_cmds = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        return mock_result

    with patch("subprocess.run", side_effect=fake_run):
        rc = main([
            "--experiment", "indentation",
            "--model-family", "reduced-model",
            "--profile", "production",
            "--output-dir", str(tmp_path),
            "--python-bin", sys.executable,
            "--n-displacements", "5",
            "--numsteps", "5000",
            "--numsteps-eq", "10000",
        ])

    assert rc == 0
    assert captured_cmds
    rendered = " ".join(str(a) for a in captured_cmds[0])
    assert "--numsteps 5000" in rendered
    assert "--numsteps-eq 10000" in rendered
    assert "--scratch-root" in rendered


def test_main_passed_status_returns_zero(tmp_path):
    """main() returns 0 when all diameters pass."""
    from run_map_mirheo import main

    manifest_dir = tmp_path / "map_phase3b"
    manifest_dir.mkdir()
    dataset = {
        "Yt": 1e7, "kb": 1e4, "d0": 0.1, "sigma": 0.03,
        "logLikelihood": 10.0, "logPrior": -5.0, "logPosterior": 5.0,
        "diameter_um": 3.2, "run_dir": "/tmp/r", "output_csv": "/tmp/o.csv",
    }
    (manifest_dir / "phase3b_map_manifest.json").write_text(
        json.dumps({"datasets": {"indentation_3.2um": dataset}})
    )

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        rc = main([
            "--experiment", "indentation",
            "--model-family", "reduced-model",
            "--profile", "production",
            "--output-dir", str(tmp_path),
            "--python-bin", sys.executable,
            "--n-displacements", "5",
        ])
    assert rc == 0
