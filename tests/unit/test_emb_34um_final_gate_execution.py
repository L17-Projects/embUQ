from __future__ import annotations

import importlib.util
import json
import types
import sys
from pathlib import Path

import pytest


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    script_path = (
        repo_root
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "run_emb_34um_final_gate_candidate.py"
    )
    spec = importlib.util.spec_from_file_location("run_emb_34um_final_gate_candidate", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_candidate_manifest(
    path: Path,
    *,
    force_grid: list[float] | None = None,
    use_legacy_yt: bool = False,
    ka: float = 160_000.0,
) -> None:
    request_payload = {
        "schema_version": "meso_uq.dpd_sampling.emb_34um_request.v1",
        "request_type": "emb_34um_full_force_sweep",
        "candidate_id": "emb-34um-test-001",
        "experiment": "indentation",
        "platform": "karolina",
        "output_root": str(path.parent / "emb" / "emb-34um-test-001"),
        "campaign_root": str(path.parent),
        "retry_limit": 3,
        "parameters": (
            {"Yt": 3.0e7, "kb": 4500.0} if use_legacy_yt else {"ka": ka, "kb": 4500.0}
        )
        | {
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
        },
        "force_grid": force_grid if force_grid is not None else [0.0, 2500.0, 5000.0],
        "force_grid_count": 3,
        "fingerprint": {
            "radp": 6.8,
            "L": 25.0,
            "fscale": 0.0074,
            "shell_th": 5.0e-9,
            "numsteps": 5000,
            "numsteps_eq": 10000,
        },
        "expected_output_paths": {
            "hdf5": str(path.parent / "emb" / "emb-34um-test-001" / "emb_34um_test.h5"),
            "request_manifest": str(
                path.parent / "emb" / "emb-34um-test-001" / "emb_34um_request_manifest.json"
            ),
        },
    }
    payload = {
        "candidate_id": "emb-34um-test-001",
        "family": "emb",
        "platform": "karolina",
        "output_root": request_payload["output_root"],
        "normalized_payload": request_payload,
        "rendered_payload": {
            "schema_version": "meso_uq.dpd_sampling.emb_render.v1",
            "family": "emb",
            "platform": "karolina",
            "candidate_id": "emb-34um-test-001",
            "request_payload": request_payload,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_candidate_request_parses_manifest_and_builds_execution_plan(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest)

    request = module.load_candidate_request(candidate_manifest)
    plan = module.execution_plan(request, retry_attempt=0)

    assert request.candidate_id == "emb-34um-test-001"
    assert request.force_grid == (0.0, 2500.0, 5000.0)
    assert request.compatibility_yt is None
    assert request.parameter_vector == (160000.0, 4500.0, 0.0, 0.0, 0.0, 0.0)
    assert plan["force_grid_count"] == 3
    assert plan["retry_limit"] == 3
    assert plan["expected_hdf5_path"].endswith("emb_34um_test.h5")


def test_load_candidate_request_supports_legacy_yt_parameter_parsing(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, use_legacy_yt=True)

    request = module.load_candidate_request(candidate_manifest)
    expected_ka = module._legacy_yt_to_ka(3.0e7, defaults=module._load_emb_default_parameters(), candidate_id="legacy")

    assert request.compatibility_yt == 3.0e7
    assert request.parameter_vector == (expected_ka, 4500.0, 0.0, 0.0, 0.0, 0.0)
    assert request.parameters["b1"] == 0.0
    assert request.parameters["b2"] == 0.0
    assert request.parameters["a3"] == 0.0
    assert request.parameters["a4"] == 0.0


def test_status_manifest_writes_runtime_sidecar_for_ingestion(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest)
    request = module.load_candidate_request(candidate_manifest)

    module._write_status_manifest(
        request,
        status="failed",
        retry_attempt=3,
        extra={"error_message": "runtime failed"},
    )

    request_status = json.loads(request.expected_request_manifest_path.read_text(encoding="utf-8"))
    runtime_status = json.loads((request.output_root / "emb_34um_runtime_status.json").read_text(encoding="utf-8"))
    assert request_status["status"] == "failed"
    assert runtime_status["retry_count"] == 3
    assert runtime_status["retry_limit"] == 3
    assert runtime_status["error_message"] == "runtime failed"


def test_write_outputs_records_ka_kb_parameter_vector(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest)
    request = module.load_candidate_request(candidate_manifest)
    sample = {
        "Reference Evaluations": [1.0, 2.0, 3.0],
        "Standard Deviation": [0.1, 0.2, 0.3],
    }

    class _FakeH5File:
        def __init__(self, *_: object, **__: object) -> None:
            self.attrs = {}

        def __enter__(self) -> "_FakeH5File":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def create_dataset(self, *_: object, **__: object) -> None:
            return None

    fake_module = types.SimpleNamespace(File=lambda *a, **k: _FakeH5File())
    sys.modules["h5py"] = fake_module

    module._write_outputs(request, sample, retry_attempt=1)

    result = json.loads((request.output_root / "emb_34um_result.json").read_text(encoding="utf-8"))
    assert result["parameter_names"] == ["ka", "kb", "b1", "b2", "a3", "a4"]
    assert result["parameters"] == list(request.parameter_vector)
    assert "d0" not in result["parameter_names"]
    assert "sigma" not in result["parameter_names"]


def test_select_candidate_manifest_and_srun_command_are_deterministic(tmp_path: Path) -> None:
    module = _load_runner_module()
    first = tmp_path / "emb" / "candidate-001" / "dpd_sampling_candidate_manifest.json"
    second = tmp_path / "emb" / "candidate-002" / "dpd_sampling_candidate_manifest.json"
    summary = tmp_path / "emb_34um_batch_summary.json"
    summary.write_text(
        json.dumps({"rendered_candidate_manifests": [str(first), str(second)]}),
        encoding="utf-8",
    )

    assert module.select_candidate_manifest(summary, 1) == second
    with pytest.raises(IndexError, match="outside rendered candidate range"):
        module.select_candidate_manifest(summary, 2)

    command = module.build_srun_command(
        runner_script="runner.py",
        candidate_manifest=second,
        retry_attempt=2,
        python_executable="python3",
        dry_run=True,
    )
    assert command == [
        "srun",
        "--ntasks=2",
        "python3",
        "runner.py",
        "--candidate-manifest",
        str(second),
        "--retry-attempt",
        "2",
        "--dry-run",
    ]


def test_candidate_manifest_validation_rejects_bad_force_grid(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, force_grid=[])

    with pytest.raises(ValueError, match="force_grid must not be empty"):
        module.load_candidate_request(candidate_manifest)


def test_karolina_sbatch_executes_runner_and_keeps_render_only_mode() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "platforms" / "karolina" / "sbatch" / "emb_34um_active_learning_array.sbatch"
    text = script.read_text(encoding="utf-8")

    assert "source /scratch/project/eu-26-17/eubrieucb/mesouq/load_mesouq_karolina.sh" in text
    assert 'EXECUTION_MODE="${EXECUTION_MODE:-execute}"' in text
    assert '"${EXECUTION_MODE}" == "render-only"' in text
    assert "run_emb_34um_final_gate_candidate.py" in text
    assert "srun --ntasks=2" in text
    assert "--mark-failed" in text
    assert "MODE=full requires submission with an sbatch --array range." in text
