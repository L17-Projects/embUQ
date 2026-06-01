from __future__ import annotations

import importlib.util
import json
import math
import types
import sys
from pathlib import Path
from typing import Any

import pytest


_DEFAULT_CANONICAL_FORCE_GRID = [
    0.0,
    714.2857142857143,
    1428.5714285714287,
    2142.857142857143,
    2857.1428571428573,
    3571.4285714285716,
    4285.714285714286,
    5000.0,
]


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


def _runtime_fingerprint_payload(
    *,
    radp: float = 6.60,
    shell_th: float = 3.75e-9,
    bpress: float = -91.0,
) -> dict[str, Any]:
    lx = float(math.ceil(2.0 * radp + 6.0))
    lz = float(math.ceil(2.0 * radp + 10.0))
    return {
        "radp": radp,
        "Lx": lx,
        "Ly": lx,
        "Lz": lz,
        "L": lz,
        "fscale": 0.0074,
        "shell_th": shell_th,
        "numsteps": 5000,
        "numsteps_eq": 10000,
        "direct_stiffness_override": True,
        "bpress": bpress,
    }


def _write_candidate_manifest(
    path: Path,
    *,
    force_grid: list[float] | None = None,
    use_legacy_yt: bool = False,
    ka: float = 60_000.0,
    kb: float = 4500.0,
    fingerprint: dict[str, object] | None = None,
) -> None:
    runtime_fingerprint = _runtime_fingerprint_payload()
    fingerprint_payload = dict(runtime_fingerprint if fingerprint is None else fingerprint)
    request_runtime_payload = dict(runtime_fingerprint)
    request_runtime_payload.update(fingerprint_payload)
    request_payload = {
        "schema_version": "meso_uq.dpd_sampling.emb_34um_request.v1",
        "request_type": "emb_34um_full_force_sweep",
        "candidate_id": "emb-34um-test-001",
        "experiment": "indentation",
        "platform": "karolina",
        "output_root": str(path.parent / "emb" / "emb-34um-test-001"),
        "campaign_root": str(path.parent),
        "retry_limit": 0,
        "parameters": (
            {"Yt": 3.0e7, "kb": kb} if use_legacy_yt else {"ka": ka, "kb": kb}
        )
        | {
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
            "radp": request_runtime_payload["radp"],
            "shell_th": request_runtime_payload["shell_th"],
            "bpress": request_runtime_payload["bpress"],
        },
        "force_grid": force_grid if force_grid is not None else _DEFAULT_CANONICAL_FORCE_GRID,
        "force_grid_count": len(_DEFAULT_CANONICAL_FORCE_GRID),
        "fingerprint": fingerprint_payload,
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
    assert request.force_grid == tuple(_DEFAULT_CANONICAL_FORCE_GRID)
    assert request.compatibility_yt is None
    assert request.parameter_vector == (60000.0, 4500.0, 0.0, 0.0, 0.0, 0.0)
    assert request.fingerprint["Lx"] == 20.0
    assert request.fingerprint["Ly"] == 20.0
    assert request.fingerprint["Lz"] == 24.0
    assert request.fingerprint["radp"] == 6.60
    assert request.fingerprint["shell_th"] == 3.75e-9
    assert request.fingerprint["direct_stiffness_override"] is True
    assert request.fingerprint["bpress"] == -91.0
    assert plan["force_grid_count"] == 8
    assert plan["retry_limit"] == 0
    assert plan["expected_hdf5_path"].endswith("emb_34um_test.h5")


def test_load_candidate_request_supports_legacy_yt_parameter_parsing(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, use_legacy_yt=True)

    request = module.load_candidate_request(candidate_manifest)
    expected_defaults = module._load_emb_default_parameters()
    expected_defaults["shell_th"] = 3.75e-9
    expected_ka = module._legacy_yt_to_ka(3.0e7, defaults=expected_defaults, candidate_id="legacy")

    assert request.compatibility_yt == 3.0e7
    assert request.parameter_vector == (expected_ka, 4500.0, 0.0, 0.0, 0.0, 0.0)
    assert request.parameters["b1"] == 0.0
    assert request.parameters["b2"] == 0.0
    assert request.parameters["a3"] == 0.0
    assert request.parameters["a4"] == 0.0


def test_load_candidate_request_rejects_d4_payloads_missing_runtime_geometry(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    request_payload = {
        "schema_version": "meso_uq.dpd_sampling.emb_34um_request.v1",
        "request_type": "emb_34um_full_force_sweep",
        "candidate_id": "emb-34um-test-001",
        "experiment": "indentation",
        "platform": "karolina",
        "output_root": str(candidate_manifest.parent / "emb" / "emb-34um-test-001"),
        "campaign_root": str(candidate_manifest.parent),
        "retry_limit": 0,
        "parameters": {
            "ka": 60_000.0,
            "kb": 4500.0,
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
        },
        "force_grid": _DEFAULT_CANONICAL_FORCE_GRID,
        "force_grid_count": len(_DEFAULT_CANONICAL_FORCE_GRID),
        "fingerprint": {},
        "expected_output_paths": {
            "hdf5": str(candidate_manifest.parent / "emb" / "emb-34um-test-001" / "emb_34um_test.h5"),
            "request_manifest": str(
                candidate_manifest.parent / "emb" / "emb-34um-test-001" / "emb_34um_request_manifest.json"
            ),
        },
    }
    candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
    candidate_manifest.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing D4 runtime parameters"):
        module.load_candidate_request(candidate_manifest)


def test_load_candidate_request_rejects_d4_payloads_outside_runtime_geometry_bounds(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, fingerprint=_runtime_fingerprint_payload(shell_th=5.0e-9))
    payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    payload["rendered_payload"]["request_payload"]["parameters"]["shell_th"] = 5.0e-9
    payload["normalized_payload"]["parameters"]["shell_th"] = 5.0e-9
    candidate_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="outside bounds"):
        module.load_candidate_request(candidate_manifest)


def test_load_candidate_request_rejects_d4_payloads_outside_ka_bounds(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, ka=99.0, kb=4500.0)

    with pytest.raises(ValueError, match="parameters\\.ka=.*outside bounds"):
        module.load_candidate_request(candidate_manifest)


def test_load_candidate_request_rejects_d4_payloads_outside_kb_bounds(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, ka=60_000.0, kb=350.0)

    with pytest.raises(ValueError, match="parameters\\.kb=.*outside bounds"):
        module.load_candidate_request(candidate_manifest)


def test_load_candidate_request_rejects_below_operational_bounds_before_timeout_corner(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, ka=100.0, kb=400.0)

    with pytest.raises(ValueError, match="outside bounds"):
        module.load_candidate_request(candidate_manifest)


def test_load_candidate_request_rejects_runtime_risk_timeout_neighborhood(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(
        candidate_manifest,
        ka=1272.145949837055,
        kb=9011.372292672091,
        fingerprint=_runtime_fingerprint_payload(
            radp=6.489854540732141,
            shell_th=3.9170402720459855e-9,
        ),
    )

    with pytest.raises(ValueError, match="runtime-risk timeout region"):
        module.load_candidate_request(candidate_manifest)


def test_load_candidate_request_rejects_final_gate_unseen_timeout_neighborhood(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(
        candidate_manifest,
        ka=22432.81994559314,
        kb=2436.648543412221,
        fingerprint=_runtime_fingerprint_payload(
            radp=6.584751551356739,
            shell_th=3.844815260086293e-9,
        ),
    )

    with pytest.raises(ValueError, match="runtime-risk timeout region"):
        module.load_candidate_request(candidate_manifest)


def test_status_manifest_writes_runtime_sidecar_for_ingestion(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest)
    request = module.load_candidate_request(candidate_manifest)

    module._write_status_manifest(
        request,
        status="failed",
        retry_attempt=0,
        extra={"error_message": "runtime failed"},
    )

    request_status = json.loads(request.expected_request_manifest_path.read_text(encoding="utf-8"))
    runtime_status = json.loads((request.output_root / "emb_34um_runtime_status.json").read_text(encoding="utf-8"))
    assert request_status["status"] == "failed"
    assert runtime_status["retry_count"] == 0
    assert runtime_status["retry_limit"] == 0
    assert runtime_status["error_message"] == "runtime failed"


def test_clean_transient_candidate_outputs_preserves_manifests(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest)
    request = module.load_candidate_request(candidate_manifest)

    transient_dir = request.output_root / "out_hierarchical"
    transient_dir.mkdir(parents=True)
    (transient_dir / "partial.dat").write_text("partial", encoding="utf-8")
    for path in (
        request.output_root / "F_Delta.dat",
        request.output_root / "emb_34um_result.json",
        request.output_root / "emb_34um_runtime_status.json",
        request.expected_hdf5_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")
    request.expected_request_manifest_path.write_text("keep request manifest", encoding="utf-8")
    (request.output_root / "dpd_production_preflight.json").write_text("keep preflight", encoding="utf-8")

    module._clean_transient_candidate_outputs(request)

    assert not transient_dir.exists()
    assert not (request.output_root / "F_Delta.dat").exists()
    assert not (request.output_root / "emb_34um_result.json").exists()
    assert not (request.output_root / "emb_34um_runtime_status.json").exists()
    assert not request.expected_hdf5_path.exists()
    assert candidate_manifest.is_file()
    assert request.expected_request_manifest_path.read_text(encoding="utf-8") == "keep request manifest"
    assert (request.output_root / "dpd_production_preflight.json").read_text(encoding="utf-8") == "keep preflight"


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

    module._write_outputs(request, sample, retry_attempt=1, runtime_seconds=12.5)

    result = json.loads((request.output_root / "emb_34um_result.json").read_text(encoding="utf-8"))
    assert result["parameter_names"] == ["ka", "kb", "b1", "b2", "a3", "a4"]
    assert result["parameters"] == list(request.parameter_vector)
    assert request.parameters["b1"] == 0.0
    assert request.parameters["b2"] == 0.0
    assert request.parameters["a3"] == 0.0
    assert request.parameters["a4"] == 0.0
    assert result["runtime_seconds"] == 12.5
    assert result["radp"] == 6.60
    assert result["shell_th"] == 3.75e-9
    assert result["bpress"] == -91.0
    assert result["runtime_fingerprint"]["radp"] == 6.60
    assert result["runtime_fingerprint"]["shell_th"] == 3.75e-9
    assert result["runtime_fingerprint"]["Lx"] == 20.0
    assert result["runtime_fingerprint"]["Ly"] == 20.0
    assert result["runtime_fingerprint"]["Lz"] == 24.0
    assert result["box_dimensions"] == {"Lx": 20.0, "Ly": 20.0, "Lz": 24.0}
    assert "d0" not in result["parameter_names"]
    assert "sigma" not in result["parameter_names"]


def test_write_outputs_preserves_d4_fields_from_request_parameters_when_fingerprint_is_empty(
    tmp_path: Path,
) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-001" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, fingerprint={})
    request = module.load_candidate_request(candidate_manifest)
    assert request.fingerprint == {}
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

    module._write_outputs(request, sample, retry_attempt=1, runtime_seconds=12.5)

    result = json.loads((request.output_root / "emb_34um_result.json").read_text(encoding="utf-8"))
    assert result["radp"] == 6.60
    assert result["shell_th"] == 3.75e-9
    assert result["bpress"] == -91.0
    assert result["runtime_fingerprint"]["radp"] == 6.60
    assert result["runtime_fingerprint"]["shell_th"] == 3.75e-9
    assert result["runtime_fingerprint"]["Lx"] == 20.0
    assert result["runtime_fingerprint"]["Ly"] == 20.0
    assert result["runtime_fingerprint"]["Lz"] == 24.0
    assert result["runtime_fingerprint"]["fscale"] == 0.0074
    assert result["runtime_fingerprint"]["numsteps"] == 5000
    assert result["runtime_fingerprint"]["numsteps_eq"] == 10000
    assert result["runtime_fingerprint"]["direct_stiffness_override"] is True
    assert result["box_dimensions"] == {"Lx": 20.0, "Ly": 20.0, "Lz": 24.0}


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


def test_candidate_manifest_validation_rejects_noncanonical_force_grid(tmp_path: Path) -> None:
    module = _load_runner_module()
    candidate_manifest = tmp_path / "emb" / "emb-34um-test-002" / "dpd_sampling_candidate_manifest.json"
    _write_candidate_manifest(candidate_manifest, force_grid=[0.0, 2500.0, 5000.0])

    with pytest.raises(ValueError, match="must contain 8 points"):
        module.load_candidate_request(candidate_manifest)


def test_karolina_sbatch_executes_runner_and_keeps_render_only_mode() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "platforms" / "karolina" / "sbatch" / "emb_34um_active_learning_array.sbatch"
    text = script.read_text(encoding="utf-8")

    assert ("load_mesouq" + "_karolina.sh") not in text
    assert "scripts/platforms/hpc/site_env.sh" in text
    assert "mesouq_activate_site_env karolina" in text
    assert 'EXECUTION_MODE="${EXECUTION_MODE:-execute}"' in text
    assert '"${EXECUTION_MODE}" == "render-only"' in text
    assert "run_emb_34um_final_gate_candidate.py" in text
    assert 'OMPI_MCA_btl="${MESOUQ_KAROLINA_OMPI_MCA_BTL:-self,vader,tcp}"' in text
    assert '"OMPI_MCA_btl=${OMPI_MCA_btl}"' in text
    assert "srun --ntasks=2 --kill-on-bad-exit=1" in text
    assert "--mark-failed" in text
    assert "MODE=full requires submission with an sbatch --array range." in text
