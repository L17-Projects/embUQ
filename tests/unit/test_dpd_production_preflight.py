from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
LEGACY_SITE_ENV = "HPC" "_SITE"

from meso_uq.dpd_sampling.preflight import (
    DPD_PRODUCTION_PREFLIGHT_SCHEMA_VERSION,
    DPDProductionPreflightConfig,
    output_root_from_candidate_manifest,
    run_dpd_production_preflight,
    write_preflight_manifest,
)


def _candidate_manifest(path: Path, *, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate_id": "emb-34um-test-001",
                "family": "emb",
                "platform": "karolina",
                "output_root": str(output_root),
                "rendered_payload": {
                    "request_payload": {
                        "candidate_id": "emb-34um-test-001",
                        "output_root": str(output_root),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_preflight_rejects_non_scratch_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "home" / "dpd-output"
    payload = run_dpd_production_preflight(
        DPDProductionPreflightConfig(
            output_root=output_root,
            require_scratch=True,
            min_free_bytes=1,
            env={},
        )
    )

    assert payload["schema_version"] == DPD_PRODUCTION_PREFLIGHT_SCHEMA_VERSION
    assert payload["status"] == "failed"
    assert payload["checks"][0]["name"] == "scratch_output_root"
    assert payload["checks"][0]["status"] == "failed"
    assert not output_root.exists()


def test_preflight_rejects_home_path_even_if_it_contains_scratch() -> None:
    output_root = Path("/home/test-user/scratch/mesouq-output")
    payload = run_dpd_production_preflight(
        DPDProductionPreflightConfig(
            output_root=output_root,
            require_scratch=True,
            min_free_bytes=1,
            env={},
        )
    )

    assert payload["status"] == "failed"
    assert payload["checks"][0]["name"] == "scratch_output_root"
    assert payload["checks"][0]["status"] == "failed"


def test_preflight_rejects_tmp_scratch_without_explicit_scratch_root() -> None:
    output_root = Path("/tmp/scratch/candidate")
    payload = run_dpd_production_preflight(
        DPDProductionPreflightConfig(
            output_root=output_root,
            require_scratch=True,
            min_free_bytes=1,
            env={},
        )
    )

    assert payload["status"] == "failed"
    assert payload["checks"][0]["name"] == "scratch_output_root"
    assert payload["checks"][0]["status"] == "failed"


def test_preflight_passes_for_scratch_like_writable_root(tmp_path: Path) -> None:
    scratch_root = tmp_path / "scratch"
    output_root = scratch_root / "project" / "mesouq" / "runs" / "candidate"
    payload = run_dpd_production_preflight(
        DPDProductionPreflightConfig(
            output_root=output_root,
            require_scratch=True,
            allowed_scratch_roots=(scratch_root,),
            min_free_bytes=1,
            env={"SLURM_JOB_ID": "12345", "MESOUQ_SITE": "karolina"},
        )
    )

    assert payload["status"] == "passed"
    assert {check["name"]: check["status"] for check in payload["checks"]} == {
        "scratch_output_root": "passed",
        "output_root_writable": "passed",
        "free_space": "passed",
        "nested_srun_cpu_environment": "passed",
    }
    assert payload["storage"]["free_bytes"] >= 1
    assert payload["environment"]["SLURM_JOB_ID"] == "12345"
    assert payload["environment"]["MESOUQ_SITE"] == "karolina"


def test_preflight_detects_unsanitized_nested_srun_cpu_environment(tmp_path: Path) -> None:
    scratch_root = tmp_path / "scratch"
    output_root = scratch_root / "project" / "mesouq" / "runs" / "candidate"
    payload = run_dpd_production_preflight(
        DPDProductionPreflightConfig(
            output_root=output_root,
            require_scratch=True,
            allowed_scratch_roots=(scratch_root,),
            min_free_bytes=1,
            env={"SLURM_CPUS_PER_TASK": "5", "SLURM_TRES_PER_TASK": "cpu=8"},
        )
    )

    assert payload["status"] == "failed"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["nested_srun_cpu_environment"]["status"] == "failed"
    assert "SLURM_CPUS_PER_TASK=5" in checks["nested_srun_cpu_environment"]["message"]


def test_preflight_resolves_output_root_from_candidate_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "scratch" / "project" / "mesouq" / "runs" / "candidate"
    manifest = tmp_path / "candidate" / "dpd_sampling_candidate_manifest.json"
    _candidate_manifest(manifest, output_root=output_root)

    assert output_root_from_candidate_manifest(manifest) == output_root.resolve()


def test_preflight_manifest_writer_creates_parent_directories(tmp_path: Path) -> None:
    manifest_path = tmp_path / "scratch" / "project" / "preflight" / "dpd_production_preflight.json"
    payload = {
        "schema_version": DPD_PRODUCTION_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
    }

    written = write_preflight_manifest(payload, manifest_path)

    assert written == manifest_path.resolve()
    assert json.loads(written.read_text(encoding="utf-8")) == payload


def test_preflight_cli_does_not_create_default_manifest_under_unsafe_root(tmp_path: Path) -> None:
    output_root = tmp_path / "not-scratch" / "candidate"
    script = REPO_ROOT / "scripts" / "workflows" / "dpd" / "run_dpd_production_preflight.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(output_root),
            "--min-free-bytes",
            "1",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert not output_root.exists()
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert payload["manifest_path"] is None


def test_dpd_preflight_helper_has_syntax_and_site_neutral_controls() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    helper = repo_root / "scripts" / "workflows" / "dpd" / "dpd_production_preflight.sh"

    subprocess.run(["bash", "-n", str(helper)], cwd=repo_root, check=True)
    text = helper.read_text(encoding="utf-8")
    assert "mesouq_sanitize_nested_srun_cpu_env" in text
    assert "mesouq_run_dpd_production_preflight" in text
    assert 'DPD_PREFLIGHT_HDF5_DRIVER="${DPD_PREFLIGHT_HDF5_DRIVER:-serial}"' in text
    assert "--scratch-root \"${SCRATCH_ROOT}\"" in text
    assert "--scratch-root \"${MESOUQ_SCRATCH_ROOT}\"" in text
    assert "karolina" not in text.lower()


@pytest.mark.parametrize(
    ("relative_script", "site"),
    [
        ("scripts/platforms/hpc/sbatch/dpd_production_preflight_canary.sbatch", None),
        ("scripts/platforms/karolina/sbatch/dpd_production_preflight_canary.sbatch", "karolina"),
        ("scripts/platforms/vega/sbatch/dpd_production_preflight_canary.sbatch", "vega"),
    ],
)
def test_dpd_preflight_canary_sbatch_is_site_aware(relative_script: str, site: str | None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / relative_script

    subprocess.run(["bash", "-n", str(script)], cwd=repo_root, check=True)
    text = script.read_text(encoding="utf-8")
    assert LEGACY_SITE_ENV not in text
    if site is None:
        assert "This dispatcher submits a site-specific Slurm template" in text
        assert "exec sbatch" in text
        assert '--export=ALL,MESOUQ_SITE="${SITE}",REPO_ROOT="${REPO_ROOT}"' in text
        assert '"${REPO_ROOT}/scripts/platforms/${SITE}/sbatch/dpd_production_preflight_canary.sbatch"' in text
        assert "vega|karolina" in text
    else:
        assert "run_dpd_production_preflight.py" in text
        assert "--scratch-root" in text
        assert "--hdf5-smoke-test" in text
        assert "--hdf5-driver" in text
        assert 'DPD_PREFLIGHT_HDF5_DRIVER="${DPD_PREFLIGHT_HDF5_DRIVER:-serial}"' in text
        assert 'source "${DPD_PREFLIGHT_HELPER}"' in text
        assert "mesouq_sanitize_nested_srun_cpu_env" in text
        assert f'SITE="{site}"' in text
        assert f'export MESOUQ_SITE="${{SITE}}"' in text
        assert "--output-root" in text
        assert 'SCRATCH_ROOT="${SCRATCH_ROOT:-${MESOUQ_SCRATCH_ROOT:-}}"' in text
        if site == "karolina":
            assert "#SBATCH --gpus=1" in text
            assert "#SBATCH --gres=gpu" not in text
            assert 'source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"' in text
        if site == "vega":
            assert "#SBATCH --gres=gpu:1" in text
            assert "#SBATCH --gpus=" not in text
            assert 'source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"' in text
            assert 'mesouq_activate_site_env vega "${REPO_ROOT}"' in text
            assert 'PYTHON_BIN="${PYTHON_BIN:-${MESOUQ_ENV_ROOT}/bin/python}"' in text
            assert "OpenMPI/4.1.4-GCC-12.2.0" in text
            assert "openmpi/4.1.2.1" not in text


@pytest.mark.parametrize(
    "relative_script",
    [
        "scripts/platforms/karolina/sbatch/emb_34um_active_learning_array.sbatch",
        "scripts/platforms/karolina/sbatch/emb_34um_dnn_causal_validation_array.sbatch",
    ],
)
def test_karolina_emb_candidate_sbatch_runs_shared_preflight_before_candidate_execution(
    relative_script: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / relative_script
    subprocess.run(["bash", "-n", str(script)], cwd=repo_root, check=True)
    text = script.read_text(encoding="utf-8")

    assert "PREFLIGHT_SCRIPT=" in text
    assert "DPD_PREFLIGHT_HELPER=" in text
    assert 'DPD_PREFLIGHT_HDF5_DRIVER="${DPD_PREFLIGHT_HDF5_DRIVER:-serial}"' in text
    assert "run_dpd_production_preflight.py" in text
    assert "dpd_production_preflight.sh" in text
    assert 'source "${DPD_PREFLIGHT_HELPER}"' in text
    assert "mesouq_sanitize_nested_srun_cpu_env" in text
    assert "mesouq_run_dpd_production_preflight" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --exclude=acn05,acn13,acn14" in text
    assert 'MESOUQ_DPD_ARRAY_LAUNCH_STAGGER_SECONDS="${MESOUQ_DPD_ARRAY_LAUNCH_STAGGER_SECONDS:-4}"' in text
    assert "Staggering DPD array launch" in text
    assert 'OMPI_MCA_btl="${MESOUQ_KAROLINA_OMPI_MCA_BTL:-self,vader,tcp}"' in text
    assert '"OMPI_MCA_btl=${OMPI_MCA_btl}"' in text
    assert "srun --ntasks=2 --kill-on-bad-exit=1" in text
    assert "mesouq_activate_site_env karolina" in text
    assert "/scratch/project/eu-26-17/eubrieucb/mesouq/load_mesouq_karolina.sh" not in text
    assert text.index("MESOUQ_DPD_ARRAY_LAUNCH_STAGGER_SECONDS") < text.index(
        "mesouq_activate_site_env karolina"
    )
    assert text.index("mesouq_sanitize_nested_srun_cpu_env") < text.index("mesouq_run_dpd_production_preflight")
    assert text.index("mesouq_run_dpd_production_preflight") < text.index('"${PYTHON_EXECUTABLE}" "${RUNNER_SCRIPT}"')
