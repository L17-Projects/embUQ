from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_TEMPLATE = REPO_ROOT / "scripts" / "platforms" / "karolina" / "sbatch" / "full_test_suite.sbatch"


def test_karolina_full_test_suite_sbatch_records_compute_node_evidence() -> None:
    text = SBATCH_TEMPLATE.read_text(encoding="utf-8")

    assert "#SBATCH --account=eu-26-17" in text
    assert "#SBATCH --partition=qgpu_exp" in text
    assert "#SBATCH --gpus=1" in text
    assert 'source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"' in text
    assert "site override variables leak into tests" in text
    assert "-u SLURM_JOB_ID" in text
    assert "-u SLURM_ARRAY_JOB_ID" in text
    assert "MESOUQ_SITE_RUNTIME_ROOT" in text
    assert "MESOUQ_RUNS_ROOT" in text
    assert 'command=("${PYTHON_BIN}" -m pytest "${PYTEST_ARGS[@]}")' in text
    assert 'SUMMARY_JSON="${OUTPUT_ROOT}/full_test_suite_summary.json"' in text
    assert '"test_environment"' in text
    assert '"site_overrides_unset"' in text
    assert '"slurm_job_identity_hidden_from_pytest"' in text
    assert r"\d+(?:\.\d+)?s" in text
    assert '"mpi_gpu_applicability"' in text
    assert "does not launch mpirun/srun MPI ranks" in text
    assert "compute-node/runtime parity and CUDA visibility" in text
