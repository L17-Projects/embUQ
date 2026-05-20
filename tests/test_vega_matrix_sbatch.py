from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SBATCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "platforms" / "vega" / "sbatch"
KAROLINA_SBATCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "platforms" / "karolina" / "sbatch"

MATRIX_TEMPLATES = (
    "dnn_rebaseline_matrix.sbatch",
    "bnn_sweep_matrix.sbatch",
    "bnn_roundtrip_check.sbatch",
    "bnn_certification_matrix.sbatch",
)

ALL_TEMPLATES = MATRIX_TEMPLATES + ("promote_certified_bnn.sbatch",)


def _read(name: str) -> str:
    return (SBATCH_DIR / name).read_text(encoding="utf-8")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_matrix_template_exists(template: str) -> None:
    assert (SBATCH_DIR / template).exists(), f"Missing template: {template}"


def test_karolina_validation_matrix_template_uses_only_karolina_runtime_paths() -> None:
    text = (KAROLINA_SBATCH_DIR / "validation_matrix.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --account=eu-26-17" in text
    assert "#SBATCH --partition=qgpu" in text
    assert "#SBATCH --gpus=1" in text
    assert 'source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${MESOUQ_RUNS_ROOT}/validation_matrix/${RUN_TAG}}"' in text
    assert 'PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"' in text
    assert 'RUN_MAP_MIRHEO="${RUN_MAP_MIRHEO:-false}"' in text
    assert 'MAP_MIRHEO_N_DISPLACEMENTS="${MAP_MIRHEO_N_DISPLACEMENTS:-1}"' in text
    assert 'SKIP_RELEASE_MANIFEST="${SKIP_RELEASE_MANIFEST:-true}"' in text
    assert "--run-map-mirheo" in text
    assert "--skip-release-manifest" in text
    assert "_vega/" not in text


def test_vega_validation_matrix_template_matches_workflow_only_defaults() -> None:
    text = (SBATCH_DIR / "validation_matrix.sbatch").read_text(encoding="utf-8")

    assert 'RUN_MAP_MIRHEO="${RUN_MAP_MIRHEO:-false}"' in text
    assert 'MAP_MIRHEO_N_DISPLACEMENTS="${MAP_MIRHEO_N_DISPLACEMENTS:-1}"' in text
    assert 'SKIP_RELEASE_MANIFEST="${SKIP_RELEASE_MANIFEST:-true}"' in text
    assert "--run-map-mirheo" in text
    assert "--skip-release-manifest" in text


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_matrix_templates_use_login_shell_repo_venv_and_repo_pythonpath(template: str) -> None:
    text = _read(template)
    assert text.startswith("#!/bin/bash -l\n")
    assert 'VENV_DIR="${VENV_DIR:-${REPO_ROOT}/_vega/venv}"' in text
    assert 'source "${VENV_DIR}/bin/activate"' in text
    assert 'export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in text
    assert 'source "${REPO_ROOT}/_vega/korali/env.sh"' in text


@pytest.mark.parametrize("template", ("bnn_sweep_matrix.sbatch", "bnn_certification_matrix.sbatch"))
def test_gpu_templates_redirect_scheduler_logs_into_output_root(template: str) -> None:
    text = _read(template)
    assert "#SBATCH --output=/dev/null" in text
    assert "#SBATCH --error=/dev/null" in text
    assert 'LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"' in text
    assert 'mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"' in text
    assert 'exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out")' in text
    assert '2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)' in text


def test_promote_template_redirects_scheduler_logs_into_certification_root() -> None:
    text = _read("promote_certified_bnn.sbatch")
    assert "#SBATCH --output=/dev/null" in text
    assert "#SBATCH --error=/dev/null" in text
    assert 'DEFAULT_LOG_BASE="${CERTIFICATION_ROOT:-${REPO_ROOT}/_runs/vega/bnn_promotion_preflight}"' in text
    assert 'LOG_ROOT="${LOG_ROOT:-${DEFAULT_LOG_BASE}/logs}"' in text
    assert 'mkdir -p "${LOG_ROOT}"' in text
    assert 'exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out")' in text
    assert '2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)' in text
    assert text.index('exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out")') < text.index(
        'if [[ -z "${CERTIFICATION_ROOT}" ]]; then'
    )


@pytest.mark.parametrize("template", ("dnn_rebaseline_matrix.sbatch", "bnn_roundtrip_check.sbatch"))
def test_matrix_templates_keep_scheduler_logs_visible_and_mirror_logs_into_output_root(template: str) -> None:
    text = _read(template)
    assert "#SBATCH --output=%x-%j.out" in text
    assert "#SBATCH --error=%x-%j.err" in text
    assert 'LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"' in text
    assert 'mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"' in text
    assert 'exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out")' in text
    assert '2> >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.err" >&2)' in text


def test_dnn_rebaseline_template_dispatches_cpu_runner_with_only_seed_and_resume_knobs() -> None:
    text = _read("dnn_rebaseline_matrix.sbatch")
    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "CUDA/12.2.2" not in text
    assert "openmpi" not in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/dnn_rebaseline/${RUN_TAG}}"' in text
    assert 'SELECTIONS="${SELECTIONS:-${ONLY:-}}"' in text
    assert 'SEEDS="${SEEDS:-${SEED:-}}"' in text
    assert 'selection_args+=(--only "${selection}")' in text
    assert 'seed_args+=(--seed "${seed}")' in text
    assert "--resume" in text
    assert "--no-resume" in text
    assert "run_dnn_rebaseline_matrix.py" in text


def test_bnn_sweep_template_dispatches_gpu_runner_without_openmpi() -> None:
    text = _read("bnn_sweep_matrix.sbatch")
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "CUDA/12.2.2" in text
    assert "openmpi" not in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/bnn_sweep/${RUN_TAG}}"' in text
    assert 'DNN_ROOT="${DNN_ROOT:-}"' in text
    assert 'SELECTIONS="${SELECTIONS:-${ONLY:-}}"' in text
    assert 'SEEDS="${SEEDS:-${SEED:-}}"' in text
    assert 'ARCHITECTURE_SOURCE="${ARCHITECTURE_SOURCE:-explicit}"' in text
    assert 'STAGE1_PRIOR_SCALES="${STAGE1_PRIOR_SCALES:-}"' in text
    assert 'STAGE1_OBS_NOISE_PRIOR_SCALES="${STAGE1_OBS_NOISE_PRIOR_SCALES:-}"' in text
    assert 'STAGE1_LRS="${STAGE1_LRS:-}"' in text
    assert 'DEVICE="${DEVICE:-cuda}"' in text
    assert 'REQUIRE_PARITY="${REQUIRE_PARITY:-false}"' in text
    assert 'selection_args+=(--only "${selection}")' in text
    assert 'seed_args+=(--seed "${seed}")' in text
    assert '--architecture-source "${ARCHITECTURE_SOURCE}"' in text
    assert '--dnn-root "${DNN_ROOT}"' in text
    assert "--stage1-prior-scales" in text
    assert "--stage1-obs-noise-prior-scales" in text
    assert "--stage1-lrs" in text
    assert "--require-parity" in text
    assert "--no-require-parity" in text
    assert "--max-walltime-seconds" in text
    assert "torch.cuda.is_available()" in text
    assert "run_bnn_sweep_matrix.py" in text


def test_bnn_certification_template_dispatches_gpu_runner_with_seeded_roots() -> None:
    text = _read("bnn_certification_matrix.sbatch")
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "CUDA/12.2.2" in text
    assert "openmpi" not in text
    assert 'DNN_ROOT="${DNN_ROOT:-}"' in text
    assert 'BNN_ROOT="${BNN_ROOT:-}"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/bnn_certification/${RUN_TAG}}"' in text
    assert 'SELECTIONS="${SELECTIONS:-${ONLY:-}}"' in text
    assert 'SEEDS="${SEEDS:-${SEED:-}}"' in text
    assert 'DEVICE="${DEVICE:-cuda}"' in text
    assert 'selection_args+=(--only "${selection}")' in text
    assert 'seed_args+=(--seed "${seed}")' in text
    assert "--predictive-mc-chunk-size" in text
    assert "--bootstrap-resamples" in text
    assert "--acceptance-upper-bound" in text
    assert "--resume" in text
    assert "--no-resume" in text
    assert "torch.cuda.is_available()" in text
    assert "run_bnn_certification_matrix.py" in text
    assert text.index('exec > >(tee -a "${LOG_ROOT}/slurm-${JOB_TOKEN}.out")') < text.index(
        'if [[ -z "${DNN_ROOT}" || -z "${BNN_ROOT}" ]]; then'
    )


def test_bnn_roundtrip_template_dispatches_gpu_runner_with_selection_and_reload_knobs() -> None:
    text = _read("bnn_roundtrip_check.sbatch")
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "CUDA/12.2.2" in text
    assert "openmpi" not in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/${SITE}/bnn_roundtrip/${RUN_TAG}}"' in text
    assert 'SELECTION="${SELECTION:-indentation_3.2um}"' in text
    assert 'command+=(--reload-tol-rel "${RELOAD_TOL_REL}")' in text
    assert 'command+=(--predictive-mc-chunk-size "${PREDICTIVE_MC_CHUNK_SIZE}")' in text
    assert "torch.cuda.is_available()" in text
    assert "run_bnn_roundtrip_check.py" in text


def test_promote_template_dispatches_cpu_runner_with_certification_root_and_dry_run() -> None:
    text = _read("promote_certified_bnn.sbatch")
    assert "#SBATCH --partition=cpu" in text
    assert "CUDA/12.2.2" not in text
    assert "openmpi" not in text
    assert 'CERTIFICATION_ROOT="${CERTIFICATION_ROOT:-}"' in text
    assert 'DRY_RUN="${DRY_RUN:-false}"' in text
    assert 'command+=(--manifest-path "${MANIFEST_PATH}")' in text
    assert 'command+=(--dry-run)' in text
    assert "promote_certified_bnn.py" in text


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/platforms/vega/run_dnn_rebaseline_matrix.py",
        "scripts/platforms/vega/run_bnn_sweep_matrix.py",
        "scripts/platforms/vega/run_bnn_certification_matrix.py",
        "scripts/platforms/vega/run_bnn_roundtrip_check.py",
        "scripts/platforms/vega/promote_certified_bnn.py",
    ],
)
def test_vega_python_wrappers_dispatch_to_hpc(monkeypatch, relative_path: str) -> None:
    module = _load_module(Path(relative_path), f"dispatch_{Path(relative_path).stem}_test")
    captured: list[list[str]] = []
    expected_abs_target = str((Path(relative_path).resolve().parent.parent / "hpc" / Path(relative_path).name).resolve())

    def fake_call(command):  # noqa: ANN001
        captured.append(list(command))
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)
    rc = module.main(["--flag", "value"])
    assert rc == 0
    assert captured == [[sys.executable, expected_abs_target, "--site", "vega", "--flag", "value"]]


def test_site_python_wrappers_reject_conflicting_site(monkeypatch, capsys) -> None:
    module = _load_module(
        Path("scripts/platforms/vega/run_bnn_sweep_matrix.py"),
        "dispatch_conflicting_site_test",
    )

    def fail_call(command):  # noqa: ANN001
        raise AssertionError(f"unexpected dispatch: {command}")

    monkeypatch.setattr(module.subprocess, "call", fail_call)

    rc = module.main(["--site", "karolina"])

    assert rc == 2
    assert "compatibility entrypoint" in capsys.readouterr().err


def test_site_python_wrappers_accept_matching_explicit_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_bnn_sweep_matrix.py"),
        "dispatch_matching_site_test",
    )
    captured: list[list[str]] = []

    def fake_call(command):  # noqa: ANN001
        captured.append(list(command))
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)

    rc = module.main(["--site", "karolina", "--flag", "value"])

    assert rc == 0
    assert captured
    joined = " ".join(captured[0])
    assert "scripts/platforms/hpc/run_bnn_sweep_matrix.py" in joined
    assert joined.count("--site karolina") == 1
