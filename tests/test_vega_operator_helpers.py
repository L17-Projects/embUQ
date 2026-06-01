import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_latest_state(latest_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {
                "Variables": [
                    {"Name": "Yt"},
                    {"Name": "kb"},
                    {"Name": "d0"},
                    {"Name": "[Sigma]"},
                ],
                "Results": {
                    "Posterior Sample Database": [
                        [1.0, 2.0, 0.1, 0.01],
                        [1.2, 2.3, 0.15, 0.02],
                    ],
                    "Posterior Sample LogLikelihood Database": [-10.0, -8.0],
                    "Posterior Sample LogPrior Database": [-1.0, -0.5],
                },
            }
        ),
        encoding="utf-8",
    )


def test_run_inference_stage_builds_phase2_command_with_profile_and_model_family(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_inference_stage.py", "run_inference_stage_test"
    )
    captured = {}

    def fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--structure",
            "emb",
            "--experiment",
            "compression",
            "--model-family",
            "full-model",
            "--profile",
            "validation",
            "--stage",
            "phase2",
            "--cpu-ranks",
            "4",
            "--output-dir",
            str(tmp_path / "run"),
            "--python-bin",
            "python",
        ]
    )

    assert rc == 0
    assert captured["cwd"] == str(repo_root)
    assert captured["command"][:6] == ["mpirun", "--bind-to", "none", "--oversubscribe", "-np", "4"]
    assert str(repo_root / "inference" / "scripts" / "run_phase_2.py") in captured["command"]
    assert "--phase2-backend" in captured["command"]
    assert captured["command"][-1] == "cpu-mpi"
    assert (
        str(
            repo_root
            / "inference"
            / "configs"
            / "validation"
            / "validation_config_compression.yaml"
        )
        in captured["command"]
    )


def test_run_inference_stage_phase2_production_defaults_to_native_cuda(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_inference_stage.py",
        "run_inference_stage_production_phase2_native_cuda_test",
    )
    captured = {}

    def fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--experiment",
            "compression",
            "--model-family",
            "full-model",
            "--profile",
            "production",
            "--stage",
            "phase2",
            "--output-dir",
            str(tmp_path / "run"),
            "--python-bin",
            "python",
        ]
    )

    assert rc == 0
    assert captured["cwd"] == str(repo_root)
    assert captured["command"][0] == "python"
    assert "mpirun" not in captured["command"]
    assert captured["command"][-2:] == ["--phase2-backend", "native-cuda"]


def test_run_propagation_uses_explicit_stage_wrapper(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_propagation.py", "run_propagation_test"
    )
    captured = {}

    def fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--experiment",
            "indentation",
            "--model-family",
            "reduced-model",
            "--profile",
            "production",
            "--stage",
            "phase3b",
            "--output-dir",
            str(tmp_path / "run"),
            "--python-bin",
            "python",
        ]
    )

    assert rc == 0
    assert captured["cwd"] == str(repo_root)
    assert captured["command"][1] == str(
        repo_root / "propagation" / "scripts" / "run_phase3b_propagation.py"
    )
    assert (
        str(repo_root / "reduced" / "configs" / "production" / "reduced_config_indentation.yaml")
        in captured["command"]
    )


def test_run_inference_stage_uses_reduced_phase1_wrapper(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_inference_stage.py",
        "run_inference_stage_reduced_test",
    )
    captured = {}

    def fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--experiment",
            "compression",
            "--model-family",
            "reduced-model",
            "--profile",
            "production",
            "--stage",
            "phase1",
            "--output-dir",
            str(tmp_path / "run"),
            "--python-bin",
            "python",
            "--restart",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert captured["cwd"] == str(repo_root)
    assert captured["command"][1] == str(repo_root / "reduced" / "scripts" / "run_phase_1.py")
    assert "--restart" in captured["command"]
    assert "--dry_run" in captured["command"]
    assert (
        str(repo_root / "reduced" / "configs" / "production" / "reduced_config_compression.yaml")
        in captured["command"]
    )


def test_reduced_phase2_wrapper_delegates_to_main_driver(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "reduced" / "scripts" / "run_phase_2.py", "reduced_phase2_wrapper_test"
    )
    captured = {}

    def fake_call(command, cwd=None):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)
    monkeypatch.setattr(module.sys, "argv", ["run_phase_2.py"])

    rc = module.main()

    assert rc == 0
    assert captured["cwd"] == str(repo_root)
    assert captured["command"][1] == str(repo_root / "inference" / "scripts" / "run_phase_2.py")
    assert (
        str(repo_root / "reduced" / "configs" / "production" / "reduced_config_compression.yaml")
        in captured["command"]
    )


def test_extract_map_writes_manifest_for_single_selected_dataset(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    fake_postprocess = types.ModuleType("meso_uq.postprocess")

    class _FakeRow:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return dict(self._payload)

    class _FakeFrame:
        def __init__(self, payload):
            self.iloc = [_FakeRow(payload)]

    def _fake_extract_map_from_directory(run_dir, output_csv):
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("Yt,kb\n1.0,2.0\n", encoding="utf-8")
        return _FakeFrame({"Yt": 1.0, "kb": 2.0})

    fake_postprocess.extract_map_from_directory = _fake_extract_map_from_directory
    monkeypatch.setitem(sys.modules, "meso_uq.postprocess", fake_postprocess)

    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "extract_map.py",
        "extract_map_test",
    )

    output_root = tmp_path / "workflow"
    _write_latest_state(output_root / "results_phase_3b" / "compression_2.1um" / "latest")

    rc = module.main(
        [
            "--experiment",
            "compression",
            "--model-family",
            "full-model",
            "--profile",
            "validation",
            "--stage",
            "phase3b",
            "--config",
            str(
                repo_root
                / "inference"
                / "configs"
                / "validation"
                / "validation_config_compression.yaml"
            ),
            "--output-dir",
            str(output_root),
            "--dataset",
            "compression_2.1um",
        ]
    )

    assert rc == 0
    manifest_path = output_root / "map_phase3b" / "phase3b_map_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["structure"] == "emb"
    assert "compression_2.1um" in manifest["datasets"]
    assert Path(manifest["datasets"]["compression_2.1um"]["output_csv"]).exists()


def test_run_inference_stage_rejects_gv_runtime_before_dispatch(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_inference_stage.py",
        "run_inference_stage_gv_runtime_rejection_test",
    )

    def fake_run(command, cwd=None, check=False):
        raise AssertionError("subprocess.run should not be reached for unsupported GV runtime")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="GV workflow runtime/config resolution is not implemented yet"):
        module.main(
            [
                "--structure",
                "gv",
                "--experiment",
                "stretching",
                "--model-family",
                "full-model",
                "--profile",
                "validation",
                "--stage",
                "phase1",
                "--output-dir",
                str(tmp_path / "run"),
                "--python-bin",
                "python",
            ]
        )


def test_run_propagation_rejects_gv_runtime_before_dispatch_with_override(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "hpc" / "run_propagation.py",
        "run_propagation_gv_runtime_rejection_test",
    )
    config_path = tmp_path / "gv_config.yaml"
    config_path.write_text("structure: gv\nexperiment: stretching\n", encoding="utf-8")

    def fake_run(command, cwd=None, check=False):
        raise AssertionError("subprocess.run should not be reached for unsupported GV propagation")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="GV workflow propagation is not implemented yet"):
        module.main(
            [
                "--structure",
                "gv",
                "--experiment",
                "stretching",
                "--model-family",
                "full-model",
                "--profile",
                "validation",
                "--stage",
                "phase1",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "run"),
                "--python-bin",
                "python",
            ]
        )


def test_vega_sbatch_templates_expose_model_family_and_profile_axes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template_dir = repo_root / "scripts" / "platforms" / "vega" / "sbatch"
    templates = sorted(template_dir.glob("*.sbatch"))
    fixed_scope_templates = {
        "dpd_production_preflight_canary.sbatch",
        "gv_paper_figure_replay.sbatch",
        "train_dnn_arch_array.sbatch",
        "train_dnn_surrogates.sbatch",
    }

    assert templates
    for template in templates:
        if template.name in fixed_scope_templates:
            continue
        text = template.read_text(encoding="utf-8")
        assert (
            "MODEL_FAMILY" in text
            or "MODEL_FAMILIES" in text
            or "SELECTION" in text
            or "SELECTIONS" in text
        )
        assert (
            "PROFILE" in text or "PROFILES" in text or "SELECTION" in text or "SELECTIONS" in text
        )
        assert 'source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"' in text
        assert "mesouq_activate_site_env vega" in text
        assert ("_vega" + "/") not in text


def test_gv_paper_figure_replay_template_uses_public_command_and_gv_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "scripts" / "platforms" / "vega" / "sbatch" / "gv_paper_figure_replay.sbatch"

    text = template.read_text(encoding="utf-8")

    assert 'REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"' in text
    assert 'CAMPAIGN_ID="${CAMPAIGN_ID:-}"' in text
    assert 'LANES="${LANES:-}"' in text
    assert 'PAPER_EXACT="${PAPER_EXACT:-0}"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/gv/figure_replay/${CAMPAIGN_ID}}"' in text
    assert 'STRETCHING_POINT_START="${STRETCHING_POINT_START:-}"' in text
    assert 'STRETCHING_POINT_STOP="${STRETCHING_POINT_STOP:-}"' in text
    assert 'BUCKLING_TIMEOUT_SECONDS="${BUCKLING_TIMEOUT_SECONDS:-}"' in text
    assert "run_paper_figure_replay.py" in text
    assert 'command+=(--lane "${lane}")' in text
    assert "command+=(--paper-exact)" in text
    assert 'command+=(--stretching-point-start "${STRETCHING_POINT_START}")' in text
    assert 'command+=(--stretching-point-stop "${STRETCHING_POINT_STOP}")' in text
    assert 'command+=(--buckling-timeout-seconds "${BUCKLING_TIMEOUT_SECONDS}")' in text
    assert 'source "${REPO_ROOT}/scripts/platforms/hpc/site_env.sh"' in text
    assert "mesouq_activate_site_env vega" in text
    assert ("_vega" + "/") not in text
    assert "OpenMPI/4.1.4-GCC-12.2.0" in text
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --ntasks=2" in text
    assert 'MESOUQ_GV_MPI_RANKS="${MESOUQ_GV_MPI_RANKS:-2}"' in text
    assert 'MESOUQ_GV_EIGENMODES_MPI_RANKS="${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}"' in text
    assert 'MESOUQ_GV_EIGENMODES_DOMAIN_RANKS="${MESOUQ_GV_EIGENMODES_DOMAIN_RANKS:-1,1,1}"' in text
    assert "#SBATCH --gres=gpu:1" in text


def test_gv_paper_figure_replay_submitter_sets_lane_aware_walltime() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    submitter = repo_root / "scripts" / "platforms" / "vega" / "submit_gv_paper_figure_replay.sh"

    text = submitter.read_text(encoding="utf-8")

    assert "GV_PAPER_REPLAY_TIME_LIMIT" in text
    assert 'exec sbatch --time="${TIME_LIMIT}"' in text
    assert "STRETCHING_POINT_START" in text
    assert "STRETCHING_POINT_STOP" in text
    assert "BUCKLING_TIMEOUT_SECONDS" in text
    assert "torsion)" in text and 'echo "01:00:00"' in text
    assert "buckling)" in text and 'echo "04:00:00"' in text
    assert "eigenmodes)" in text and 'echo "08:00:00"' in text
    assert "count <= 30" in text and 'echo "03:00:00"' in text


def test_production_sanity_template_uses_public_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "scripts" / "platforms" / "vega" / "sbatch" / "production_sanity.sbatch"

    text = template.read_text(encoding="utf-8")

    assert "run_production_sanity.py" in text
    assert "SELECTIONS" in text
    assert "ALL_LANES" in text
    assert 'PHASE2_CPU_RANKS="1"' in text
    assert "--phase2-backend" in text


def test_validation_matrix_template_uses_public_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "scripts" / "platforms" / "vega" / "sbatch" / "validation_matrix.sbatch"

    text = template.read_text(encoding="utf-8")

    assert "run_validation_matrix.py" in text
    assert "MODEL_FAMILIES" in text
    assert "EXPERIMENTS" in text
    assert "#SBATCH --mem=64000" in text
    assert "mesouq_activate_site_env vega" in text
    assert ("_vega" + "/") not in text


def test_acceptance_template_uses_public_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "scripts" / "platforms" / "vega" / "sbatch" / "acceptance.sbatch"

    text = template.read_text(encoding="utf-8")

    assert "run_vega_acceptance.py" in text
    assert "SELECTIONS" in text
    assert "compression:reduced-model:validation" in text
    assert "mesouq_activate_site_env vega" in text
    assert ("_vega" + "/") not in text



def test_bootstrap_korali_preserves_venv_python_path_for_dependency_installs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "platforms" / "hpc" / "bootstrap_korali.sh"

    text = script.read_text(encoding="utf-8")
    path_block = text[text.index('if [[ "$python_bin" == */* ]]; then') : text.index("export PATH=")]

    assert 'python_bin_dir="$(cd "$(dirname "$python_bin")" && pwd -P)"' in path_block
    assert 'python_bin="${python_bin_dir}/$(basename "$python_bin")"' in path_block
    assert 'python_bin="$(readlink -f "$python_bin")"' not in path_block
    assert 'export PATH="$python_bin_dir${PATH:+:$PATH}"' in text

def test_vega_bootstrap_scripts_resolve_repo_root_after_platforms_move() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    korali = repo_root / "scripts" / "platforms" / "hpc" / "bootstrap_korali.sh"
    mirheo = repo_root / "scripts" / "platforms" / "hpc" / "bootstrap_mirheo.sh"
    legacy_korali = repo_root / "scripts" / "vega" / "bootstrap_korali.sh"
    legacy_mirheo = repo_root / "scripts" / "vega" / "bootstrap_mirheo.sh"

    korali_text = korali.read_text(encoding="utf-8")
    mirheo_text = mirheo.read_text(encoding="utf-8")
    legacy_korali_text = legacy_korali.read_text(encoding="utf-8")
    legacy_mirheo_text = legacy_mirheo.read_text(encoding="utf-8")

    for text in (korali_text, mirheo_text):
        assert 'script_path="$(readlink -f "${BASH_SOURCE[0]}")"' in text
        assert 'script_dir="$(cd "$(dirname "$script_path")" && pwd)"' in text
        assert 'repo_root="$(cd "${script_dir}/../../.." && pwd)"' in text
    for text in (legacy_korali_text, legacy_mirheo_text):
        assert 'exec bash "${script_dir}/../hpc/bootstrap_' in text
        assert '--site vega "$@"' in text
