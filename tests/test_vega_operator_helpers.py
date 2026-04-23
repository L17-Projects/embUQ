import importlib.util
import json
from pathlib import Path


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
        repo_root / "scripts" / "vega" / "run_inference_stage.py", "run_inference_stage_test"
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


def test_run_propagation_uses_explicit_stage_wrapper(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_propagation.py", "run_propagation_test"
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
        repo_root / "scripts" / "vega" / "run_inference_stage.py",
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


def test_extract_map_writes_manifest_for_single_selected_dataset(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "vega" / "extract_map.py", "extract_map_test")

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
    assert "compression_2.1um" in manifest["datasets"]
    assert Path(manifest["datasets"]["compression_2.1um"]["output_csv"]).exists()


def test_vega_sbatch_templates_expose_model_family_and_profile_axes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template_dir = repo_root / "scripts" / "vega" / "sbatch"
    templates = sorted(template_dir.glob("*.sbatch"))

    assert templates
    for template in templates:
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
        assert "_vega/korali/env.sh" in text


def test_production_sanity_template_uses_public_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "scripts" / "vega" / "sbatch" / "production_sanity.sbatch"

    text = template.read_text(encoding="utf-8")

    assert "run_production_sanity.py" in text
    assert "SELECTIONS" in text
    assert "ALL_LANES" in text


def test_validation_matrix_template_uses_public_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "scripts" / "vega" / "sbatch" / "validation_matrix.sbatch"

    text = template.read_text(encoding="utf-8")

    assert "run_validation_matrix.py" in text
    assert "MODEL_FAMILIES" in text
    assert "EXPERIMENTS" in text
    assert "_vega/korali/env.sh" in text


def test_acceptance_template_uses_public_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "scripts" / "vega" / "sbatch" / "acceptance.sbatch"

    text = template.read_text(encoding="utf-8")

    assert "run_vega_acceptance.py" in text
    assert "SELECTIONS" in text
    assert "compression:reduced-model:validation" in text
    assert "_vega/korali/env.sh" in text
