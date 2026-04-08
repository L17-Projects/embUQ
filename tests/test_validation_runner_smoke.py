import importlib.util
import json
from pathlib import Path

import yaml


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_phase3b_latest(latest_path: Path):
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


def _write_propagation_summary(summary_csv: Path):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.write_text("x,mean\n0.0,0.0\n1.0,1.0\n2.0,2.0\n", encoding="utf-8")


class _FakeExperiment:
    name = "compression"
    diameters = [2.1, 2.9, 3.0]

    def dataset_name(self, diameter_um):
        return f"compression_{diameter_um}um"

    def get_reference_points(self, diameter_um):
        return [0.0, 1.0, 2.0]

    def get_reference_data(self, diameter_um):
        return [0.0, 1.0, 2.0]


def test_validation_runner_smoke_creates_summary_and_artifacts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "vega" / "run_validation_suite.py", "run_gpu_validation_suite_test")
    workflow_name = "compression:reduced-model:validation"

    class Result:
        def __init__(self, returncode=0):
            self.returncode = returncode
            self.stdout = "ok"
            self.stderr = ""

    def fake_run(command, cwd=None, env=None, check=False):
        command_str = " ".join(str(part) for part in command)
        if "run_phase_3b.py" in command_str:
            out_idx = command.index("--output-dir") + 1
            results_dir = Path(command[out_idx])
            for d in [2.1, 2.9, 3.0]:
                _write_phase3b_latest(results_dir / "results_phase_3b" / f"compression_{d}um" / "latest")
        if "run_phase3b_propagation.py" in command_str:
            out_idx = command.index("--output-dir") + 1
            results_dir = Path(command[out_idx])
            for d in [2.1, 2.9, 3.0]:
                _write_propagation_summary(results_dir / "propagation_phase3b" / f"compression_{d}um" / "summary.csv")
        return Result(0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_load_experiment_spec", lambda config_path, experiment_name: _FakeExperiment())

    output_root = tmp_path / "runner"
    summary = module.run_workflow(
        workflow_name=workflow_name,
        workflow_spec=dict(module.WORKFLOW_CONFIGS[workflow_name]),
        output_root=output_root,
        python_bin="python",
        korali_pythonpath=None,
        cpu_ranks=1,
        population_size=8,
    )

    workflow_dir = output_root / "compression__reduced-model__validation_8"
    assert summary["workflow"] == "compression__reduced-model__validation_8"
    assert summary["workflow_base_name"] == workflow_name
    assert summary["model_family"] == "reduced-model"
    assert summary["profile"] == "validation"
    assert summary["selection"] == "compression:reduced-model:validation"
    assert (workflow_dir / "summary.json").exists()
    assert (workflow_dir / "map_phase3b" / "all_diameters_map.json").exists()
    assert (workflow_dir / "overlay_uq_ref" / "uq_overlay_2.1um.png").exists()
    assert (workflow_dir / "posteriors_phase3b" / "posterior_marginals_2.1um.png").exists()


def test_validation_runner_defaults_to_validation_configs_and_preserves_population(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "vega" / "run_validation_suite.py", "run_gpu_validation_suite_defaults_test")

    compression_reduced = module.WORKFLOW_CONFIGS["compression:reduced-model:validation"]["config"]
    indentation_reduced = module.WORKFLOW_CONFIGS["indentation:reduced-model:validation"]["config"]
    assert "validation" in str(compression_reduced)
    assert "validation" in str(indentation_reduced)

    derived_path = tmp_path / "config.yaml"
    derived = module._write_derived_config(compression_reduced, derived_path, population_size=None)
    with open(compression_reduced, "rb") as handle:
        base = yaml.load(handle, Loader=yaml.CLoader)

    for key in ("pop_size", "hbi_pop_size", "phase3a_pop_size", "phase3b_pop_size"):
        assert derived[key] == base[key]
    assert derived_path.name == "config.yaml"


def test_validation_runner_accepts_legacy_aliases_for_compatibility():
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "vega" / "run_validation_suite.py", "run_gpu_validation_suite_alias_test")

    assert module._resolve_validation_selection("compression_reduced") == "compression:reduced-model:validation"
    assert module._resolve_validation_selection("indentation_full") == "indentation:full-model:validation"
