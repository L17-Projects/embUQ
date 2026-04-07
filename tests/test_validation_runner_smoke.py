import importlib.util
import json
from pathlib import Path


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


def test_validation_runner_smoke_creates_summary_and_artifacts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "inference" / "scripts" / "run_gpu_validation_suite.py", "run_gpu_validation_suite_test")

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
            with open(command[command.index("--config") + 1], "rb") as handle:
                import yaml
                config = yaml.load(handle, Loader=yaml.CLoader)
            from meso_uq.experiments import load_experiments
            experiments = [exp for exp in load_experiments(config, repo_root) if exp.enabled]
            for exp in experiments:
                for d in exp.diameters:
                    _write_phase3b_latest(results_dir / "results_phase_3b" / exp.dataset_name(d) / "latest")
        if "run_phase3b_propagation.py" in command_str:
            out_idx = command.index("--output-dir") + 1
            results_dir = Path(command[out_idx])
            with open(command[command.index("--config") + 1], "rb") as handle:
                import yaml
                config = yaml.load(handle, Loader=yaml.CLoader)
            from meso_uq.experiments import load_experiments
            experiments = [exp for exp in load_experiments(config, repo_root) if exp.enabled]
            for exp in experiments:
                for d in exp.diameters:
                    _write_propagation_summary(results_dir / "propagation_phase3b" / exp.dataset_name(d) / "summary.csv")
        return Result(0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    output_root = tmp_path / "runner"
    summary = module.run_workflow(
        workflow_name="compression_reduced",
        workflow_spec=dict(module.WORKFLOW_CONFIGS["compression_reduced"]),
        output_root=output_root,
        python_bin="python",
        korali_pythonpath=None,
        cpu_ranks=1,
        population_size=8,
    )

    workflow_dir = output_root / "compression_reduced_8"
    assert summary["workflow"] == "compression_reduced_8"
    assert (workflow_dir / "summary.json").exists()
    assert (workflow_dir / "map_phase3b" / "all_diameters_map.json").exists()
    assert (workflow_dir / "overlay_uq_ref" / "uq_overlay_2.1um.png").exists()
    assert (workflow_dir / "posteriors_phase3b" / "posterior_marginals_2.1um.png").exists()
