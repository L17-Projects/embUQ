from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from meso_uq.experiments import ExperimentSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "emb" / "uq_emb" / "run_forward_canary.py"
REPLAY_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "emb" / "uq_emb" / "run_hbi_replay.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("uq_emb_forward_canary", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_replay_module():
    spec = importlib.util.spec_from_file_location("uq_emb_hbi_replay", REPLAY_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment(tmp_path: Path) -> ExperimentSpec:
    return ExperimentSpec(
        structure="emb",
        name="compression",
        geometries=["diameter_2.1um"],
        data_dir=tmp_path / "data",
        data_prefix="compression_data_",
        surrogate_dir=tmp_path / "surrogates",
        enabled=True,
        prior_d0=[0.0, 0.5],
        prior_sigma=[0.02, 0.1],
        controls=["default"],
        prior_ka=[3500.0, 34000.0],
        prior_kb=[101.0, 1100.0],
        surrogate_parameterization="direct_ka_kb",
    )


def test_parameter_batch_uses_experiment_priors(tmp_path: Path) -> None:
    module = _load_module()

    batch = module._parameter_batch(
        {
            "prior_ka": [1.0, 2.0],
            "prior_kb": [3.0, 4.0],
            "prior_d0": [0.0, 1.0],
            "prior_sigma": [0.0, 1.0],
        },
        _experiment(tmp_path),
        2.1,
    )

    assert batch.shape == (3, 4)
    np.testing.assert_allclose(batch[:, 0], [15700.0, 18750.0, 21800.0])
    np.testing.assert_allclose(batch[:, 1], [500.6, 600.5, 700.4])
    np.testing.assert_allclose(batch[:, 2], [0.2, 0.25, 0.3])
    np.testing.assert_allclose(batch[:, 3], [0.052, 0.06, 0.068])


def test_summarize_batch_rejects_nonfinite_and_nonpositive_uncertainty() -> None:
    module = _load_module()
    valid = {
        "Batch Reference Evaluations": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        "Batch Standard Deviation": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
    }

    result = module._summarize_batch(valid, expected_rows=3, expected_columns=2)

    assert result["shape"] == [3, 2]
    assert result["predictions"] == valid["Batch Reference Evaluations"]
    assert result["standard_deviations"] == valid["Batch Standard Deviation"]
    assert result["prediction_min"] == 1.0
    assert result["standard_deviation_max"] == pytest.approx(0.4)

    invalid = dict(valid)
    invalid["Batch Standard Deviation"] = [[0.1, 0.2], [0.2, 0.0], [0.3, 0.4]]
    with pytest.raises(ValueError, match="strictly positive"):
        module._summarize_batch(invalid, expected_rows=3, expected_columns=2)


def test_mechanical_artifact_receipt_hashes_the_selected_dnn(tmp_path: Path) -> None:
    module = _load_module()
    experiment = _experiment(tmp_path)
    trained = experiment.surrogate_dir / "2.1um" / "trained"
    trained.mkdir(parents=True)
    model = trained / "microbubble_force_BEST.pkl"
    model.write_bytes(b"frozen-model")

    artifacts = module._mechanical_artifacts(experiment, 2.1)

    assert artifacts == [
        {
            "path": str(model),
            "sha256": "079797e75d879c9c7945ebc83cb6e8a3490fdebdd4b8a25c6d6ff45914ff3592",
            "size_bytes": len(b"frozen-model"),
        }
    ]


@pytest.mark.parametrize(
    ("agent", "experiment"),
    (("sonovue", "indentation"), ("definity", "compression")),
)
def test_hbi_replay_builds_accepted_three_stage_sequence(
    monkeypatch, tmp_path: Path, agent: str, experiment: str
) -> None:
    module = _load_replay_module()
    config_path = tmp_path / f"{agent}.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_load_and_validate_config",
        lambda _path: ({}, agent, 10000),
    )

    resolved_agent, population, commands = module.build_replay_commands(
        config_path=config_path,
        output_root=tmp_path / "run",
        python_bin="/verified/python",
        stages=["phase1", "phase2", "phase3b"],
    )

    assert resolved_agent == agent
    assert population == 10000
    assert len(commands) == 3
    assert all("/verified/python" in command for command in commands)
    assert all(str(config_path.resolve()) in command for command in commands)
    assert all(str((tmp_path / "run").resolve()) in command for command in commands)
    assert all(experiment not in command for command in commands)
    assert "--phase2-backend" in commands[1]
    assert "native-cuda" in commands[1]
    assert commands[0][-2:] == ["--device", "gpu"]
    assert commands[2][-2:] == ["--device", "gpu"]
