from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from meso_uq.agents.emb import (
    emb_config_candidate_paths,
    get_emb_generation_workflow,
    list_emb_generation_workflows,
    resolve_emb_config_file,
)
from meso_uq.core import AgentFamily, Modality
from meso_uq.simulation import (
    build_emb_hpc_sbatch_text,
    generate_emb_simulation,
    generate_parameter_payloads,
    parse_parameter_sweeps,
)


def test_emb_generation_workflow_contracts_cover_compression_and_indentation() -> None:
    workflows = {workflow.modality: workflow for workflow in list_emb_generation_workflows()}

    assert set(workflows) == {Modality.COMPRESSION, Modality.INDENTATION}
    assert workflows[Modality.COMPRESSION].family is AgentFamily.EMB
    assert workflows[Modality.COMPRESSION].legacy_root == "emb/compression"
    assert workflows[Modality.COMPRESSION].generation_script == "emb/compression/src/generate.py"
    assert workflows[Modality.COMPRESSION].parameters_script == "emb/compression/src/parameters.py"
    assert workflows[Modality.COMPRESSION].indentation_mass_multiplier == pytest.approx(1.0)
    assert workflows[Modality.INDENTATION].legacy_root == "emb/indentation"
    assert workflows[Modality.INDENTATION].generation_script == "emb/indentation/src/generate.py"
    assert workflows[Modality.INDENTATION].parameters_script == "emb/indentation/src/parameters.py"
    assert workflows[Modality.INDENTATION].indentation_mass_multiplier == pytest.approx(5.0)

    parameter_files = workflows[Modality.COMPRESSION].parameter_files
    assert parameter_files.default_template == "parameters-default.emb.yaml"
    assert parameter_files.generated_default_file("00001") == "parameters-default00001.yaml"
    assert parameter_files.runtime_prms_file("00001") == "parameters.prms00001.yaml"


def test_emb_config_resolution_preserves_cwd_and_legacy_script_fallbacks(tmp_path: Path) -> None:
    cwd_root = tmp_path / "cwd"
    cwd_config = cwd_root / "inference" / "configs" / "production" / "inference_config_compression.yaml"
    cwd_config.parent.mkdir(parents=True)
    cwd_config.write_text("debug: 0\n", encoding="utf-8")

    assert (
        resolve_emb_config_file(Modality.COMPRESSION, purpose="generation", cwd=cwd_root)
        == str(cwd_config)
    )

    repo_root = tmp_path / "repo"
    anchor = repo_root / "emb" / "compression" / "src" / "generate.py"
    anchor.parent.mkdir(parents=True)
    file_config = repo_root / "inference" / "configs" / "production" / "inference_config_compression.yaml"
    file_config.parent.mkdir(parents=True)
    file_config.write_text("debug: 0\n", encoding="utf-8")

    assert (
        resolve_emb_config_file(
            Modality.COMPRESSION,
            purpose="generation",
            anchor_file=anchor,
            cwd=tmp_path / "elsewhere",
        )
        == str(file_config)
    )

    candidate_paths = emb_config_candidate_paths(
        Modality.INDENTATION,
        purpose="generation",
        anchor_file=repo_root / "emb" / "indentation" / "src" / "generate.py",
        cwd=cwd_root,
    )
    assert (
        cwd_root / "../../../inference/configs/production/inference_config_indentation.yaml"
    ) in candidate_paths
    assert cwd_root / "inference/configs/production/inference_config_indentation.yaml" in candidate_paths
    assert repo_root / "inference/configs/production/inference_config_indentation.yaml" in candidate_paths


def test_parameter_sweep_payload_order_matches_legacy_generation_loop() -> None:
    sweeps = parse_parameter_sweeps([["buck", "1.0", "2.0", "2"], ["steps", "3", "5", "2"]])
    payloads = list(generate_parameter_payloads({"buck": 10.0, "steps": 10}, sweeps))

    assert payloads == [
        {"buck": 1.0, "steps": 3},
        {"buck": 2.0, "steps": 3},
        {"buck": 1.0, "steps": 5},
        {"buck": 2.0, "steps": 5},
    ]
    assert all(isinstance(payload["steps"], int) for payload in payloads)


def test_shared_emb_generation_writes_legacy_artifact_shape(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"debug": 0}), encoding="utf-8")
    source = tmp_path / "source"
    simu = tmp_path / "simu"
    source.mkdir()
    simu.mkdir()
    (source / "parameters-default.emb.yaml").write_text(
        yaml.safe_dump({"buck": 10.0, "steps": 10}),
        encoding="utf-8",
    )

    result = generate_emb_simulation(
        modality=Modality.COMPRESSION,
        source_path=f"{source}/",
        simu_path=f"{simu}/",
        par=[["buck", "1.0", "2.0", "2"]],
        obj="emb",
        forward=None,
        hysteresis=None,
        parallel=True,
        g=2,
        N=1,
        first=None,
        numJobs=1,
        config_file=config,
    )

    assert result.config_file == str(config)
    assert result.parameter_count == 2
    assert result.simulation_count == 2
    assert (simu / "parameter" / "parameters-default00001.yaml").exists()
    assert (simu / "parameter" / "parameters-default00002.yaml").exists()
    assert "bash run.sh --equil 00001eq 4" in (simu / "commands.txt").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2" in (simu / "run_HPC.sbatch").read_text(encoding="utf-8")


def test_shared_generation_rejects_invalid_sweep_width() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        parse_parameter_sweeps([["buck", "1.0", "2.0", "0"]])


def test_sbatch_text_keeps_legacy_hpc_shape() -> None:
    sbatch = build_emb_hpc_sbatch_text(num_gpus=1, num_nodes=2, ntasks_per_node=2, total_mem=20)

    assert '#SBATCH --job-name="KeyserSoze"' in sbatch
    assert "#SBATCH --nodes=2" in sbatch
    assert sbatch.endswith("bash commands.txt\n")


def test_unknown_emb_generation_workflow_reports_supported_modalities() -> None:
    with pytest.raises(ValueError, match="compression, indentation"):
        get_emb_generation_workflow("buckling")
