from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import (
    CandidateGenerationConfig,
    CandidateParameterDimension,
    generate_candidate_batch,
    validate_active_learning_candidates,
    write_candidate_generation_artifacts,
)
from meso_uq.dpd_sampling.boundary import build_dpd_sampling_batch_request


_VALID_GV_MATERIAL_PARAMETERS = {
    "ka": 1.1,
    "kb": 1.2,
    "mu": 0.9,
    "b1": 0.1,
    "b2": 0.2,
    "a3": 0.3,
    "a4": 0.4,
    "mu_l": 0.5,
    "c": 0.6,
}


def _gv_candidate_generation_config(
    *,
    strategy: str = "prior_exploration",
    count: int = 4,
) -> CandidateGenerationConfig:
    return CandidateGenerationConfig(
        family="gv",
        experiment="stretching",
        dimensions=(
            CandidateParameterDimension(
                name="ka",
                path=("material_parameters", "ka"),
                lower=0.8,
                upper=1.4,
                posterior_mean=1.1,
                posterior_std=0.08,
                units="DPD energy / area",
            ),
            CandidateParameterDimension(
                name="radGV",
                path=("geometry", "radGV"),
                lower=1.75,
                upper=2.25,
                posterior_mean=2.0,
                posterior_std=0.04,
                units="DPD length",
            ),
        ),
        count=count,
        strategy=strategy,
        seed=17,
        candidate_prefix="gv-al",
        payload_template={
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _VALID_GV_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0, "tot_force": [500.0, 750.0]},
        },
        metadata={"source_issue": "MES-7"},
    )


def _png_has_signature(path: Path) -> bool:
    return path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_candidate_generation_config_roundtrip_and_validation() -> None:
    config = _gv_candidate_generation_config()
    loaded = CandidateGenerationConfig.from_dict(config.as_dict())

    assert loaded == config
    assert loaded.payload_key == "gv_launch"

    with pytest.raises(ValueError, match="family"):
        CandidateGenerationConfig(
            family="unsupported",
            experiment="stretching",
            dimensions=config.dimensions,
            count=1,
        )

    with pytest.raises(ValueError, match="strategy"):
        CandidateGenerationConfig(
            family="gv",
            experiment="stretching",
            dimensions=config.dimensions,
            count=1,
            strategy="unknown",
        )

    with pytest.raises(ValueError, match="lower must be < upper"):
        CandidateParameterDimension(
            name="bad",
            path=("material_parameters", "ka"),
            lower=1.0,
            upper=1.0,
        )


def test_candidate_generation_config_preserves_python_and_json_integer_counts() -> None:
    config = _gv_candidate_generation_config(count=2)
    json_payload = json.loads(json.dumps(config.as_dict()))

    loaded = CandidateGenerationConfig.from_dict(json_payload)

    assert config.count == 2
    assert loaded.count == 2
    assert len(generate_candidate_batch(json_payload).candidates) == 2


@pytest.mark.parametrize("raw_count", [1.9, 1.0, "1", True, False])
def test_candidate_generation_config_rejects_non_integer_raw_count(raw_count: object) -> None:
    payload = _gv_candidate_generation_config().as_dict()
    payload["count"] = raw_count

    with pytest.raises(ValueError, match="count must be a positive integer"):
        CandidateGenerationConfig.from_dict(payload)
    with pytest.raises(ValueError, match="count must be a positive integer"):
        generate_candidate_batch(payload)


def test_generate_candidate_batch_emits_deterministic_family_scoped_gv_payloads(tmp_path: Path) -> None:
    config = _gv_candidate_generation_config()
    result = generate_candidate_batch(config)
    repeat = generate_candidate_batch(config)

    assert [candidate.as_dict() for candidate in result.candidates] == [
        candidate.as_dict() for candidate in repeat.candidates
    ]
    assert result.config.family == "gv"
    assert result.config.experiment == "stretching"
    assert result.config.strategy == "prior_exploration"
    assert len(result.candidates) == 4
    assert result.candidates[0].candidate_id == "gv-al-gv-stretching-prior_exploration-seed-17-0000"
    assert result.candidates[-1].candidate_id == "gv-al-gv-stretching-prior_exploration-seed-17-0003"

    report = validate_active_learning_candidates(result.candidates)
    assert [candidate.candidate_id for candidate in report.valid_candidates] == [
        candidate.candidate_id for candidate in result.candidates
    ]
    assert report.rejected_candidate_ids == ()

    first = result.candidates[0]
    assert first.parameters["family"] == "gv"
    assert set(first.parameters) == {"family", "gv_launch"}
    launch_payload = first.parameters["gv_launch"]
    assert launch_payload["experiment"] == "stretching"
    assert launch_payload["controls"]["tot_force"] == [500.0, 750.0]
    assert launch_payload["material_parameters"]["kb"] == _VALID_GV_MATERIAL_PARAMETERS["kb"]
    assert 0.8 <= launch_payload["material_parameters"]["ka"] <= 1.4
    assert 1.75 <= launch_payload["geometry"]["radGV"] <= 2.25
    assert first.metadata["source"] == "active_learning_candidate_generation"
    assert first.metadata["source_issue"] == "MES-7"
    assert first.metadata["parameter_paths"]["ka"] == ["material_parameters", "ka"]

    request = build_dpd_sampling_batch_request(
        result.candidates,
        run_id="candidate-generation-gv",
        iteration=0,
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        campaign_root=tmp_path / "_runs" / "active_learning" / "candidate-generation-gv" / "iterations" / "iter_0000",
        provenance_tags={"source_issue": "MES-7"},
    )

    assert request.family == "gv"
    assert len(request.candidate_manifests) == len(result.candidates)
    assert all(manifest.expected_hdf5_datasets for manifest in request.candidate_manifests)


@pytest.mark.parametrize(
    "strategy",
    ["posterior_region", "posterior_boundary", "prior_exploration", "validity_boundary"],
)
def test_candidate_generation_strategies_emit_in_bounds_values(strategy: str) -> None:
    result = generate_candidate_batch(_gv_candidate_generation_config(strategy=strategy, count=6))

    for generated in result.generated_values:
        assert 0.8 <= generated["ka"] <= 1.4
        assert 1.75 <= generated["radGV"] <= 2.25


def test_generate_candidate_batch_supports_emb_family_scoped_payloads() -> None:
    result = generate_candidate_batch(
        CandidateGenerationConfig(
            family="emb",
            experiment="indentation",
            dimensions=(
                CandidateParameterDimension(
                    name="shell_modulus",
                    path=("physical_parameters", "shell_modulus"),
                    lower=0.2,
                    upper=0.9,
                    posterior_mean=0.5,
                    posterior_std=0.1,
                ),
            ),
            count=2,
            strategy="posterior_region",
            seed=19,
            candidate_prefix="emb-al",
            payload_template={
                "physical_parameters": {"viscosity": 1.2},
                "controls": {"indentation_depth": 0.35},
            },
        )
    )

    assert [candidate.parameters["family"] for candidate in result.candidates] == ["emb", "emb"]
    for candidate in result.candidates:
        assert set(candidate.parameters) == {"family", "emb_launch"}
        launch_payload = candidate.parameters["emb_launch"]
        assert launch_payload["experiment"] == "indentation"
        assert "physical_parameters" in launch_payload
        assert "parameter_vector" not in launch_payload
        assert "reduced_parameter_vector" not in launch_payload

    report = validate_active_learning_candidates(
        result.candidates,
        allowed_families=("emb",),
        required_parameter_paths={"emb": (("experiment",), ("physical_parameters",))},
    )
    assert len(report.valid_candidates) == 2
    assert report.rejected_candidate_ids == ()


def test_generate_candidate_batch_uses_config_experiment_over_stale_template() -> None:
    config = _gv_candidate_generation_config()
    config_with_stale_template = CandidateGenerationConfig(
        **{
            **config.as_dict(),
            "payload_template": {
                **dict(config.payload_template),
                "experiment": "stale-template-experiment",
            },
        }
    )

    result = generate_candidate_batch(config_with_stale_template)
    assert all(
        candidate.parameters["gv_launch"]["experiment"] == config_with_stale_template.experiment
        for candidate in result.candidates
    )


def test_generate_candidate_batch_scopes_candidate_ids_by_family_experiment_and_seed() -> None:
    gv_result = generate_candidate_batch(_gv_candidate_generation_config(count=1))
    emb_result = generate_candidate_batch(
        CandidateGenerationConfig(
            family="emb",
            experiment="stretching",
            dimensions=(
                CandidateParameterDimension(
                    name="shell_modulus",
                    path=("physical_parameters", "shell_modulus"),
                    lower=0.2,
                    upper=0.9,
                ),
            ),
            count=1,
            strategy="prior_exploration",
            seed=17,
            candidate_prefix="gv-al",
            payload_template={"physical_parameters": {"viscosity": 1.2}},
        )
    )
    gv_other_experiment_result = generate_candidate_batch(
        CandidateGenerationConfig(
            **{
                **_gv_candidate_generation_config(count=1).as_dict(),
                "experiment": "indentation",
            }
        )
    )
    gv_other_seed_result = generate_candidate_batch(
        CandidateGenerationConfig(
            **{
                **_gv_candidate_generation_config(count=1).as_dict(),
                "seed": 23,
            }
        )
    )

    candidate_ids = {
        gv_result.candidates[0].candidate_id,
        emb_result.candidates[0].candidate_id,
        gv_other_experiment_result.candidates[0].candidate_id,
        gv_other_seed_result.candidates[0].candidate_id,
    }
    assert len(candidate_ids) == 4


def test_write_candidate_generation_artifacts_generates_manifest_report_plot_and_sidecar(tmp_path: Path) -> None:
    result = generate_candidate_batch(_gv_candidate_generation_config(count=3))
    artifacts = write_candidate_generation_artifacts(
        output_root=tmp_path / "_runs" / "active_learning",
        run_id="candidate-generation-artifacts",
        iteration=2,
        result=result,
        include_plot=True,
    )

    assert artifacts.artifact_dir == (
        tmp_path
        / "_runs"
        / "active_learning"
        / "candidate-generation-artifacts"
        / "iterations"
        / "iter_0002"
    )
    assert artifacts.manifest_path.is_file()
    assert artifacts.report_path.is_file()
    assert artifacts.plot_path.is_file()
    assert artifacts.plot_path.stat().st_size > 0
    assert _png_has_signature(artifacts.plot_path)
    assert artifacts.plot_sidecar_path.is_file()

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    sidecar = json.loads(artifacts.plot_sidecar_path.read_text(encoding="utf-8"))
    assert manifest["candidate_count"] == 3
    assert report["family_distribution"] == {"gv": 3}
    assert report["strategy"] == "prior_exploration"
    assert set(report["candidate_hashes"]) == set(manifest["candidate_ids"])
    assert sidecar == report


def test_write_candidate_generation_artifacts_emits_fallback_png_when_plot_disabled(tmp_path: Path) -> None:
    result = generate_candidate_batch(_gv_candidate_generation_config(count=1))
    artifacts = write_candidate_generation_artifacts(
        output_root=tmp_path / "_runs" / "active_learning",
        run_id="candidate-generation-fallback",
        iteration=0,
        result=result,
        include_plot=False,
    )

    assert artifacts.plot_path.is_file()
    assert _png_has_signature(artifacts.plot_path)


def test_candidate_generation_module_import_does_not_load_matplotlib() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module('meso_uq.active_learning.candidate_generation'); "
                "print('matplotlib' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )

    assert proc.stdout.strip() == "False"
