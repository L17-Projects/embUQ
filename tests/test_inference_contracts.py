from __future__ import annotations

import pytest

from meso_uq.core import (
    AgentFamily,
    InferenceBackend,
    Modality,
    ModelBackend,
    NoiseModelKind,
    Platform,
    RequirementState,
    RuntimeRequirement,
    RuntimeRequirementKind,
)
from meso_uq.inference.contracts import (
    InferenceContract,
    InferenceLayer,
    InferenceSupportState,
    LikelihoodComponentContract,
    PosteriorArtifactContract,
    PosteriorStorageKind,
    PriorContract,
    SamplerBackendContract,
    unsupported_gv_inference_contract,
)


def _emb_contract(*, uncertainty_fields: tuple[str, ...] = ("epistemic",)) -> InferenceContract:
    return InferenceContract(
        contract_id="emb-compression-dry-run",
        agent_family=AgentFamily.EMB,
        modality=Modality.COMPRESSION,
        dataset_id="emb:compression:diameter_2.0um:default",
        surrogate_backend=ModelBackend.BNN,
        sampler=SamplerBackendContract(backend=InferenceBackend.DRY_RUN),
        priors=(
            PriorContract("population_mu", InferenceLayer.POPULATION, (0.0, 1.0)),
            PriorContract("sigma", InferenceLayer.MEASUREMENT, (0.01, 0.5)),
        ),
        likelihood_components=(
            LikelihoodComponentContract(
                name="measurement",
                noise_model=NoiseModelKind.MEASUREMENT_ERROR,
                layer=InferenceLayer.MEASUREMENT,
                observables=("force",),
            ),
            LikelihoodComponentContract(
                name="surrogate-error",
                noise_model=NoiseModelKind.SURROGATE_ERROR,
                layer=InferenceLayer.SURROGATE_ERROR,
                observables=("force",),
                required_surrogate_uncertainty=("epistemic",),
            ),
        ),
        platform=Platform.KAROLINA,
        surrogate_uncertainty_fields=uncertainty_fields,
    )


def test_inference_contract_serializes_layers_and_validates_supported_combo() -> None:
    contract = _emb_contract()

    contract.require_supported()
    payload = contract.as_dict()

    assert payload["agent_family"] == "emb"
    assert payload["modality"] == "compression"
    assert payload["sampler"]["backend"] == "dry_run"
    assert {item["layer"] for item in payload["likelihood_components"]} == {
        "measurement",
        "surrogate_error",
    }


def test_inference_contract_rejects_missing_surrogate_uncertainty() -> None:
    contract = _emb_contract(uncertainty_fields=())

    errors = contract.validation_errors()

    assert any("requires surrogate uncertainty fields" in error for error in errors)
    with pytest.raises(ValueError, match="surrogate uncertainty"):
        contract.require_supported()


def test_sampler_backend_contract_reports_actionable_missing_dependency() -> None:
    sampler = SamplerBackendContract(
        backend=InferenceBackend.KORALI,
        requirements=(
            RuntimeRequirement(
                name="korali",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.EXTERNAL,
                package="korali",
                description="bootstrap the Korali runtime before sampling",
            ),
        ),
    )

    messages = sampler.missing_requirement_messages()

    assert messages == ("korali requires korali: bootstrap the Korali runtime before sampling",)


def test_unsupported_gv_inference_contract_fails_explicitly() -> None:
    contract = unsupported_gv_inference_contract(
        contract_id="gv-buckling-hbi",
        modality=Modality.BUCKLING,
        dataset_id="gv:buckling:ribbed:default",
        reason="GV posterior calibration awaits validated surrogate/noise contracts.",
    )

    assert contract.support_state is InferenceSupportState.UNSUPPORTED
    assert "GV posterior calibration" in contract.validation_errors()[0]


def test_posterior_artifact_contract_is_manifestable_and_rejects_private_absolute_path() -> None:
    artifact = PosteriorArtifactContract(
        artifact_id="posterior-summary",
        path="artifacts/posteriors/summary.json",
        storage_kind=PosteriorStorageKind.SUMMARY,
        parameters=("mu", "sigma"),
    )

    assert artifact.as_manifest_record()["artifact_class"] == "posterior"

    with pytest.raises(ValueError, match="relative"):
        PosteriorArtifactContract(
            artifact_id="bad",
            path="/tmp/private/posterior.json",
            storage_kind=PosteriorStorageKind.SAMPLES,
            parameters=("mu",),
        )
