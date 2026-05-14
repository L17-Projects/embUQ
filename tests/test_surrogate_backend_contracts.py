from __future__ import annotations

import pytest

from meso_uq.core import (
    AgentFamily,
    Modality,
    ModelBackend,
    RequirementState,
    RuntimeRequirement,
    RuntimeRequirementKind,
)
from meso_uq.surrogates import (
    SurrogateBackendMetadata,
    SurrogateCheckpoint,
    SurrogateCheckpointMetadata,
    SurrogatePrediction,
    SurrogateDependencyError,
    SurrogateLoadRequest,
    SurrogatePredictionRequest,
    SurrogateTrainingRequest,
    clear_surrogate_backend_registry,
    list_surrogate_backends,
    register_surrogate_backend,
    resolve_surrogate_backend,
)


class _DeterministicToyBackend:
    metadata = SurrogateBackendMetadata(
        backend=ModelBackend.DNN,
        label="deterministic toy backend",
        supported_agent_families=(AgentFamily.EMB,),
        supported_modalities=(Modality.COMPRESSION,),
        runtime_requirements=(),
    )

    def train(self, request: SurrogateTrainingRequest) -> SurrogateCheckpoint:
        return SurrogateCheckpoint(
            uri=request.checkpoint_uri,
            metadata=SurrogateCheckpointMetadata(
                checkpoint_id="toy-dnn",
                backend=request.backend,
                agent_family=request.agent_family,
                modality=request.modality,
            ),
        )

    def load(self, request: SurrogateLoadRequest):
        return self

    def predict(self, request: SurrogatePredictionRequest):  # noqa: ANN001
        return SurrogatePrediction(mean=request.features)


class _UncertaintyToyBackend:
    metadata = SurrogateBackendMetadata(
        backend=ModelBackend.PYRO_BNN,
        label="uncertainty toy backend",
        supported_agent_families=(AgentFamily.EMB,),
        supported_modalities=(Modality.INDENTATION,),
        supports_epistemic=True,
        supports_aleatoric=True,
        supports_surrogate_error=True,
        runtime_requirements=(
            RuntimeRequirement(
                name="optional_fake_dep",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.OPTIONAL,
                package="optional_fake_dep",
            ),
        ),
    )

    def train(self, request: SurrogateTrainingRequest) -> SurrogateCheckpoint:
        return SurrogateCheckpoint(
            uri=request.checkpoint_uri,
            metadata=SurrogateCheckpointMetadata(
                checkpoint_id="toy-pyro-bnn",
                backend=request.backend,
                agent_family=request.agent_family,
                modality=request.modality,
            ),
        )

    def load(self, request: SurrogateLoadRequest):
        return self

    def predict(self, request: SurrogatePredictionRequest):  # noqa: ANN001
        mean = tuple(3.0 * value for value in request.features)
        variance = tuple(0.1 for _ in request.features)
        return SurrogatePrediction(
            mean=mean,
            epistemic=variance,
            aleatoric=variance,
            surrogate_error=variance,
        )


class _MissingDependencyToyBackend:
    metadata = SurrogateBackendMetadata(
        backend=ModelBackend.BNN,
        label="missing dependency toy backend",
        supported_agent_families=(AgentFamily.EMB,),
        supported_modalities=(Modality.COMPRESSION,),
        runtime_requirements=(
            RuntimeRequirement(
                name="missing_dep",
                kind=RuntimeRequirementKind.PYTHON_PACKAGE,
                state=RequirementState.REQUIRED,
                package="missing_dep",
            ),
        ),
    )

    def train(self, request: SurrogateTrainingRequest) -> SurrogateCheckpoint:
        return SurrogateCheckpoint(
            uri=request.checkpoint_uri,
            metadata=SurrogateCheckpointMetadata(
                checkpoint_id="toy-bnn",
                backend=request.backend,
                agent_family=request.agent_family,
                modality=request.modality,
            ),
        )

    def load(self, request: SurrogateLoadRequest):
        return self

    def predict(self, request: SurrogatePredictionRequest):  # noqa: ANN001
        return SurrogatePrediction(
            mean=request.features,
        )


@pytest.fixture(autouse=True)
def fresh_backend_registry() -> None:
    clear_surrogate_backend_registry()
    yield
    clear_surrogate_backend_registry()


def _training_request(*, backend: ModelBackend, family: AgentFamily, modality: Modality) -> SurrogateTrainingRequest:
    return SurrogateTrainingRequest(
        backend=backend,
        agent_family=family,
        modality=modality,
        training_data_uri="/tmp/data.csv",
        checkpoint_uri="/tmp/model.chk",
    )


def _load_request(*, backend: ModelBackend, family: AgentFamily, modality: Modality) -> SurrogateLoadRequest:
    return SurrogateLoadRequest(
        backend=backend,
        agent_family=family,
        modality=modality,
        checkpoint_uri="/tmp/model.chk",
    )


def test_contract_registry_resolves_fake_backends_with_expected_outputs():
    deterministic_backend = _DeterministicToyBackend()
    uncertainty_backend = _UncertaintyToyBackend()
    register_surrogate_backend(deterministic_backend)
    register_surrogate_backend(uncertainty_backend)
    assert tuple(list_surrogate_backends()) == (ModelBackend.DNN, ModelBackend.PYRO_BNN)

    resolved_dnn = resolve_surrogate_backend(
        "dnn",
        agent_family=AgentFamily.EMB,
        modality=Modality.COMPRESSION,
        availability={},
    )
    dnn_checkpoint = resolved_dnn.train(_training_request(
        backend=ModelBackend.DNN,
        family=AgentFamily.EMB,
        modality=Modality.COMPRESSION,
    ))
    dnn_prediction = resolved_dnn.load(
        _load_request(backend=ModelBackend.DNN, family=AgentFamily.EMB, modality=Modality.COMPRESSION),
    ).predict(SurrogatePredictionRequest(features=(1.0, 2.0)))

    assert dnn_prediction.mean == (1.0, 2.0)
    assert dnn_prediction.is_deterministic
    assert dnn_checkpoint.uri == "/tmp/model.chk"

    resolved_pyro = resolve_surrogate_backend(
        ModelBackend.PYRO_BNN,
        agent_family="emb",
        modality="indentation",
        availability={},
    )
    pyro_checkpoint = resolved_pyro.train(_training_request(
        backend=ModelBackend.PYRO_BNN,
        family=AgentFamily.EMB,
        modality=Modality.INDENTATION,
    ))
    pyro_prediction = resolved_pyro.load(
        _load_request(backend=ModelBackend.PYRO_BNN, family=AgentFamily.EMB, modality=Modality.INDENTATION),
    ).predict(SurrogatePredictionRequest(features=(2.0,)))

    assert pyro_checkpoint.metadata.checkpoint_id == "toy-pyro-bnn"
    assert pyro_prediction.mean == (6.0,)
    assert pyro_prediction.epistemic == (0.1,)
    assert pyro_prediction.aleatoric == (0.1,)
    assert pyro_prediction.surrogate_error == (0.1,)
    assert resolved_pyro.metadata.supports_epistemic
    assert resolved_pyro.metadata.supports_aleatoric
    assert resolved_pyro.metadata.supports_surrogate_error


def test_registry_rejects_required_dependency_unavailable():
    register_surrogate_backend(_MissingDependencyToyBackend())

    with pytest.raises(SurrogateDependencyError, match="missing required dependencies"):
        resolve_surrogate_backend(
            ModelBackend.BNN,
            agent_family=AgentFamily.EMB,
            modality=Modality.COMPRESSION,
            availability={},
        )
