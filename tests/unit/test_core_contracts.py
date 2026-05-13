from __future__ import annotations

import json

import pytest

from meso_uq.core import (
    AgentFamily,
    ArtifactReference,
    ArtifactClass,
    DatasetSourceMetadata,
    InferenceBackend,
    ManifestMetadata,
    Modality,
    ModelBackend,
    NoiseModelKind,
    Platform,
    RuntimeRequirement,
    RuntimeRequirementKind,
    RunMetadata,
    SurrogateIdentifier,
    coerce_agent_family,
    coerce_modality,
)


def test_core_identifiers_are_stable_strings():
    assert AgentFamily.EMB.value == "emb"
    assert AgentFamily.GV.value == "gv"
    assert Modality.COMPRESSION.value == "compression"
    assert Modality.STRETCHING.value == "stretching"
    assert Modality.SHEAR_FLOW.value == "shear_flow"
    assert ModelBackend.DNN.value == "dnn"
    assert ModelBackend.PYRO_BNN.value == "pyro_bnn"
    assert InferenceBackend.KORALI.value == "korali"
    assert NoiseModelKind.SURROGATE_ERROR.value == "surrogate_error"
    assert Platform.KAROLINA.value == "karolina"
    assert Platform.GENERIC_SLURM.value == "generic_slurm"
    assert ArtifactClass.RUN_MANIFEST.value == "run_manifest"
    assert ArtifactClass.SURROGATE_CHECKPOINT.value == "surrogate_checkpoint"


def test_invalid_core_identifiers_raise_clear_errors():
    with pytest.raises(ValueError, match="Unsupported agent family 'bubble'"):
        coerce_agent_family("bubble")
    with pytest.raises(ValueError, match="Unsupported modality 'swelling'"):
        coerce_modality("swelling")


def test_manifest_metadata_serializes_round_trip():
    run = RunMetadata(
        run_id="run-001",
        agent_family="gv",
        modality="torsion",
        model_backend="synthetic",
        inference_backend="korali",
        noise_model="surrogate_error",
        platform="workstation",
        experiment_id="gv-torsion-smoke",
        code_version="test-sha",
        config_hash="config-hash",
        data_hash="data-hash",
        environment={"python": "3.11"},
        artifact_classes=("reference", "runtime_manifest"),
        tags=("smoke",),
        metadata={"geometry": "default_gv"},
    )
    manifest = ManifestMetadata(
        schema_version="meso_uq.manifest.v1",
        manifest_kind="run_manifest",
        generated_by="unit-test",
        run=run,
        metadata={"issue": "MES-137"},
    )

    payload = manifest.to_json()
    decoded = json.loads(payload)
    restored = ManifestMetadata.from_json(payload)

    assert decoded["run"]["agent_family"] == "gv"
    assert decoded["run"]["inference_backend"] == "korali"
    assert decoded["run"]["noise_model"] == "surrogate_error"
    assert decoded["run"]["environment"] == {"python": "3.11"}
    assert decoded["run"]["artifact_classes"] == ["reference", "runtime_manifest"]
    assert restored == manifest


def test_dataset_artifact_and_surrogate_identifiers_round_trip():
    dataset = DatasetSourceMetadata(
        dataset_id="emb-compression-2.1um",
        source="compression/surrogate/diameters/2.1um/data/F_Delta.dat",
        artifact_class="reference",
        checksum="sha256:test",
        unit_system="micrometer",
    )
    artifact = ArtifactReference(
        artifact_id="emb-compression-2.1um-dnn",
        artifact_class="surrogate_checkpoint",
        uri="compression/surrogate/diameters/2.1um/trained/microbubble_force_BEST.pkl",
        checksum="sha256:model",
    )
    surrogate = SurrogateIdentifier(
        surrogate_id="emb-compression-2.1um-dnn",
        agent_family="emb",
        modality="compression",
        backend="dnn",
        artifact=artifact,
    )
    requirement = RuntimeRequirement(
        name="torch",
        kind=RuntimeRequirementKind.PYTHON_PACKAGE,
        package="torch",
    )

    assert DatasetSourceMetadata.from_dict(dataset.as_dict()) == dataset
    assert ArtifactReference.from_dict(artifact.as_dict()) == artifact
    assert SurrogateIdentifier.from_dict(surrogate.as_dict()) == surrogate
    assert requirement.as_dict()["kind"] == "python_package"
