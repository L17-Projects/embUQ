from __future__ import annotations

import json

import pytest

from meso_uq.core import (
    AgentFamily,
    ArtifactClass,
    ManifestMetadata,
    Modality,
    ModelBackend,
    Platform,
    RunMetadata,
    coerce_agent_family,
    coerce_modality,
)


def test_core_identifiers_are_stable_strings():
    assert AgentFamily.EMB.value == "emb"
    assert AgentFamily.GV.value == "gv"
    assert Modality.COMPRESSION.value == "compression"
    assert Modality.SHEAR_FLOW.value == "shear_flow"
    assert ModelBackend.DNN.value == "dnn"
    assert Platform.KAROLINA.value == "karolina"
    assert ArtifactClass.RUN_MANIFEST.value == "run_manifest"


def test_invalid_core_identifiers_raise_clear_errors():
    with pytest.raises(ValueError, match="Unsupported agent family 'bubble'"):
        coerce_agent_family("bubble")
    with pytest.raises(ValueError, match="Unsupported modality 'stretching'"):
        coerce_modality("stretching")


def test_manifest_metadata_serializes_round_trip():
    run = RunMetadata(
        run_id="run-001",
        agent_family="gv",
        modality="torsion",
        model_backend="synthetic",
        platform="workstation",
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
    assert decoded["run"]["artifact_classes"] == ["reference", "runtime_manifest"]
    assert restored == manifest
