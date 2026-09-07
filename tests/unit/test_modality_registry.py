from __future__ import annotations

import pytest

from meso_uq.core import AgentFamily, Modality
from meso_uq.modalities import assert_modality_family, get_modality_descriptor, list_modality_descriptors


def test_modality_registry_lists_metadata_descriptors():
    descriptors = list_modality_descriptors()
    descriptor_ids = {descriptor.modality.value for descriptor in descriptors}

    assert descriptor_ids == {"compression", "indentation"}
    assert all(descriptor.smoke_level == "metadata" for descriptor in descriptors)


def test_modality_registry_can_filter_by_family():
    emb = list_modality_descriptors(AgentFamily.EMB)
    gv = list_modality_descriptors("gv")

    assert tuple(descriptor.modality for descriptor in emb) == (Modality.COMPRESSION, Modality.INDENTATION)
    assert gv == ()


def test_modality_descriptor_serialization_is_metadata_only():
    descriptor = get_modality_descriptor("compression")
    payload = descriptor.as_dict()
    restored = type(descriptor).from_dict(payload)

    assert payload["family"] == "emb"
    assert payload["metadata"]["legacy_root"] == "emb/compression"
    assert payload["runtime_requirements"][0]["name"] == "torch"
    assert payload["input_controls"] == ["diameter_um", "displacement"]
    assert payload["observables"] == {"force": "force", "displacement": "length"}
    assert restored == descriptor


def test_modality_family_mismatch_has_clear_error():
    with pytest.raises(ValueError, match="Modality 'compression' belongs to agent family 'emb', not 'gv'"):
        assert_modality_family("compression", "gv")
