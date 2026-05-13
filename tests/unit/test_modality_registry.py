from __future__ import annotations

import pytest

from meso_uq.core import AgentFamily, Modality
from meso_uq.modalities import assert_modality_family, get_modality_descriptor, list_modality_descriptors


def test_modality_registry_lists_metadata_descriptors():
    descriptors = list_modality_descriptors()
    descriptor_ids = {descriptor.modality.value for descriptor in descriptors}

    assert descriptor_ids == {
        "compression",
        "indentation",
        "buckling",
        "torsion",
        "eigenmodes",
        "shear_flow",
    }
    assert all(descriptor.smoke_level == "metadata" for descriptor in descriptors)


def test_modality_registry_can_filter_by_family():
    emb = list_modality_descriptors(AgentFamily.EMB)
    gv = list_modality_descriptors("gv")

    assert tuple(descriptor.modality for descriptor in emb) == (Modality.COMPRESSION, Modality.INDENTATION)
    assert tuple(descriptor.modality.value for descriptor in gv) == (
        "buckling",
        "torsion",
        "eigenmodes",
        "shear_flow",
    )


def test_modality_descriptor_serialization_is_metadata_only():
    descriptor = get_modality_descriptor("shear_flow")
    payload = descriptor.as_dict()
    restored = type(descriptor).from_dict(payload)

    assert payload["family"] == "gv"
    assert payload["metadata"]["known_runtime_status"] == "staging_out_of_scope"
    assert restored == descriptor


def test_modality_family_mismatch_has_clear_error():
    with pytest.raises(ValueError, match="Modality 'buckling' belongs to agent family 'gv', not 'emb'"):
        assert_modality_family("buckling", "emb")
