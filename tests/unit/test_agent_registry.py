from __future__ import annotations

import pytest

from meso_uq.agents import (
    get_agent_definition,
    list_agent_definitions,
    list_agent_modalities,
    resolve_agent_modality,
    supported_modalities_for_agent,
)
from meso_uq.core import AgentFamily, Modality


def test_agent_registry_contains_emb_and_gv_definitions():
    definitions = {definition.family: definition for definition in list_agent_definitions()}

    assert set(definitions) == {AgentFamily.EMB, AgentFamily.GV}
    assert tuple(item.value for item in definitions[AgentFamily.EMB].supported_modalities) == (
        "compression",
        "indentation",
    )
    assert tuple(item.value for item in definitions[AgentFamily.GV].supported_modalities) == (
        "buckling",
        "torsion",
        "eigenmodes",
        "shear_flow",
    )


def test_agent_registry_resolves_supported_combinations():
    agent, modality = resolve_agent_modality("emb", "compression")

    assert agent.family is AgentFamily.EMB
    assert modality.modality is Modality.COMPRESSION
    assert supported_modalities_for_agent("gv") == (
        Modality.BUCKLING,
        Modality.TORSION,
        Modality.EIGENMODES,
        Modality.SHEAR_FLOW,
    )
    assert tuple(descriptor.modality for descriptor in list_agent_modalities("emb")) == (
        Modality.COMPRESSION,
        Modality.INDENTATION,
    )


def test_agent_registry_rejects_invalid_family_identifier():
    with pytest.raises(ValueError, match="Unsupported agent family 'vesicle'"):
        get_agent_definition("vesicle")


def test_agent_registry_rejects_unsupported_family_modality_combinations():
    with pytest.raises(ValueError, match="Agent family 'emb' does not support modality 'buckling'"):
        resolve_agent_modality("emb", "buckling")
    with pytest.raises(ValueError, match="Agent family 'gv' does not support modality 'compression'"):
        resolve_agent_modality("gv", "compression")
