from __future__ import annotations

import pytest

from meso_uq.agents import (
    classify_agent_modality_support,
    get_agent_definition,
    list_agent_definitions,
    list_agent_modalities,
    missing_dependency_requirements_for_agent,
    resolve_agent_modality,
    resolve_agent_family_identifier,
    runtime_requirements_for_agent,
    supported_modalities_for_agent,
)
from meso_uq.core import AgentFamily, Modality, RequirementState


def test_agent_registry_contains_emb_definition():
    definitions = {definition.family: definition for definition in list_agent_definitions()}

    assert set(definitions) == {AgentFamily.EMB}
    assert tuple(item.value for item in definitions[AgentFamily.EMB].supported_modalities) == (
        "compression",
        "indentation",
    )
    assert definitions[AgentFamily.EMB].default_for_legacy is True


def test_agent_registry_resolves_supported_combinations():
    agent, modality = resolve_agent_modality("encapsulated_microbubble", "compression")

    assert agent.family is AgentFamily.EMB
    assert modality.modality is Modality.COMPRESSION
    assert supported_modalities_for_agent("emb") == (Modality.COMPRESSION, Modality.INDENTATION)
    assert tuple(descriptor.modality for descriptor in list_agent_modalities("emb")) == (
        Modality.COMPRESSION,
        Modality.INDENTATION,
    )
    assert resolve_agent_family_identifier("microbubble") is AgentFamily.EMB


def test_agent_registry_rejects_invalid_family_identifier():
    with pytest.raises(ValueError, match="Unsupported agent family 'cell'"):
        get_agent_definition("cell")


def test_agent_registry_rejects_unsupported_family_modality_combinations():
    with pytest.raises(ValueError, match="Agent family 'emb' does not support modality 'buckling'"):
        resolve_agent_modality("emb", "buckling")
    with pytest.raises(ValueError, match="Agent family 'gv' is not registered"):
        resolve_agent_modality("gv", "compression")


def test_agent_registry_reports_support_and_dependency_state():
    status = classify_agent_modality_support("emb", "compression")
    unsupported = classify_agent_modality_support("emb", "shear_flow")
    requirements = runtime_requirements_for_agent("emb")
    missing = missing_dependency_requirements_for_agent("emb", {"torch": False, "pyro": False, "korali": False})

    assert status["status"] == "supported"
    assert unsupported["status"] == "unsupported"
    assert any(requirement.state is RequirementState.EXTERNAL for requirement in requirements)
    assert tuple(requirement.name for requirement in missing) == ("torch", "korali")
