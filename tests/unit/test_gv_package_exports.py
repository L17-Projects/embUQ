from __future__ import annotations

import pytest

import meso_uq.structures.gv as gv


def test_gv_package_lazy_numerical_generation_exports() -> None:
    assert gv.GVNumericalGenerationResult.__name__ == "GVNumericalGenerationResult"
    assert callable(gv.generate_gv_numerical_data)
    exported = dir(gv)
    assert "GVNumericalGenerationResult" in exported
    assert "generate_gv_numerical_data" in exported


def test_gv_package_rejects_unknown_lazy_export() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        gv.__getattr__("not_an_export")


def test_gv_package_exports_sampling_api() -> None:
    exported = dir(gv)
    assert gv.sample_gv.__name__ == "sample_gv"
    assert gv.GVMaterialGeometry.__name__ == "GVMaterialGeometry"
    assert gv.GVRuntimeOptions.__name__ == "GVRuntimeOptions"
    assert gv.GVSweep.__name__ == "GVSweep"
    assert gv.GVSampleResult.__name__ == "GVSampleResult"
    assert gv.GV_SAMPLING_EXPERIMENTS == ("stretching", "buckling", "torsion", "eigenmodes")
    assert "sample_gv" in exported
    assert "GVMaterialGeometry" in exported
    assert "GVRuntimeOptions" in exported
    assert "GVSweep" in exported


def test_gv_package_exports_launch_api() -> None:
    exported = dir(gv)
    assert gv.GVLaunchRequest.__name__ == "GVLaunchRequest"
    assert gv.GVLaunchRunManifest.__name__ == "GVLaunchRunManifest"
    assert gv.GVLaunchCampaignManifest.__name__ == "GVLaunchCampaignManifest"
    assert gv.GVLaunchSchedulerScript.__name__ == "GVLaunchSchedulerScript"
    assert gv.GVLaunchRenderedCampaign.__name__ == "GVLaunchRenderedCampaign"
    assert gv.GVActiveLearningLaunchCandidate.__name__ == "GVActiveLearningLaunchCandidate"
    assert gv.GVActiveLearningLaunchHandoff.__name__ == "GVActiveLearningLaunchHandoff"
    assert callable(gv.validate_gv_launch_request)
    assert callable(gv.build_gv_launch_campaign_manifest)
    assert callable(gv.render_gv_launch_campaign)
    assert callable(gv.render_gv_launch_campaigns)
    assert callable(gv.build_gv_active_learning_launch_handoff)
    assert callable(gv.render_gv_active_learning_launch_handoff)
    assert "GVLaunchRequest" in exported
    assert "GVLaunchRunManifest" in exported
    assert "GVLaunchCampaignManifest" in exported
    assert "GVLaunchSchedulerScript" in exported
    assert "GVLaunchRenderedCampaign" in exported
    assert "GVActiveLearningLaunchCandidate" in exported
    assert "GVActiveLearningLaunchHandoff" in exported
    assert "validate_gv_launch_request" in exported
    assert "build_gv_launch_campaign_manifest" in exported
    assert "render_gv_launch_campaign" in exported
    assert "render_gv_launch_campaigns" in exported
    assert "build_gv_active_learning_launch_handoff" in exported
    assert "render_gv_active_learning_launch_handoff" in exported
