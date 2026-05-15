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
    assert callable(gv.validate_gv_launch_request)
    assert callable(gv.build_gv_launch_campaign_manifest)
    assert "GVLaunchRequest" in exported
    assert "GVLaunchRunManifest" in exported
    assert "GVLaunchCampaignManifest" in exported
    assert "validate_gv_launch_request" in exported
    assert "build_gv_launch_campaign_manifest" in exported
