from .controls import ALL_GV_CONTROLS, EXPERIMENT_CONTROLS
from .experiments import (
    GV_EXPERIMENT_LAYOUT_DIRS,
    GV_EXPERIMENTS,
    gv_experiment_layout,
    gv_experiment_layouts,
)
from .geometries import DEFAULT_GV_GEOMETRY, build_geometry, geometry_id
from .observables import GV_OBSERVABLE_SCHEMAS
from .paper_replay_constants import (
    GV_PAPER_REPLAY_FIGURE_TARGETS,
    GV_PAPER_REPLAY_PAPER_PDF,
    GV_PAPER_REPLAY_PROFILE_ID,
    GV_PAPER_REPLAY_SCHEMA_VERSION,
    GV_PAPER_REPLAY_SI_PDF,
    GVPaperReplayGeometry,
    GVPaperReplayProfile,
    GVPaperReplayProvenance,
    GVPaperReplayValue,
    canonical_runtime_source_path,
    load_gv_paper_replay_profile,
    validate_gv_paper_replay_profile,
)
from .parameters import GV_NOISE_MODEL, GV_PARAMETER_CONTRACT
from ..registry import StructureSpec


GV_STRUCTURE = StructureSpec(
    name="gv",
    description="Gas-vesicle structure contracts imported from the GV staging scripts.",
    parameter_contract=GV_PARAMETER_CONTRACT,
    geometries=(DEFAULT_GV_GEOMETRY,),
    experiments=GV_EXPERIMENTS,
    metadata={
        "source_of_truth": "MesoUQ GV structure implementation plan",
        "control_policy": "GV controls are design inputs and excluded from calibrated vectors.",
        "observable_schema": GV_OBSERVABLE_SCHEMAS,
    },
)


__all__ = [
    "ALL_GV_CONTROLS",
    "DEFAULT_GV_GEOMETRY",
    "EXPERIMENT_CONTROLS",
    "GV_EXPERIMENTS",
    "GV_EXPERIMENT_LAYOUT_DIRS",
    "GV_NOISE_MODEL",
    "GV_PARAMETER_CONTRACT",
    "GV_STRUCTURE",
    "build_geometry",
    "geometry_id",
    "GV_OBSERVABLE_SCHEMAS",
    "gv_experiment_layout",
    "gv_experiment_layouts",
    "GV_PAPER_REPLAY_FIGURE_TARGETS",
    "GV_PAPER_REPLAY_PAPER_PDF",
    "GV_PAPER_REPLAY_PROFILE_ID",
    "GV_PAPER_REPLAY_SCHEMA_VERSION",
    "GV_PAPER_REPLAY_SI_PDF",
    "GVPaperReplayGeometry",
    "GVPaperReplayProfile",
    "GVPaperReplayProvenance",
    "GVPaperReplayValue",
    "canonical_runtime_source_path",
    "load_gv_paper_replay_profile",
    "validate_gv_paper_replay_profile",
    "GVSampleResult",
    "GVMaterialGeometry",
    "GVRuntimeOptions",
    "GVSweep",
    "GV_SAMPLING_EXPERIMENTS",
    "sample_gv",
    "GVNumericalGenerationResult",
    "generate_gv_numerical_data",
    "GVLaunchRequest",
    "GVLaunchRunManifest",
    "GVLaunchCampaignManifest",
    "GVLaunchSchedulerScript",
    "GVLaunchRenderedCampaign",
    "validate_gv_launch_request",
    "build_gv_launch_campaign_manifest",
    "render_gv_launch_campaign",
    "render_gv_launch_campaigns",
]


def __getattr__(name: str):
    if name == "GVNumericalGenerationResult":
        from .generator import GVNumericalGenerationResult

        return GVNumericalGenerationResult
    if name == "generate_gv_numerical_data":
        from .generator import generate_gv_numerical_data

        return generate_gv_numerical_data
    if name == "sample_gv":
        from .sampling import sample_gv

        return sample_gv
    if name == "GVSampleResult":
        from .sampling import GVSampleResult

        return GVSampleResult
    if name == "GVMaterialGeometry":
        from .sampling import GVMaterialGeometry

        return GVMaterialGeometry
    if name == "GVRuntimeOptions":
        from .sampling import GVRuntimeOptions

        return GVRuntimeOptions
    if name == "GVSweep":
        from .sampling import GVSweep

        return GVSweep
    if name == "GV_SAMPLING_EXPERIMENTS":
        from .sampling import GV_SAMPLING_EXPERIMENTS

        return GV_SAMPLING_EXPERIMENTS
    if name == "GVLaunchRequest":
        from .launch import GVLaunchRequest

        return GVLaunchRequest
    if name == "GVLaunchRunManifest":
        from .launch import GVLaunchRunManifest

        return GVLaunchRunManifest
    if name == "GVLaunchCampaignManifest":
        from .launch import GVLaunchCampaignManifest

        return GVLaunchCampaignManifest
    if name == "GVLaunchSchedulerScript":
        from .launch import GVLaunchSchedulerScript

        return GVLaunchSchedulerScript
    if name == "GVLaunchRenderedCampaign":
        from .launch import GVLaunchRenderedCampaign

        return GVLaunchRenderedCampaign
    if name == "validate_gv_launch_request":
        from .launch import validate_gv_launch_request

        return validate_gv_launch_request
    if name == "build_gv_launch_campaign_manifest":
        from .launch import build_gv_launch_campaign_manifest

        return build_gv_launch_campaign_manifest
    if name == "render_gv_launch_campaign":
        from .launch import render_gv_launch_campaign

        return render_gv_launch_campaign
    if name == "render_gv_launch_campaigns":
        from .launch import render_gv_launch_campaigns

        return render_gv_launch_campaigns
    raise AttributeError(f"module 'meso_uq.structures.gv' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        list(globals())
        + [
            "GVMaterialGeometry",
            "GVRuntimeOptions",
            "GVSweep",
            "GVSampleResult",
            "GV_SAMPLING_EXPERIMENTS",
            "GVNumericalGenerationResult",
            "generate_gv_numerical_data",
            "sample_gv",
            "GVLaunchRequest",
            "GVLaunchRunManifest",
            "GVLaunchCampaignManifest",
            "GVLaunchSchedulerScript",
            "GVLaunchRenderedCampaign",
            "validate_gv_launch_request",
            "build_gv_launch_campaign_manifest",
            "render_gv_launch_campaign",
            "render_gv_launch_campaigns",
        ]
    )
