from .controls import ALL_GV_CONTROLS, EXPERIMENT_CONTROLS
from .experiments import GV_EXPERIMENTS
from .geometries import DEFAULT_GV_GEOMETRY, build_geometry, geometry_id
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
    },
)


__all__ = [
    "ALL_GV_CONTROLS",
    "DEFAULT_GV_GEOMETRY",
    "EXPERIMENT_CONTROLS",
    "GV_EXPERIMENTS",
    "GV_NOISE_MODEL",
    "GV_PARAMETER_CONTRACT",
    "GV_STRUCTURE",
    "build_geometry",
    "geometry_id",
]
