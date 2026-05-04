from .controls import ALL_GV_CONTROLS, EXPERIMENT_CONTROLS
from .experiments import GV_EXPERIMENTS
from .geometries import DEFAULT_GV_GEOMETRY, build_geometry, geometry_id
from .observables import GV_OBSERVABLE_SCHEMAS
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
    "GV_NOISE_MODEL",
    "GV_PARAMETER_CONTRACT",
    "GV_STRUCTURE",
    "build_geometry",
    "geometry_id",
    "GV_OBSERVABLE_SCHEMAS",
    "GVNumericalGenerationResult",
    "generate_gv_numerical_data",
]


def __getattr__(name: str):
    if name == "GVNumericalGenerationResult":
        from .generator import GVNumericalGenerationResult

        return GVNumericalGenerationResult
    if name == "generate_gv_numerical_data":
        from .generator import generate_gv_numerical_data

        return generate_gv_numerical_data
    raise AttributeError(f"module 'meso_uq.structures.gv' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + ["GVNumericalGenerationResult", "generate_gv_numerical_data"])
