from .common import (
    GVNumericalPostprocessResult,
    build_gv_numerical_manifest,
    validate_numeric_channels,
    write_numerical_dataset_artifacts,
)
from .buckling import (
    parse_buckling_fixture_channels,
    process_buckling_numerical_dataset,
)
from .eigenmodes import (
    parse_eigenmodes_fixture_channels,
    process_eigenmodes_numerical_dataset,
)
from .stretching import (
    parse_stretching_fixture_channels,
    process_stretching_numerical_dataset,
)
from .torsion import (
    parse_torsion_fixture_channels,
    process_torsion_numerical_dataset,
)


POSTPROCESSORS = {
    "gv:buckling": process_buckling_numerical_dataset,
    "gv:eigenmodes": process_eigenmodes_numerical_dataset,
    "gv:stretching": process_stretching_numerical_dataset,
    "gv:torsion": process_torsion_numerical_dataset,
}

__all__ = [
    "GVNumericalPostprocessResult",
    "POSTPROCESSORS",
    "build_gv_numerical_manifest",
    "parse_buckling_fixture_channels",
    "parse_eigenmodes_fixture_channels",
    "parse_stretching_fixture_channels",
    "parse_torsion_fixture_channels",
    "process_buckling_numerical_dataset",
    "process_eigenmodes_numerical_dataset",
    "process_stretching_numerical_dataset",
    "process_torsion_numerical_dataset",
    "validate_numeric_channels",
    "write_numerical_dataset_artifacts",
]
