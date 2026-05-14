from __future__ import annotations

import pytest

from meso_uq.structures import get_structure
from meso_uq.structures.gv import (
    GV_EXPERIMENT_LAYOUT_DIRS,
    gv_experiment_layout,
    gv_experiment_layouts,
)


def test_gv_experiment_metadata_covers_expected_lanes() -> None:
    gv = get_structure("gv")
    assert [experiment.name for experiment in gv.experiments] == [
        "stretching",
        "buckling",
        "torsion",
        "eigenmodes",
        "shear_flow",
    ]
    assert gv.get_experiment("stretching").control_names == ("tot_force", "bpress")
    assert gv.get_experiment("buckling").control_names == ("buck", "bpress")
    assert gv.get_experiment("torsion").control_names == ("theta",)
    assert gv.get_experiment("eigenmodes").control_names == ("bpress",)


def test_shear_flow_requires_experimental_opt_in() -> None:
    gv = get_structure("gv")
    with pytest.raises(ValueError, match="requires include_experimental=True"):
        gv.get_experiment("shear_flow")

    experiment = gv.get_experiment("shear_flow", include_experimental=True)
    assert experiment.experimental is True
    assert experiment.control_names == ("ptan", "afsi", "bpress")


def test_gv_experiment_lookup_rejects_unknown_name() -> None:
    gv = get_structure("gv")
    with pytest.raises(KeyError, match="Unknown experiment"):
        gv.get_experiment("missing")


def test_gv_experiment_layout_contract_is_normalized() -> None:
    gv = get_structure("gv")
    layouts = gv_experiment_layouts()

    assert GV_EXPERIMENT_LAYOUT_DIRS == ("src", "evalkit", "surrogate")
    assert set(layouts) == {experiment.name for experiment in gv.experiments}
    assert layouts["stretching"] == {
        "src": "gv/stretching/src",
        "evalkit": "gv/stretching/evalkit",
        "surrogate": "gv/stretching/surrogate",
    }
    with pytest.raises(KeyError, match="Unknown GV experiment"):
        gv_experiment_layout("missing")
