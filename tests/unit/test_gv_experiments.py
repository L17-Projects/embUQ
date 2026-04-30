from __future__ import annotations

import pytest

from meso_uq.structures import get_structure


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
