from __future__ import annotations

from meso_uq.structures.gv.postprocessing import POSTPROCESSORS


CORE_GV_POSTPROCESSOR_EXPERIMENTS = (
    "gv:stretching",
    "gv:buckling",
    "gv:torsion",
    "gv:eigenmodes",
)


def test_postprocessors_registry_contains_exact_core_gv_experiments() -> None:
    assert set(POSTPROCESSORS.keys()) == set(CORE_GV_POSTPROCESSOR_EXPERIMENTS)


def test_postprocessors_registry_excludes_shear_flow() -> None:
    assert "gv:shear_flow" not in POSTPROCESSORS


def test_postprocessors_registry_entries_are_postprocessors() -> None:
    for experiment, processor in POSTPROCESSORS.items():
        assert experiment in CORE_GV_POSTPROCESSOR_EXPERIMENTS
        assert callable(processor)
        assert processor.__name__.startswith("process_")
