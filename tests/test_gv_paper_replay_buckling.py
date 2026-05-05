from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.paper_replay_lanes import buckling as buckling_module
from meso_uq.structures.gv.paper_replay_lanes.buckling import (
    BUCKLING_FIGURE_ID,
    plan_buckling_paper_replay_lane,
    plot_buckling_paper_replay,
    postprocess_buckling_paper_replay_lane,
    run_buckling_paper_replay_lane,
)


_BASE_MATERIAL_PARAMETERS = {
    "ka": 1.1,
    "kb": 1.2,
    "mu": 0.9,
    "b1": 0.0,
    "b2": 0.0,
    "a3": 0.0,
    "a4": 0.0,
    "mu_l": 0.5,
    "c": 0.6,
}


def test_plan_buckling_paper_replay_lane_uses_pressure_sweep_and_records_assumptions() -> None:
    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        pressure_differences=(-88.0, -91.0, -94.0),
        buck=0.75,
    )

    assert plan.figure_id == BUCKLING_FIGURE_ID
    assert plan.experiment == "buckling"
    assert plan.controls == {"buck": 0.75, "bpress": (-88.0, -91.0, -94.0)}
    assert plan.runtime_options.controls == {"buck": 0.75}
    assert plan.runtime_options.output_root == "_runs/gv/paper_replay/buckling"
    assert len(plan.mapping_assumptions) >= 2


def test_postprocess_buckling_paper_replay_lane_requires_relative_volume_channel() -> None:
    with pytest.raises(ValueError, match="requires channels: bpress, relative_volume"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"buck": 0.75},
                "channels": {
                    "bpress": [-88.0, -91.0],
                    "buckling_response": [1.0, 0.8],
                },
            }
        )


def test_postprocess_buckling_paper_replay_lane_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"buck": 0.75},
                "channels": {
                    "bpress": [-88.0, -91.0],
                    "relative_volume": [1.0, np.nan],
                    "buckling_response": [1.0, 0.8],
                },
            }
        )


def test_postprocess_buckling_paper_replay_lane_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"buck": 0.75},
                "channels": {
                    "bpress": [-88.0, -91.0],
                    "relative_volume": [1.0],
                    "buckling_response": [0.1, 0.2],
                },
            }
        )


def test_run_buckling_paper_replay_lane_uses_injected_sampler() -> None:
    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        pressure_differences=(-88.0, -91.0, -94.0),
    )

    captured: dict[str, object] = {}

    def fake_sampler(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "experiment": "buckling",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"buck": 0.75},
            "channels": {
                "bpress": [-88.0, -91.0, -94.0],
                "relative_volume": [1.0, 0.96, 0.9],
                "buckling_response": [1.0, 0.95, 0.82],
            },
        }

    result = run_buckling_paper_replay_lane(plan, sampler=fake_sampler)

    assert captured["experiment"] == "buckling"
    assert captured["controls"] == {"buck": 0.75, "bpress": (-88.0, -91.0, -94.0)}
    assert result.axis == "bpress"
    assert np.array_equal(result.channels["bpress"], np.array([-88.0, -91.0, -94.0]))
    assert np.array_equal(result.channels["relative_volume"], np.array([1.0, 0.96, 0.9]))
    assert result.provenance["lane_plan"]["figure_id"] == BUCKLING_FIGURE_ID


def test_buckling_paper_replay_accepts_object_payload_and_writes_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sample_result = SimpleNamespace(
        experiment="buckling",
        geometry=SimpleNamespace(radGV=2.0, height=14.28),
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"buck": 0.75, "pressure_difference": -91.0},
        channels={
            "pressure_difference": [-88.0, -91.0, -94.0],
            "volumetric_strain": [1.0, 0.96, 0.9],
            "shape_amplitude": [0.0, 0.1, 0.3],
        },
        manifest={"dataset_id": "gv__buckling__fixture"},
        runtime_manifests=({"control_id": "pressure_-91"},),
        plan_manifests=({"experiment": "buckling"},),
        work_dirs=(tmp_path / "_runs" / "work",),
        runtime_seconds=12.5,
        status="passed",
    )

    result = postprocess_buckling_paper_replay_lane(sample_result)
    manifest = result.to_manifest()
    plot_path = plot_buckling_paper_replay(
        result,
        output_path="_runs/gv/paper_replay/buckling/relative_volume.png",
    )

    assert manifest["channels"]["relative_volume"] == [1.0, 0.96, 0.9]
    assert result.controls["buck"] == 0.75
    assert result.controls["bpress"] == -91.0
    assert result.provenance["runtime_seconds"] == 12.5
    assert result.provenance["runtime_manifests"] == [{"control_id": "pressure_-91"}]
    assert plot_path.is_file()


def test_plot_buckling_paper_replay_rejects_non_runs_output() -> None:
    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        pressure_differences=(-88.0, -91.0),
    )
    result = postprocess_buckling_paper_replay_lane(
        {
            "experiment": "buckling",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"buck": 0.75},
            "channels": {
                "bpress": [-88.0, -91.0],
                "relative_volume": [1.0, 0.96],
                "buckling_response": [1.0, 0.95],
            },
        },
        plan=plan,
    )

    with pytest.raises(ValueError, match="under _runs/"):
        plot_buckling_paper_replay(result, output_path="paper_replay/buckling.png")


def test_plan_buckling_paper_replay_rejects_output_root_traversal() -> None:
    with pytest.raises(ValueError, match="under"):
        plan_buckling_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            pressure_differences=(-88.0, -91.0),
            output_root="_runs/../outside",
        )


def test_plan_buckling_paper_replay_rejects_invalid_material_and_sweep_inputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        plan_buckling_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            pressure_differences=(-88.0,),
        )

    bad_parameters = dict(_BASE_MATERIAL_PARAMETERS)
    bad_parameters["ka"] = 0.0
    with pytest.raises(ValueError, match="Material parameter 'ka'"):
        plan_buckling_paper_replay_lane(
            material_parameters=bad_parameters,
            radGV=2.0,
            height=14.28,
            pressure_differences=(-88.0, -91.0),
        )

    duplicate_parameters = dict(_BASE_MATERIAL_PARAMETERS)
    duplicate_parameters["muL"] = duplicate_parameters["mu_l"]
    with pytest.raises(ValueError, match="duplicated"):
        plan_buckling_paper_replay_lane(
            material_parameters=duplicate_parameters,
            radGV=2.0,
            height=14.28,
            pressure_differences=(-88.0, -91.0),
        )


def test_buckling_private_coercion_and_metadata_error_paths() -> None:
    class ManifestOnly:
        def as_manifest(self):
            return {
                "channels": {"bpress": [-88.0], "relative_volume": [1.0]},
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"buck": 0.75},
            }

    assert buckling_module._coerce_mapping(ManifestOnly(), context="fixture")["controls"] == {"buck": 0.75}

    with pytest.raises(ValueError, match="1D"):
        buckling_module._coerce_finite_1d([[1.0]], name="pressure_differences")
    with pytest.raises(ValueError, match="at least one"):
        buckling_module._coerce_finite_1d([], name="pressure_differences")
    with pytest.raises(ValueError, match="finite"):
        buckling_module._coerce_finite_1d([float("nan")], name="pressure_differences")
    with pytest.raises(ValueError, match="scalar"):
        buckling_module._coerce_scalar([1.0, 2.0], name="buck")
    with pytest.raises(ValueError, match="finite"):
        buckling_module._coerce_scalar(float("nan"), name="buck")

    with pytest.raises(ValueError, match="Unexpected"):
        buckling_module._validate_paper_replay_material_parameters({**_BASE_MATERIAL_PARAMETERS, "bad": 1.0})
    with pytest.raises(ValueError, match=">= 0"):
        buckling_module._validate_paper_replay_material_parameters({**_BASE_MATERIAL_PARAMETERS, "ka": -1.0})
    with pytest.raises(ValueError, match="Missing required"):
        buckling_module._validate_paper_replay_material_parameters(
            {name: value for name, value in _BASE_MATERIAL_PARAMETERS.items() if name != "ka"}
        )

    with pytest.raises(ValueError, match="geometry metadata"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"buck": 0.75},
                "channels": {"bpress": [-88.0], "relative_volume": [1.0], "buckling_response": [0.1]},
            }
        )
    with pytest.raises(ValueError, match="material_parameters"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "controls": {"buck": 0.75},
                "channels": {"bpress": [-88.0], "relative_volume": [1.0], "buckling_response": [0.1]},
            }
        )
    with pytest.raises(ValueError, match="scalar controls"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "channels": {"bpress": [-88.0], "relative_volume": [1.0], "buckling_response": [0.1]},
            }
        )
