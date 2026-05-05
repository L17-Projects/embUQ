from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.paper_replay_lanes import eigenmodes as eigenmodes_module
from meso_uq.structures.gv.paper_replay_lanes.eigenmodes import (
    EIGENMODES_FIGURE_ID,
    plan_eigenmodes_paper_replay_lane,
    plot_eigenmodes_paper_replay,
    postprocess_eigenmodes_paper_replay_lane,
    run_eigenmodes_paper_replay_lane,
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


def test_plan_eigenmodes_paper_replay_lane_records_mode_count() -> None:
    plan = plan_eigenmodes_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        bpress=-92.0,
        mode_count=12,
    )

    assert plan.figure_id == EIGENMODES_FIGURE_ID
    assert plan.controls == {"bpress": -92.0}
    assert plan.mode_count == 12
    assert plan.runtime_options.output_root == "_runs/gv/paper_replay/eigenmodes"


def test_postprocess_eigenmodes_paper_replay_lane_requires_spectrum_channel() -> None:
    with pytest.raises(ValueError, match="requires at least one spectral channel"):
        postprocess_eigenmodes_paper_replay_lane(
            {
                "experiment": "eigenmodes",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
                "channels": {
                    "mode_index": [0, 1],
                    "eigenvectors": [[1.0], [0.0]],
                },
            }
        )


def test_postprocess_eigenmodes_paper_replay_lane_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        postprocess_eigenmodes_paper_replay_lane(
            {
                "experiment": "eigenmodes",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
                "channels": {
                    "eigenvalues": [1.0, np.nan],
                    "eigenvectors": [[1.0], [0.0]],
                },
            }
        )


def test_run_eigenmodes_paper_replay_lane_uses_injected_sampler() -> None:
    plan = plan_eigenmodes_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        bpress=-91.0,
        mode_count=3,
    )

    captured: dict[str, object] = {}

    def fake_sampler(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "experiment": "eigenmodes",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "eigenfrequencies": [10.0, 20.0, 30.0, 40.0],
                "eigenvalues": [1.0, 4.0, 9.0, 16.0],
                "eigenvectors": [[1.0], [0.0], [0.5], [0.25]],
            },
        }

    result = run_eigenmodes_paper_replay_lane(plan, sampler=fake_sampler)

    assert captured["experiment"] == "eigenmodes"
    assert captured["controls"] == {"bpress": (-91.0,)}
    assert result.axis == "mode_index"
    assert np.array_equal(result.channels["mode_index"], np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(result.channels["eigenfrequencies"], np.array([10.0, 20.0, 30.0]))
    assert result.provenance["lane_plan"]["mode_count"] == 3


def test_eigenmodes_paper_replay_accepts_object_payload_and_writes_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sample_result = SimpleNamespace(
        experiment="eigenmodes",
        geometry=SimpleNamespace(radGV=2.0, height=14.28),
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"pressure_difference": -91.0},
        channels={
            "eigenvalues": [1.0, 4.0, 9.0, 16.0],
            "eigenvectors": [[1.0], [0.5], [0.25], [0.125]],
        },
        manifest={"dataset_id": "gv__eigenmodes__fixture"},
        runtime_manifests=({"control_id": "bpress_-91"},),
        plan_manifests=({"experiment": "eigenmodes"},),
        work_dirs=(tmp_path / "_runs" / "work",),
        runtime_seconds=8.5,
        status="passed",
    )

    result = postprocess_eigenmodes_paper_replay_lane(sample_result)
    manifest = result.to_manifest()
    plot_path = plot_eigenmodes_paper_replay(
        result,
        output_path="_runs/gv/paper_replay/eigenmodes/spectrum.png",
    )

    assert manifest["channels"]["mode_index"][:3] == [0.0, 1.0, 2.0]
    assert result.controls == {"bpress": -91.0}
    assert result.provenance["runtime_seconds"] == 8.5
    assert result.provenance["runtime_manifests"] == [{"control_id": "bpress_-91"}]
    assert plot_path.is_file()


def test_plot_eigenmodes_paper_replay_rejects_non_runs_output() -> None:
    plan = plan_eigenmodes_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
    )
    result = postprocess_eigenmodes_paper_replay_lane(
        {
            "experiment": "eigenmodes",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "eigenfrequencies": [10.0, 20.0],
                "eigenvalues": [1.0, 4.0],
            },
        },
        plan=plan,
    )

    with pytest.raises(ValueError, match="under _runs/"):
        plot_eigenmodes_paper_replay(result, output_path="paper_replay/eigenmodes.png")


def test_plan_eigenmodes_paper_replay_rejects_output_root_traversal() -> None:
    with pytest.raises(ValueError, match="under"):
        plan_eigenmodes_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            output_root="_runs/../outside",
        )


def test_plan_eigenmodes_paper_replay_rejects_invalid_material_and_controls() -> None:
    with pytest.raises(ValueError, match="mode_count"):
        plan_eigenmodes_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            mode_count=0,
        )

    bad_parameters = dict(_BASE_MATERIAL_PARAMETERS)
    bad_parameters["mu_l"] = -1.0
    with pytest.raises(ValueError, match="mu_l"):
        plan_eigenmodes_paper_replay_lane(
            material_parameters=bad_parameters,
            radGV=2.0,
            height=14.28,
        )

    duplicate_parameters = dict(_BASE_MATERIAL_PARAMETERS)
    duplicate_parameters["muL"] = duplicate_parameters["mu_l"]
    with pytest.raises(ValueError, match="duplicated"):
        plan_eigenmodes_paper_replay_lane(
            material_parameters=duplicate_parameters,
            radGV=2.0,
            height=14.28,
        )


def test_eigenmodes_private_coercion_and_metadata_error_paths() -> None:
    class ManifestOnly:
        def as_manifest(self):
            return {
                "channels": {"eigenvalues": [1.0, 4.0]},
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
            }

    assert eigenmodes_module._coerce_mapping(ManifestOnly(), context="fixture")["controls"] == {"bpress": -91.0}

    with pytest.raises(ValueError, match="scalar"):
        eigenmodes_module._coerce_scalar([1.0, 2.0], name="bpress")
    with pytest.raises(ValueError, match="finite"):
        eigenmodes_module._coerce_scalar(float("nan"), name="bpress")
    with pytest.raises(ValueError, match="Unexpected"):
        eigenmodes_module._validate_paper_replay_material_parameters({**_BASE_MATERIAL_PARAMETERS, "bad": 1.0})
    with pytest.raises(ValueError, match=">= 0"):
        eigenmodes_module._validate_paper_replay_material_parameters({**_BASE_MATERIAL_PARAMETERS, "ka": -1.0})
    with pytest.raises(ValueError, match="Missing required"):
        eigenmodes_module._validate_paper_replay_material_parameters(
            {name: value for name, value in _BASE_MATERIAL_PARAMETERS.items() if name != "ka"}
        )

    with pytest.raises(ValueError, match="geometry metadata"):
        postprocess_eigenmodes_paper_replay_lane(
            {
                "experiment": "eigenmodes",
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
                "channels": {"eigenvalues": [1.0, 4.0]},
            }
        )
    with pytest.raises(ValueError, match="material_parameters"):
        postprocess_eigenmodes_paper_replay_lane(
            {
                "experiment": "eigenmodes",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "controls": {"bpress": -91.0},
                "channels": {"eigenvalues": [1.0, 4.0]},
            }
        )
    with pytest.raises(ValueError, match="scalar control"):
        postprocess_eigenmodes_paper_replay_lane(
            {
                "experiment": "eigenmodes",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "channels": {"eigenvalues": [1.0, 4.0]},
            }
        )
