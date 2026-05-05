from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.paper_replay_lanes import stretching
from meso_uq.structures.gv.sampling import GVRuntimeOptions


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


class _FakeFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, path: Path, *args, **kwargs) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("plot", encoding="utf-8")


class _FakeAxis:
    def __init__(self) -> None:
        self.transAxes = object()

    def plot(self, *args, **kwargs) -> None:
        return None

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return None

    def set_ylabel(self, *_args, **_kwargs) -> None:
        return None

    def set_title(self, *_args, **_kwargs) -> None:
        return None

    def grid(self, *_args, **_kwargs) -> None:
        return None

    def legend(self, *_args, **_kwargs) -> None:
        return None

    def text(self, *_args, **_kwargs) -> None:
        return None


class _FakePyplot:
    def subplots(self, *args, **kwargs):
        return _FakeFigure(), _FakeAxis()

    def close(self, *_args, **_kwargs) -> None:
        return None


def test_run_stretching_paper_replay_lane_generates_plot_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stretching, "_import_matplotlib_pyplot", lambda: _FakePyplot())

    plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        raw_provenance={"issue": "MES-112"},
    )

    captured: dict[str, object] = {}

    def _sampler(**kwargs):
        captured.update(kwargs)
        return {
            "controls": {"tot_force": 500.0, "bpress": -91.0},
            "channels": {
                "tot_force": [500.0, 750.0, 1000.0],
                "force": [5.0, 7.5, 10.0],
                "displacement": [0.05, 0.09, 0.14],
                "radius_change": [0.0, 0.01, 0.015],
            },
            "manifest": {"status": "synthetic"},
        }

    result = stretching.run_stretching_paper_replay_lane(plan, sampling_callable=_sampler)

    assert captured["experiment"] == "stretching"
    assert captured["controls"] == {
        "tot_force": (500.0, 5000.0, 15000.0),
        "bpress": -91.0,
    }
    assert isinstance(captured["runtime_options"], GVRuntimeOptions)
    assert result.summary["axis_channel"] == "displacement"
    assert result.summary["response_channel"] == "force"
    assert result.plot_path is not None and result.plot_path.is_file()
    assert result.plot_metadata_path is not None and result.plot_metadata_path.is_file()
    assert result.plot_path.as_posix().startswith("_runs/gv/paper_replay/stretching/plots/")
    metadata = json.loads(result.plot_metadata_path.read_text(encoding="utf-8"))
    assert metadata["manifest"]["experiment"] == "stretching"
    assert metadata["summary"]["sample_count"] == 3


def test_stretching_paper_replay_accepts_object_payload_and_reference_curve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stretching, "_import_matplotlib_pyplot", lambda: _FakePyplot())
    plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"tot_force": np.array([500.0, 750.0]), "bpress": -91.0},
        raw_provenance={"issue": "MES-112"},
    )
    sample_result = SimpleNamespace(
        experiment="stretching",
        geometry=SimpleNamespace(radGV=2.0, height=14.28),
        controls={"tot_force": 500.0, "bpress": -91.0},
        channels={
            "tot_force": np.array([500.0, 750.0]),
            "force": np.array([5.0, 7.5]),
            "displacement": np.array([0.05, 0.09]),
        },
        manifest={"dataset_id": "gv__stretching__fixture"},
        runtime_manifests=({"control_id": "force_500"},),
        plan_manifests=({"experiment": "stretching"},),
        execution_manifests=({"status": "passed"},),
        work_dirs=(tmp_path / "_runs" / "work",),
        runtime_seconds=4.5,
        status="passed",
    )

    result = stretching.postprocess_stretching_paper_replay_lane(sample_result, plan=plan)
    plotted = stretching.plot_stretching_paper_replay_lane(
        result,
        reference_curve={"displacement": [0.0, 0.1], "force": [0.0, 8.0]},
    )

    assert result.raw_sample_result["geometry"] == {"radGV": 2.0, "height": 14.28}
    assert result.raw_sample_result["runtime_seconds"] == 4.5
    assert "sweep_500_750" in result.manifest["control_id"]
    assert plotted.plot_path is not None and plotted.plot_path.is_file()


def test_run_stretching_paper_replay_lane_can_skip_plot() -> None:
    plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )

    result = stretching.run_stretching_paper_replay_lane(
        plan,
        sampling_callable=lambda **_kwargs: {
            "channels": {
                "tot_force": [500.0],
                "force": [5.0],
                "displacement": [0.05],
            }
        },
    )

    assert result.plot_path is None
    assert result.summary["sample_count"] == 1


def test_postprocess_stretching_paper_replay_lane_rejects_missing_channels() -> None:
    plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )

    with pytest.raises(ValueError, match="requires at least channels"):
        stretching.postprocess_stretching_paper_replay_lane(
            {"channels": {"tot_force": [500.0], "force": [5.0]}},
            plan=plan,
        )


def test_postprocess_stretching_paper_replay_lane_rejects_non_finite_channels() -> None:
    plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )

    with pytest.raises(ValueError, match="non-finite"):
        stretching.postprocess_stretching_paper_replay_lane(
            {
                "channels": {
                    "tot_force": [500.0, 750.0],
                    "force": [5.0, np.nan],
                    "displacement": [0.05, 0.09],
                }
            },
            plan=plan,
        )


def test_stretching_plot_rejects_bad_reference_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stretching, "_import_matplotlib_pyplot", lambda: _FakePyplot())
    plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )
    result = stretching.postprocess_stretching_paper_replay_lane(
        {
            "channels": {
                "tot_force": [500.0],
                "force": [5.0],
                "displacement": [0.05],
            }
        },
        plan=plan,
    )

    with pytest.raises(ValueError, match="reference_curve"):
        stretching.plot_stretching_paper_replay_lane(
            result,
            reference_curve={"displacement": [0.0, 0.1]},
        )


def test_stretching_helpers_reject_invalid_inputs() -> None:
    assert stretching._import_matplotlib_pyplot().__name__.endswith("pyplot")

    with pytest.raises(ValueError, match="raw_provenance"):
        stretching.plan_stretching_paper_replay_lane(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            raw_provenance=object(),
        )

    with pytest.raises(ValueError, match="must be numeric"):
        stretching.plan_stretching_paper_replay_lane(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters={**_BASE_MATERIAL_PARAMETERS, "ka": object()},
        )

    with pytest.raises(ValueError, match="non-empty sweep"):
        stretching.plan_stretching_paper_replay_lane(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"tot_force": (), "bpress": -91.0},
        )

    with pytest.raises(ValueError, match="sample_result"):
        stretching.postprocess_stretching_paper_replay_lane(
            object(),
            plan=stretching.plan_stretching_paper_replay_lane(
                campaign_id="paper-stretch",
                geometry_radius=2.0,
                geometry_height=14.28,
                material_parameters=_BASE_MATERIAL_PARAMETERS,
            ),
        )


def test_stretching_plan_rejects_non_runs_output_root() -> None:
    with pytest.raises(ValueError, match="must live under _runs"):
        stretching.plan_stretching_paper_replay_lane(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            output_root="artifacts/stretching",
        )

    with pytest.raises(ValueError, match="must live under _runs"):
        stretching.plan_stretching_paper_replay_lane(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            output_root="_runs/../outside",
        )
