from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.paper_replay_lanes import torsion
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


def test_run_torsion_paper_replay_lane_generates_plot_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torsion, "_import_matplotlib_pyplot", lambda: _FakePyplot())

    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        raw_provenance={"issue": "MES-114"},
    )

    captured: dict[str, object] = {}

    def _sampler(**kwargs):
        captured.update(kwargs)
        return {
            "controls": {"theta": 0.03},
            "channels": {
                "theta": [0.0, 0.03, 0.06],
                "constrained_vertex_forces": [0.0, 2.0, 4.0],
            },
            "manifest": {"status": "synthetic"},
        }

    result = torsion.run_torsion_paper_replay_lane(plan, sampling_callable=_sampler)

    assert captured["experiment"] == "torsion"
    assert captured["controls"] == {"theta": (0.01, 0.03, 0.05, 0.075, 0.1)}
    assert isinstance(captured["runtime_options"], GVRuntimeOptions)
    assert result.summary["axis_channel"] == "gamma"
    assert result.summary["response_channel"] == "sigma_phi_r"
    assert result.plot_path is not None and result.plot_path.is_file()
    assert result.plot_metadata_path is not None and result.plot_metadata_path.is_file()
    assert result.plot_path.as_posix().startswith("_runs/gv/paper_replay/torsion/plots/")
    metadata = json.loads(result.plot_metadata_path.read_text(encoding="utf-8"))
    assert metadata["manifest"]["experiment"] == "torsion"
    assert metadata["summary"]["sample_count"] == 3


def test_torsion_paper_replay_accepts_object_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torsion, "_import_matplotlib_pyplot", lambda: _FakePyplot())
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"theta": np.array([0.01, 0.03])},
        raw_provenance={"issue": "MES-114"},
    )
    sample_result = SimpleNamespace(
        experiment="torsion",
        geometry=SimpleNamespace(radGV=2.0, height=14.28),
        controls={"theta": 0.03},
        channels={
            "gamma": np.array([0.0, 0.1]),
            "sigma_phi_r": np.array([0.0, 2.0]),
        },
        manifest={"dataset_id": "gv__torsion__fixture"},
        runtime_manifests=({"control_id": "theta_0_03"},),
        plan_manifests=({"experiment": "torsion"},),
        execution_manifests=({"status": "passed"},),
        work_dirs=(tmp_path / "_runs" / "work",),
        runtime_seconds=5.5,
        status="passed",
    )

    result = torsion.postprocess_torsion_paper_replay_lane(sample_result, plan=plan)
    plotted = torsion.plot_torsion_paper_replay_lane(result)

    assert result.raw_sample_result["geometry"] == {"radGV": 2.0, "height": 14.28}
    assert result.raw_sample_result["runtime_seconds"] == 5.5
    assert result.manifest["control_id"] == "theta_sweep_0.01_0.03"
    assert plotted.plot_path is not None and plotted.plot_path.is_file()


def test_run_torsion_paper_replay_lane_can_skip_plot() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )

    result = torsion.run_torsion_paper_replay_lane(
        plan,
        sampling_callable=lambda **_kwargs: {
            "channels": {
                "gamma": [0.0, 0.1],
                "sigma_phi_r": [0.0, 2.0],
            }
        },
    )

    assert result.plot_path is None
    assert result.summary["sample_count"] == 2


def test_postprocess_torsion_paper_replay_lane_rejects_missing_channels() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )

    with pytest.raises(ValueError, match="requires a constrained-vertex force source"):
        torsion.postprocess_torsion_paper_replay_lane(
            {"channels": {"theta": [0.0, 0.03]}},
            plan=plan,
        )


def test_postprocess_torsion_paper_replay_lane_rejects_non_finite_channels() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )

    with pytest.raises(ValueError, match="non-finite"):
        torsion.postprocess_torsion_paper_replay_lane(
            {
                "channels": {
                    "theta": [0.0, np.nan],
                    "constrained_vertex_forces": [1.0, 2.0],
                }
            },
            plan=plan,
        )


def test_torsion_helpers_reject_invalid_inputs() -> None:
    assert torsion._import_matplotlib_pyplot().__name__.endswith("pyplot")

    with pytest.raises(ValueError, match="raw_provenance"):
        torsion.plan_torsion_paper_replay_lane(
            campaign_id="paper-torsion",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            raw_provenance=object(),
        )

    with pytest.raises(ValueError, match="must be numeric"):
        torsion.plan_torsion_paper_replay_lane(
            campaign_id="paper-torsion",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters={**_BASE_MATERIAL_PARAMETERS, "ka": object()},
        )

    with pytest.raises(ValueError, match="non-empty sweep"):
        torsion.plan_torsion_paper_replay_lane(
            campaign_id="paper-torsion",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"theta": ()},
        )

    with pytest.raises(ValueError, match="sample_result"):
        torsion.postprocess_torsion_paper_replay_lane(
            object(),
            plan=torsion.plan_torsion_paper_replay_lane(
                campaign_id="paper-torsion",
                geometry_radius=2.0,
                geometry_height=14.28,
                material_parameters=_BASE_MATERIAL_PARAMETERS,
            ),
        )


def test_torsion_plan_rejects_non_runs_output_root() -> None:
    with pytest.raises(ValueError, match="must live under _runs"):
        torsion.plan_torsion_paper_replay_lane(
            campaign_id="paper-torsion",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            output_root="artifacts/torsion",
        )

    with pytest.raises(ValueError, match="must live under _runs"):
        torsion.plan_torsion_paper_replay_lane(
            campaign_id="paper-torsion",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            output_root="_runs/../outside",
        )
