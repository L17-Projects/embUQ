from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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

    def errorbar(self, *args, **kwargs) -> None:
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
    assert metadata["plot_provenance"]["x_label"] == "gamma = 2 * theta * radGV / dz"


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
            "sigma_std": np.array([0.01, 0.02]),
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


def test_torsion_plan_supports_exact_paper_theta() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        paper_exact=True,
        include_plot=False,
    )

    assert plan.paper_exact is True
    assert plan.controls["theta"] == tuple(np.round(np.linspace(0.01, 0.10, 10), 2))
    assert torsion._short_plot_slug("x" * 150).startswith("x" * 96)


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


def test_torsion_paper_exact_execution_sets_runtime_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"theta": (0.01, 0.02)},
        paper_exact=True,
        include_plot=False,
        output_root="_runs/gv/figure_replay/test-torsion",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        torsion,
        "build_geometry",
        lambda **_kwargs: SimpleNamespace(id="gv_geom", parameters={"radGV": 2.0, "height": 14.28}),
    )

    class Runtime:
        work_dir = tmp_path

        def to_manifest(self) -> dict[str, object]:
            return {"work_dir": str(self.work_dir)}

    class SamplingPlan:
        def to_manifest(self) -> dict[str, object]:
            return {"control_axis": "theta"}

    monkeypatch.setattr(torsion, "plan_runtime", lambda *args, **kwargs: Runtime())
    monkeypatch.setattr(torsion, "build_sampling_plan", lambda *args, **kwargs: SamplingPlan())

    def fake_execute(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured["env"] = kwargs["env"]
        return SimpleNamespace(executed_commands=("bash commands.txt",), return_codes=(0,))

    monkeypatch.setattr(torsion, "execute_sampling_plan", fake_execute)
    monkeypatch.setattr(
        torsion,
        "extract_sampling_channels",
        lambda **_kwargs: {
            "gamma": np.array([0.01, 0.02]),
            "sigma_phi_r": np.array([0.3, 0.6]),
        },
    )

    result = torsion.run_torsion_paper_replay_lane(plan)

    assert "MESOUQ_GV_MATERIAL_OVERRIDES_JSON" in captured["env"]
    assert result.raw_sample_result["status"] == "completed"
    assert result.manifest["controls"]["theta"] == [0.01, 0.02]


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


def test_postprocess_torsion_paper_replay_reconstructs_gamma_and_sigma_from_paper_raw_payload() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"theta": np.array([0.1])},
        include_plot=False,
        paper_exact=True,
    )

    mesh_vertices = np.array(
        [
            [0.0, 0.0, -6.0],
            [1.0, 0.0, -5.5],
            [0.0, 0.0, 5.5],
            [1.0, 0.0, 6.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    bottom_forces = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]],
            [[0.0, 3.0, 0.0], [0.0, 6.0, 0.0]],
            [[0.0, 4.0, 0.0], [0.0, 8.0, 0.0]],
        ],
        dtype=float,
    )
    top_forces = np.array(
        [
            [[0.0, 1.5, 0.0], [0.0, 3.0, 0.0]],
            [[0.0, 3.0, 0.0], [0.0, 6.0, 0.0]],
            [[0.0, 4.5, 0.0], [0.0, 9.0, 0.0]],
            [[0.0, 6.0, 0.0], [0.0, 12.0, 0.0]],
        ],
        dtype=float,
    )

    result = torsion.postprocess_torsion_paper_replay_lane(
        {
            "controls": {"theta": 0.1},
            "channels": {
                "mesh_vertices": mesh_vertices,
                "anchor_min_forces": bottom_forces,
                "anchor_max_forces": top_forces,
            },
        },
        plan=plan,
    )

    dz = 5.5 - (-5.5)
    expected_gamma = 2.0 * 0.1 * 2.0 / dz
    rotation = np.cos(0.1)
    tau_bottom = np.cross(
        np.array([[0.0, 0.0, -6.0], [rotation, -np.sin(0.1), -5.5]], dtype=float),
        bottom_forces[-1],
    ).sum(axis=0)[2]
    tau_top = np.cross(
        np.array([[0.0, 0.0, 5.5], [rotation, np.sin(0.1), 6.0]], dtype=float),
        top_forces[-1],
    ).sum(axis=0)[2]
    area = 2.0 * np.pi * 2.0**2

    assert result.channels["gamma"] == pytest.approx([expected_gamma])
    assert result.channels["sigma_phi_r"] == pytest.approx(
        [0.5 * (abs(tau_bottom / area) + abs(tau_top / area))]
    )
    assert result.channels["sigma_std"].shape == (1,)
    assert result.manifest["raw_provenance"]["paper_exact"] is True


def test_postprocess_torsion_paper_replay_passes_through_canonical_gamma_sigma_and_sigma_std() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )

    result = torsion.postprocess_torsion_paper_replay_lane(
        {
            "controls": {"theta": 0.03},
            "channels": {
                "gamma": [0.1, 0.2],
                "sigma_phi_r": [1.5, 1.75],
                "sigma_std": [0.05, 0.06],
            },
        },
        plan=plan,
    )

    assert result.channels["gamma"] == pytest.approx([0.1, 0.2])
    assert result.channels["sigma_phi_r"] == pytest.approx([1.5, 1.75])
    assert result.channels["sigma_std"] == pytest.approx([0.05, 0.06])


def test_postprocess_torsion_paper_replay_hard_fails_on_incomplete_raw_anchor_payload() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"theta": np.array([0.1])},
        include_plot=False,
    )

    with pytest.raises(ValueError, match="requires both anchor_min/anchor_max force payloads"):
        torsion.postprocess_torsion_paper_replay_lane(
            {
                "controls": {"theta": 0.1},
                "channels": {
                    "mesh_vertices": np.array(
                        [
                            [0.0, 0.0, -6.0],
                            [1.0, 0.0, -5.5],
                            [0.0, 0.0, 5.5],
                            [1.0, 0.0, 6.0],
                        ],
                        dtype=float,
                    ),
                    "anchor_min_forces": np.ones((4, 2, 3), dtype=float),
                },
            },
            plan=plan,
        )


def test_postprocess_torsion_paper_replay_hard_fails_when_anchor_mesh_is_missing() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"theta": np.array([0.1])},
        include_plot=False,
    )

    with pytest.raises(ValueError, match="requires mesh_vertices"):
        torsion.postprocess_torsion_paper_replay_lane(
            {
                "controls": {"theta": 0.1},
                "channels": {
                    "anchor_min_forces": np.ones((4, 1, 3), dtype=float),
                    "anchor_max_forces": np.ones((4, 1, 3), dtype=float),
                },
            },
            plan=plan,
        )


def test_postprocess_torsion_paper_replay_hard_fails_when_anchor_regions_cannot_be_reconstructed() -> None:
    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"theta": np.array([0.1])},
        include_plot=False,
    )

    with pytest.raises(ValueError, match="could not reconstruct anchor regions"):
        torsion.postprocess_torsion_paper_replay_lane(
            {
                "controls": {"theta": 0.1},
                "channels": {
                    "mesh_vertices": np.array(
                        [
                            [0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0],
                        ],
                        dtype=float,
                    ),
                    "anchor_min_forces": np.ones((4, 1, 3), dtype=float),
                    "anchor_max_forces": np.ones((4, 1, 3), dtype=float),
                },
            },
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


def test_torsion_helpers_reject_invalid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    matplotlib = ModuleType("matplotlib")
    pyplot = ModuleType("matplotlib.pyplot")
    matplotlib.use = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
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


def test_torsion_helper_branches_cover_scalar_json_and_shape_validation() -> None:
    assert torsion._jsonable({"value": np.float64(1.25), "flag": np.bool_(True)}) == {
        "value": 1.25,
        "flag": True,
    }
    with pytest.raises(ValueError, match="must be finite"):
        torsion._coerce_float_mapping({"ka": float("inf")}, name="material_parameters")
    with pytest.raises(ValueError, match="contain only finite values"):
        torsion._coerce_control_value([0.01, float("nan")], name="theta")
    with pytest.raises(ValueError, match="must be numeric"):
        torsion._coerce_control_value(object(), name="theta")
    with pytest.raises(ValueError, match="must be finite"):
        torsion._coerce_control_value(float("nan"), name="theta")
    assert torsion._coerce_control_value(0.03, name="theta") == pytest.approx(0.03)
    assert torsion._representative_controls({"theta": 0.03}) == {"theta": 0.03}
    assert torsion._control_identifier({"theta": 0.03}) == "theta_0.03"

    plan = torsion.plan_torsion_paper_replay_lane(
        campaign_id="paper-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )
    with pytest.raises(ValueError, match="matching shapes"):
        torsion.postprocess_torsion_paper_replay_lane(
            {
                "channels": {
                    "gamma": [0.0, 0.1],
                    "sigma_phi_r": [0.0],
                }
            },
            plan=plan,
        )
    with pytest.raises(ValueError, match="requires control theta"):
        torsion.TorsionPaperReplayPlan(
            campaign_id="paper-torsion",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={},
        )
    with pytest.raises(ValueError, match="sigma_std and sigma_phi_r"):
        torsion.postprocess_torsion_paper_replay_lane(
            {
                "controls": {"theta": 0.03},
                "channels": {
                    "gamma": [0.0, 0.1],
                    "sigma_phi_r": [0.0, 1.0],
                    "sigma_std": [0.1],
                },
            },
            plan=plan,
        )
