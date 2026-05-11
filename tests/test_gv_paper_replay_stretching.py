from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    def suptitle(self, *args, **kwargs) -> None:
        return None

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

    def set_xlim(self, *_args, **_kwargs) -> None:
        return None

    def set_ylim(self, *_args, **_kwargs) -> None:
        return None

    def grid(self, *_args, **_kwargs) -> None:
        return None

    def legend(self, *_args, **_kwargs) -> None:
        return None

    def text(self, *_args, **_kwargs) -> None:
        return None


class _FakePyplot:
    def subplots(self, *args, **kwargs):
        if args[:2] == (1, 2):
            return _FakeFigure(), (_FakeAxis(), _FakeAxis())
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
                "epsilon_zz": [0.0, 0.01, 0.02],
                "epsilon_phi_plot": [0.0, 0.003, 0.006],
                "std_epsilon_zz": [0.0, 0.0005, 0.0008],
                "std_epsilon_phi": [0.0, 0.0002, 0.0003],
                "force": [5.0, 7.5, 10.0],
                "displacement": [0.05, 0.09, 0.14],
            },
            "manifest": {"status": "synthetic"},
        }

    result = stretching.run_stretching_paper_replay_lane(plan, sampling_callable=_sampler)

    assert captured["experiment"] == "stretching"
    assert captured["controls"] == {
        "tot_force": tuple(np.linspace(500.0, 50_000.0, 9)),
        "bpress": -91.0,
    }
    assert isinstance(captured["runtime_options"], GVRuntimeOptions)
    assert result.summary["axis_channel"] == "epsilon_zz"
    assert result.summary["auxiliary_axis_channel"] == "minus_epsilon_phi"
    assert result.summary["response_channel"] == "sigma_zz"
    np.testing.assert_allclose(
        result.channels["sigma_zz"],
        np.asarray([500.0, 750.0, 1000.0]) / (2.0 * np.pi * 2.0),
    )
    assert result.summary["has_auxiliary_force_displacement"] is True
    assert result.plot_path is not None and result.plot_path.is_file()
    assert result.plot_metadata_path is not None and result.plot_metadata_path.is_file()
    assert result.plot_path.as_posix().startswith("_runs/gv/paper_replay/stretching/plots/")
    metadata = json.loads(result.plot_metadata_path.read_text(encoding="utf-8"))
    assert metadata["manifest"]["experiment"] == "stretching"
    assert metadata["manifest"]["raw_provenance"]["paper_protocol_reference_only"] is True
    assert metadata["manifest"]["raw_provenance"]["canonical_replay_data_source"] == "Mirheo reruns"
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
            "epsilon_zz": np.array([0.0, 0.01]),
            "minus_epsilon_phi": np.array([0.0, 0.004]),
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
        reference_curve={
            "sigma_zz": [0.0, 10.0],
            "epsilon_zz": [0.0, 0.01],
            "epsilon_phi_plot": [0.0, 0.004],
            "displacement": [0.0, 0.1],
            "force": [0.0, 8.0],
        },
    )

    assert result.raw_sample_result["geometry"] == {"radGV": 2.0, "height": 14.28}
    assert result.raw_sample_result["runtime_seconds"] == 4.5
    assert "sweep_500_750" in result.manifest["control_id"]
    np.testing.assert_allclose(result.channels["epsilon_phi_plot"], [0.0, 0.004])
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
                "epsilon_zz": [0.0],
                "epsilon_phi_plot": [0.0],
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

    with pytest.raises(ValueError, match="requires epsilon_zz"):
        stretching.postprocess_stretching_paper_replay_lane(
            {"channels": {"tot_force": [500.0], "epsilon_phi_plot": [0.0]}},
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
                    "epsilon_zz": [0.0, np.nan],
                    "epsilon_phi_plot": [0.0, 0.01],
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
                "epsilon_zz": [0.0],
                "epsilon_phi_plot": [0.0],
                "force": [5.0],
                "displacement": [0.05],
            }
        },
        plan=plan,
    )

    with pytest.raises(ValueError, match="reference_curve"):
        stretching.plot_stretching_paper_replay_lane(
            result,
            reference_curve={"epsilon_zz": [0.0, 0.1]},
        )


def test_stretching_helpers_reject_invalid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    matplotlib = ModuleType("matplotlib")
    pyplot = ModuleType("matplotlib.pyplot")
    matplotlib.use = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
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


def test_stretching_plan_uses_exact_and_default_force_sweeps() -> None:
    default_plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
    )
    exact_plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        paper_exact=True,
    )

    assert default_plan.paper_exact is False
    assert exact_plan.paper_exact is True
    assert len(default_plan.controls["tot_force"]) == 9
    assert len(exact_plan.controls["tot_force"]) == 80
    assert default_plan.controls["tot_force"][0] == 500.0
    assert default_plan.controls["tot_force"][-1] == 50_000.0
    assert exact_plan.controls["tot_force"][0] == 500.0
    assert exact_plan.controls["tot_force"][-1] == 50_000.0


def test_stretching_plan_can_slice_force_points() -> None:
    sliced = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        paper_exact=True,
        point_start=2,
        point_stop=5,
    )

    assert len(sliced.controls["tot_force"]) == 3
    assert sliced.controls["tot_force"] == tuple(np.linspace(500.0, 50_000.0, 80)[2:5])

    with pytest.raises(ValueError, match="stop > start"):
        stretching.plan_stretching_paper_replay_lane(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            paper_exact=True,
            point_start=4,
            point_stop=4,
        )
    with pytest.raises(ValueError, match=">= 0"):
        stretching.plan_stretching_paper_replay_lane(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            paper_exact=True,
            point_start=-1,
        )
    with pytest.raises(ValueError, match="selected no control"):
        stretching.plan_stretching_paper_replay_lane(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            paper_exact=True,
            point_start=200,
            point_stop=201,
        )


def test_stretching_helper_branches_cover_scalar_json_and_shape_validation() -> None:
    assert stretching._jsonable({"value": np.float64(1.25), "flag": np.bool_(True)}) == {
        "value": 1.25,
        "flag": True,
    }
    with pytest.raises(ValueError, match="must be finite"):
        stretching._coerce_float_mapping({"ka": float("inf")}, name="material_parameters")
    with pytest.raises(ValueError, match="contain only finite values"):
        stretching._coerce_control_value([500.0, float("nan")], name="tot_force")
    with pytest.raises(ValueError, match="must be numeric"):
        stretching._coerce_control_value(object(), name="tot_force")
    with pytest.raises(ValueError, match="must be finite"):
        stretching._coerce_control_value(float("nan"), name="tot_force")

    plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        include_plot=False,
    )
    with pytest.raises(ValueError, match="share the same shape"):
        stretching.postprocess_stretching_paper_replay_lane(
            {
                "channels": {
                    "tot_force": [500.0, 750.0],
                    "epsilon_zz": [0.0, 0.01],
                    "epsilon_phi_plot": [0.0],
                }
            },
            plan=plan,
        )


def test_stretching_helpers_cover_control_aliases_and_missing_control_validation() -> None:
    assert stretching._representative_controls(
        {"tot_force": (500.0, 750.0), "bpress": -91.0}
    ) == {"tot_force": 500.0, "bpress": -91.0}
    assert stretching._control_identifier({}) == "default"
    with pytest.raises(ValueError, match="tot_force and bpress"):
        stretching.StretchingPaperReplayPlan(
            campaign_id="paper-stretch",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"tot_force": (500.0,)},
        )


def test_stretching_existing_point_probe_handles_missing_and_failed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert stretching._try_extract_existing_stretching_point(
        work_dir=tmp_path / "missing",
        tot_force=500.0,
        bpress=-91.0,
        geometry=stretching.GVMaterialGeometry(radGV=2.0, height=14.28),
    ) is None

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fail_extract(**_kwargs: object) -> dict[str, np.ndarray]:
        raise RuntimeError("not ready")

    monkeypatch.setattr(stretching, "extract_sampling_channels", fail_extract)
    assert stretching._try_extract_existing_stretching_point(
        work_dir=work_dir,
        tot_force=500.0,
        bpress=-91.0,
        geometry=stretching.GVMaterialGeometry(radGV=2.0, height=14.28),
    ) is None


def test_stretching_normalization_covers_alias_channels_and_optional_shape_checks() -> None:
    derived = stretching._normalize_stretching_channels(
        {
            "tot_force": [500.0, 750.0],
            "mean_length": [10.0, 10.5],
            "std_length": [0.0, 0.1],
            "mean_radius": [2.0, 1.9],
            "std_radius": [0.0, 0.05],
        },
        geometry_radius=2.0,
    )
    np.testing.assert_allclose(derived["epsilon_zz"], [0.0, 0.05])
    np.testing.assert_allclose(derived["epsilon_phi_plot"], [-0.0, 0.05])
    np.testing.assert_allclose(derived["std_epsilon_zz"], [0.0, 0.01])
    np.testing.assert_allclose(derived["std_epsilon_phi"], [0.0, 0.025])

    normalized = stretching._normalize_stretching_channels(
        {
            "sigma_zz": [10.0, 20.0],
            "axial_strain": [0.0, 0.01],
            "hoop_strain": [0.0, -0.004],
            "force": [5.0, 6.0],
        },
        geometry_radius=2.0,
    )
    np.testing.assert_allclose(normalized["epsilon_zz"], [0.0, 0.01])
    np.testing.assert_allclose(normalized["epsilon_phi_plot"], [-0.0, 0.004])
    np.testing.assert_allclose(normalized["minus_epsilon_phi"], [-0.0, 0.004])

    with pytest.raises(ValueError, match="requires a channel mapping"):
        stretching._extract_channel_mapping({"channels": [1.0, 2.0]})
    with pytest.raises(ValueError, match="requires sigma_zz or tot_force"):
        stretching._normalize_stretching_channels(
            {"epsilon_zz": [0.0], "epsilon_phi_plot": [0.0]},
            geometry_radius=2.0,
        )
    with pytest.raises(ValueError, match="requires epsilon_phi_plot or minus_epsilon_phi"):
        stretching._normalize_stretching_channels(
            {"sigma_zz": [10.0], "strain_z": [0.0]},
            geometry_radius=2.0,
        )
    assert stretching._first_available({"strain_phi": np.array([1.0])}, ("epsilon_phi", "strain_phi")).tolist() == [1.0]


def test_stretching_plot_handles_single_panel_and_force_only_reference(
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
        include_plot=False,
    )
    result = stretching.postprocess_stretching_paper_replay_lane(
        {
            "channels": {
                "sigma_zz": [10.0, 12.0],
                "longitudinal_strain": [0.0, 0.01],
                "minus_epsilon_phi": [0.0, 0.003],
            }
        },
        plan=plan,
    )
    plotted = stretching.plot_stretching_paper_replay_lane(
        result,
        reference_curve={"displacement": [0.0, 0.1], "force": [0.0, 8.0]},
    )
    assert plotted.plot_path is not None and plotted.plot_path.is_file()
    assert plotted.plot_metadata_path is not None and plotted.plot_metadata_path.is_file()


def test_stretching_plot_handles_reference_stress_strain_and_auxiliary_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stretching, "_import_matplotlib_pyplot", lambda: _FakePyplot())
    work_dir = tmp_path / "_runs" / "work"
    parameter_dir = work_dir / "parameter"
    parameter_dir.mkdir(parents=True)
    (parameter_dir / "parameters-default00001.yaml").write_text(
        "\n".join(
            [
                "ul: 1.0",
                "kbol: 1.0",
                "t0: 1.0",
                "shell_th: 1.0",
                "th_fac: 1.0",
                "fscale: 1.0",
                "Yl: 2.0",
                "nu: 0.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (parameter_dir / "parameters00001.yaml").write_text("ue: 1.0\n", encoding="utf-8")
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
                "tot_force": [500.0, 750.0],
                "strain_z": [0.0, 0.01],
                "epsilon_phi": [0.0, -0.004],
                "force": [5.0, 7.5],
                "displacement": [0.05, 0.09],
            },
            "work_dirs": (work_dir,),
        },
        plan=plan,
    )
    plotted = stretching.plot_stretching_paper_replay_lane(
        result,
        reference_curve={
            "sigma_zz": [0.0, 10.0],
            "epsilon_zz": [0.0, 0.01],
            "minus_epsilon_phi": [0.0, 0.004],
            "displacement": [0.0, 0.1],
            "force": [0.0, 8.0],
        },
    )
    assert plotted.plot_path is not None and plotted.plot_path.is_file()


def test_stretching_paper_exact_execution_sets_runtime_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = stretching.plan_stretching_paper_replay_lane(
        campaign_id="paper-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"tot_force": (500.0, 750.0), "bpress": -91.0},
        paper_exact=True,
        include_plot=False,
        output_root="_runs/gv/figure_replay/test-stretching",
    )
    captured: dict[str, object] = {"planned": []}

    monkeypatch.setattr(
        stretching,
        "build_geometry",
        lambda **_kwargs: SimpleNamespace(id="gv_geom", parameters={"radGV": 2.0, "height": 14.28}),
    )

    class Runtime:
        def __init__(self, work_dir: Path) -> None:
            self.work_dir = work_dir

        def to_manifest(self) -> dict[str, object]:
            return {"work_dir": str(self.work_dir)}

    class SamplingPlan:
        def __init__(self, value: float) -> None:
            self.value = value

        def to_manifest(self) -> dict[str, object]:
            return {"control_axis": "tot_force", "value": self.value}

    def fake_plan_runtime(*_args: object, **kwargs: object) -> Runtime:
        controls = dict(kwargs["controls"])
        work_dir = tmp_path / f"tot_force_{int(float(controls['tot_force']))}"
        work_dir.mkdir(parents=True, exist_ok=True)
        captured["planned"].append(float(controls["tot_force"]))
        return Runtime(work_dir)

    monkeypatch.setattr(stretching, "plan_runtime", fake_plan_runtime)
    monkeypatch.setattr(
        stretching,
        "build_sampling_plan",
        lambda runtime, *_args, **_kwargs: SamplingPlan(float(Path(runtime.work_dir).name.split("_")[-1])),
    )

    def fake_execute(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured["env"] = kwargs["env"]
        return SimpleNamespace(executed_commands=("bash commands.txt",), return_codes=(0,))

    monkeypatch.setattr(stretching, "execute_sampling_plan", fake_execute)
    monkeypatch.setattr(
        stretching,
        "_try_extract_existing_stretching_point",
        lambda **kwargs: (
            {
                "tot_force": np.array([500.0]),
                "force": np.array([500.0]),
                "bpress": np.array([-91.0]),
                "displacement": np.array([0.0]),
                "mean_length": np.array([1.0]),
                "std_length": np.array([0.1]),
                "mean_radius": np.array([2.0]),
                "std_radius": np.array([0.1]),
                "reference_length": np.array([1.0]),
                "reference_radius": np.array([2.0]),
            }
            if kwargs["tot_force"] == 500.0
            else None
        ),
    )
    monkeypatch.setattr(
        stretching,
        "extract_sampling_channels",
        lambda **kwargs: {
            "tot_force": np.array([kwargs["controls"]["tot_force"]]),
            "force": np.array([kwargs["controls"]["tot_force"]]),
            "bpress": np.array([kwargs["controls"]["bpress"]]),
            "displacement": np.array([0.25]),
            "mean_length": np.array([1.25]),
            "std_length": np.array([0.05]),
            "mean_radius": np.array([1.9]),
            "std_radius": np.array([0.05]),
            "reference_length": np.array([1.0]),
            "reference_radius": np.array([2.0]),
        },
    )

    result = stretching.run_stretching_paper_replay_lane(plan)

    assert captured["env"]["MESOUQ_GV_PAPER_EXACT"] == "1"
    assert captured["planned"] == [500.0, 750.0]
    assert result.raw_sample_result["status"] == "completed"
    assert result.plan.controls["bpress"] == pytest.approx(-91.0)
    np.testing.assert_allclose(result.channels["tot_force"], [500.0, 750.0])
    assert [item["status"] for item in result.raw_sample_result["execution_manifests"]] == ["reused_existing", "completed"]
    assert len(result.raw_sample_result["plan_manifests"]) == 1
