from __future__ import annotations

import sys
from types import ModuleType
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
from meso_uq.structures.gv.runtime.base import DryRunCommand, RuntimeDryRun


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

    def text(self, *args, **kwargs) -> None:
        return None

    def tight_layout(self, *args, **kwargs) -> None:
        return None

    def savefig(self, path: Path, *args, **kwargs) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("plot", encoding="utf-8")


class _FakeAxis:
    def __init__(self) -> None:
        self.axvlines: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def plot(self, *args, **kwargs) -> None:
        return None

    def errorbar(self, *args, **kwargs) -> None:
        return None

    def set_xlabel(self, *args, **kwargs) -> None:
        return None

    def set_ylabel(self, *args, **kwargs) -> None:
        return None

    def set_title(self, *args, **kwargs) -> None:
        return None

    def set_xlim(self, *args, **kwargs) -> None:
        return None

    def set_ylim(self, *args, **kwargs) -> None:
        return None

    def grid(self, *args, **kwargs) -> None:
        return None

    def legend(self, *args, **kwargs) -> None:
        return None

    def axvline(self, *args, **kwargs) -> None:
        self.axvlines.append((args, kwargs))


class _FakePyplot:
    def subplots(self, *args, **kwargs):
        if args[:2] == (1, 2):
            return _FakeFigure(), (_FakeAxis(), _FakeAxis())
        return _FakeFigure(), _FakeAxis()

    def close(self, *args, **kwargs) -> None:
        return None


def _install_fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    matplotlib = ModuleType("matplotlib")
    pyplot = ModuleType("matplotlib.pyplot")
    matplotlib.use = lambda *_args, **_kwargs: None
    pyplot.subplots = _FakePyplot().subplots
    pyplot.close = _FakePyplot().close
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)


def test_plan_buckling_paper_replay_lane_uses_pressure_sweep_and_records_assumptions() -> None:
    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
    )

    assert plan.figure_id == BUCKLING_FIGURE_ID
    assert plan.experiment == "buckling"
    assert len(plan.controls["buck"]) == 20
    assert plan.controls["buck"][0] == pytest.approx(0.0)
    assert plan.controls["buck"][-1] == pytest.approx(0.75)
    assert plan.controls["pressure_difference"][0] == pytest.approx(0.0)
    assert plan.controls["pressure_difference"][-1] == pytest.approx(68.175)
    assert plan.controls["bpress"] == pytest.approx(-91.0)
    assert plan.runtime_options.controls == {"bpress": -91.0}
    assert plan.runtime_options.output_root == "_runs/gv/paper_replay/buckling"
    assert "Mirheo reruns" in plan.mapping_assumptions[-1]


def test_plan_buckling_paper_replay_lane_uses_exact_paper_sweep_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "MESOUQ_GV_BUCKLING_MEMBRANE_BPRESS_MODE",
        "MESOUQ_GV_BUCKLING_FLUID_MODE",
        "MESOUQ_GV_BUCKLING_FLUID_STABILIZATION",
        "MESOUQ_GV_BUCKLING_PIN_OBJECT",
        "MESOUQ_GV_BUCKLING_ODPD_AMP_SCALE",
    ):
        monkeypatch.delenv(name, raising=False)

    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        paper_exact=True,
    )

    assert len(plan.controls["buck"]) == 25
    assert len(plan.controls["pressure_difference"]) == 25
    assert plan.controls["pressure_difference"][-1] == pytest.approx(68.175)
    assert plan.paper_protocol is not None
    assert plan.paper_protocol["mesh_winding"] == "archived_paper_winding"
    assert plan.paper_protocol["fluid_coupling"] == "dropped_imported"
    assert plan.paper_protocol["environment"] == {
        "MESOUQ_GV_PAPER_EXACT": "1",
        "MESOUQ_GV_BUCKLING_MEMBRANE_BPRESS_MODE": "runtime",
        "MESOUQ_GV_BUCKLING_FLUID_MODE": "dropped",
        "MESOUQ_GV_BUCKLING_FLUID_STABILIZATION": "0.0",
        "MESOUQ_GV_BUCKLING_PIN_OBJECT": "1",
        "MESOUQ_GV_BUCKLING_ODPD_AMP_SCALE": "1.0",
    }
    assert plan.paper_protocol["pressure_mapping"]["delta_p_formula"] == "Delta_p = 90.9 * buck"
    assert plan.paper_protocol["pressure_mapping"]["delta_p_scale"] == pytest.approx(90.9)
    manifest = plan.to_manifest()
    assert manifest["paper_protocol"]["environment"]["MESOUQ_GV_BUCKLING_FLUID_STABILIZATION"] == "0.0"


def test_plan_buckling_paper_replay_lane_records_operator_protocol_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MESOUQ_GV_BUCKLING_FLUID_STABILIZATION", "0.25")
    monkeypatch.setenv("MESOUQ_GV_BUCKLING_PIN_OBJECT", "0")

    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        paper_exact=True,
    )

    assert plan.paper_protocol is not None
    assert plan.paper_protocol["environment"]["MESOUQ_GV_BUCKLING_FLUID_STABILIZATION"] == "0.25"
    assert plan.paper_protocol["environment"]["MESOUQ_GV_BUCKLING_PIN_OBJECT"] == "0"
    assert plan.paper_protocol["operator_overrides"] == {
        "MESOUQ_GV_BUCKLING_FLUID_STABILIZATION": "0.25",
        "MESOUQ_GV_BUCKLING_PIN_OBJECT": "0",
    }


def test_plan_buckling_paper_replay_lane_preserves_legacy_fluid_stabilization_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MESOUQ_GV_BUCKLING_FLUID_MODE", "legacy")
    monkeypatch.delenv("MESOUQ_GV_BUCKLING_FLUID_STABILIZATION", raising=False)

    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        paper_exact=True,
    )

    assert plan.paper_protocol is not None
    assert plan.paper_protocol["environment"]["MESOUQ_GV_BUCKLING_FLUID_MODE"] == "legacy"
    assert plan.paper_protocol["environment"]["MESOUQ_GV_BUCKLING_FLUID_STABILIZATION"] == "1.0"
    assert plan.paper_protocol["operator_overrides"] == {
        "MESOUQ_GV_BUCKLING_FLUID_MODE": "legacy",
    }


def test_plan_buckling_paper_replay_lane_accepts_extended_sweep_request() -> None:
    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        buck=1.1,
        buck_point_count=37,
        paper_exact=True,
    )

    assert len(plan.controls["buck"]) == 37
    assert plan.controls["buck"][-1] == pytest.approx(1.1)
    assert plan.controls["pressure_difference"][-1] == pytest.approx(99.99)

    with pytest.raises(ValueError, match="at least 2"):
        plan_buckling_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            buck=1.1,
            buck_point_count=1,
        )

    with pytest.raises(ValueError, match="only valid"):
        plan_buckling_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            buck=(0.0, 0.75),
            buck_point_count=5,
        )


def test_plan_buckling_rejects_inconsistent_explicit_pressure_differences() -> None:
    with pytest.raises(ValueError, match="at least two pressure-difference"):
        plan_buckling_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            buck=(0.0, 0.75),
            pressure_differences=[0.0],
        )

    with pytest.raises(ValueError, match="match the buck sweep length"):
        plan_buckling_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            buck=(0.0, 0.375, 0.75),
            pressure_differences=[0.0, 68.175],
        )

def test_postprocess_buckling_paper_replay_lane_requires_relative_volume_channel() -> None:
    with pytest.raises(ValueError, match="requires channels: buck, pressure_difference, relative_volume"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
                "channels": {
                    "buck": [0.0, 0.75],
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
                "controls": {"bpress": -91.0},
                "channels": {
                    "buck": [0.0, 0.75],
                    "pressure_difference": [0.0, 68.175],
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
                "controls": {"bpress": -91.0},
                "channels": {
                    "buck": [0.0, 0.75],
                    "pressure_difference": [0.0],
                    "relative_volume": [1.0],
                    "buckling_response": [0.1, 0.2],
                },
            }
        )


def test_postprocess_buckling_paper_replay_normalizes_mean_volume_to_first_control() -> None:
    result = postprocess_buckling_paper_replay_lane(
        {
            "experiment": "buckling",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "buck": [0.0, 0.375, 0.75],
                "mean_volume": [100.0, 98.0, 95.0],
                "std_volume": [1.0, 2.0, 3.0],
                "deformation_amplitude": [0.0, 0.1, 0.3],
            },
        }
    )

    assert np.allclose(result.channels["pressure_difference"], np.array([0.0, 34.0875, 68.175]))
    assert np.allclose(result.channels["relative_volume"], np.array([1.0, 0.98, 0.95]))
    assert np.allclose(result.channels["relative_volume_std"], np.array([0.01, 0.02, 0.03]))


def test_postprocess_buckling_paper_replay_uses_measured_zero_pressure_volume_reference() -> None:
    result = postprocess_buckling_paper_replay_lane(
        {
            "experiment": "buckling",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "buck": [0.5, 0.0, 0.75],
                "mean_volume": [99.0, 110.0, 88.0],
                "std_volume": [1.0, 2.0, 3.0],
                "initial_volume": [100.0, 100.0, 100.0],
                "deformation_amplitude": [0.2, 0.0, 0.4],
            },
        }
    )

    assert np.allclose(result.channels["reference_volume"], np.array([110.0, 110.0, 110.0]))
    assert np.allclose(result.channels["relative_volume"], np.array([0.9, 1.0, 0.8]))
    assert np.allclose(result.channels["relative_volume_std"], np.array([1.0, 2.0, 3.0]) / 110.0)


def test_run_buckling_paper_replay_lane_uses_injected_sampler() -> None:
    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
    )

    captured: dict[str, object] = {}

    def fake_sampler(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "experiment": "buckling",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "buck": [0.0, 0.375, 0.75],
                "relative_volume": [1.0, 0.96, 0.9],
                "buckling_response": [1.0, 0.95, 0.82],
            },
        }

    result = run_buckling_paper_replay_lane(plan, sampler=fake_sampler)

    assert captured["experiment"] == "buckling"
    assert captured["controls"]["bpress"] == pytest.approx(-91.0)
    assert len(captured["controls"]["buck"]) == 20
    assert "pressure_difference" not in captured["controls"]
    assert result.axis == "pressure_difference"
    assert np.allclose(result.channels["buck"], np.array([0.0, 0.375, 0.75]))
    assert np.allclose(result.channels["pressure_difference"], np.array([0.0, 34.0875, 68.175]))
    assert np.allclose(result.channels["relative_volume"], np.array([1.0, 0.96, 0.9]))
    assert result.provenance["lane_plan"]["figure_id"] == BUCKLING_FIGURE_ID


def test_buckling_paper_replay_accepts_object_payload_and_writes_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _install_fake_matplotlib(monkeypatch)
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
                "fscale: 1.0",
                "radGV: 2.0",
                "nu: 0.3",
                "Yl: 2.0",
                "Yt: 3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (parameter_dir / "parameters00001.yaml").write_text("ue: 1.0\n", encoding="utf-8")
    sample_result = SimpleNamespace(
        experiment="buckling",
        geometry=SimpleNamespace(radGV=2.0, height=14.28),
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"bpress": -91.0},
        channels={
            "buck": [0.0, 0.375, 0.75],
            "volumetric_strain": [1.0, 0.96, 0.9],
            "relative_volume_std": [0.01, 0.02, 0.03],
            "shape_amplitude": [0.0, 0.1, 0.3],
        },
        manifest={"dataset_id": "gv__buckling__fixture"},
        runtime_manifests=({"control_id": "pressure_-91"},),
        plan_manifests=({"experiment": "buckling"},),
        work_dirs=(work_dir,),
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
    assert manifest["channels"]["pressure_difference"] == pytest.approx([0.0, 34.0875, 68.175])
    assert result.controls["bpress"] == pytest.approx(-91.0)
    assert result.provenance["runtime_seconds"] == 12.5
    assert result.provenance["runtime_manifests"] == [{"control_id": "pressure_-91"}]
    assert "Mirheo reruns" in result.provenance["canonical_replay_source"]
    assert plot_path.is_file()


def test_plot_buckling_paper_replay_rejects_non_runs_output() -> None:
    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
    )
    result = postprocess_buckling_paper_replay_lane(
        {
            "experiment": "buckling",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "buck": [0.0, 0.75],
                "pressure_difference": [0.0, 68.175],
                "relative_volume": [1.0, 0.96],
                "buckling_response": [1.0, 0.95],
            },
        },
        plan=plan,
    )

    with pytest.raises(ValueError, match="under _runs/"):
        plot_buckling_paper_replay(result, output_path="paper_replay/buckling.png")


def test_plot_buckling_paper_replay_handles_missing_theory_without_errorbars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _install_fake_matplotlib(monkeypatch)
    result = postprocess_buckling_paper_replay_lane(
        {
            "experiment": "buckling",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "work_dirs": (tmp_path / "_runs" / "missing-parameters",),
            "channels": {
                "buck": [0.0, 0.75],
                "pressure_difference": [0.0, 68.175],
                "relative_volume": [1.0, 0.96],
                "buckling_response": [1.0, 0.95],
            },
        },
    )

    output = plot_buckling_paper_replay(
        result,
        output_path="_runs/gv/paper_replay/buckling/no_theory.png",
    )

    assert output.is_file()


def test_plan_buckling_paper_replay_rejects_output_root_traversal() -> None:
    with pytest.raises(ValueError, match="under"):
        plan_buckling_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            output_root="_runs/../outside",
        )


def test_plan_buckling_paper_replay_accepts_external_runs_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "scratch" / "runs"
    output_root = runs_root / "gv" / "figure_replay" / "campaign" / "lanes" / "buckling"
    monkeypatch.setenv("MESOUQ_RUNS_ROOT", str(runs_root))

    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        output_root=output_root,
    )

    assert Path(plan.runtime_options.output_root) == output_root.resolve()


def test_plan_buckling_paper_replay_rejects_invalid_material_and_sweep_inputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        plan_buckling_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            pressure_differences=(68.175,),
        )

    bad_parameters = dict(_BASE_MATERIAL_PARAMETERS)
    bad_parameters["ka"] = 0.0
    with pytest.raises(ValueError, match="Material parameter 'ka'"):
        plan_buckling_paper_replay_lane(
            material_parameters=bad_parameters,
            radGV=2.0,
            height=14.28,
        )

    duplicate_parameters = dict(_BASE_MATERIAL_PARAMETERS)
    duplicate_parameters["muL"] = duplicate_parameters["mu_l"]
    with pytest.raises(ValueError, match="duplicated"):
        plan_buckling_paper_replay_lane(
            material_parameters=duplicate_parameters,
            radGV=2.0,
            height=14.28,
        )


def test_buckling_private_coercion_and_metadata_error_paths() -> None:
    class ManifestOnly:
        def as_manifest(self):
            return {
                "channels": {"buck": [0.0], "pressure_difference": [0.0], "relative_volume": [1.0]},
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
            }

    assert buckling_module._coerce_mapping(ManifestOnly(), context="fixture")["controls"] == {"bpress": -91.0}

    with pytest.raises(ValueError, match="1D"):
        buckling_module._coerce_finite_1d([[1.0]], name="pressure_differences")
    with pytest.raises(ValueError, match="at least one"):
        buckling_module._coerce_finite_1d([], name="pressure_differences")
    with pytest.raises(ValueError, match="finite"):
        buckling_module._coerce_finite_1d([float("nan")], name="pressure_differences")
    with pytest.raises(ValueError, match="scalar"):
        buckling_module._coerce_scalar([1.0, 2.0], name="buck")
    assert buckling_module._coerce_scalar(np.array([0.75]), name="buck") == pytest.approx(0.75)
    with pytest.raises(ValueError, match="finite"):
        buckling_module._coerce_scalar(float("nan"), name="buck")


def test_buckling_postprocess_plan_fallback_and_control_aliases() -> None:
    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
    )

    result = postprocess_buckling_paper_replay_lane(
        {
            "channels": {
                "pressure_difference": [0.0, 68.175],
                "relative_volume": [1.0, 0.96],
                "buckling_response": [1.0, 0.95],
            },
            "controls": {"pressure_difference": (0.0, 68.175), "bpress": -91.0},
            "runtime_manifests": [{"control_id": "pressure_-91"}],
        },
        plan=plan,
    )

    assert result.geometry == {"radGV": 2.0, "height": 14.28}
    assert result.material_parameters == _BASE_MATERIAL_PARAMETERS
    assert result.controls == {"bpress": -91.0}
    assert np.allclose(result.channels["buck"], np.array([0.0, 0.75]))
    assert result.provenance["runtime_manifests"] == [{"control_id": "pressure_-91"}]

    legacy = postprocess_buckling_paper_replay_lane(
        {
            "experiment": "buckling",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"pressure": -91.0},
            "channels": {
                "bpress": [0.0, 68.175],
                "relative_volume": [1.0, 0.96],
                "buckling_response": [1.0, 0.95],
            },
        }
    )
    assert legacy.controls == {"bpress": -91.0}
    assert np.allclose(legacy.channels["buck"], [0.0, 0.75])
    assert np.allclose(legacy.channels["pressure_difference"], [0.0, 68.175])

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
                "controls": {"bpress": -91.0},
                "channels": {
                    "buck": [0.0],
                    "pressure_difference": [0.0],
                    "relative_volume": [1.0],
                    "buckling_response": [0.1],
                },
            }
        )
    with pytest.raises(ValueError, match="material_parameters"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "controls": {"bpress": -91.0},
                "channels": {
                    "buck": [0.0],
                    "pressure_difference": [0.0],
                    "relative_volume": [1.0],
                    "buckling_response": [0.1],
                },
            }
        )
    with pytest.raises(ValueError, match="scalar controls"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "channels": {
                    "buck": [0.0],
                    "pressure_difference": [0.0],
                    "relative_volume": [1.0],
                    "buckling_response": [0.1],
                },
            }
        )
    with pytest.raises(ValueError, match="positive measured zero-pressure volume"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
                "channels": {
                    "buck": [0.0, 0.75],
                    "mean_volume": [0.0, 1.0],
                    "buckling_response": [0.1, 0.2],
                },
            }
        )
    with pytest.raises(ValueError, match="std_volume and mean_volume"):
        postprocess_buckling_paper_replay_lane(
            {
                "experiment": "buckling",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
                "channels": {
                    "buck": [0.0, 0.75],
                    "mean_volume": [1.0, 0.95],
                    "std_volume": [0.01],
                    "buckling_response": [0.1, 0.2],
                },
            }
        )


def test_buckling_paper_exact_execution_sets_runtime_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "MESOUQ_GV_BUCKLING_MEMBRANE_BPRESS_MODE",
        "MESOUQ_GV_BUCKLING_FLUID_MODE",
        "MESOUQ_GV_BUCKLING_FLUID_STABILIZATION",
        "MESOUQ_GV_BUCKLING_PIN_OBJECT",
        "MESOUQ_GV_BUCKLING_ODPD_AMP_SCALE",
    ):
        monkeypatch.delenv(name, raising=False)

    plan = plan_buckling_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        buck=(0.0, 0.75),
        paper_exact=True,
        output_root="_runs/gv/figure_replay/test-buckling",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        buckling_module,
        "build_geometry",
        lambda **_kwargs: SimpleNamespace(id="gv_geom", parameters={"radGV": 2.0, "height": 14.28}),
    )

    runtime = RuntimeDryRun(
        structure="gv",
        experiment="buckling",
        geometry="gv_geom",
        controls={"buck": 0.0, "bpress": -91.0},
        control_sweeps=(),
        control_id="buck_0_0_75__bpress_-91",
        dataset_id="gv__buckling__fixture",
        provenance_root=str(tmp_path),
        source_root=str(tmp_path),
        legacy_import_root="",
        output_root=str(tmp_path),
        work_dir=str(tmp_path),
        commands=(
            DryRunCommand(
                argv=("python3", "generate.py", "-p", "buck", "0", "0.75", "2", "--object", "gv", "--forward"),
                cwd=str(tmp_path),
                description="generate",
            ),
        ),
        source_files=(),
        source_manifest=str(tmp_path / "source_manifest.json"),
        generated_subdirs=(),
        runtime_package="mirheo",
        sweep_mode="forward",
    )

    monkeypatch.setattr(buckling_module, "plan_runtime", lambda *args, **kwargs: runtime)
    class SamplingPlan:
        def to_manifest(self) -> dict[str, object]:
            return {"control_axis": "buck"}

    def fake_build_sampling_plan(runtime_arg: RuntimeDryRun, *args: object, **kwargs: object) -> SamplingPlan:
        captured["generate_argv"] = runtime_arg.commands[0].argv
        captured["first_restart"] = runtime_arg.first_restart
        return SamplingPlan()

    monkeypatch.setattr(buckling_module, "build_sampling_plan", fake_build_sampling_plan)

    def fake_execute(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured["env"] = kwargs["env"]
        return SimpleNamespace(executed_commands=("bash commands.txt",), return_codes=(0,))

    monkeypatch.setattr(buckling_module, "execute_sampling_plan", fake_execute)
    monkeypatch.setattr(
        buckling_module,
        "extract_sampling_channels",
        lambda **_kwargs: {
            "buck": np.array([0.0, 0.75]),
            "pressure_difference": np.array([0.0, 68.175]),
            "relative_volume": np.array([1.0, 0.96]),
            "deformation_amplitude": np.array([0.0, 0.2]),
        },
    )

    result = run_buckling_paper_replay_lane(plan)

    assert captured["env"]["MESOUQ_GV_PAPER_EXACT"] == "1"
    assert captured["env"]["MESOUQ_GV_BUCKLING_MEMBRANE_BPRESS_MODE"] == "runtime"
    assert captured["env"]["MESOUQ_GV_BUCKLING_FLUID_MODE"] == "dropped"
    assert captured["env"]["MESOUQ_GV_BUCKLING_FLUID_STABILIZATION"] == "0.0"
    assert captured["env"]["MESOUQ_GV_BUCKLING_PIN_OBJECT"] == "1"
    assert captured["env"]["MESOUQ_GV_BUCKLING_ODPD_AMP_SCALE"] == "1.0"
    assert "--first" not in captured["generate_argv"]
    assert captured["first_restart"] is False
    assert result.provenance["status"] == "completed"
    assert result.controls["bpress"] == pytest.approx(-91.0)
    assert result.provenance["paper_protocol"]["environment"] == {
        name: captured["env"][name]
        for name in (
            "MESOUQ_GV_PAPER_EXACT",
            "MESOUQ_GV_BUCKLING_MEMBRANE_BPRESS_MODE",
            "MESOUQ_GV_BUCKLING_FLUID_MODE",
            "MESOUQ_GV_BUCKLING_FLUID_STABILIZATION",
            "MESOUQ_GV_BUCKLING_PIN_OBJECT",
            "MESOUQ_GV_BUCKLING_ODPD_AMP_SCALE",
        )
    }
    assert (
        result.provenance["lane_plan"]["paper_protocol"]["environment"]["MESOUQ_GV_BUCKLING_FLUID_MODE"]
        == "dropped"
    )
