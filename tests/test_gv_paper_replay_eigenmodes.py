from __future__ import annotations

import sys
from types import ModuleType
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


class _FakeFigure:
    def text(self, *args, **kwargs) -> None:
        return None

    def tight_layout(self, *args, **kwargs) -> None:
        return None

    def savefig(self, path: Path, *args, **kwargs) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("plot", encoding="utf-8")


class _FakeAxis:
    def __init__(self) -> None:
        self.off = False
        self.transAxes = object()

    def plot(self, *args, **kwargs) -> None:
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

    def set_yticks(self, *args, **kwargs) -> None:
        return None

    def annotate(self, *args, **kwargs) -> None:
        return None

    def text(self, *args, **kwargs) -> None:
        return None

    def axvline(self, *args, **kwargs) -> None:
        return None

    def legend(self, *args, **kwargs) -> None:
        return None

    def grid(self, *args, **kwargs) -> None:
        return None

    def axis(self, value: str) -> None:
        if value == "off":
            self.off = True


class _FakePyplot:
    def subplots(self, nrows: int = 1, ncols: int = 1, *args, **kwargs):
        figure = _FakeFigure()
        if nrows == 1 and ncols == 1:
            return figure, _FakeAxis()
        axes = np.empty((nrows, ncols), dtype=object)
        for row in range(nrows):
            for col in range(ncols):
                axes[row, col] = _FakeAxis()
        return figure, axes if nrows > 1 else axes[0]

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
    assert plan.paper_exact is False
    assert plan.runtime_options.output_root == "_runs/gv/paper_replay/eigenmodes"


def test_plan_eigenmodes_paper_replay_lane_records_timeout() -> None:
    plan = plan_eigenmodes_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        timeout_seconds=36000,
    )

    assert plan.runtime_options.timeout_seconds == 36000
    assert plan.to_manifest()["runtime_options"]["timeout_seconds"] == 36000


def test_plan_eigenmodes_paper_replay_lane_exact_replay_forces_30_modes() -> None:
    plan = plan_eigenmodes_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        mode_count=12,
        paper_exact=True,
    )

    assert plan.mode_count == 30
    assert plan.paper_exact is True


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
                    "kBT": 4.0,
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
                "eigenvalues": [1.0, 4.0, 9.0, 16.0],
                "kBT": 9.0,
                "eigenvectors": [[1.0], [0.0], [0.5], [0.25]],
            },
        }

    result = run_eigenmodes_paper_replay_lane(plan, sampler=fake_sampler)

    assert captured["experiment"] == "eigenmodes"
    assert captured["controls"] == {"bpress": (-91.0,)}
    assert result.axis == "mode_index"
    assert np.array_equal(result.channels["mode_index"], np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(result.channels["eigenvalues"], np.array([1.0, 4.0, 9.0]))
    assert np.allclose(result.channels["frequency"], np.array([1.0, 1.5, 3.0]))
    assert np.array_equal(result.channels["eigenvectors"], np.array([[1.0], [0.0], [0.5]]))
    assert result.selected_modes["surface_mode_indices"] == [0, 4, 6, 7, 18, 24]
    assert result.provenance["lane_plan"]["mode_count"] == 3


def test_eigenmodes_paper_exact_execution_sets_runtime_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = plan_eigenmodes_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        paper_exact=True,
        output_root="_runs/gv/figure_replay/test-eigenmodes",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(eigenmodes_module, "build_geometry", lambda **_kwargs: SimpleNamespace(id="gv_geom"))

    class Runtime:
        work_dir = tmp_path

        def to_manifest(self) -> dict[str, object]:
            return {"work_dir": str(self.work_dir)}

    class SamplingPlan:
        def to_manifest(self) -> dict[str, object]:
            return {"control_axis": "bpress"}

    monkeypatch.setattr(eigenmodes_module, "plan_runtime", lambda *args, **kwargs: Runtime())
    monkeypatch.setattr(eigenmodes_module, "build_sampling_plan", lambda *args, **kwargs: SamplingPlan())

    def fake_execute(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured["env"] = kwargs["env"]
        return SimpleNamespace(executed_commands=("bash commands.txt",), return_codes=(0,))

    monkeypatch.setattr(eigenmodes_module, "execute_sampling_plan", fake_execute)
    monkeypatch.setattr(
        eigenmodes_module,
        "extract_sampling_channels",
        lambda **_kwargs: {
            "mode_index": np.array([0.0, 1.0, 2.0]),
            "eigenvalues": np.array([9.0, 4.0, 1.0]),
            "kBT": np.array([9.0]),
            "eigenvectors": np.arange(27.0).reshape(3, 9),
        },
    )

    result = run_eigenmodes_paper_replay_lane(plan)

    assert captured["env"]["MESOUQ_GV_PAPER_EXACT"] == "1"
    assert result.provenance["status"] == "completed"
    assert result.controls["bpress"] == pytest.approx(-91.0)
    assert result.provenance["lane_plan"]["paper_exact"] is True


def test_eigenmodes_paper_replay_accepts_object_payload_and_writes_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _install_fake_matplotlib(monkeypatch)
    sample_result = SimpleNamespace(
        experiment="eigenmodes",
        geometry=SimpleNamespace(radGV=2.0, height=14.28),
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"pressure_difference": -91.0},
        channels={
            "eigenvalues": [1.0, 4.0, 9.0, 16.0],
            "kBT": 9.0,
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
    assert manifest["channels"]["frequency"][:4] == [0.75, 1.0, 1.5, 3.0]
    assert result.controls == {"bpress": -91.0}
    assert result.selected_modes["axial_mode_indices"] == [0, 6, 20]
    assert result.provenance["runtime_seconds"] == 8.5
    assert result.provenance["runtime_manifests"] == [{"control_id": "bpress_-91"}]
    assert result.provenance["protocol_references"]["canonical_replay_data"].startswith(
        "Canonical eigenmode replay data come from Mirheo reruns"
    )
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
                "eigenvalues": [1.0, 4.0],
                "kBT": 4.0,
            },
        },
        plan=plan,
    )

    with pytest.raises(ValueError, match="under _runs/"):
        plot_eigenmodes_paper_replay(result, output_path="paper_replay/eigenmodes.png")


def test_eigenmodes_helpers_cover_aliases_fallbacks_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _install_fake_matplotlib(monkeypatch)

    normalized = eigenmodes_module._normalize_eigenmode_channels(
        {
            "channels": {
                "lambdas": [9.0, 4.0, 1.0],
                "metadata_frequency": [100.0],
                "eigenvectors": np.arange(18.0).reshape(2, 9),
                "mesh_vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "faces": [[0, 1, 2]],
            },
            "metadata": {"kbt": 9.0},
        },
        mode_count=2,
    )
    np.testing.assert_allclose(normalized["lambdas"], [9.0, 4.0])
    np.testing.assert_allclose(normalized["frequency"], [1.0, 1.5])
    assert normalized["reference_positions"].shape == (3, 3)
    assert normalized["mesh_faces"].shape == (1, 3)

    plan = plan_eigenmodes_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        bpress=-91.0,
        mode_count=2,
    )
    fallback_result = postprocess_eigenmodes_paper_replay_lane(
        {
            "experiment": "eigenmodes",
            "channels": {"frequency": [0.25, 0.5]},
            "controls": {"background_pressure": -90.0},
        },
        plan=plan,
    )
    assert fallback_result.geometry == plan.geometry
    assert fallback_result.material_parameters == plan.material_parameters
    assert fallback_result.controls == {"bpress": -90.0}

    manual_result = eigenmodes_module.EigenmodesPaperReplayResult(
        figure_id=EIGENMODES_FIGURE_ID,
        experiment="eigenmodes",
        axis="mode_index",
        controls={"bpress": -91.0},
        geometry={"radGV": 2.0, "height": 14.28},
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        channels={
            "mode_index": np.array([0.0, 1.0]),
            "frequency": np.array([1.0, 2.0]),
            "eigenvectors": np.arange(36.0).reshape(2, 18),
            "reference_positions": np.array(
                [
                    [-2.0, 0.0, -3.0],
                    [-1.0, 0.0, -2.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [3.0, 0.0, 2.0],
                ],
            ),
        },
        selected_modes={
            "surface_mode_indices": [0, 4, 6, 7, 18, 24],
            "surface_mode_indices_available": [],
            "axial_mode_indices": [0, 6, 20],
            "axial_mode_indices_available": [0],
        },
        provenance={"paper_exact": False},
    )
    plot_path = plot_eigenmodes_paper_replay(
        manual_result,
        output_path="_runs/gv/paper_replay/eigenmodes/manual_modes.png",
    )
    assert plot_path.is_file()

    selected_result = eigenmodes_module.EigenmodesPaperReplayResult(
        figure_id=EIGENMODES_FIGURE_ID,
        experiment="eigenmodes",
        axis="mode_index",
        controls={"bpress": -91.0},
        geometry={"radGV": 2.0, "height": 14.28},
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        channels=manual_result.channels,
        selected_modes={
            "surface_mode_indices": [0, 4, 6, 7, 18, 24],
            "surface_mode_indices_available": [0],
            "axial_mode_indices": [0, 6, 20],
            "axial_mode_indices_available": [0],
        },
        provenance={"paper_exact": False},
    )
    assert plot_eigenmodes_paper_replay(
        selected_result,
        output_path="_runs/gv/paper_replay/eigenmodes/selected_modes.png",
    ).is_file()


def test_eigenmodes_panel_h_reconstructs_paper_coordinate_frame() -> None:
    centered_reference = np.array(
        [
            [-2.0, 0.0, -3.0],
            [-1.0, 0.0, -2.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 2.0],
        ],
        dtype=float,
    )

    paper_reference = eigenmodes_module._paper_reference_coordinates(centered_reference)
    selected = eigenmodes_module._paper_axial_slice_indices(paper_reference)

    assert selected.tolist() == [3, 4, 5]
    np.testing.assert_allclose(paper_reference[selected, 0], [13.5, 14.5, 15.5])
    np.testing.assert_allclose(paper_reference[selected, 1], [12.5, 12.5, 12.5])
    with pytest.raises(ValueError, match="panel \\(h\\)"):
        eigenmodes_module._paper_axial_slice_indices(np.zeros((3, 3), dtype=float))


def test_eigenmodes_helpers_reject_invalid_payload_shapes_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eigenmodes_module,
        "_normalize_eigenmode_channels",
        lambda *_args, **_kwargs: {"mode_index": np.array([0.0])},
    )
    with pytest.raises(ValueError, match="requires at least one spectral channel"):
        postprocess_eigenmodes_paper_replay_lane(
            {
                "experiment": "eigenmodes",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
                "channels": {"frequency": [1.0]},
            }
        )

    with pytest.raises(ValueError, match="requires kBT"):
        eigenmodes_module._resolve_frequency(
            {"channels": {}},
            {},
            {"eigenvalues": np.array([1.0])},
            mode_count=1,
        )
    with pytest.raises(ValueError, match="must be positive"):
        eigenmodes_module._resolve_frequency(
            {"channels": {}, "manifest": {"kBT": 1.0}},
            {},
            {"eigenvalues": np.array([0.0])},
            mode_count=1,
        )
    assert eigenmodes_module._resolve_frequency(
        {"channels": {}},
        {"EigenFrequencies": [3.0, 4.0]},
        {},
        mode_count=1,
    ).tolist() == [3.0]
    with pytest.raises(ValueError, match="finite 1D data"):
        eigenmodes_module._select_frequency_payload({"frequency": [[1.0], [2.0]]})

    with pytest.raises(ValueError, match="must be a 2D array"):
        eigenmodes_module._extract_auxiliary_channel(
            {"mesh_vertices": [0.0, 1.0, 2.0]},
            "reference_positions",
            mode_count=1,
        )
    with pytest.raises(ValueError, match="must be a 2D array"):
        eigenmodes_module._extract_auxiliary_channel(
            {"faces": [0, 1, 2]},
            "mesh_faces",
            mode_count=1,
        )

    class _ManifestLike:
        def as_manifest(self):
            return {"channels": {"frequency": [1.0]}, "geometry": {"radGV": 2.0, "height": 14.28}}

    manifest = eigenmodes_module._coerce_mapping(_ManifestLike(), context="sample_result")
    assert manifest["channels"]["frequency"] == [1.0]
    with pytest.raises(ValueError, match="sample_result must be a mapping"):
        eigenmodes_module._coerce_mapping(object(), context="sample_result")

    with pytest.raises(ValueError, match="must be a scalar value"):
        eigenmodes_module._coerce_scalar([1.0, 2.0], name="bpress")
    with pytest.raises(ValueError, match="must be finite"):
        eigenmodes_module._coerce_scalar(np.nan, name="bpress")
    with pytest.raises(ValueError, match="must be finite and > 0"):
        eigenmodes_module._validate_paper_replay_material_parameters(
            {**_BASE_MATERIAL_PARAMETERS, "ka": 0.0}
        )
    with pytest.raises(ValueError, match="scalar control 'bpress'"):
        eigenmodes_module._resolve_controls({}, None)


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
                "channels": {"eigenvalues": [1.0, 4.0], "kBT": 4.0},
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "controls": {"bpress": -91.0},
            }

    assert eigenmodes_module._coerce_mapping(ManifestOnly(), context="fixture")["controls"] == {"bpress": -91.0}

    with pytest.raises(ValueError, match="scalar"):
        eigenmodes_module._coerce_scalar([1.0, 2.0], name="bpress")
    assert eigenmodes_module._coerce_scalar(np.array([-91.0]), name="bpress") == pytest.approx(-91.0)
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
                "channels": {"eigenvalues": [1.0, 4.0], "kBT": 4.0},
            }
        )
    with pytest.raises(ValueError, match="material_parameters"):
        postprocess_eigenmodes_paper_replay_lane(
            {
                "experiment": "eigenmodes",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "controls": {"bpress": -91.0},
                "channels": {"eigenvalues": [1.0, 4.0], "kBT": 4.0},
            }
        )
    with pytest.raises(ValueError, match="scalar control"):
        postprocess_eigenmodes_paper_replay_lane(
            {
                "experiment": "eigenmodes",
                "geometry": {"radGV": 2.0, "height": 14.28},
                "material_parameters": _BASE_MATERIAL_PARAMETERS,
                "channels": {"eigenvalues": [1.0, 4.0], "kBT": 4.0},
            }
        )


def test_postprocess_eigenmodes_requires_spectrum_and_mode_index_even_with_plan() -> None:
    plan = plan_eigenmodes_paper_replay_lane(
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        radGV=2.0,
        height=14.28,
        bpress=-91.0,
        mode_count=2,
    )

    with pytest.raises(ValueError, match="at least one spectral channel"):
        postprocess_eigenmodes_paper_replay_lane(
            {"channels": {"mode_index": [0.0, 1.0]}},
            plan=plan,
        )

    with pytest.raises(ValueError, match="mode_index"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                eigenmodes_module,
                "_normalize_eigenmode_channels",
                lambda *_args, **_kwargs: {"frequency": np.asarray([1.0, 2.0], dtype=float)},
            )
            postprocess_eigenmodes_paper_replay_lane(
                {"channels": {"eigenvalues": [1.0, 4.0]}},
                plan=plan,
            )


def test_postprocess_eigenmodes_derives_frequency_from_sorted_kbt_over_lambda() -> None:
    result = postprocess_eigenmodes_paper_replay_lane(
        {
            "experiment": "eigenmodes",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "lambdas": [9.0, 1.0, 4.0],
                "kBT": 9.0,
                "eigenvectors": np.arange(27.0).reshape(3, 9),
            },
        },
        plan=plan_eigenmodes_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            mode_count=3,
        ),
    )

    assert np.array_equal(result.channels["eigenvalues"], np.array([9.0, 1.0, 4.0]))
    assert np.allclose(result.channels["frequency"], np.array([1.0, 1.5, 3.0]))
    assert np.array_equal(result.channels["eigenvectors"], np.arange(27.0).reshape(3, 9))


def test_plot_eigenmodes_paper_replay_rejects_full_mode_shape_without_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _install_fake_matplotlib(monkeypatch)
    result = postprocess_eigenmodes_paper_replay_lane(
        {
            "experiment": "eigenmodes",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "eigenvalues": [1.0, 4.0, 9.0],
                "kBT": 9.0,
                "eigenvectors": np.arange(9.0).reshape(3, 3),
            },
        },
        plan=plan_eigenmodes_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            mode_count=3,
        ),
    )

    with pytest.raises(ValueError, match="Full eigenmode shape plot requested"):
        plot_eigenmodes_paper_replay(
            result,
            output_path="_runs/gv/paper_replay/eigenmodes/full.png",
            full_mode_shape_plot=True,
        )


def test_plot_eigenmodes_paper_replay_writes_selected_surface_modes_when_source_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _install_fake_matplotlib(monkeypatch)
    result = postprocess_eigenmodes_paper_replay_lane(
        {
            "experiment": "eigenmodes",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "eigenvalues": np.arange(1.0, 31.0),
                "kBT": 9.0,
                "eigenvectors": np.arange(30.0 * 18.0).reshape(30, 18),
                "reference_positions": np.array(
                    [
                        [-2.0, 0.0, -3.0],
                        [-1.0, 0.0, -2.0],
                        [0.0, 0.0, -1.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 1.0],
                        [3.0, 0.0, 2.0],
                    ],
                ),
            },
        },
        plan=plan_eigenmodes_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            paper_exact=True,
        ),
    )

    output = plot_eigenmodes_paper_replay(
        result,
        output_path="_runs/gv/paper_replay/eigenmodes/selected.png",
    )

    assert output.is_file()
    assert result.selected_modes["surface_mode_indices_available"] == [0, 4, 6, 7, 18, 24]


def test_plot_eigenmodes_paper_replay_writes_paper_panels_with_mesh_faces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    reference_positions = np.array(
        [
            [13.0, 12.5, 5.0],
            [13.5, 12.5, 7.0],
            [14.0, 12.5, 9.0],
            [12.7, 12.0, 6.0],
            [14.2, 13.0, 8.0],
            [13.2, 13.5, 10.0],
        ],
        dtype=float,
    )
    eigenvectors = np.zeros((30, reference_positions.size), dtype=float)
    for mode in range(30):
        vectors = np.zeros_like(reference_positions)
        vectors[:, 0] = 0.01 * (mode + 1)
        vectors[:, 1] = np.linspace(-0.02, 0.02, reference_positions.shape[0])
        vectors[:, 2] = 0.005 * np.sin(np.arange(reference_positions.shape[0]) + mode)
        eigenvectors[mode] = vectors.reshape(-1)

    result = postprocess_eigenmodes_paper_replay_lane(
        {
            "experiment": "eigenmodes",
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": _BASE_MATERIAL_PARAMETERS,
            "controls": {"bpress": -91.0},
            "channels": {
                "eigenvalues": np.arange(1.0, 31.0),
                "kBT": 9.0,
                "eigenvectors": eigenvectors,
                "reference_positions": reference_positions,
                "mesh_faces": np.array([(0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4)]),
            },
        },
        plan=plan_eigenmodes_paper_replay_lane(
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            radGV=2.0,
            height=14.28,
            paper_exact=True,
        ),
    )

    output = plot_eigenmodes_paper_replay(
        result,
        output_path="_runs/gv/paper_replay/eigenmodes/paper_panels.png",
        full_mode_shape_plot=True,
    )

    assert output.is_file()
    assert result.selected_modes["axial_mode_indices_available"] == [0, 6, 20]
