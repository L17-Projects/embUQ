from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from meso_uq.experiments import ExperimentSpec
from meso_uq.inference import gv_hbi


def _experiment(
    tmp_path: Path,
    *,
    structure: str = "gv",
    name: str = "torsion",
    enabled: bool = True,
    controls: list[str] | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        structure=structure,
        name=name,
        geometries=["gv_rad2_height14_28"],
        data_dir=tmp_path,
        data_prefix=f"{name}_data_",
        surrogate_dir=tmp_path,
        enabled=enabled,
        prior_d0=[0.0, 0.5],
        prior_sigma=[0.01, 0.1],
        controls=controls or ["theta_0.03"],
    )


def _gv_config(manifest_path: Path) -> dict[str, object]:
    return {
        "structure": "gv",
        "structures": ["gv"],
        "experimental_gv_hbi": True,
        "surrogate": {"backend": "dnn"},
        "gv_surrogate_manifest": str(manifest_path),
        "prior_ka": [0.1, 1.1],
        "prior_kb": [0.2, 1.2],
        "prior_mu": [0.3, 1.3],
        "prior_b1": [0.4, 1.4],
        "prior_b2": [0.5, 1.5],
        "prior_a3": [0.6, 1.6],
        "prior_a4": [0.7, 1.7],
        "prior_mu_l": [0.8, 1.8],
        "prior_c": [0.9, 1.9],
        "prior_sigma": [0.01, 0.1],
        "hyperprior_mu_ka": [0.1, 1.1],
        "hyperprior_sigma_ka": [0.01, 0.2],
        "hyperprior_mu_kb": [0.2, 1.2],
        "hyperprior_sigma_kb": [0.01, 0.2],
        "hyperprior_mu_mu": [0.3, 1.3],
        "hyperprior_sigma_mu": [0.01, 0.2],
        "hyperprior_mu_b1": [0.4, 1.4],
        "hyperprior_sigma_b1": [0.01, 0.2],
        "hyperprior_mu_b2": [0.5, 1.5],
        "hyperprior_sigma_b2": [0.01, 0.2],
        "hyperprior_mu_a3": [0.6, 1.6],
        "hyperprior_sigma_a3": [0.01, 0.2],
        "hyperprior_mu_a4": [0.7, 1.7],
        "hyperprior_sigma_a4": [0.01, 0.2],
        "hyperprior_mu_mu_l": [0.8, 1.8],
        "hyperprior_sigma_mu_l": [0.01, 0.2],
        "hyperprior_mu_c": [0.9, 1.9],
        "hyperprior_sigma_c": [0.01, 0.2],
    }


def _surrogate_manifest(
    tmp_path: Path,
    *,
    payload_updates: dict[str, object] | None = None,
    artifacts: dict[str, str] | None = None,
) -> Path:
    model_path = tmp_path / "model.pkl"
    model_path.write_text("artifact", encoding="utf-8")
    reference_path = tmp_path / "reference.json"
    reference_path.write_text("{}", encoding="utf-8")
    manifest = {
        "structure": "gv",
        "experiment": "torsion",
        "geometry": "gv_rad2_height14_28",
        "controls": {"theta": 0.03},
        "reference_kind": "synthetic",
        "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
        "backend": "dnn",
        "artifacts": artifacts
        if artifacts is not None
        else {
            "model_path": str(model_path),
            "reference_manifest": str(reference_path),
        },
    }
    if payload_updates:
        manifest.update(payload_updates)
    manifest_path = tmp_path / "surrogate_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_gv_hbi_setup_helper_selection_and_gate_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _experiment(tmp_path)
    manifest_path = _surrogate_manifest(tmp_path)
    dataset_id = experiment.dataset_name("gv_rad2_height14_28", control="theta_0.03")

    assert gv_hbi._as_bool("yes") is True
    assert gv_hbi._as_bool("off") is False
    assert gv_hbi._as_bool(1) is True
    monkeypatch.setenv(gv_hbi.GV_HBI_EXPERIMENTAL_FLAG, "1")
    assert gv_hbi._enabled_by({}) == f"env:{gv_hbi.GV_HBI_EXPERIMENTAL_FLAG}"
    monkeypatch.delenv(gv_hbi.GV_HBI_EXPERIMENTAL_FLAG)
    assert gv_hbi._enabled_by({"experimental": {"gv_hbi": "true"}}) == "config:experimental.gv_hbi"

    assert gv_hbi._resolve_repo_path(tmp_path, "surrogate_manifest.json") == manifest_path
    assert gv_hbi._load_json(manifest_path, label="surrogate manifest")["structure"] == "gv"
    with pytest.raises(FileNotFoundError, match="Missing GV surrogate manifest"):
        gv_hbi._load_json(tmp_path / "missing.json", label="surrogate manifest")
    non_object_json = tmp_path / "list.json"
    non_object_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        gv_hbi._load_json(non_object_json, label="surrogate manifest")

    with pytest.raises(ValueError, match="only GV experiments"):
        gv_hbi._require_gv_only([_experiment(tmp_path, structure="emb", name="compression")])
    with pytest.raises(ValueError, match="must be a mapping"):
        gv_hbi._surrogate_manifest_map({"gv_surrogate_manifests": ["bad"]})
    assert gv_hbi._surrogate_manifest_map({"gv_surrogate_manifests": {dataset_id: manifest_path}}) == {
        dataset_id: str(manifest_path)
    }
    assert gv_hbi._single_surrogate_manifest({}) is None
    assert gv_hbi._single_surrogate_manifest({"surrogate_manifest": manifest_path}) == str(manifest_path)

    assert gv_hbi._inline_surrogate_manifest_map({}, [experiment]) == {}
    inline = gv_hbi._inline_surrogate_manifest_map(
        {
            "experiments": [
                "not-a-mapping",
                {"structure": "emb", "name": "torsion", "surrogate_manifest": "wrong"},
                {"structure": "gv", "name": "torsion"},
                {"structure": "gv", "name": "torsion", "surrogate_manifest": manifest_path},
            ]
        },
        [experiment],
    )
    assert inline == {dataset_id: str(manifest_path)}

    assert gv_hbi._dataset_manifest_path(
        {}, dataset_id=dataset_id, dataset_count=1, inline_manifest_map=inline
    ) == str(manifest_path)
    assert gv_hbi._dataset_manifest_path(
        {"gv_surrogate_manifests": {dataset_id: manifest_path}},
        dataset_id=dataset_id,
        dataset_count=2,
        inline_manifest_map={},
    ) == str(manifest_path)
    assert gv_hbi._dataset_manifest_path(
        {"gv_surrogate_manifest": manifest_path},
        dataset_id=dataset_id,
        dataset_count=1,
        inline_manifest_map={},
    ) == str(manifest_path)
    with pytest.raises(FileNotFoundError, match="Missing GV surrogate manifest selection"):
        gv_hbi._dataset_manifest_path({}, dataset_id=dataset_id, dataset_count=2, inline_manifest_map={})


def test_gv_hbi_requires_structure_qualified_dataset_ids() -> None:
    assert (
        gv_hbi._require_structure_qualified_dataset_id("gv:torsion:gv_rad2_height14_28:theta_0.03")
        is None
    )
    with pytest.raises(ValueError, match="structure-qualified"):
        gv_hbi._require_structure_qualified_dataset_id("torsion:gv_rad2_height14_28:theta_0.03")
    with pytest.raises(ValueError, match="structure-qualified"):
        gv_hbi._require_structure_qualified_dataset_id("gv:torsion::theta_0.03")


def test_gv_hbi_rejects_unqualified_dataset_id_in_manifest_map() -> None:
    with pytest.raises(ValueError, match="structure-qualified"):
        gv_hbi._surrogate_manifest_map(
            {"gv_surrogate_manifests": {"torsion:gv_rad2_height14_28:theta_0.03": "foo.json"}}
        )


def test_gv_hbi_contract_helpers_reject_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ContractWithoutSigma:
        calibrated_names = gv_hbi.GV_PHASE1_CALIBRATED_PARAMETERS
        nuisance_names: tuple[str, ...] = ()
        noise_model = None

    class _ContractWithWrongCalibrated:
        calibrated_names = ("ka",)
        nuisance_names = gv_hbi.GV_PHASE1_NUISANCE_PARAMETERS
        noise_model = None

    class _StructureWithoutSigma:
        parameter_contract = _ContractWithoutSigma()

    class _StructureWithWrongCalibrated:
        parameter_contract = _ContractWithWrongCalibrated()

    monkeypatch.setattr(gv_hbi, "get_structure", lambda _name: _StructureWithWrongCalibrated())
    with pytest.raises(ValueError, match="calibrated contract mismatch"):
        gv_hbi._require_gv_calibrated_contract()

    monkeypatch.setattr(gv_hbi, "get_structure", lambda _name: _StructureWithoutSigma())
    with pytest.raises(ValueError, match="missing required nuisance parameter"):
        gv_hbi._parameter_contract_manifest()

    with pytest.raises(ValueError, match="multiplicative sigma"):
        gv_hbi._require_gv_noisy_model({"noise_model": "sigma"}, label="test payload")


def test_gv_hbi_reference_manifest_defaults_to_multiplicative_noise_model() -> None:
    payload = {
        "structure": "gv",
        "experiment": "torsion",
        "geometry": "gv_rad2_height14_28",
        "controls": {"theta": 0.03},
        "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
        "reference_kind": "synthetic",
        "calibrated_parameter_names": list(gv_hbi.GV_PHASE1_CALIBRATED_PARAMETERS),
        "noise_model": {
            "kind": "multiplicative",
            "parameter": "sigma",
            "description": "Apply multiplicative observation noise.",
        },
    }
    validated = gv_hbi._validate_reference_manifest(
        payload,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        dataset_id="gv:torsion:gv_rad2_height14_28:theta_0.03",
        control="theta_0.03",
        expected_controls={"theta": 0.03},
    )
    assert validated["noise_model"] == gv_hbi.GV_PHASE1_NOISE_MODEL


def test_gv_hbi_requires_default_multiplicative_noise_model_when_missing() -> None:
    assert gv_hbi._require_gv_noisy_model({}, label="GV reference manifest") == gv_hbi.GV_PHASE1_NOISE_MODEL


def test_gv_hbi_loads_gv_dnn_model_state_with_fake_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = object()

    def fake_load_model_states(path: str) -> tuple[object, list[float], list[float], list[float], list[float]]:
        assert path == str(tmp_path / "state.pkl")
        return expected, [1.0, 2.0], [3.0, 4.0], [5.0], [6.0]

    monkeypatch.setitem(sys.modules, "meso_uq.surrogate.model", types.SimpleNamespace(load_model_states=fake_load_model_states))

    model, xshift, xscale, yshift, yscale = gv_hbi._load_gv_dnn_model_state(tmp_path / "state.pkl")
    assert model is expected
    assert np.array_equal(xshift, np.array([1.0, 2.0]))
    assert np.array_equal(xscale, np.array([3.0, 4.0]))
    assert np.array_equal(yshift, np.array([5.0]))
    assert np.array_equal(yscale, np.array([6.0]))


@pytest.mark.parametrize(
    ("payload_updates", "artifacts", "match"),
    (
        ({"structure": "emb"}, None, "structure='gv'"),
        ({"experiment": "stretching"}, None, "experiment mismatch"),
        ({"geometry": "gv_rad3_height16"}, None, "geometry mismatch"),
        ({"dataset_id": "gv:torsion:other:theta_0.03"}, None, "dataset mismatch"),
        ({"backend": "bnn"}, None, "requires a DNN surrogate manifest"),
        ({"artifacts": []}, None, "missing artifacts"),
        ({}, {}, "Missing GV surrogate artifact path"),
    ),
)
def test_gv_hbi_rejects_invalid_surrogate_manifests(
    tmp_path: Path,
    payload_updates: dict[str, object],
    artifacts: dict[str, str] | None,
    match: str,
) -> None:
    experiment = _experiment(tmp_path)
    manifest_path = _surrogate_manifest(tmp_path, payload_updates=payload_updates, artifacts=artifacts)
    payload = gv_hbi._load_json(manifest_path, label="surrogate manifest")

    with pytest.raises((FileNotFoundError, ValueError), match=match):
        gv_hbi._validate_surrogate_manifest(
            payload,
            manifest_path=manifest_path,
            experiment=experiment,
            geometry="gv_rad2_height14_28",
            control="theta_0.03",
            dataset_id="gv:torsion:gv_rad2_height14_28:theta_0.03",
        )


def test_gv_hbi_rejects_missing_reference_manifest_file(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    model_path = tmp_path / "model.pkl"
    model_path.write_text("artifact", encoding="utf-8")
    manifest_path = _surrogate_manifest(
        tmp_path,
        artifacts={
            "model_path": str(model_path),
            "reference_manifest": str(tmp_path / "missing_reference.json"),
        },
    )

    with pytest.raises(FileNotFoundError, match="Missing GV reference manifest"):
        gv_hbi._validate_surrogate_manifest(
            gv_hbi._load_json(manifest_path, label="surrogate manifest"),
            manifest_path=manifest_path,
            experiment=experiment,
            geometry="gv_rad2_height14_28",
            control="theta_0.03",
            dataset_id="gv:torsion:gv_rad2_height14_28:theta_0.03",
        )


@pytest.mark.parametrize(
    ("payload_updates", "match"),
    (
        ({"controls": {"theta": 0.07}}, "controls mismatch"),
        ({"structure": "emb"}, "structure='gv'"),
        ({"experiment": "stretching"}, "experiment mismatch"),
        ({"geometry": "gv_rad3_height16"}, "geometry mismatch"),
        ({"dataset_id": "gv:torsion:other:theta_0.03"}, "dataset mismatch"),
        ({"noise_model": {"kind": "additive", "parameter": "sigma"}}, "multiplicative sigma noise"),
        ({"calibrated_parameter_names": ["theta", "kb"]}, "must match the GV parameter contract"),
    ),
)
def test_gv_hbi_rejects_invalid_reference_manifests(
    payload_updates: dict[str, object],
    match: str,
) -> None:
    payload = {
        "structure": "gv",
        "experiment": "torsion",
        "geometry": "gv_rad2_height14_28",
        "controls": {"theta": 0.03},
        "reference_kind": "synthetic",
        "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
        "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
        "calibrated_parameter_names": ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"],
    }
    payload.update(payload_updates)

    with pytest.raises(ValueError, match=match):
        gv_hbi._validate_reference_manifest(
            payload,
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            dataset_id="gv:torsion:gv_rad2_height14_28:theta_0.03",
            control="theta_0.03",
            expected_controls={"theta": 0.03},
        )


def test_gv_hbi_rejects_reference_manifest_that_calibrates_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Experiment:
        control_names = ("theta",)

    class _Contract:
        calibrated_names = ("ka", "kb", "theta")

    class _Structure:
        parameter_contract = _Contract()

        @staticmethod
        def get_experiment(_name: str, include_experimental: bool = True):
            return _Experiment()

    monkeypatch.setattr(gv_hbi, "get_structure", lambda _name: _Structure())
    monkeypatch.setattr(
        gv_hbi,
        "_require_gv_calibrated_contract",
        lambda: ("ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c", "theta"),
    )

    with pytest.raises(ValueError, match="controls separate from calibrated parameters"):
        gv_hbi._validate_reference_manifest(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
                "calibrated_parameter_names": ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c", "theta"],
            },
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            dataset_id="gv:torsion:gv_rad2_height14_28:theta_0.03",
            control="theta_0.03",
            expected_controls={"theta": 0.03},
        )


def test_gv_hbi_validates_manifest_control_shapes() -> None:
    with pytest.raises(ValueError, match="surrogate manifest controls must be a mapping"):
        gv_hbi._validate_gv_control_values(
            controls=["theta=0.03"],
            experiment_name="torsion",
            label="GV surrogate manifest",
        )

    with pytest.raises(ValueError, match="unknown GV controls"):
        gv_hbi._validate_gv_control_values(
            controls={"theta": 0.03, "extra": 1.0},
            experiment_name="torsion",
            label="GV surrogate manifest",
        )

    with pytest.raises(ValueError, match="missing GV controls"):
        gv_hbi._validate_gv_control_values(
            controls={"tot_force": 500.0},
            experiment_name="stretching",
            label="GV surrogate manifest",
        )


def test_gv_hbi_resolves_optional_artifacts_and_requires_existing_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifests" / "surrogate.json"
    manifest_path.parent.mkdir()
    artifact_path = tmp_path / "manifests" / "artifact.pkl"
    artifact_path.write_text("artifact", encoding="utf-8")

    assert gv_hbi._resolve_optional_manifest_artifact(manifest_path, None) is None
    assert gv_hbi._resolve_optional_manifest_artifact(manifest_path, "") is None
    assert gv_hbi._resolve_optional_manifest_artifact(manifest_path, "artifact.pkl") == str(artifact_path.resolve())
    assert gv_hbi._require_existing_path(str(artifact_path), label="surrogate artifact") == str(artifact_path.resolve())

    with pytest.raises(FileNotFoundError, match="Missing GV surrogate artifact path"):
        gv_hbi._require_existing_path(None, label="surrogate artifact path")
    with pytest.raises(FileNotFoundError, match="Missing GV surrogate artifact"):
        gv_hbi._require_existing_path(str(tmp_path / "missing.pkl"), label="surrogate artifact")


def test_gv_hbi_artifact_probe_reports_loaded_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "model.pkl"
    artifact_path.write_text("artifact", encoding="utf-8")

    class _Model:
        pass

    def _fake_load_model_states(path: str):
        assert path == str(artifact_path)
        return _Model(), [0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.0], [1.0]

    monkeypatch.setitem(
        sys.modules,
        "meso_uq.surrogate.model",
        types.SimpleNamespace(load_model_states=_fake_load_model_states),
    )

    assert gv_hbi._build_execution_artifact_probe(artifact_path) == {
        "status": "loaded",
        "model_class": "_Model",
        "input_dim": 3,
        "output_dim": 1,
    }


def test_gv_hbi_artifact_probe_skips_when_model_dependency_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "model.pkl"
    artifact_path.write_text("artifact", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "meso_uq.surrogate.model", None)

    probe = gv_hbi._build_execution_artifact_probe(artifact_path)

    assert probe["status"] == "skipped_missing_dependency"
    assert "meso_uq.surrogate.model" in probe["reason"]


def test_gv_hbi_builds_reference_series_and_surrogate_inputs(tmp_path: Path) -> None:
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "ka,kb,mu,b1,b2,a3,a4,mu_l,c,radius,height,theta,torsion_coord,torsion_response,source_curve_id\n"
        "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2.0,14.28,0.03,0.0,1.0,curve_0\n"
        "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2.0,14.28,0.03,0.5,1.5,curve_0\n"
        "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,14.28,0.03,0.0,2.0,curve_1\n",
        encoding="utf-8",
    )

    series = gv_hbi._reference_series_from_dataset({}, dataset_csv, controls={"theta": 0.03})

    assert series["axis_column"] == "torsion_coord"
    assert series["target_column"] == "torsion_response"
    assert series["reference_points"] == [0.0, 0.5]
    assert series["reference_data"] == [1.0, 1.5]

    matrix = gv_hbi._build_gv_dnn_input_matrix(
        sample_parameters={
            "ka": 1.1,
            "kb": 1.2,
            "mu": 1.3,
            "b1": 1.4,
            "b2": 1.5,
            "a3": 1.6,
            "a4": 1.7,
            "mu_l": 1.8,
            "c": 1.9,
            "sigma": 0.05,
        },
        reference_rows=series["rows"],
        input_columns=series["input_columns"],
        controls={"theta": 0.03},
        geometry_parameters={"radius": 2.0, "height": 14.28},
    )

    assert matrix.shape == (2, 13)
    assert matrix[0].tolist() == pytest.approx(
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 14.28, 0.03, 0.0]
    )


def test_gv_hbi_rejects_invalid_reference_dataset_shapes(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("ka,response\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no rows"):
        gv_hbi._reference_series_from_dataset({}, empty_csv, controls={})

    no_axis_csv = tmp_path / "no_axis.csv"
    no_axis_csv.write_text(
        "ka,kb,mu,b1,b2,a3,a4,mu_l,c,radius,height,theta,torsion_response\n"
        "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2.0,14.28,0.03,1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one observable-axis"):
        gv_hbi._reference_series_from_dataset({}, no_axis_csv, controls={"theta": 0.03})

    with pytest.raises(ValueError, match="geometry_spec.parameters"):
        gv_hbi._geometry_parameters({"geometry_spec": {}})

    with pytest.raises(ValueError, match="geometry parameters are missing"):
        gv_hbi._geometry_parameters({"geometry_spec": {"parameters": {"radius": 2.0}}})

    no_response_csv = tmp_path / "no_response.csv"
    no_response_csv.write_text("ka,axis\n1.0,0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must declare target_column"):
        gv_hbi._reference_series_from_dataset({}, no_response_csv, controls={})

    missing_input_csv = tmp_path / "missing_input.csv"
    missing_input_csv.write_text("ka,response,axis\n1.0,2.0,0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing input columns"):
        gv_hbi._reference_series_from_dataset(
            {"target_column": "response", "input_columns": ["ka", "missing_axis"]},
            missing_input_csv,
            controls={},
        )

    with pytest.raises(ValueError, match="not a calibrated material/control input"):
        gv_hbi._build_gv_dnn_input_matrix(
            sample_parameters={"ka": 1.0, "d0": 0.1},
            reference_rows=[{"axis": "0.0"}],
            input_columns=["ka", "d0", "axis"],
            controls={},
            geometry_parameters={},
        )

    with pytest.raises(ValueError, match="Cannot resolve GV surrogate input column"):
        gv_hbi._build_gv_dnn_input_matrix(
            sample_parameters={"ka": 1.0},
            reference_rows=[{"axis": "0.0"}],
            input_columns=["ka", "unknown_axis"],
            controls={},
            geometry_parameters={},
        )


def test_gv_hbi_uses_response_suffix_target_fallback(tmp_path: Path) -> None:
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "ka,arc_length,custom_response\n"
        "1.0,0.0,2.0\n"
        "1.0,1.0,3.0\n",
        encoding="utf-8",
    )

    series = gv_hbi._reference_series_from_dataset({}, dataset_csv, controls={})

    assert series["axis_column"] == "arc_length"
    assert series["target_column"] == "custom_response"
    assert series["reference_data"] == [2.0, 3.0]


def test_gv_hbi_evaluate_reference_applies_optional_d0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gv_hbi, "_predict_gv_dnn", lambda _state, _x_raw: np.asarray([1.0, 2.0]))
    sample_data = {"Parameters": [0.5, 3.0]}

    gv_hbi._evaluate_gv_dnn_reference(
        sample_data,
        {
            "variable_names": ["ka", "d0"],
            "reference_rows": [{"axis": "0.0"}, {"axis": "1.0"}],
            "input_columns": ["ka", "axis"],
            "controls": {},
            "geometry_parameters": {},
            "model_state": object(),
        },
    )

    assert sample_data["Reference Evaluations"] == [4.0, 5.0]


def test_gv_phase1_dnn_execution_rejects_controls_as_korali_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surrogate_manifest = tmp_path / "surrogate_manifest.json"
    surrogate_manifest.write_text("{}", encoding="utf-8")
    reference_manifest = tmp_path / "reference_manifest.json"
    reference_manifest.write_text("{}", encoding="utf-8")
    execution_manifest = tmp_path / "gv_phase1_execution_manifest.json"
    execution_manifest.write_text(
        json.dumps(
            {
                "dataset": {
                    "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                    "control_values": {"theta": 0.03},
                },
                "phase1": {"variable_names": ["ka", "theta", "sigma"]},
                "provenance": {
                    "surrogate_manifest": str(surrogate_manifest),
                    "reference_manifest": str(reference_manifest),
                    "dataset_csv": str(tmp_path / "dataset.csv"),
                    "surrogate_artifact": str(tmp_path / "surrogate.pkl"),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gv_hbi,
        "write_gv_phase1_execution_manifest",
        lambda *args, **kwargs: execution_manifest,
    )

    with pytest.raises(ValueError, match="GV controls must not appear"):
        gv_hbi.run_gv_phase1_dnn_execution(
            {},
            [],
            repo_root=tmp_path,
            output_root=tmp_path,
            korali_module=object(),
            engine=object(),
        )


def test_gv_hbi_predicts_with_loaded_dnn_state() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[2.0, 3.0]])
        model.bias[:] = torch.tensor([0.5])
    predictions = gv_hbi._predict_gv_dnn(
        (
            model,
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
            np.asarray([1.0]),
            np.asarray([2.0]),
        ),
        np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64),
    )
    assert predictions.tolist() == pytest.approx([18.0, 2.0])


def test_gv_hbi_predicts_with_mocked_torch_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTensor:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def detach(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._values

    class FakeTorchInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def as_tensor(value: np.ndarray, dtype: object) -> np.ndarray:
            assert dtype == "float32"
            return np.asarray(value, dtype=np.float64)

        @staticmethod
        def inference_mode() -> FakeTorchInferenceMode:
            return FakeTorchInferenceMode()

    class FakeModel:
        def eval(self) -> None:
            return None

        def __call__(self, x_norm: np.ndarray) -> FakeTensor:
            return FakeTensor(np.sum(x_norm, axis=1) * 2.0)

    model = FakeModel()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    predictions = gv_hbi._predict_gv_dnn(
        (
            model,
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
            np.asarray([1.0]),
            np.asarray([2.0]),
        ),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    )
    assert predictions.tolist() == pytest.approx([13.0, 29.0])


def test_gv_phase1_execution_rejects_non_surrogate_config(tmp_path: Path) -> None:
    manifest_path = _surrogate_manifest(tmp_path)
    config = _gv_config(manifest_path)
    config["use_surrogate"] = False

    with pytest.raises(ValueError, match="requires use_surrogate=true"):
        gv_hbi.write_gv_phase1_execution_manifest(
            config,
            [_experiment(tmp_path)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_gv_phase1_execution_rejects_invalid_setup_manifest_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_path = tmp_path / "setup.json"

    def _write_setup(payload: dict[str, object]) -> None:
        setup_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        gv_hbi,
        "write_gv_phase1_setup_manifest",
        lambda *args, **kwargs: setup_path,
    )
    config = _gv_config(_surrogate_manifest(tmp_path))

    _write_setup({"datasets": []})
    with pytest.raises(NotImplementedError, match="exactly one GV dataset lane"):
        gv_hbi.write_gv_phase1_execution_manifest(
            config,
            [_experiment(tmp_path)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )

    _write_setup({"datasets": [{"dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03"}]})
    with pytest.raises(ValueError, match="missing surrogate metadata"):
        gv_hbi.write_gv_phase1_execution_manifest(
            config,
            [_experiment(tmp_path)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_gv_phase1_execution_rejects_missing_artifacts_and_bad_training_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_path = tmp_path / "setup.json"
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
            }
        ),
        encoding="utf-8",
    )
    artifact_path = tmp_path / "model.pkl"
    artifact_path.write_text("artifact", encoding="utf-8")
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("theta,response\n0.03,0.1\n", encoding="utf-8")
    training_report_path = tmp_path / "training.json"
    training_report_path.write_text("[]", encoding="utf-8")
    surrogate_path = tmp_path / "surrogate.json"

    setup_payload = {
        "enabled_by": "config:experimental_gv_hbi",
        "output_root": str(tmp_path / "out"),
        "phase1": {"variable_names": ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c", "sigma"]},
        "controls": {"names": ["theta"], "configured": ["theta_0.03"], "policy": "fixed"},
        "datasets": [
            {
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
                "experiment": "torsion",
                "experiment_id": "gv:torsion",
                "geometry": "gv_rad2_height14_28",
                "control": "theta_0.03",
                "surrogate": {
                    "manifest": str(surrogate_path),
                    "reference_manifest": str(reference_path),
                    "artifact": str(artifact_path),
                    "backend": "dnn",
                    "control_values": {"theta": 0.03},
                },
            }
        ],
    }
    setup_path.write_text(json.dumps(setup_payload), encoding="utf-8")
    monkeypatch.setattr(
        gv_hbi,
        "write_gv_phase1_setup_manifest",
        lambda *args, **kwargs: setup_path,
    )
    config = _gv_config(tmp_path / "config_manifest_unused.json")

    surrogate_path.write_text(json.dumps({"structure": "gv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing artifacts"):
        gv_hbi.write_gv_phase1_execution_manifest(
            config,
            [_experiment(tmp_path)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )

    surrogate_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "artifacts": {
                    "reference_manifest": str(reference_path),
                    "model_path": str(artifact_path),
                    "dataset_csv": str(dataset_path),
                    "training_report": str(training_report_path),
                },
            }
        ),
        encoding="utf-8",
    )
    original_load_json = gv_hbi._load_json

    def _load_json_with_non_object_training_report(path: Path, *, label: str):
        if Path(path) == training_report_path:
            return []
        return original_load_json(path, label=label)

    monkeypatch.setattr(gv_hbi, "_load_json", _load_json_with_non_object_training_report)
    with pytest.raises(ValueError, match="training report must be a JSON object"):
        gv_hbi.write_gv_phase1_execution_manifest(
            config,
            [_experiment(tmp_path)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_gv_hbi_build_manifest_rejects_empty_selection_and_control_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _surrogate_manifest(tmp_path)
    config = _gv_config(manifest_path)

    with pytest.raises(ValueError, match="No enabled GV experiments"):
        gv_hbi.build_gv_phase1_setup_manifest(
            config,
            [_experiment(tmp_path, enabled=False)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )

    monkeypatch.setattr(gv_hbi, "_gv_dataset_entries", lambda *args, **kwargs: [])
    monkeypatch.setattr(gv_hbi, "phase1_prior_specs", lambda _config: [("theta", [0.0, 1.0])])
    monkeypatch.setattr(gv_hbi, "phase2_hyperprior_specs", lambda _config: [])
    monkeypatch.setattr(gv_hbi, "active_hierarchical_variable_names", lambda _config: [])
    with pytest.raises(ValueError, match="GV controls must not appear"):
        gv_hbi.build_gv_phase1_setup_manifest(
            config,
            [_experiment(tmp_path)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_gv_hbi_build_manifest_rejects_controls_in_calibrated_or_nuisance_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _surrogate_manifest(tmp_path)
    config = _gv_config(manifest_path)
    class _FakeExperiment:
        control_names = ("sigma",)

    class _FakeContract:
        calibrated_names = gv_hbi.GV_PHASE1_CALIBRATED_PARAMETERS
        nuisance_names = gv_hbi.GV_PHASE1_NUISANCE_PARAMETERS

    class _FakeStructure:
        parameter_contract = _FakeContract()

        @staticmethod
        def get_experiment(_name: str, include_experimental: bool = True) -> _FakeExperiment:
            return _FakeExperiment()

    original_get_structure = gv_hbi.get_structure
    monkeypatch.setattr(
        gv_hbi,
        "get_structure",
        lambda name: _FakeStructure() if name == "gv" else original_get_structure(name),
    )
    monkeypatch.setattr(gv_hbi, "_gv_dataset_entries", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        gv_hbi,
        "phase1_prior_specs",
        lambda _config: [
            ("ka", [0.1, 1.1]),
            ("kb", [0.2, 1.2]),
            ("mu", [0.3, 1.3]),
            ("b1", [0.4, 1.4]),
            ("b2", [0.5, 1.5]),
            ("a3", [0.6, 1.6]),
            ("a4", [0.7, 1.7]),
            ("mu_l", [0.8, 1.8]),
            ("c", [0.9, 1.9]),
        ],
    )
    monkeypatch.setattr(gv_hbi, "phase2_hyperprior_specs", lambda _config: [])
    monkeypatch.setattr(gv_hbi, "active_hierarchical_variable_names", lambda _config: [])

    with pytest.raises(ValueError, match="GV controls must not appear in calibrated or nuisance variables"):
        gv_hbi.build_gv_phase1_setup_manifest(
            config,
            [_experiment(tmp_path)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_gv_hbi_build_manifest_rejects_missing_required_nuisance_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _surrogate_manifest(tmp_path)
    config = _gv_config(manifest_path)

    class _FakeContract:
        calibrated_names = gv_hbi.GV_PHASE1_CALIBRATED_PARAMETERS
        nuisance_names: tuple[str, ...] = ()

    class _FakeStructure:
        parameter_contract = _FakeContract()

    monkeypatch.setattr(gv_hbi, "get_structure", lambda _name: _FakeStructure())
    monkeypatch.setattr(gv_hbi, "_gv_dataset_entries", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        gv_hbi,
        "phase1_prior_specs",
        lambda _config: [
            ("ka", [0.1, 1.1]),
            ("kb", [0.2, 1.2]),
            ("mu", [0.3, 1.3]),
            ("b1", [0.4, 1.4]),
            ("b2", [0.5, 1.5]),
            ("a3", [0.6, 1.6]),
            ("a4", [0.7, 1.7]),
            ("mu_l", [0.8, 1.8]),
            ("c", [0.9, 1.9]),
        ],
    )
    monkeypatch.setattr(gv_hbi, "phase2_hyperprior_specs", lambda _config: [])

    with pytest.raises(ValueError, match="missing required nuisance parameter"):
        gv_hbi.build_gv_phase1_setup_manifest(
            config,
            [_experiment(tmp_path)],
            repo_root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_gv_hbi_bounds_payload_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="Unsupported bounds spec shape"):
        gv_hbi._bounds_payload([("ka", [0.1, 1.1], [0.01, 0.2], "extra")])
