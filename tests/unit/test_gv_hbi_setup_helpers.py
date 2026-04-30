from __future__ import annotations

import json
from pathlib import Path

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


def test_gv_hbi_bounds_payload_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="Unsupported bounds spec shape"):
        gv_hbi._bounds_payload([("ka", [0.1, 1.1], [0.01, 0.2], "extra")])
