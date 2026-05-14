from __future__ import annotations

from pathlib import Path

import pytest

from meso_uq.core import AgentFamily, Modality, ModelBackend
from meso_uq.surrogate.emb_workflows import (
    DatasetSplitMetadata,
    best_artifact_suffix,
    get_emb_surrogate_workflow,
    list_emb_surrogate_workflows,
    resolve_emb_surrogate_backend,
    run_emb_bnn_training_cli,
    run_emb_dnn_training_cli,
)


def test_emb_surrogate_workflow_specs_cover_compression_and_indentation() -> None:
    specs = {spec.modality: spec for spec in list_emb_surrogate_workflows()}

    assert set(specs) == {Modality.COMPRESSION, Modality.INDENTATION}
    assert specs[Modality.COMPRESSION].family is AgentFamily.EMB
    assert specs[Modality.COMPRESSION].input_cols == ("Yt", "kb", "b1", "b2", "a3", "a4", "disp")
    assert specs[Modality.COMPRESSION].target_col == "F"
    assert specs[Modality.INDENTATION].input_cols == ("Yt", "kb", "b1", "b2", "a3", "a4", "F")
    assert specs[Modality.INDENTATION].target_col == "disp"
    assert specs[Modality.INDENTATION].has_indentation_loader_knobs is True


def test_emb_surrogate_checkpoint_metadata_is_backend_aware() -> None:
    spec = get_emb_surrogate_workflow(Modality.INDENTATION)
    metadata = spec.checkpoint_metadata

    assert metadata["agent_family"] == "emb"
    assert metadata["modality"] == "indentation"
    assert metadata["default_dnn_out"] == "trained/microbubble_disp_BEST.pkl"
    assert metadata["default_bnn_out"] == "trained/microbubble_displacement_BNN.pt"
    assert metadata["supported_backends"] == ["dnn", "bnn", "pyro_bnn"]


def test_emb_surrogate_backend_resolution_and_invalid_inputs() -> None:
    assert resolve_emb_surrogate_backend("dnn") is ModelBackend.DNN
    assert resolve_emb_surrogate_backend(ModelBackend.BNN) is ModelBackend.BNN

    with pytest.raises(ValueError, match="Unsupported EMB surrogate backend"):
        resolve_emb_surrogate_backend("dpd")
    with pytest.raises(ValueError, match="compression, indentation"):
        get_emb_surrogate_workflow("buckling")


def test_dataset_split_metadata_validates_fraction() -> None:
    assert DatasetSplitMetadata(seed=7, val_fraction=0.25).as_dict() == {
        "seed": 7,
        "val_fraction": 0.25,
    }
    with pytest.raises(ValueError, match="val_fraction"):
        DatasetSplitMetadata(seed=7, val_fraction=1.0)


def test_run_dnn_training_cli_uses_shared_compression_contract(tmp_path: Path) -> None:
    data = tmp_path / "compression.dat"
    out = tmp_path / "out.pkl"
    report = tmp_path / "report.json"
    data.write_text("placeholder", encoding="utf-8")
    captured: dict[str, object] = {}

    def reader(path: str, *, curve_axis_name: str, value_name: str):
        captured["reader"] = (path, curve_axis_name, value_name)
        return {"df": True}

    def trainer(df, **kwargs):
        captured["trainer"] = (df, kwargs)
        return {"out": kwargs["out_path"], "train_loss": 1.0, "val_loss": 2.0}

    result = run_emb_dnn_training_cli(
        Modality.COMPRESSION,
        argv=[str(data), "--out", str(out), "--report-path", str(report), "--seed", "11"],
        reader=reader,
        trainer=trainer,
    )

    assert result["out"] == str(out)
    assert captured["reader"] == (str(data), "disp", "F")
    _, kwargs = captured["trainer"]
    assert kwargs["input_cols"] == ["Yt", "kb", "b1", "b2", "a3", "a4", "disp"]
    assert kwargs["target_col"] == "F"
    assert kwargs["seed"] == 11


def test_run_bnn_training_cli_preserves_indentation_loader_knobs(tmp_path: Path) -> None:
    data = tmp_path / "indentation.dat"
    dnn = tmp_path / "dnn.pkl"
    out = tmp_path / "out.pt"
    data.write_text("placeholder", encoding="utf-8")
    dnn.write_text("placeholder", encoding="utf-8")
    captured: dict[str, object] = {}

    def reader(path: str, *, disp_source: str, rupture_ratio_threshold):
        captured["reader"] = (path, disp_source, rupture_ratio_threshold)
        return {"df": True}

    def trainer(df, **kwargs):
        captured["trainer"] = (df, kwargs)
        return {
            "out": kwargs["out_path"],
            "training": {
                "final_val_rmse": 1.0,
                "parity_dnn_rmse": 1.0,
                "parity_ratio": 1.0,
                "step_count": 1,
                "stop_reason": "max_steps",
            },
        }

    result = run_emb_bnn_training_cli(
        Modality.INDENTATION,
        argv=[
            str(data),
            "--dnn-reference",
            str(dnn),
            "--out",
            str(out),
            "--disp-source",
            "displacement",
            "--rupture-ratio",
            "0",
            "--obs-noise",
            "0.33",
        ],
        reader=reader,
        trainer=trainer,
    )

    assert result["out"] == str(out.resolve())
    path, disp_source, rupture_ratio = captured["reader"]
    assert path == str(data.resolve())
    assert disp_source == "displacement"
    assert rupture_ratio is None
    _, kwargs = captured["trainer"]
    assert kwargs["input_cols"] == ["Yt", "kb", "b1", "b2", "a3", "a4", "F"]
    assert kwargs["target_col"] == "disp"
    assert kwargs["obs_noise_prior_scale"] == pytest.approx(0.33)


def test_best_artifact_suffix_matches_legacy_group_holdout_names() -> None:
    assert best_artifact_suffix("dnn") == ".pkl"
    assert best_artifact_suffix("bnn") == ".pt"
