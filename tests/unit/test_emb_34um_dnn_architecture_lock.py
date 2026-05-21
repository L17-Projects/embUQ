from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from meso_uq.active_learning.emb_34um_dnn_architecture_lock import (
    EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_SEEDS,
    EMB_34UM_DNN_ARCHITECTURE_LOCK_MANIFEST_FILENAME,
    EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_FILENAME,
    EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_SIDECAR_FILENAME,
    EMB_34UM_DNN_ARCHITECTURE_LOCK_REPORT_FILENAME,
    EMB_34UM_DNN_ARCHITECTURE_LOCK_SCHEMA_VERSION,
    EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE,
    build_emb_34um_dnn_causal_architecture_lock_manifest,
    write_emb_34um_dnn_causal_architecture_lock_artifacts,
)


def test_build_manifest_records_fixed_dnn_and_seeds() -> None:
    seeds = tuple(range(11, 21))
    manifest = build_emb_34um_dnn_causal_architecture_lock_manifest(
        selected_architecture=EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE,
        ensemble_size=10,
        ensemble_seeds=seeds,
        timing_canary={"required": True},
        source_evidence={"workflow": "unit-test"},
    )

    assert manifest["schema_version"] == EMB_34UM_DNN_ARCHITECTURE_LOCK_SCHEMA_VERSION
    assert manifest["selected_architecture"] == EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE
    assert manifest["ensemble_size"] == 10
    assert manifest["ensemble_seeds"] == list(seeds)
    evidence = manifest["source_evidence"]
    assert evidence["selected_architecture"] == EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE
    assert evidence["workflow"] == "unit-test"
    assert manifest["timing_canary"]["required"] is True
    assert manifest["timing_canary"]["point_count"] == 8


def test_build_manifest_rejects_non_fixed_architecture() -> None:
    with pytest.raises(ValueError, match="fixed DNN architecture"):
        build_emb_34um_dnn_causal_architecture_lock_manifest(
            selected_architecture="cnn_small",
            timing_canary={"required": True},
        )


def test_write_manifest_artifacts_and_plot_output(tmp_path: Path) -> None:
    output_root = tmp_path / "arch-lock"
    artifacts = write_emb_34um_dnn_causal_architecture_lock_artifacts(
        output_root,
        ensemble_seeds=EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_SEEDS,
        timing_canary={"required": True},
        source_evidence={"workflow": "unit-test"},
        training_hyperparameters={
            "batch_size": 32,
            "epochs": 3,
            "learning_rate": 1e-4,
        },
        training_history=(
            {"epoch": 0, "train_loss": 1.0, "validation_loss": 1.2},
            {"epoch": 1, "train_loss": 0.6, "validation_loss": 0.9},
        ),
    )

    manifest_path = output_root / EMB_34UM_DNN_ARCHITECTURE_LOCK_MANIFEST_FILENAME
    report_path = output_root / EMB_34UM_DNN_ARCHITECTURE_LOCK_REPORT_FILENAME
    plot_path = output_root / EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_FILENAME
    sidecar_path = output_root / EMB_34UM_DNN_ARCHITECTURE_LOCK_PLOT_SIDECAR_FILENAME

    assert artifacts.manifest_path == manifest_path
    assert artifacts.report_path == report_path
    assert artifacts.plot_path == plot_path
    assert artifacts.plot_sidecar_path == sidecar_path
    assert manifest_path.is_file()
    assert report_path.is_file()
    assert plot_path.is_file()
    assert sidecar_path.is_file()
    assert plot_path.stat().st_size > 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert report["selected_architecture"] == manifest["selected_architecture"]
    assert report["ensemble_signature"]["count"] == manifest["ensemble_size"]
    assert report["training_history_summary"]["epoch_count"] == 2
    assert sidecar["evidence_summary"]["selected_architecture"] == manifest["selected_architecture"]
    assert sidecar["status"] in {"rendered", "fallback_png"}


def test_write_artifacts_canary_override_reason(tmp_path: Path) -> None:
    output_root = tmp_path / "arch-lock-override"
    artifacts = write_emb_34um_dnn_causal_architecture_lock_artifacts(
        output_root,
        timing_canary={"required": True, "point_count": 5, "override_reason": "resource constrained"},
        source_evidence={"workflow": "unit-test"},
        training_history=(
            {"epoch": 0, "train_loss": 0.8, "validation_loss": 0.9},
        ),
        include_plot=False,
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["timing_canary"]["point_count"] == 5
    assert manifest["timing_canary"]["override_reason"] == "resource constrained"


def test_plot_sidecar_is_written_even_without_history(tmp_path: Path) -> None:
    output_root = tmp_path / "arch-lock-no-history"
    artifacts = write_emb_34um_dnn_causal_architecture_lock_artifacts(
        output_root,
        timing_canary={"required": True},
        include_plot=False,
    )

    sidecar = json.loads(artifacts.plot_sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "fallback_png"
    assert sidecar["reason"] == "disabled_or_no_history"
    assert sidecar["epoch_count"] == 0
    assert artifacts.plot_path.stat().st_size > 0
