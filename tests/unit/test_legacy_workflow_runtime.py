from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

from meso_uq.core import Modality
from meso_uq.workflows import legacy
from meso_uq.workflows.legacy import (
    legacy_evalkit_import_paths,
    list_legacy_workflow_surfaces,
    prepend_legacy_evalkit_paths,
    resolve_legacy_inference_config_path,
    resolve_legacy_project_root,
    resolve_legacy_relative_path,
    resolve_legacy_surrogate_backend,
    resolve_legacy_surrogate_runtime,
    resolve_legacy_surrogate_trained_dir,
    warn_legacy_workflow_surface,
)


def test_legacy_project_root_resolves_from_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project = tmp_path / "repo"
    (project / "compression" / "src").mkdir(parents=True)
    workdir = project / "nested" / "child"
    workdir.mkdir(parents=True)

    monkeypatch.chdir(workdir)

    assert (
        resolve_legacy_project_root(marker_parts=("compression", "src"), start_path=Path.cwd())
        == project
    )


def test_legacy_project_root_uses_anchor_when_cwd_is_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "repo"
    anchor = project / "compression" / "evalkit" / "posterior_compression.py"
    marker = project / "compression" / "src"
    anchor.parent.mkdir(parents=True)
    marker.mkdir(parents=True)
    anchor.write_text("# anchor\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(elsewhere)

    assert resolve_legacy_project_root(
        marker_parts=("compression", "src"),
        start_path=Path.cwd(),
        anchor_file=anchor,
    ) == project


def test_legacy_project_root_raises_with_marker_name(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="indentation/src"):
        resolve_legacy_project_root(
            marker_parts=("indentation", "src"),
            start_path=tmp_path,
            max_parent_depth=0,
        )


def test_legacy_inference_config_prefers_env_override(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    override = project / "custom.yaml"
    override.parent.mkdir()
    override.write_text("surrogate: {}\n", encoding="utf-8")

    assert resolve_legacy_inference_config_path(
        project,
        Modality.COMPRESSION,
        env={"HUQ_INFERENCE_CONFIG": "custom.yaml"},
    ) == override


def test_legacy_inference_config_falls_back_to_modality_path(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    config = project / "inference" / "configs" / "production" / "inference_config_indentation.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("surrogate: {}\n", encoding="utf-8")

    assert resolve_legacy_inference_config_path(project, "indentation", env={}) == config


def test_legacy_surrogate_runtime_validation() -> None:
    runtime = resolve_legacy_surrogate_runtime(
        {"surrogate": {"backend": "BNN", "predictive_mc_samples": 16, "predictive_mc_chunk_size": 4}}
    )
    assert runtime.as_tuple() == ("bnn", 16, 4)

    with pytest.raises(ValueError, match="Unsupported surrogate backend"):
        resolve_legacy_surrogate_runtime({"surrogate": {"backend": "foo"}})
    with pytest.raises(ValueError, match="predictive_mc_samples"):
        resolve_legacy_surrogate_runtime({"surrogate": {"predictive_mc_samples": 0}})


def test_legacy_surrogate_backend_ignores_bnn_mc_knobs_for_backend_only_callers() -> None:
    assert (
        resolve_legacy_surrogate_backend(
            {
                "surrogate": {
                    "backend": "dnn",
                    "predictive_mc_samples": None,
                    "predictive_mc_chunk_size": None,
                }
            }
        )
        == "dnn"
    )

    with pytest.raises(ValueError, match="Unsupported surrogate backend"):
        resolve_legacy_surrogate_backend({"surrogate": {"backend": "foo"}})


def test_legacy_surrogate_trained_dir_is_modality_specific(tmp_path: Path) -> None:
    assert resolve_legacy_surrogate_trained_dir(tmp_path, "compression", 2.9) == (
        tmp_path / "compression" / "surrogate" / "diameters" / "2.9um" / "trained"
    )
    assert resolve_legacy_surrogate_trained_dir(tmp_path, "indentation", 3.2) == (
        tmp_path / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained"
    )


def test_legacy_relative_path_resolves_against_project_root(tmp_path: Path) -> None:
    assert resolve_legacy_relative_path(tmp_path, "out") == (tmp_path / "out").resolve()


def test_legacy_evalkit_path_prepend_preserves_historical_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "path", ["sentinel"])

    inserted = prepend_legacy_evalkit_paths(tmp_path)

    assert inserted == tuple(str(path) for path in legacy_evalkit_import_paths(tmp_path))
    assert sys.path[:4] == [
        str(tmp_path / "indentation" / "evalkit"),
        str(tmp_path / "indentation"),
        str(tmp_path / "compression" / "evalkit"),
        str(tmp_path / "compression"),
    ]


def test_legacy_surface_inventory_covers_major_workflow_families() -> None:
    surfaces = list_legacy_workflow_surfaces()
    families = {surface.family for surface in surfaces}
    assert {"compression", "indentation", "inference", "reduced", "propagation"} <= families
    assert all(surface.replacement_api for surface in surfaces)


def test_legacy_warning_is_once_per_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy._WARNED_SURFACES.clear()
    monkeypatch.delenv("MESOUQ_SUPPRESS_LEGACY_WARNINGS", raising=False)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warn_legacy_workflow_surface("propagation/scripts/run_phase1_propagation.py")
        warn_legacy_workflow_surface("propagation/scripts/run_phase1_propagation.py")

    assert len(seen) == 1
    assert "meso_uq.postprocess.propagation" in str(seen[0].message)


def test_legacy_warning_can_be_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy._WARNED_SURFACES.clear()
    monkeypatch.setenv("MESOUQ_SUPPRESS_LEGACY_WARNINGS", "1")

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warn_legacy_workflow_surface("propagation/scripts/run_phase3b_propagation.py")

    assert seen == []
