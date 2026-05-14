from __future__ import annotations

from pathlib import Path

import pytest

from meso_uq.structures.gv.staging import (
    MissingTemplateError,
    UnsafeOutputPathError,
    UnsupportedStagingTargetError,
    reserve_run_directory,
    stage_gv_runtime,
)
from meso_uq.structures.gv.staging_manifest import TemplateSpec


def test_staging_path_construction_and_copy_from_temp_source(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    (source_root / "templates").mkdir(parents=True)
    (source_root / "templates" / "commands.txt").write_text("echo dry-run\n", encoding="utf-8")
    (source_root / "templates" / "run.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    manifest = stage_gv_runtime(
        modality="gv",
        experiment="stretching",
        output_root=output_root,
        mirheo_source_root=source_root,
        templates=(
            TemplateSpec("templates/commands.txt", "runtime/commands.txt"),
            TemplateSpec("templates/run.sh", "runtime/run.sh"),
        ),
        run_id="run-contract-001",
        dry_run=False,
    )

    assert manifest.run_root == output_root / "gv" / "stretching" / "run-contract-001"
    staged_paths = {entry.relative_destination_path: entry for entry in manifest.staged_files}
    assert (manifest.run_root / "runtime" / "commands.txt").is_file()
    assert (manifest.run_root / "runtime" / "run.sh").is_file()
    assert staged_paths["runtime/commands.txt"].size_bytes > 0


def test_missing_template_raises_contract_error(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(MissingTemplateError):
        stage_gv_runtime(
            modality="gv",
            experiment="stretching",
            output_root=tmp_path / "output",
            mirheo_source_root=source_root,
            templates=(TemplateSpec("templates/missing.txt", "runtime/missing.txt"),),
            run_id="run-missing",
            dry_run=True,
        )


def test_rejects_template_source_that_escapes_mirheo_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")

    with pytest.raises(UnsafeOutputPathError, match="source escapes Mirheo source root"):
        stage_gv_runtime(
            modality="gv",
            experiment="stretching",
            output_root=tmp_path / "output",
            mirheo_source_root=source_root,
            templates=(TemplateSpec("../outside.txt", "runtime/outside.txt"),),
            run_id="run-source-escape",
            dry_run=True,
        )


def test_rejects_unsafe_output_path_in_repo_source_tree() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "docs"

    with pytest.raises(UnsafeOutputPathError):
        stage_gv_runtime(
            modality="gv",
            experiment="stretching",
            output_root=repo_root / "src" / "unsafe",
            mirheo_source_root=source_root,
            templates=(),
            run_id="run-unsafe",
            dry_run=True,
        )


def test_rejects_existing_non_empty_output_directory_by_default(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    output_root.mkdir()
    (output_root / "already_there.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(UnsafeOutputPathError):
        stage_gv_runtime(
            modality="gv",
            experiment="stretching",
            output_root=output_root,
            mirheo_source_root=source_root,
            templates=(),
            run_id="run-occupied",
            dry_run=True,
        )


def test_rejects_unsupported_modality_and_experiment(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(UnsupportedStagingTargetError):
        stage_gv_runtime(
            modality="hbi",
            experiment="stretching",
            output_root=tmp_path / "output-a",
            mirheo_source_root=source_root,
            templates=(),
        )
    with pytest.raises(UnsupportedStagingTargetError):
        stage_gv_runtime(
            modality="gv",
            experiment="unknown-experiment",
            output_root=tmp_path / "output-b",
            mirheo_source_root=source_root,
            templates=(),
        )


def test_reserve_run_directory_is_concurrent_safe_and_unique(tmp_path: Path) -> None:
    first_id, first_path = reserve_run_directory(
        output_root=tmp_path / "output",
        modality="gv",
        experiment="stretching",
    )
    second_id, second_path = reserve_run_directory(
        output_root=tmp_path / "output",
        modality="gv",
        experiment="stretching",
    )

    assert first_id != second_id
    assert first_path.is_dir()
    assert second_path.is_dir()
    assert first_path != second_path


def test_dry_run_does_not_materialize_templates(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    (source_root / "templates").mkdir(parents=True)
    (source_root / "templates" / "commands.txt").write_text("echo only-plan\n", encoding="utf-8")

    manifest = stage_gv_runtime(
        modality="gv",
        experiment="stretching",
        output_root=output_root,
        mirheo_source_root=source_root,
        templates=(TemplateSpec("templates/commands.txt", "runtime/commands.txt"),),
        run_id="run-dry",
        dry_run=True,
    )

    assert manifest.dry_run
    assert manifest.run_root == output_root / "gv" / "stretching" / "run-dry"
    assert not manifest.run_root.exists()
