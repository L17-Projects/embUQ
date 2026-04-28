from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _write_certification_inputs(
    certification_root: Path,
    *,
    certified: bool = True,
    candidates: list[dict[str, object]] | None = None,
    per_dataset_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    certification_root.mkdir(parents=True, exist_ok=True)
    source = certification_root / "candidate.pt"
    target = certification_root / "trained" / "tracked.pt"
    source.write_text("artifact", encoding="utf-8")

    if candidates is None:
        candidates = [
            {
                "dataset_name": "compression_2.1um",
                "candidate_seed": 101,
                "candidate_artifact_path": str(source),
                "tracked_bnn_artifact_path": str(target),
            }
        ]
    (certification_root / "promotion_candidates.json").write_text(
        json.dumps(candidates, indent=2),
        encoding="utf-8",
    )
    rows = per_dataset_rows
    if rows is None:
        rows = [
            {
                "dataset_name": "compression_2.1um",
                "certified": certified,
            }
        ]
    pd.DataFrame(rows).to_csv(certification_root / "certification_per_dataset.csv", index=False)
    return source, target


def test_promote_certified_bnn_dry_run_writes_manifest_without_copy(tmp_path: Path) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/promote_certified_bnn.py"),
        "promote_certified_bnn_dry_run_test",
    )
    certification_root = tmp_path / "cert"
    _source, target = _write_certification_inputs(certification_root)

    rc = module.main(["--certification-root", str(certification_root), "--dry-run"])
    assert rc == 0
    assert not target.exists()
    manifest = json.loads((certification_root / "promotion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"


def test_promote_certified_bnn_copies_certified_artifact(tmp_path: Path) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/promote_certified_bnn.py"),
        "promote_certified_bnn_copy_test",
    )
    certification_root = tmp_path / "cert"
    source, target = _write_certification_inputs(certification_root)

    rc = module.main(["--certification-root", str(certification_root)])
    assert rc == 0
    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_promote_certified_bnn_rejects_uncertified_datasets(tmp_path: Path) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/promote_certified_bnn.py"),
        "promote_certified_bnn_reject_test",
    )
    certification_root = tmp_path / "cert"
    _write_certification_inputs(certification_root, certified=False)

    with pytest.raises(SystemExit, match="uncertified datasets remain"):
        module.main(["--certification-root", str(certification_root)])


def test_promote_certified_bnn_requires_explicit_certified_row(tmp_path: Path) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/promote_certified_bnn.py"),
        "promote_certified_bnn_missing_row_test",
    )
    certification_root = tmp_path / "cert"
    _write_certification_inputs(
        certification_root,
        per_dataset_rows=[{"dataset_name": "compression_3.2um", "certified": True}],
    )

    with pytest.raises(SystemExit, match="lacks explicit certified row"):
        module.main(["--certification-root", str(certification_root)])


def test_promote_certified_bnn_prevalidates_all_sources_before_copy(tmp_path: Path) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/promote_certified_bnn.py"),
        "promote_certified_bnn_prevalidate_test",
    )
    certification_root = tmp_path / "cert"
    first_source = certification_root / "candidate_a.pt"
    first_target = certification_root / "trained" / "tracked_a.pt"
    missing_source = certification_root / "candidate_b.pt"
    second_target = certification_root / "trained" / "tracked_b.pt"
    certification_root.mkdir(parents=True, exist_ok=True)
    first_source.write_text("artifact-a", encoding="utf-8")
    first_target.parent.mkdir(parents=True, exist_ok=True)
    first_target.write_text("old-a", encoding="utf-8")

    _write_certification_inputs(
        certification_root,
        candidates=[
            {
                "dataset_name": "compression_2.1um",
                "candidate_seed": 101,
                "candidate_artifact_path": str(first_source),
                "tracked_bnn_artifact_path": str(first_target),
            },
            {
                "dataset_name": "compression_3.2um",
                "candidate_seed": 102,
                "candidate_artifact_path": str(missing_source),
                "tracked_bnn_artifact_path": str(second_target),
            },
        ],
        per_dataset_rows=[
            {"dataset_name": "compression_2.1um", "certified": True},
            {"dataset_name": "compression_3.2um", "certified": True},
        ],
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        module.main(["--certification-root", str(certification_root)])

    assert first_target.read_text(encoding="utf-8") == "old-a"
    assert not second_target.exists()
    assert not (certification_root / "promotion_manifest.json").exists()


def test_promote_certified_bnn_rolls_back_if_late_replace_fails(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/promote_certified_bnn.py"),
        "promote_certified_bnn_replace_rollback_test",
    )
    certification_root = tmp_path / "cert"
    first_source = certification_root / "candidate_a.pt"
    first_target = certification_root / "trained" / "tracked_a.pt"
    second_source = certification_root / "candidate_b.pt"
    second_target = certification_root / "trained" / "tracked_b.pt"
    certification_root.mkdir(parents=True, exist_ok=True)
    first_source.write_text("artifact-a", encoding="utf-8")
    second_source.write_text("artifact-b", encoding="utf-8")
    first_target.parent.mkdir(parents=True, exist_ok=True)
    first_target.write_text("old-a", encoding="utf-8")
    second_target.write_text("old-b", encoding="utf-8")

    _write_certification_inputs(
        certification_root,
        candidates=[
            {
                "dataset_name": "compression_2.1um",
                "candidate_seed": 101,
                "candidate_artifact_path": str(first_source),
                "tracked_bnn_artifact_path": str(first_target),
            },
            {
                "dataset_name": "compression_3.2um",
                "candidate_seed": 102,
                "candidate_artifact_path": str(second_source),
                "tracked_bnn_artifact_path": str(second_target),
            },
        ],
        per_dataset_rows=[
            {"dataset_name": "compression_2.1um", "certified": True},
            {"dataset_name": "compression_3.2um", "certified": True},
        ],
    )

    original_replace = module.Path.replace

    def fake_replace(self, target):  # noqa: ANN001
        target_path = Path(target)
        if self.suffix == ".tmp" and target_path == second_target:
            raise OSError("late replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(module.Path, "replace", fake_replace, raising=False)

    with pytest.raises(OSError, match="late replace failure"):
        module.main(["--certification-root", str(certification_root)])

    assert first_target.read_text(encoding="utf-8") == "old-a"
    assert second_target.read_text(encoding="utf-8") == "old-b"
    assert not (certification_root / "promotion_manifest.json").exists()


def test_hpc_promote_certified_bnn_wrapper_dispatches_to_selected_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/promote_certified_bnn.py"),
        "hpc_promote_certified_bnn_dispatch_test",
    )
    monkeypatch.setenv("HPC_SITE", "karolina")
    captured: list[list[str]] = []

    def _fake_call(cmd):  # noqa: ANN001
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    rc = module.main(["--certification-root", "dummy"])
    assert rc == 0
    assert captured
    assert sys.executable in captured[0][0]
    assert "scripts/platforms/karolina/promote_certified_bnn.py" in " ".join(captured[0])


def test_hpc_promote_certified_bnn_wrapper_rejects_unknown_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/promote_certified_bnn.py"),
        "hpc_promote_certified_bnn_invalid_site_test",
    )
    monkeypatch.setenv("HPC_SITE", "unknown")
    with pytest.raises(SystemExit, match="Unsupported HPC_SITE"):
        module.main([])
