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
) -> tuple[Path, Path]:
    certification_root.mkdir(parents=True, exist_ok=True)
    source = certification_root / "candidate.pt"
    target = certification_root / "trained" / "tracked.pt"
    source.write_text("artifact", encoding="utf-8")

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
    pd.DataFrame(
        [
            {
                "dataset_name": "compression_2.1um",
                "certified": certified,
            }
        ]
    ).to_csv(certification_root / "certification_per_dataset.csv", index=False)
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
