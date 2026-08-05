from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "workflows"
    / "emb"
    / "uq_emb"
    / "compile_manuscript.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("uq_emb_manuscript_replay", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_manuscript_stages_frozen_files_and_checks_baseline(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    files = {
        "main.tex": b"main source",
        "main_marked_up.tex": b"marked source",
        "si.tex": b"si source",
        "main.pdf": b"frozen main",
        "main_marked_up.pdf": b"frozen marked",
    }
    entries = []
    for relative, content in files.items():
        path = bundle / relative
        path.write_bytes(content)
        entries.append(
            {
                "path": relative,
                "size_bytes": len(content),
                "sha256": module._sha256(path),
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"files": entries}), encoding="utf-8")

    monkeypatch.setattr(
        module,
        "verify_snapshot",
        lambda **_kwargs: {"status": "PASS", "file_count": len(entries)},
    )
    provenance_calls = []
    monkeypatch.setattr(
        module,
        "replay_receipt_provenance",
        lambda **kwargs: provenance_calls.append(kwargs) or {},
    )
    real_run = subprocess.run

    def fake_run(command, *, cwd, check, **kwargs):
        if command[0] == "git":
            return real_run(command, cwd=cwd, check=check, **kwargs)
        assert check is True
        env = kwargs["env"]
        assert env["SOURCE_DATE_EPOCH"] == "1785834096"
        if command[0] != "pdflatex":
            return
        stem = Path(command[-1]).stem
        (cwd / f"{stem}.pdf").write_bytes(f"compiled {stem}".encode())
        (cwd / f"{stem}.log").write_text("clean log\n", encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_pdf_pages", lambda _path: 1)
    monkeypatch.setattr(module, "_text_sha256", lambda path, _root: f"text-{path.stem}")
    monkeypatch.setattr(module, "_failed_log_lines", lambda _path: [])

    build_root = tmp_path / "build"
    receipt = module.compile_manuscript(
        bundle_root=bundle,
        manifest_path=manifest_path,
        build_root=build_root,
        pdflatex="pdflatex",
        bibtex="bibtex",
    )

    assert receipt["status"] == "passed"
    assert [row["target"] for row in receipt["commands"]] == list(module.TARGETS)
    assert receipt["baseline_comparison"]["main.pdf"]["text_matches"] is True
    assert receipt["baseline_comparison"]["main_marked_up.pdf"]["page_count_matches"] is True
    assert all(
        isinstance(path, Path)
        for path in provenance_calls[0]["consumed_paths"]
    )
    assert (build_root / "uq_emb_manuscript_replay_receipt.json").is_file()


def test_compile_manuscript_rejects_nonempty_build_root(tmp_path: Path) -> None:
    module = _load_module()
    build_root = tmp_path / "build"
    build_root.mkdir()
    (build_root / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    try:
        module._require_fresh_directory(build_root)
    except FileExistsError as exc:
        assert "non-empty" in str(exc)
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("Expected a non-empty build root to be rejected")


def test_compile_manuscript_rejects_build_root_inside_bundle(tmp_path: Path) -> None:
    module = _load_module()
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()

    with pytest.raises(ValueError, match="outside the frozen bundle root"):
        module.compile_manuscript(
            bundle_root=bundle_root,
            manifest_path=tmp_path / "manifest.json",
            build_root=bundle_root / "build",
            pdflatex="pdflatex",
            bibtex="bibtex",
        )

    assert not (bundle_root / "build").exists()


def test_log_audit_covers_natbib_references_and_rerun(tmp_path: Path) -> None:
    module = _load_module()
    log_path = tmp_path / "main.log"
    log_path.write_text(
        "Package natbib Warning: Citation `missing' undefined.\n"
        "LaTeX Warning: Reference `fig:missing' undefined.\n"
        "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.\n",
        encoding="utf-8",
    )

    failed = module._failed_log_lines(log_path)
    assert len(failed) == 2
    assert module._log_requires_rerun(log_path) is True
