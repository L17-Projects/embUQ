from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from meso_uq.hpc_paths import (
    default_runs_root,
    detect_hpc_site,
    ensure_canonical_output_root,
    make_run_tag,
)


def test_detect_hpc_site_prefers_explicit_env() -> None:
    assert detect_hpc_site(env={"HPC_SITE": "karolina"}) == "karolina"
    assert detect_hpc_site(env={"HPC_SITE": "vega"}) == "vega"


def test_detect_hpc_site_uses_hostname_heuristics() -> None:
    assert detect_hpc_site(env={}, hostname="login4.karolina.it4i.cz") == "karolina"
    assert detect_hpc_site(env={}, hostname="acn49") == "karolina"
    assert detect_hpc_site(env={}, hostname="vega-gpu-01") == "vega"


def test_detect_hpc_site_falls_back_to_vega() -> None:
    assert detect_hpc_site(env={}, hostname="unknown-host") == "vega"


def test_make_run_tag_format() -> None:
    ts = make_run_tag(datetime(2026, 4, 21, 12, 34, 56))
    assert ts == "20260421_123456"


def test_default_runs_root_layout() -> None:
    root = default_runs_root(
        Path("/tmp/repo"),
        "validation_matrix",
        site="karolina",
        run_tag="20260421_123456",
    )
    assert root == Path("/tmp/repo/_runs/karolina/validation_matrix/20260421_123456")


def test_default_runs_root_honors_runs_root_override(tmp_path) -> None:
    root = default_runs_root(
        Path("/tmp/repo"),
        "validation_matrix",
        site="karolina",
        run_tag="20260421_123456",
        env={"MESOUQ_RUNS_ROOT": str(tmp_path / "runs")},
    )

    assert root == (tmp_path / "runs" / "validation_matrix" / "20260421_123456").resolve()


def test_default_runs_root_rejects_unknown_site() -> None:
    with pytest.raises(ValueError, match="Unsupported site"):
        default_runs_root("/tmp/repo", "validation_matrix", site="other")


def test_ensure_canonical_output_root_requires_runs_or_paper_data_under_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="repository root"):
        ensure_canonical_output_root(repo, repo)

    bad_root = repo / "scripts" / "validation_matrix"
    with pytest.raises(ValueError, match="not canonical"):
        ensure_canonical_output_root(str(bad_root), repo)

    canonical_root = repo / "_runs" / "validation_matrix"
    assert ensure_canonical_output_root(str(canonical_root), repo) == canonical_root.resolve()
    paper_root = repo / "paper_data" / "camp1"
    assert ensure_canonical_output_root(str(paper_root), repo) == paper_root.resolve()


def test_ensure_canonical_output_root_rejects_absolute_noncanonical_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert ensure_canonical_output_root(str(tmp_path / "_runs" / "validation_matrix"), repo) == (
        tmp_path / "_runs" / "validation_matrix"
    )
    assert ensure_canonical_output_root(str(tmp_path / "mesouq" / "runs" / "validation_matrix"), repo) == (
        tmp_path / "mesouq" / "runs" / "validation_matrix"
    )
    with pytest.raises(ValueError, match="root"):
        ensure_canonical_output_root(str(tmp_path / "scratch" / "validation_matrix"), repo)
