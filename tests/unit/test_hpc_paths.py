from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from meso_uq.hpc_paths import default_runs_root, detect_hpc_site, make_run_tag


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
