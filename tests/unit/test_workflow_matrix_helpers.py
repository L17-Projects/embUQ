"""Tests for pure helper functions in scripts/platforms/hpc/run_workflow_matrix.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_SCRIPT = REPO / "scripts" / "platforms" / "hpc" / "run_workflow_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_workflow_matrix", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_workflow_matrix"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------

def test_resolve_path_absolute_unchanged(mod):
    p = mod._resolve_path("/tmp/foo")
    assert p == Path("/tmp/foo")


def test_resolve_path_relative_anchors_to_repo_root(mod):
    p = mod._resolve_path("some/relative/path")
    assert p == (REPO / "some" / "relative" / "path").resolve()


# ---------------------------------------------------------------------------
# _selection_output_root / _selection_logs_root / _selection_summary_path
# ---------------------------------------------------------------------------

def test_selection_output_root(mod):
    from meso_uq.vega_workflows import VegaWorkflowSelection
    sel = VegaWorkflowSelection("indentation", "reduced-model", "production")
    root = Path("/tmp/matrix")
    p = mod._selection_output_root(root, sel)
    assert p == root / "runs" / "indentation" / "reduced-model" / "production"


def test_selection_logs_root(mod):
    from meso_uq.vega_workflows import VegaWorkflowSelection
    sel = VegaWorkflowSelection("compression", "full-model", "production")
    root = Path("/tmp/matrix")
    p = mod._selection_logs_root(root, sel)
    assert p == root / "logs" / "compression" / "full-model" / "production"


def test_selection_summary_path(mod):
    from meso_uq.vega_workflows import VegaWorkflowSelection
    sel = VegaWorkflowSelection("indentation", "full-model", "production")
    root = Path("/tmp/matrix")
    p = mod._selection_summary_path(root, sel)
    assert p.parent == root / "summaries"
    assert p.suffix == ".json"
