from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_AMBIENT_PLATFORM_ENV = (
    "HPC_SITE",
    "MESOUQ_SITE",
    "MESOUQ_PROJECT_ID",
    "MESOUQ_RUNS_ROOT",
    "MESOUQ_SCRATCH_ROOT",
    "MESOUQ_SITE_RUNTIME_ROOT",
    "MESOUQ_RUNTIME_ROOT",
    "MESOUQ_PROVENANCE_ROOT",
    "MESOUQ_GV_ENV_SCRIPT",
    "MESOUQ_GV_VENV_ROOT",
    "MESOUQ_GV_VENV_SITE_PACKAGES",
    "MESOUQ_KAROLINA_ROOT",
)


@pytest.fixture(autouse=True)
def clear_ambient_platform_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent from a sourced Karolina/Vega shell."""

    for name in _AMBIENT_PLATFORM_ENV:
        monkeypatch.delenv(name, raising=False)
