from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


HEAVY_OPTIONAL_MODULES = ("torch", "pyro", "matplotlib", "mpi4py", "mirheo", "korali", "slurm")


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "meso_uq",
        "meso_uq.structures",
        "meso_uq.structures.gv",
        "meso_uq.surrogate.catalogs",
        "meso_uq.surrogate.gv_catalog",
        "meso_uq.surrogate.emb_catalog",
        "meso_uq.agents",
        "meso_uq.agents.emb",
        "meso_uq.modalities",
        "meso_uq.public_api",
        "meso_uq.simulation",
    ),
)
def test_metadata_imports_do_not_load_heavy_optional_dependencies(module_name: str) -> None:
    code = f"""
import importlib
import sys

importlib.import_module({module_name!r})
loaded = [name for name in {HEAVY_OPTIONAL_MODULES!r} if name in sys.modules]
assert loaded == [], loaded
"""

    result = _run_import_probe(code)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "code",
    (
        """
import sys
import meso_uq.surrogate.gv_catalog
import meso_uq.surrogate.catalogs
loaded = [name for name in ('torch', 'pyro', 'matplotlib', 'mpi4py', 'mirheo', 'korali') if name in sys.modules]
assert loaded == [], loaded
""",
        """
import sys
from meso_uq.surrogate.catalogs import iter_surrogate_catalog_entries
entries = iter_surrogate_catalog_entries(include_experimental=True)
assert entries
import meso_uq.surrogate.gv_catalog
loaded = [name for name in ('torch', 'pyro', 'matplotlib', 'mpi4py', 'mirheo', 'korali') if name in sys.modules]
assert loaded == [], loaded
""",
        """
import sys
import meso_uq.structures
import meso_uq.structures.gv
loaded = [name for name in ('torch', 'pyro', 'matplotlib', 'mpi4py', 'mirheo', 'korali') if name in sys.modules]
assert loaded == [], loaded
""",
    ),
)
def test_cycle_sensitive_metadata_import_orders_remain_light(code: str) -> None:
    result = _run_import_probe(code)

    assert result.returncode == 0, result.stderr
