from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_existing_package_import_behavior_is_preserved():
    import meso_uq

    assert meso_uq.__version__ == "0.1.0"
    assert meso_uq.__all__ == ["__version__"]
    assert not hasattr(meso_uq, "AgentFamily")


def test_public_api_exposes_narrow_contract_boundary():
    from meso_uq import public_api

    exported = set(public_api.__all__)
    assert "AgentFamily" in exported
    assert "ManifestMetadata" in exported
    assert "resolve_agent_modality" in exported
    assert public_api.resolve_agent_modality("gv", "torsion")[1].modality.value == "torsion"


def test_public_api_import_does_not_load_heavy_runtime_dependencies(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    code = """
import importlib
import sys
import meso_uq
import meso_uq.public_api
blocked = ["torch", "scipy", "matplotlib", "mpi4py", "mirheo", "korali", "pyro"]
loaded = [name for name in blocked if name in sys.modules]
assert meso_uq.__all__ == ["__version__"]
assert loaded == [], loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
