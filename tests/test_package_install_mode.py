from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

PACKAGE_INSTALL_PROBE = """\
import importlib
import pathlib
import sys


class _BoundaryBlocker:
    def __init__(self, blocked_roots):
        self._blocked_roots = set(blocked_roots)

    def find_spec(self, fullname, path=None, target=None):
        root_name = fullname.split(".", 1)[0]
        if root_name in self._blocked_roots:
            raise ModuleNotFoundError(root_name)
        return None


sys.meta_path.insert(0, _BoundaryBlocker(("agents", "artifacts", "configs", "core", "modalities")))

import meso_uq

package_root = pathlib.Path(meso_uq.__file__).resolve()
if not str(package_root).startswith(str(pathlib.Path(r"{src_root}").resolve())):
    raise RuntimeError("meso_uq imported from unexpected location: " + str(package_root))

modules = (
    "meso_uq",
    "meso_uq.public_api",
    "meso_uq.core",
    "meso_uq.agents",
    "meso_uq.modalities",
    "meso_uq.configs",
    "meso_uq.artifacts",
)

for module_name in modules:
    importlib.import_module(module_name)

print("installed-mode-import-ok")
"""


def test_public_entrypoints_import_with_src_only_and_non_repo_cwd() -> None:
    command = [sys.executable, "-c", PACKAGE_INSTALL_PROBE.format(src_root=str(SRC_ROOT))]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)

    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT.parent),
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "installed-mode-import-ok"
