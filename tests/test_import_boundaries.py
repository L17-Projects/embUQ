from __future__ import annotations

import os
import subprocess
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_IMPORT_PROBE = """\
import pathlib
import sys


class _BoundaryBlocker:
    def __init__(self, blocked_roots):
        self._blocked_roots = set(blocked_roots)

    def find_spec(self, fullname, path=None, target=None):
        root_name = fullname.split(\".\", 1)[0]
        if root_name in self._blocked_roots:
            raise ModuleNotFoundError(root_name)
        return None


sys.meta_path.insert(0, _BoundaryBlocker((\"korali\", \"mpi4py\", \"mirheo\", \"pyro\")))

import meso_uq

package_root = pathlib.Path(meso_uq.__file__).resolve()
if not str(package_root).startswith(str(pathlib.Path(r\"{src_root}\").resolve())):
    raise RuntimeError("meso_uq imported from unexpected location: " + str(package_root))

print("import-root=" + str(package_root))
"""


def test_mesouq_import_works_from_non_repo_cwd_without_optional_hpc_modules() -> None:
    command = [
        sys.executable,
        "-c",
        PACKAGE_IMPORT_PROBE.format(src_root=str(SRC_ROOT)),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    env.pop("PYTHONHOME", None)

    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().startswith("import-root=")
