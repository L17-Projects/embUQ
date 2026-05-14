from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from meso_uq.config.aliases import (
    LegacyConfigPathAlias,
    list_legacy_config_path_aliases,
    resolve_legacy_config_path,
)


def _write_legacy_configs(repo_root: Path) -> None:
    roots = [
        repo_root / "inference" / "configs" / "production" / "inference_config_compression.yaml",
        repo_root / "inference" / "configs" / "validation" / "validation_config_compression.yaml",
        repo_root / "reduced" / "configs" / "production" / "reduced_config_compression.yaml",
        repo_root / "reduced" / "configs" / "validation" / "validation_config_compression.yaml",
        repo_root / "examples" / "configs" / "compression_full.yaml",
    ]
    for path in roots:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("emb_diameters: [2.1]\n", encoding="utf-8")


def test_list_legacy_config_path_aliases_includes_expected_roots(tmp_path: Path) -> None:
    _write_legacy_configs(tmp_path)
    aliases = list_legacy_config_path_aliases(project_root=tmp_path)
    assert {alias.legacy_root for alias in aliases} == {
        "inference/configs/production",
        "inference/configs/validation",
        "reduced/configs/production",
        "reduced/configs/validation",
        "examples/configs",
    }


def test_resolve_legacy_config_path_known_roots(tmp_path: Path) -> None:
    _write_legacy_configs(tmp_path)
    assert (
        resolve_legacy_config_path(
            project_root=tmp_path,
            legacy_path="inference/configs/production/inference_config_compression.yaml",
        ).canonical_path
        == tmp_path
        / "inference"
        / "configs"
        / "production"
        / "inference_config_compression.yaml"
    )
    assert (
        resolve_legacy_config_path(
            project_root=tmp_path,
            legacy_path="reduced/configs/validation",
        ).canonical_path
        == tmp_path / "reduced" / "configs" / "validation"
    )
    assert (
        resolve_legacy_config_path(
            project_root=tmp_path,
            legacy_path="examples/configs/compression_full.yaml",
        ).alias
        == LegacyConfigPathAlias(
            legacy_root="examples/configs",
            canonical_root="examples/configs",
            description="example configuration root (refreshed/archived during migration)",
            retirement="Keep while examples/configs refresh/archive decisions are pending.",
        )
    )


def test_resolve_legacy_config_path_rejects_unknown_alias(tmp_path: Path) -> None:
    _write_legacy_configs(tmp_path)
    with pytest.raises(ValueError, match="Unknown legacy config path"):
        resolve_legacy_config_path(
            project_root=tmp_path,
            legacy_path="legacy/configs/production/inference_config_compression.yaml",
        )


def test_resolve_legacy_config_path_rejects_path_traversal(tmp_path: Path) -> None:
    _write_legacy_configs(tmp_path)
    with pytest.raises(ValueError, match="path contains path traversal"):
        resolve_legacy_config_path(
            project_root=tmp_path,
            legacy_path="inference/configs/production/../validation/validation_config_compression.yaml",
        )


def test_resolve_legacy_config_path_rejects_private_absolute_path() -> None:
    with pytest.raises(ValueError, match="forbidden private root"):
        resolve_legacy_config_path(
            project_root=Path("/tmp/mesouq"),
            legacy_path="/ceph/hpc/home/eubrieucb/inference/configs/production/inference_config.yaml",
        )


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


def test_legacy_config_aliases_import_is_dependency_light() -> None:
    code = """
import sys
import meso_uq.config
import meso_uq.config.aliases

loaded = [
    name
    for name in ("torch", "pyro", "matplotlib", "mpi4py", "mirheo", "korali")
    if name in sys.modules
]
assert loaded == [], loaded
"""
    result = _run_import_probe(code)
    assert result.returncode == 0, result.stderr
