from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    "uq_emb_forward_canary.sbatch",
    "uq_emb_hbi_replay.sbatch",
)


def _fake_sbatch(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    receipt = tmp_path / "sbatch_args.txt"
    executable = bin_dir / "sbatch"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$MESOUQ_SITE\" \"$REPO_ROOT\" \"$@\" > \"$SBATCH_RECEIPT\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir, receipt


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("site", ("karolina", "vega"))
def test_neutral_uq_emb_launcher_submits_selected_site_wrapper(
    tmp_path: Path,
    launcher: str,
    site: str,
) -> None:
    bin_dir, receipt = _fake_sbatch(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO_ROOT": str(REPO_ROOT),
        "SBATCH_RECEIPT": str(receipt),
    }
    env.pop("MESOUQ_SITE", None)
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/platforms/hpc/sbatch" / launcher),
            "--site",
            site,
            "--partition=diagnostic",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = receipt.read_text(encoding="utf-8").splitlines()
    assert lines == [
        site,
        str(REPO_ROOT),
        "--partition=diagnostic",
        str(REPO_ROOT / f"scripts/platforms/{site}/sbatch/{launcher}"),
    ]


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_neutral_uq_emb_launcher_exports_defaulted_repo_root(
    tmp_path: Path,
    launcher: str,
) -> None:
    bin_dir, receipt = _fake_sbatch(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SBATCH_RECEIPT": str(receipt),
    }
    env.pop("REPO_ROOT", None)
    env.pop("MESOUQ_SITE", None)
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/platforms/hpc/sbatch" / launcher),
            "--site",
            "karolina",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = receipt.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "karolina",
        str(REPO_ROOT),
        str(REPO_ROOT / f"scripts/platforms/karolina/sbatch/{launcher}"),
    ]


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_neutral_uq_emb_launcher_requires_explicit_site(
    tmp_path: Path,
    launcher: str,
) -> None:
    bin_dir, receipt = _fake_sbatch(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO_ROOT": str(REPO_ROOT),
        "SBATCH_RECEIPT": str(receipt),
    }
    env.pop("MESOUQ_SITE", None)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/platforms/hpc/sbatch" / launcher)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Missing site selector" in result.stderr
    assert not receipt.exists()


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_neutral_uq_emb_launcher_rejects_nested_slurm_submission(
    tmp_path: Path,
    launcher: str,
) -> None:
    bin_dir, receipt = _fake_sbatch(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO_ROOT": str(REPO_ROOT),
        "SBATCH_RECEIPT": str(receipt),
        "SLURM_JOB_ID": "12345",
    }
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/platforms/hpc/sbatch" / launcher),
            "--site",
            "karolina",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must run on a login node" in result.stderr
    assert not receipt.exists()
