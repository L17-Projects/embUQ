import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_runtime(tmp_path: Path) -> tuple[Path, Path]:
    runtime_root = tmp_path / "runtime"
    env_root = runtime_root / "env"
    bin_dir = env_root / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("python", "python3"):
        executable = bin_dir / name
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    (env_root / "env.sh").write_text(
        'export PATH="${MESOUQ_ENV_ROOT}/bin:${PATH}"\n',
        encoding="utf-8",
    )
    return runtime_root, env_root


def test_site_env_resolves_path_python_bin_before_executable_check(tmp_path: Path) -> None:
    runtime_root, env_root = _make_runtime(tmp_path)
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)

    script = f"""
set -euo pipefail
unset HPC_SITE MESOUQ_SITE MESOUQ_ENV_ROOT MESOUQ_ENV_SCRIPT MESOUQ_GV_ENV_SCRIPT PYTHONPATH
export MESOUQ_SITE_RUNTIME_ROOT={runtime_root}
export PYTHON_BIN=python3
source {REPO_ROOT / 'scripts/platforms/hpc/site_env.sh'}
mesouq_activate_site_env vega {repo_root}
printf '%s\n' "$PYTHON_BIN"
"""

    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=True)

    assert result.stdout.strip() == str(env_root / "bin" / "python3")


def test_site_env_defaults_plain_python_to_canonical_env_python(tmp_path: Path) -> None:
    runtime_root, env_root = _make_runtime(tmp_path)
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)

    script = f"""
set -euo pipefail
unset HPC_SITE MESOUQ_SITE MESOUQ_ENV_ROOT MESOUQ_ENV_SCRIPT MESOUQ_GV_ENV_SCRIPT PYTHONPATH
export MESOUQ_SITE_RUNTIME_ROOT={runtime_root}
export PYTHON_BIN=python
source {REPO_ROOT / 'scripts/platforms/hpc/site_env.sh'}
mesouq_activate_site_env vega {repo_root}
printf '%s\n' "$PYTHON_BIN"
"""

    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=True)

    assert result.stdout.strip() == str(env_root / "bin" / "python")
