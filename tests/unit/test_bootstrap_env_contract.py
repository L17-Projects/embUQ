from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_env_import_smoke_uses_canonical_env_activation() -> None:
    text = (REPO_ROOT / "scripts/platforms/hpc/bootstrap_env.sh").read_text(encoding="utf-8")
    marker = "Validating meso_uq import through canonical env activation."

    assert marker in text
    smoke_block = text[text.index(marker) :]

    assert 'source "$ENV_SCRIPT"' in smoke_block
    assert "import meso_uq" in smoke_block
    assert smoke_block.index('source "$ENV_SCRIPT"') < smoke_block.index("import meso_uq")


def test_korali_skip_python_deps_keeps_korali_build_deps_enabled() -> None:
    text = (REPO_ROOT / "scripts/platforms/hpc/bootstrap_env.sh").read_text(encoding="utf-8")

    assert 'korali_args=(--site "$SITE" --python-bin "$env_python")' in text
    assert 'if [[ "$install_python_deps" -eq 1 ]]; then\n    korali_args+=(--skip-python-build-deps)' in text


def test_korali_bootstrap_installs_and_checks_mpi4py_build_dependency() -> None:
    text = (REPO_ROOT / "scripts/platforms/hpc/bootstrap_korali.sh").read_text(encoding="utf-8")

    assert '"mpi4py>=4.1.1"' in text
    assert "import mpi4py" in text
