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
