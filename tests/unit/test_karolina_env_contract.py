from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_karolina_env_sets_scratch_safe_provenance_root_default() -> None:
    env_script = REPO_ROOT / "scripts" / "platforms" / "karolina" / "env_karolina.sh"
    contents = env_script.read_text(encoding="utf-8")

    assert 'MESOUQ_PROVENANCE_ROOT="${MESOUQ_PROVENANCE_ROOT:-${MESOUQ_SCRATCH_ROOT}/provenance}"' in contents


def test_karolina_env_requires_explicit_site_runtime_root() -> None:
    env_script = REPO_ROOT / "scripts" / "platforms" / "karolina" / "env_karolina.sh"
    contents = env_script.read_text(encoding="utf-8")

    retired_default = 'MESOUQ_SITE_RUNTIME_ROOT="${MESOUQ_SITE_RUNTIME_ROOT:-${MESOUQ_SCRATCH_ROOT}' + '/runtime}"'
    assert retired_default not in contents
    assert 'MESOUQ_SITE_RUNTIME_ROOT must be set before sourcing env_karolina.sh' in contents
    assert 'export MESOUQ_SITE_RUNTIME_ROOT' in contents
