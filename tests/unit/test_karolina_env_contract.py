from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_karolina_env_sets_scratch_safe_provenance_root_default() -> None:
    env_script = REPO_ROOT / "scripts" / "platforms" / "karolina" / "env_karolina.sh"
    contents = env_script.read_text(encoding="utf-8")

    assert 'MESOUQ_PROVENANCE_ROOT="${MESOUQ_PROVENANCE_ROOT:-${MESOUQ_SCRATCH_ROOT}/provenance}"' in contents
