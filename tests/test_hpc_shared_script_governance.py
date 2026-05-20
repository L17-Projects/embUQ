from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = REPO_ROOT / "scripts" / "platforms" / "hpc"
FORBIDDEN_SITE_ALIAS_PATTERNS = (
    "scripts/vega",
    "scripts/karolina",
    '"scripts" / "vega"',
    '"scripts" / "karolina"',
    "'scripts' / 'vega'",
    "'scripts' / 'karolina'",
)


def test_shared_hpc_entrypoints_do_not_reference_site_alias_roots() -> None:
    offenders: list[str] = []
    for path in sorted(HPC_ROOT.rglob("*")):
        if path.suffix not in {".py", ".sh", ".sbatch"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_SITE_ALIAS_PATTERNS:
            if pattern in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {pattern!r}")

    assert not offenders, "Shared HPC entrypoints must not route through site alias roots.\n" + "\n".join(offenders)
