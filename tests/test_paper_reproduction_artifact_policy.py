from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = REPO_ROOT / "docs" / "PAPER_REPRODUCTION_ARTIFACT_POLICY.md"
PAPERS_README = REPO_ROOT / "papers" / "README.md"

GENERATED_PAPER_PATH_PARTS = {
    "checkpoints",
    "figures",
    "logs",
    "map_outputs",
    "mirheo_runs",
    "posterior_samples",
    "rendered_reports",
    "run_logs",
    "samples",
    "scratch",
}

GENERATED_PAPER_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".hdf5",
    ".nc",
    ".npy",
    ".npz",
    ".pth",
    ".pt",
}


def _tracked_files_under_papers() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "papers"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def test_paper_reproduction_policy_documents_external_artifact_boundary() -> None:
    policy = POLICY.read_text(encoding="utf-8")
    readme = PAPERS_README.read_text(encoding="utf-8")

    normalized_policy = policy.lower()
    for required in (
        "PAPER_DATA_ROOT",
        "manifest",
        "posterior samples",
        "checkpoints",
        "generated figures",
        "source scripts",
        "tiny fixtures",
    ):
        assert required.lower() in normalized_policy

    assert "PAPER_DATA_ROOT" in readme
    assert "papers/huq_emb" in readme


def test_generated_paper_payloads_are_not_tracked_in_papers_tree() -> None:
    offenders: list[str] = []

    for path in _tracked_files_under_papers():
        parts = set(path.parts)
        if parts & GENERATED_PAPER_PATH_PARTS:
            offenders.append(str(path))
            continue
        if path.suffix.lower() in GENERATED_PAPER_SUFFIXES:
            offenders.append(str(path))

    assert not offenders, (
        "Generated paper payloads must live under PAPER_DATA_ROOT or artifact "
        f"manifests, not git: {offenders}"
    )
