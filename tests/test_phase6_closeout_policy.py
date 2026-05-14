from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEANUP_RECORD = REPO_ROOT / "docs" / "PHASE6_GENERATED_CLEANUP_RECORD.md"

APPROVED_GENERATED_OUTPUTS = (
    "_out",
    "_runs",
    "_init_compression_20260514",
    "out_hierarchical",
    "_ci",
    "logs/mesouq-validation-matrix-12345.out",
    "runtime/gv/staged/source_manifest.json",
    "mesouq-validation-matrix-12345.out",
    "mesouq-validation-matrix-12345.err",
    "validation-matrix-12345.out",
    "validation-matrix-12345.err",
)

TRACKED_GENERATED_GLOBS = (
    "_out",
    "_out/**",
    "_runs",
    "_runs/**",
    "_init_compression_*",
    "out_hierarchical",
    "out_hierarchical/**",
    "_ci",
    "_ci/**",
    "runtime",
    "runtime/**",
)

REQUIRED_RECORD_TEXT = (
    "_out/",
    "_runs/",
    "_init_compression_*",
    "out_hierarchical/",
    "_ci/",
    "root Slurm scheduler logs",
    "root `runtime/` staging output",
    "ignored by repository policy",
    "untracked local artifacts",
    "extern/korali/",
    "curated surrogate manifests",
    "source, configuration, documentation, and test paths",
    "paper-specific source scripts",
    "compatibility-shim documentation",
    "none of the approved generated roots are tracked",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_phase6_cleanup_record_documents_approved_scope() -> None:
    record = CLEANUP_RECORD.read_text(encoding="utf-8")

    for text in REQUIRED_RECORD_TEXT:
        assert text in record


def test_phase6_approved_generated_outputs_are_ignored() -> None:
    for output in APPROVED_GENERATED_OUTPUTS:
        result = _git("check-ignore", "--quiet", output)

        assert result.returncode == 0, f"{output} is not ignored: {result.stderr}"


def test_phase6_generated_roots_are_not_tracked() -> None:
    result = _git("ls-files", "--", *TRACKED_GENERATED_GLOBS)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
