from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_native_cuda_phase2_baseline_doc_tracks_current_source_and_strategy() -> None:
    doc = (REPO_ROOT / "docs" / "NATIVE_CUDA_PHASE2_BASELINE.md").read_text(
        encoding="utf-8"
    )
    status_doc = (REPO_ROOT / "docs" / "PHASE2_BACKEND_STATUS.md").read_text(
        encoding="utf-8"
    )
    docs_index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    psi_source = (
        REPO_ROOT / "extern" / "korali" / "source" / "modules" / "problem" / "hierarchical" / "psi" / "psi.cpp"
    ).read_text(encoding="utf-8")
    meson_build = (REPO_ROOT / "extern" / "korali" / "meson.build").read_text(
        encoding="utf-8"
    )
    normalized_doc = " ".join(doc.split())

    for phrase in (
        "not a production-support claim",
        "CPU-MPI remains the maintained fallback",
        "Meson-built CUDA object/module delivery path",
        "current implementation is NVRTC-based",
        "HUQ_PSI_NATIVE_CUDA_PROFILE_JSONL",
        "Validation evidence must state which delivery mode was actually used",
    ):
        assert phrase in normalized_doc

    for source_marker, doc_marker in {
        "kPsiNativeCudaKernelSource": "kPsiNativeCudaKernelSource",
        "nvrtcCreateProgram": "NVRTC",
        "nvrtcCompileProgram": "NVRTC",
        "cuModuleLoadData": "cuModuleLoadData",
        "HUQ_PSI_NATIVE_CUDA_PROFILE_JSONL": "HUQ_PSI_NATIVE_CUDA_PROFILE_JSONL",
    }.items():
        assert source_marker in psi_source
        assert doc_marker in doc

    for build_marker in ("native_cuda_batch", "_KORALI_USE_CUDA_BATCH", "nvrtc.h"):
        assert build_marker in meson_build
        assert build_marker in doc

    assert "NATIVE_CUDA_PHASE2_BASELINE.md" in status_doc
    assert "NATIVE_CUDA_PHASE2_BASELINE.md" in docs_index
