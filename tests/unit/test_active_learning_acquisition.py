from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import (
    ACQUISITION_MANIFEST_FILENAME,
    ACQUISITION_PLOT_FILENAME,
    ACQUISITION_PLOT_SIDECAR_FILENAME,
    ACQUISITION_REPORT_FILENAME,
    Candidate,
    score_acquisition_candidates,
    write_acquisition_artifacts,
)


def _png_has_signature(path: Path) -> bool:
    return path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", data[16:24])


def _png_chunks(path: Path):
    data = path.read_bytes()
    cursor = 8
    while cursor < len(data):
        length = struct.unpack(">I", data[cursor : cursor + 4])[0]
        chunk_type = data[cursor + 4 : cursor + 8]
        chunk_data = data[cursor + 8 : cursor + 8 + length]
        yield chunk_type, chunk_data
        cursor += length + 12


def _fallback_png_nonwhite_pixels(path: Path) -> int:
    data = path.read_bytes()
    width, height = _png_dimensions(path)
    assert data[24] == 8
    assert data[25] == 2
    idat = b"".join(chunk_data for chunk_type, chunk_data in _png_chunks(path) if chunk_type == b"IDAT")
    raw = zlib.decompress(idat)
    stride = width * 3
    nonwhite = 0
    cursor = 0
    for _row in range(height):
        assert raw[cursor] == 0
        row = raw[cursor + 1 : cursor + 1 + stride]
        for offset in range(0, len(row), 3):
            if row[offset : offset + 3] != b"\xff\xff\xff":
                nonwhite += 1
        cursor += stride + 1
    return nonwhite


def test_score_acquisition_candidates_ties_are_deterministic_by_id_and_stable_hash() -> None:
    candidates = (
        Candidate(candidate_id="zeta", parameters={"uncertainty": 1.1, "family": "gv", "gv_launch": {"tag": 1}}),
        Candidate(candidate_id="alpha", parameters={"uncertainty": 1.1, "family": "gv", "gv_launch": {"tag": 2}}),
        Candidate(candidate_id="beta", parameters={"uncertainty": 0.8}),
    )

    first = score_acquisition_candidates(candidates, policy="uncertainty")
    second = score_acquisition_candidates(candidates, policy="uncertainty")

    assert first.ranked_candidate_ids == ("alpha", "zeta", "beta")
    assert second.ranked_candidate_ids == first.ranked_candidate_ids
    assert first.scores[0].score == first.scores[1].score == 1.1


def test_score_acquisition_candidates_rejects_missing_fields_with_reasons() -> None:
    candidates = (
        Candidate(candidate_id="valid", parameters={"uncertainty": 0.6}),
        Candidate(candidate_id="missing", parameters={"notes": 1}, metadata={"notes": "none"}),
    )

    result = score_acquisition_candidates(candidates, policy="uncertainty")

    assert [item.candidate_id for item in result.scores] == ["valid"]
    assert result.rejected_candidate_ids == ("missing",)
    assert "missing" in result.rejection_reasons
    assert "uncertainty" in result.rejection_reasons["missing"][0]


def test_score_acquisition_candidates_rejects_missing_diversity_paths_with_reasons() -> None:
    candidates = (
        Candidate(candidate_id="valid", parameters={"features": {"x": 1.0, "y": 2.0}}),
        Candidate(candidate_id="missing-y", parameters={"features": {"x": 1.0}}),
    )

    result = score_acquisition_candidates(
        candidates,
        policy="diversity",
        diversity_paths=("features.x", "features.y"),
    )

    assert [item.candidate_id for item in result.scores] == ["valid"]
    assert result.rejected_candidate_ids == ("missing-y",)
    assert result.rejection_reasons["missing-y"] == ("diversity_paths.features.y",)


def test_score_acquisition_candidates_diversity_uses_nearest_neighbor_distance() -> None:
    candidates = (
        Candidate(candidate_id="duplicate", parameters={"features": {"x": 0.0}}),
        Candidate(candidate_id="novel", parameters={"features": {"x": 5.0}}),
    )
    reference_candidates = (
        Candidate(candidate_id="observed-duplicate", parameters={"features": {"x": 0.0}}),
        Candidate(candidate_id="far-observed", parameters={"features": {"x": 100.0}}),
    )

    result = score_acquisition_candidates(
        candidates,
        policy="diversity",
        diversity_paths=("features.x",),
        reference_candidates=reference_candidates,
    )

    scores = {item.candidate_id: item.score for item in result.scores}
    assert scores == {"novel": 5.0, "duplicate": 0.0}
    assert result.ranked_candidate_ids == ("novel", "duplicate")


def test_score_acquisition_candidates_rejects_bad_reference_diversity_paths() -> None:
    candidates = (
        Candidate(candidate_id="valid", parameters={"features": {"x": 2.0, "y": 3.0}}),
    )
    reference_candidates = (
        Candidate(candidate_id="observed-missing", parameters={"features": {"x": 0.0}}),
        Candidate(candidate_id="observed-invalid", parameters={"features": {"x": "bad", "y": 1.0}}),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"reference_candidates.*observed-missing.*diversity_paths\.features\.y"
            r".*observed-invalid.*diversity_paths\.features\.x"
        ),
    ):
        score_acquisition_candidates(
            candidates,
            policy="diversity",
            diversity_paths=("features.x", "features.y"),
            reference_candidates=reference_candidates,
        )


def test_score_acquisition_candidates_supports_family_neutral_gv_and_emb_payloads() -> None:
    candidates = (
        Candidate(
            candidate_id="gv-one",
            parameters={
                "family": "gv",
                "gv_launch": {"acquisition_uncertainty": 0.9},
            },
        ),
        Candidate(
            candidate_id="emb-one",
            parameters={
                "family": "emb",
                "emb_launch": {"acquisition_uncertainty": 0.7},
            },
        ),
        Candidate(
            candidate_id="raw-one",
            parameters={"acquisition_uncertainty": 0.5},
        ),
    )

    result = score_acquisition_candidates(candidates, policy="uncertainty")

    assert result.rejected_candidate_ids == ()
    assert [item.candidate_id for item in result.scores] == ["gv-one", "emb-one", "raw-one"]


def test_write_acquisition_artifacts_emits_manifest_plot_and_sidecar(tmp_path: Path) -> None:
    candidates = (
        Candidate(candidate_id="c1", parameters={"uncertainty": 1.0}),
        Candidate(candidate_id="c2", parameters={"uncertainty": 0.5}),
    )
    result = score_acquisition_candidates(candidates, policy="uncertainty")
    artifacts = write_acquisition_artifacts(
        output_root=tmp_path / "_runs" / "active_learning",
        run_id="acq-artifacts",
        iteration=3,
        result=result,
        include_plot=True,
    )

    assert artifacts.manifest_path == (
        tmp_path / "_runs" / "active_learning" / "acq-artifacts" / "iterations" / "iter_0003" / ACQUISITION_MANIFEST_FILENAME
    )
    assert artifacts.report_path == (
        tmp_path / "_runs" / "active_learning" / "acq-artifacts" / "iterations" / "iter_0003" / ACQUISITION_REPORT_FILENAME
    )
    assert artifacts.plot_path == (
        tmp_path / "_runs" / "active_learning" / "acq-artifacts" / "iterations" / "iter_0003" / ACQUISITION_PLOT_FILENAME
    )
    assert artifacts.plot_sidecar_path == (
        tmp_path / "_runs" / "active_learning" / "acq-artifacts" / "iterations" / "iter_0003" / ACQUISITION_PLOT_SIDECAR_FILENAME
    )
    assert artifacts.manifest_path.is_file()
    assert artifacts.report_path.is_file()
    assert artifacts.plot_path.is_file()
    assert artifacts.plot_sidecar_path.is_file()
    assert artifacts.plot_path.stat().st_size > 0
    assert _png_has_signature(artifacts.plot_path)
    assert _png_dimensions(artifacts.plot_path)[0] > 1
    assert _png_dimensions(artifacts.plot_path)[1] > 1

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    sidecar = json.loads(artifacts.plot_sidecar_path.read_text(encoding="utf-8"))
    assert manifest["candidate_count"] == 2
    assert manifest["scored_count"] == 2
    assert manifest["policy"] == "uncertainty"
    assert report["weights"] == {}
    assert report["family_distribution"] == {"unknown": 2}
    assert sidecar["plot"] == ACQUISITION_PLOT_FILENAME


def test_write_acquisition_artifacts_plot_fallback_preserves_score_shape(tmp_path: Path) -> None:
    candidates = (
        Candidate(candidate_id="c1", parameters={"uncertainty": 1.0}),
        Candidate(candidate_id="c2", parameters={"uncertainty": 0.25}),
        Candidate(candidate_id="c3", parameters={"uncertainty": 0.75}),
    )
    result = score_acquisition_candidates(candidates, policy="uncertainty")
    artifacts = write_acquisition_artifacts(
        output_root=tmp_path / "_runs" / "active_learning",
        run_id="acq-fallback-artifacts",
        iteration=0,
        result=result,
        include_plot=False,
    )

    assert _png_dimensions(artifacts.plot_path) == (360, 220)
    assert _fallback_png_nonwhite_pixels(artifacts.plot_path) > 1_000
    sidecar = json.loads(artifacts.plot_sidecar_path.read_text(encoding="utf-8"))
    assert [item["candidate_id"] for item in sidecar["scores"]] == ["c1", "c3", "c2"]


def test_active_learning_acquisition_module_import_does_not_load_matplotlib() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module('meso_uq.active_learning.acquisition'); "
                "print('matplotlib' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.stdout.strip() == "False"
